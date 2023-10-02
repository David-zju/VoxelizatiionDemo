#include "geom.h"

#include <set>
#include <thread>
#include "deps/lodepng/lodepng.h"

auto add_midpoint(vector<double> &arr) {
    vector<double> ret;
    for (int i = 0, n = arr.size(); i < n - 1; i ++) {
        ret.push_back(arr[i]);
        ret.push_back((arr[i] + arr[i + 1]) / 2);
    }
    ret.push_back(arr.back());
    return ret;
}

__global__ void kernel_get_faces_in_cells(
    buffer<double3> verts, buffer<int4> faces,
    buffer<double> xs, buffer<double> ys, int dir,
    buffer<int> len, buffer<int4> out) {
    for (int n = cuIdx(x); n < faces.len; n += cuDim(x)) {
        auto face = faces[n];
        auto a = verts[face.x], b = verts[face.y], c = verts[face.z];
        auto p0 = rotate(fmin(a, b, c), dir), p1 = rotate(fmax(a, b, c), dir);
        for (int i = 0, nx = xs.len; i < nx - 1; i ++) {
            auto x0 = xs[i], x1 = xs[i + 1];
            for (int j = 0, ny = ys.len; j < ny - 1; j ++) {
                auto y0 = ys[j], y1 = ys[j + 1];
                if (p0.x < x1 && p0.y < y1 && p1.x > x0 && p1.y > y0) {
                    auto next = atomicAdd(len.ptr + i + j * nx, 1);
                    if (next < out.len) {
                        out[next] = face;
                    }
                }
            }
        }
    }
}

struct face_groups_t {
    int dir;
    double tol;
    vector<double> xs, ys;
    vector<int> offset;
    vector<int4> faces;
};
auto get_faces_in_cells(device_vector<double3> &verts, device_vector<int4> &faces, grid_t &grid, int dir) {
    auto &xv = dir == 0 ? grid.ys : dir == 1 ? grid.zs : grid.xs;
    auto &yv = dir == 0 ? grid.zs : dir == 1 ? grid.xs : grid.ys;
    device_vector xs(xv), ys(yv);

    double tol = 1;
    for (int i = 0; i + 1 < xv.size(); i ++) tol = min(xv[i + 1] - xv[i], tol);
    for (int i = 0; i + 1 < yv.size(); i ++) tol = min(yv[i + 1] - yv[i], tol);
    tol = tol * 1e-6;

    face_groups_t ret { dir, tol, xv, yv };
    ret.offset.resize(xs.len * ys.len + 1);
    device_vector<int> offset(ret.offset);
    kernel_get_faces_in_cells CU_DIM(1024, 128) (verts, faces, xs, ys, dir, offset, { });
    CUDA_ASSERT(cudaGetLastError());

    offset.copy_to(ret.offset);
    exclusive_scan(ret.offset.begin(), ret.offset.end(), ret.offset.begin(), 0);
    device_vector<int4> out(ret.offset.back());

    offset.copy_from(ret.offset);
    kernel_get_faces_in_cells CU_DIM(1024, 128) (verts, faces, xs, ys, dir, offset, out);
    CUDA_ASSERT(cudaGetLastError());

    out.copy_to(ret.faces);
    return ret;
}

__host__ __device__ __forceinline__ auto lerp(double a, double b, double f) {
    return a + (b - a) * f;
}
__host__ __device__ __forceinline__ auto interp(double2 a, double2 b, double x) {
    auto f = (x - a.x) / (b.x - a.x);
    return lerp(a.y, b.y, f);
}
__host__ __device__ __forceinline__ auto interp(double3 a, double3 b, double x) {
    auto f = (x - a.x) / (b.x - a.x);
    return double2 { lerp(a.y, b.y, f), lerp(a.z, b.z, f) };
}
__host__ __device__ __forceinline__ auto ordered(double x0, double x, double x1) {
    return x0 != x1 && ((x0 <= x && x <= x1) || (x1 <= x && x <= x0));
}
__host__ __device__ __forceinline__ auto point_in_triangle(double2 p, double3 A, double3 B, double3 C, int dir, double tol) {
    double2 u = { 0 }, v = { 0 };
    auto a = rotate(A, dir), b = rotate(B, dir), c = rotate(C, dir);
    auto AB = ordered(a.x, p.x, b.x),
         BC = ordered(b.x, p.x, c.x),
         CA = ordered(c.x, p.x, a.x);
    if (CA && AB) {
        u = interp(a, b, p.x); v = interp(a, c, p.x);
    } else if (AB && BC) {
        u = interp(b, c, p.x); v = interp(b, a, p.x);
    } else if (BC && CA) {
        u = interp(c, a, p.x); v = interp(c, b, p.x);
    }
    // Note: ensure watertight intersections
    u.y = floor(u.y / tol) * tol;
    v.y = floor(v.y / tol) * tol;
    p.y = floor(p.y / tol) * tol + tol / 2;
    if (ordered(u.x, p.y, v.x)) {
        return interp(u, v, p.y);
    }
    return DBL_MAX;
}

struct cast_joint_t {
    unsigned short geom;
    float pos;
};
auto operator<(const cast_joint_t &a, const cast_joint_t &b) {
    return a.geom == b.geom ? a.pos < b.pos : a.geom < b.geom;
}
__global__ void kernel_cast_in_cells(
    buffer<double3> verts, buffer<int4> faces,
    buffer<double> xs, buffer<double> ys, int dir,
    buffer<int> offset, double ext, double tol,
    buffer<int> len, buffer<cast_joint_t> out) {
    int u = blockIdx.x, v = blockIdx.y, w = u + gridDim.x * v,
        begin = offset[w], end = offset[w + 1];
    if (u >= xs.len - 1 || v >= ys.len - 1) {
        return;
    }
    int2 dim = { (int) blockDim.x * 2, (int) blockDim.y * 2 };
    #pragma unroll
    for (int i0 = 0; i0 < 2; i0 ++) for (int j0 = 0; j0 < 2; j0 ++) {
        int i = threadIdx.x * 2 + i0, j = threadIdx.y * 2 + j0,
            k = i + u * dim.x + (j + v * dim.y) * gridDim.x * dim.x;
        double2 p = {
            lerp(xs[u], xs[u + 1], lerp(ext, 1 - ext, 1. * i / (dim.x - 1))),
            lerp(ys[v], ys[v + 1], lerp(ext, 1 - ext, 1. * j / (dim.y - 1))),
        };
        for (int n = begin; n < end; n ++) {
            auto face = faces[n];
            auto pos = point_in_triangle(p, verts[face.x], verts[face.y], verts[face.z], dir, tol);
            if (pos != DBL_MAX) {
                auto next = atomicAdd(len.ptr + k, 1);
                if (next < out.len) {
                    out[next] = { (unsigned short) face.w, (float) pos };
                }
            }
        }
    }
}

struct cast_dexel_t {
    int dir;
    double ext;
    int2 dim;
    vector<double> xs, ys;
    vector<int> offset;
    vector<cast_joint_t> joints;
    auto sort_joints(int num = thread::hardware_concurrency()) {
        vector<thread> pool;
        for (int i = 0, j = num; i < j; i ++) {
            pool.push_back(thread([this](int i, int j) {
                auto ptr = joints.data();
                for (int m = i, n = offset.size(); m < n - 1; m += j) {
                    sort(ptr + offset[m], ptr + offset[m + 1]);
                }
            }, i, j));
        }
        for (auto &item : pool) {
            item.join();
        }
    }
    auto get_verts(int mode) {
        vector<double3> verts;
        for (int u = 0; u + 1 < xs.size(); u ++) for (int v = 0; v + 1 < ys.size(); v ++) {
            int w = u + xs.size() * v;
            for (int i = 0; i < dim.x; i ++) for (int j = 0; j < dim.y; j ++) {
                double2 p = {
                    lerp(xs[u], xs[u + 1], lerp(ext, 1 - ext, 1. * i / (dim.x - 1))),
                    lerp(ys[v], ys[v + 1], lerp(ext, 1 - ext, 1. * j / (dim.y - 1))),
                };
                int k = i + u * dim.x + (j + v * dim.y) * xs.size() * dim.x,
                    begin = offset[k], end = offset[k + 1];
                if (mode == 0) {
                    for (int q = begin; q < end; q ++) {
                        auto &a = joints[q];
                        verts.push_back(revert(double3 { p.x, p.y, a.pos }, dir));
                    }
                } else {
                    for (int q = begin; q < end - 1; q ++) {
                        auto &a = joints[q], &b = joints[q + 1];
                        if (a.geom == b.geom) {
                            auto dist = (b.pos - a.pos) * ext;
                            verts.push_back(revert(double3 { p.x, p.y, a.pos + dist }, dir));
                            verts.push_back(revert(double3 { p.x, p.y, b.pos - dist }, dir));
                            q ++;
                        }
                    }
                }
            }
        }
        return verts;
    }
    auto dump_gltf(string file, int mode) {
        auto verts = get_verts(mode);
        ::dump_gltf(verts, file, mode);
    }
};
struct device_group {
    int dir;
    double tol;
    device_vector<double> xs, ys;
    device_vector<int> offset;
    device_vector<int4> faces;
    device_group(face_groups_t &group) :
        dir(group.dir), tol(group.tol),
        xs(group.xs), ys(group.ys),
        offset(group.offset), faces(group.faces) {
    }
    auto cast(device_vector<double3> &verts, int2 dim, double ext = 1e-3) {
        dim3 gridDim((int) xs.len, (int) ys.len, 1),
             blockDim(dim.x / 2, dim.y / 2, 1);

        cast_dexel_t ret { dir, ext, dim, xs.copy(), ys.copy() };
        ret.offset.resize(xs.len * ys.len * dim.x * dim.y + 1);
        device_vector len(ret.offset);
        kernel_cast_in_cells CU_DIM(gridDim, blockDim) (verts, faces, xs, ys, dir, offset, ext, tol, len, { });
        CUDA_ASSERT(cudaGetLastError());

        len.copy_to(ret.offset);
        exclusive_scan(ret.offset.begin(), ret.offset.end(), ret.offset.begin(), 0);
        device_vector<cast_joint_t> out(ret.offset.back());

        len.copy_from(ret.offset);
        kernel_cast_in_cells CU_DIM(gridDim, blockDim) (verts, faces, xs, ys, dir, offset, ext, tol, len, out);
        CUDA_ASSERT(cudaGetLastError());

        out.copy_to(ret.joints);
        ret.sort_joints();
        return ret;
    }
};

struct pixel_t {
    unsigned short geom;
};

template <typename T>
__global__ void kernel_reset(buffer<T> arr, T val) {
    for (int i = cuIdx(x); i < arr.len; i += cuDim(x)) {
        arr[i] = val;
    }
}

__global__ void kernel_render_dexel(
        int start, int width, double ext,
        buffer<double> points,
        buffer<int> offset, buffer<cast_joint_t> joints,
        buffer<pixel_t> out) {
    int j = blockIdx.x, w = start + j,
        begin = offset[w], end = offset[w + 1],
        ns = width / points.len;
    for (int i = threadIdx.x, n = ns * (points.len - 1); i < n; i += blockDim.x) {
        auto u = i / ns, v = i % ns;
        auto pos = lerp(points[u], points[u + 1], lerp(ext, 1 - ext, 1. * v / (ns - 1)));
        for (int q = begin; q < end - 1; q ++) {
            auto &a = joints[q], &b = joints[q + 1];
            if (a.geom == b.geom) {
                if (ordered(a.pos, pos, b.pos)) {
                    out[i + j * width] = { a.geom };
                }
                q ++;
            }
        }
    }
}
auto dump_png(vector<pixel_t> &img, int width, int height, int grid, string file, map<unsigned short, int3> &colors) {
    vector<unsigned char> buf(width * height * 4);
    for (int i = 0; i < width; i ++) {
        for (int j = 0; j < height; j ++) {
            auto k = i + j * width;
            auto p = buf.data() + k * 4;
            auto g = img[k].geom;
            if (!colors.count(g)) {
                colors[g] = int3 { rand(), rand(), rand() };
            }
            auto c = colors[g];
            auto m =
                i % grid == 0 || i % grid == grid - 1 ||
                j % grid == 0 || j % grid == grid - 1;
            p[0] = c.x;
            p[1] = c.y;
            p[2] = m ? 255 : c.z;
            p[3] = 255;
        }
    }
    lodepng::encode(file, buf, width, height);
}

int main(int argc, char *argv[]) {
    map<string, string> args;
    map<string, vector<string>> args_list;
    args["dump-cast-mode"] = "pointcloud";
    args["dump-render-axis"] = "xyz";
    args["ext"] = "0.001";
    for (int i = 0; i < argc; i ++) {
        if (!strncmp(argv[i], "--", 2)) {
            auto key = argv[i] + 2;
            auto val = "on";
            if (i + 1 < argc && strncmp(argv[i + 1], "--", 2)) {
                val = argv[i + 1];
                i ++;
            }
            args[key] = val;
            args_list[key].push_back(val);
        }
    }

    grid_t grid;
    if (args.count("load-grid")) {
        auto load = args["load-grid"];
        if (load.substr(load.size() - 5) == ".json") {
            grid.from_json(load);
        } else {
            fprintf(stderr, "FATAL: don't know how to load %s\n", load.c_str());
            exit(-1);
        }
    }
    for (auto _ : args_list["grid-add-midpoint"]) {
        grid.xs = add_midpoint(grid.xs);
        grid.ys = add_midpoint(grid.ys);
        grid.zs = add_midpoint(grid.zs);
    }
    auto nx = grid.xs.size(), ny = grid.ys.size(), nz = grid.zs.size();
    printf("INFO: loaded %zu x %zu x %zu (%zu) grids\n", nx, ny, nz, nx * ny * nz);

    mesh_t mesh;
    int geom_start = 0;
    for (auto load : args_list["load-mesh"]) {
        mesh_t loaded;
        auto vert_start = (int) mesh.verts.size();
        auto geom_max = 0;
        if (load.substr(load.size() - 4) == ".obj") {
            loaded.from_obj(load);
        } else {
            fprintf(stderr, "FATAL: don't know how to load %s\n", load.c_str());
            exit(-1);
        }
        for (auto &vert : loaded.verts) {
            mesh.verts.push_back(vert);
        }
        for (auto &face : loaded.faces) {
            mesh.faces.push_back({
                face.x + vert_start,
                face.y + vert_start,
                face.z + vert_start,
                face.w + geom_start,
            });
            geom_max = max(geom_start, face.w);
        }
        geom_start = geom_max + 1;
    }
    if (args.count("mesh-geometry")) {
        map<int, vector<int4>> mesh_map;
        for (auto face : mesh.faces) {
            mesh_map[face.w].push_back(face);
        }
        for (auto &[geom, faces] : mesh_map) {
            printf("INFO: got geom %d with %zu faces\n", geom, faces.size());
        }
        vector<int> geoms;
        string geom_list;
        for (auto list : args_list["mesh-geometry"]) {
            geom_list += list;
            replace(list.begin(), list.end(), ',', ' ');
            int val;
            stringstream ss(list);
            while (ss >> val) geoms.push_back(val);
        }
        printf("WARN: filter mesh only to show geom %s\n", geom_list.c_str());
        mesh.faces.resize(0);
        for (auto geom : geoms) {
            for (auto face : mesh_map[geom]) {
                mesh.faces.push_back(face);
            }
        }
    }
    printf("INFO: loaded %zu faces with %zu vertices\n", mesh.faces.size(), mesh.verts.size());

    int dump_mode = 0;
    if (args["dump-cast-mode"] == "pointcloud") {
        dump_mode = 0;
    } else if (args["dump-cast-mode"] == "dexel") {
        dump_mode = 1;
    } else {
        fprintf(stderr, "WARN: don't know how to dump %s, using pointcloud\n", args["dump-cast-mode"].c_str());
    }

    int2 cast_pixels = { 2, 2 };
    if (args.count("cast-pixels")) {
        auto pixels = args["cast-pixels"];
        replace(pixels.begin(), pixels.end(), ',', ' ');
        if (pixels.find(' ') == string::npos) {
            pixels = pixels + " " + pixels;
        }
        stringstream ss(pixels);
        ss >> cast_pixels.x >> cast_pixels.y;
    }
    if (cast_pixels.x <= 1 || cast_pixels.y <= 1 || cast_pixels.x % 2 || cast_pixels.y % 2) {
        fprintf(stderr, "FATAL: pixel size should be larger than 2 x 2 and even. Got %d x %d\n", cast_pixels.x, cast_pixels.y);
        exit(-1);
    }
    if (cast_pixels.x * cast_pixels.y > 4096) {
        fprintf(stderr, "FATAL: pixel.x * pixel.y should be less than 4096 due to CUDA limits. Got %d x %d\n", cast_pixels.x, cast_pixels.y);
        exit(-1);
    }
    printf("INFO: using pixel size %d x %d\n", cast_pixels.x, cast_pixels.y);

    auto all_start = clock_now();
    device_vector verts(mesh.verts);
    device_vector faces(mesh.faces);
    cast_dexel_t dexels[3];

    auto dump_cast = args["dump-cast"];
    auto ext = stod(args["ext"]);
    vector<double3> cast_points;
    for (int dir = 0; dir < 3; dir ++) {
        auto axis = ("xyz")[dir];

        auto group_start = clock_now();
        auto face_groups = get_faces_in_cells(verts, faces, grid, dir);
        printf("PERF: on %c group %zu in %f s\n", axis, face_groups.faces.size(), seconds_since(group_start));

        auto cast_start = clock_now();
        auto joint_count = 0;
        device_group groups(face_groups);
        auto &dexel = dexels[dir] = groups.cast(verts, cast_pixels, ext);
        joint_count += dexel.joints.size();
        if (dump_cast.size()) for (auto &v : dexel.get_verts(dump_mode)) {
            cast_points.push_back(v);
        }
        if (cast_pixels.x != cast_pixels.y) {
            auto dexel = groups.cast(verts, { cast_pixels.y, cast_pixels.x });
            joint_count += dexel.joints.size();
            if (dump_cast.size()) for (auto &v : dexel.get_verts(dump_mode)) {
                cast_points.push_back(v);
            }
        }
        printf("PERF: on %c cast %d in %f s\n", axis, joint_count, seconds_since(cast_start));
    }
    if (dump_cast.size()) {
        if (cast_points.size()) {
            if (dump_cast.substr(dump_cast.size() - 5) == ".gltf") {
                printf("INFO: saved to %s\n", dump_cast.c_str());
                dump_gltf(cast_points, dump_cast, dump_mode);
            } else {
                fprintf(stderr, "WARN: don't know how to dump %s\n", dump_cast.c_str());
            }
        } else {
            printf("WARN: nothing to dump\n");
        }
    }
    printf("PERF: cast done in %f s\n", seconds_since(all_start));

    all_start = clock_now();
    auto dump_render = args["dump-render"];
    replace(dump_render.begin(), dump_render.end(), '\\', '/');
    if (dump_render.size() && dump_render.back() != '/') {
        dump_render = dump_render + "/";
    }
    vector<pixel_t> dump_buffer;
    map<unsigned short, int3> dump_colors = { { 0xffff, { 0, 0, 0 } } };
    auto dump_render_axis = args["dump-render-axis"];
    for (int dir = 0; dir < 3; dir ++) {
        auto sz = rotate(int3 { (int) nx, (int) ny, (int) nz }, dir);
        auto width = sz.x * cast_pixels.x, height = sz.y * cast_pixels.x;
        device_vector points(dexels[dir].xs);

        auto &dexel = dexels[dir == 0 ? 1 : dir == 1 ? 2 : 0];
        device_vector offset(dexel.offset);
        device_vector joints(dexel.joints);
        device_vector<pixel_t> pixels(width * height);

        auto axis = ("xyz")[dir];
        auto render_start = clock_now();
        for (int i = 0; i < sz.z; i ++) {
            kernel_reset CU_DIM(1024, 128) (pixels, { 0xffff });
            auto start = i * cast_pixels.y * height;
            kernel_render_dexel CU_DIM(height, 1024) (start, width, ext, points, offset, joints, pixels);
            CUDA_ASSERT(cudaGetLastError());
            if (start > height) {
                kernel_render_dexel CU_DIM(height, 1024) (start - height, width, ext, points, offset, joints, pixels);
                CUDA_ASSERT(cudaGetLastError());
            }
            if (dump_render.size() && dump_render_axis.find(axis) != string::npos) {
                pixels.copy_to(dump_buffer);
                auto file = dump_render + string(1, axis) + "-" + to_string(i) + ".png";
                dump_png(dump_buffer, width, height, cast_pixels.x, file, dump_colors);
            }
        }
        CUDA_ASSERT(cudaDeviceSynchronize());
        printf("PERF: on %c render %d in %f s\n", axis, sz.z, seconds_since(render_start));
    }
    printf("PERF: render done in %f s\n", seconds_since(all_start));

    return 0;
}
