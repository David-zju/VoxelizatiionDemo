#include "geom.h"

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
    vector<double> xs, ys;
    vector<int> offset;
    vector<int4> faces;
};
auto get_faces_in_cells(device_vector<double3> &verts, device_vector<int4> &faces, grid_t &grid, int dir) {
    auto &xv = dir == 0 ? grid.ys : dir == 1 ? grid.zs : grid.xs;
    auto &yv = dir == 0 ? grid.zs : dir == 1 ? grid.xs : grid.ys;
    face_groups_t ret { dir, xv, yv };

    device_vector xs(xv), ys(yv);
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
    return a * (1. - f) + b * f;
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
    return (x0 < x1 && x0 <= x && x <= x1) || (x1 < x0 && x1 <= x && x <= x0);
}
__host__ __device__ __forceinline__ auto point_in_triangle(double2 p, double3 A, double3 B, double3 C, int dir) {
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
    if (ordered(u.x, p.y, v.x)) {
        return interp(u, v, p.y);
    }
    return DBL_MAX;
}

struct cast_joint_t {
    unsigned short solid;
    float pos;
};
auto operator<(const cast_joint_t &a, const cast_joint_t &b) {
    return a.solid == b.solid ? a.pos < b.pos : a.solid < b.solid;
}
__global__ void kernel_cast_in_cells(
    buffer<double3> verts, buffer<int4> faces,
    buffer<double> xs, buffer<double> ys, int dir,
    buffer<int> offset, double tol,
    buffer<int> len, buffer<cast_joint_t> out) {
    int u = blockIdx.x, v = blockIdx.y, w = u + gridDim.x * v;
    if (u >= xs.len - 1 || v >= ys.len - 1) {
        return;
    }
    int i = threadIdx.x, j = threadIdx.y, k = i + blockDim.x * j;
    double2 p = {
        lerp(xs[u] + tol, xs[u + 1] - tol, 1. * i / (blockDim.x - 1)),
        lerp(ys[v] + tol, ys[v + 1] - tol, 1. * j / (blockDim.y - 1)),
    };
    for (int m0 = offset[w], m1 = offset[w + 1]; m0 < m1; m0 ++) {
        auto face = faces[m0];
        auto pos = point_in_triangle(p, verts[face.x], verts[face.y], verts[face.z], dir);
        if (pos != DBL_MAX) {
            auto next = atomicAdd(len.ptr + w * blockDim.x * blockDim.y + k, 1);
            if (next < out.len) {
                out[next] = { (unsigned short) face.w, (float) pos };
            }
        }
    }
}

struct cast_dexel_t {
    int dir;
    double tol;
    int2 dim;
    vector<double> xs, ys;
    vector<int> offset;
    vector<cast_joint_t> joints;
    auto &sort_joints() {
        auto ptr = joints.data();
        for (int m = 0, n = offset.size(); m < n - 1; m ++) {
            auto begin = offset[m], end = offset[m + 1];
            sort(ptr + begin, ptr + end);
        }
        return *this;
    }
    auto get_verts(int mode) {
        vector<double3> verts;
        for (int u = 0; u + 1 < xs.size(); u ++) for (int v = 0; v + 1 < ys.size(); v ++) {
            int w = u + xs.size() * v;
            for (int i = 0; i < dim.x; i ++) for (int j = 0; j < dim.y; j ++) {
                int k = i + dim.x * j;
                double2 p = {
                    lerp(xs[u] + tol, xs[u + 1] - tol, 1. * i / (dim.x - 1)),
                    lerp(ys[v] + tol, ys[v + 1] - tol, 1. * j / (dim.y - 1)),
                };
                auto m = w * dim.x * dim.y + k;
                auto begin = offset[m], end = offset[m + 1];
                if (mode == 0) {
                    for (int q = begin; q < end; q ++) {
                        auto &a = joints[q];
                        verts.push_back(revert({ p.x, p.y, a.pos }, dir));
                    }
                } else {
                    for (int q = begin; q < end - 1; q ++) {
                        auto &a = joints[q], &b = joints[q + 1];
                        if (a.solid == b.solid) {
                            verts.push_back(revert({ p.x, p.y, a.pos + tol }, dir));
                            verts.push_back(revert({ p.x, p.y, b.pos - tol }, dir));
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
    device_vector<double> xs, ys;
    device_vector<int> offset;
    device_vector<int4> faces;
    device_group(face_groups_t &group) :
        dir(group.dir),
        xs(group.xs), ys(group.ys),
        offset(group.offset), faces(group.faces) {
    }
    auto cast(device_vector<double3> &verts, int2 dim, double tol) {
        dim3 gridDim((int) xs.len, (int) ys.len, 1),
             blockDim(dim.x, dim.y, 1);

        cast_dexel_t ret { dir, tol, dim, xs.copy(), ys.copy() };
        ret.offset.resize(xs.len * ys.len * blockDim.x * blockDim.y + 1);
        device_vector len(ret.offset);
        kernel_cast_in_cells CU_DIM(gridDim, blockDim) (verts, faces, xs, ys, dir, offset, tol, len, { });
        CUDA_ASSERT(cudaGetLastError());

        len.copy_to(ret.offset);
        exclusive_scan(ret.offset.begin(), ret.offset.end(), ret.offset.begin(), 0);
        device_vector<cast_joint_t> out(ret.offset.back());

        len.copy_from(ret.offset);
        kernel_cast_in_cells CU_DIM(gridDim, blockDim) (verts, faces, xs, ys, dir, offset, tol, len, out);
        CUDA_ASSERT(cudaGetLastError());

        out.copy_to(ret.joints);
        return ret;
    }
};

int main() {
    grid_t grid;
    grid.from_json("data\\MaterialAndGridLines.json");
    grid.xs = add_midpoint(grid.xs); //grid.xs = add_midpoint(grid.xs);
    grid.ys = add_midpoint(grid.ys); //grid.ys = add_midpoint(grid.ys);
    grid.zs = add_midpoint(grid.zs); //grid.zs = add_midpoint(grid.zs);
    auto nx = grid.xs.size(), ny = grid.ys.size(), nz = grid.zs.size();
    printf("INFO: loaded %zu x %zu x %zu (%zu) grids\n", nx, ny, nz, nx * ny * nz);

    mesh_t mesh;
    mesh.from_obj("data\\toStudent_EM3.obj");
    if (0) {
        map<int, vector<int4>> mesh_map;
        for (auto face : mesh.faces) {
            mesh_map[face.w].push_back(face);
        }
        for (auto &[geom, faces] : mesh_map) {
            printf("INFO: got geom %d with %zu faces\n", geom, faces.size());
        }
        int geom = 52;
        printf("WARN: filter mesh only to show g == %d\n", geom);
        mesh.faces.resize(0);
        for (auto face : mesh_map[geom]) {
            mesh.faces.push_back(face);
        }
    }
    printf("INFO: loaded %zu faces with %zu vertices\n", mesh.faces.size(), mesh.verts.size());

    if (0) {
        grid.xs = { -2, -1, 0, 1, 2 };
        grid.ys = { -2, -1, 0, 1, 2 };
        grid.zs = { -2, -1, 0, 1, 2 };
        mesh.verts = {
            { 0, 0, 0 },
            { 0.5, 0, 0 },
            { 0, 0.7, 0 },
            { 0, 0, 0.9 },
        };
        mesh.faces = {
            { 0, 1, 2, 0 },
            { 0, 1, 3, 0 },
            { 0, 2, 3, 0 },
            { 1, 2, 3, 0 },
        };
    }

    // Note: 0 for point cloud; 1 for lines
    int dump_mode = 0;
    // Note: we will try to make grid from this dimension
    int2 cast_pixels = { 16, 2 };
    vector<int2> cast_dims = { cast_pixels };
    if (cast_pixels.x != cast_pixels.y) {
        cast_dims.push_back({ cast_pixels.y, cast_pixels.x });
    }

    device_vector verts(mesh.verts);
    device_vector faces(mesh.faces);
    auto all_start = clock_now();
    vector<double3> dexels;
    for (int dir = 0; dir < 3; dir ++) {
        auto axis = ("xyz")[dir];
        auto group_start = clock_now();
        auto face_groups = get_faces_in_cells(verts, faces, grid, dir);
        printf("PERF: on %c group %zu in %f s\n", axis, face_groups.faces.size(), seconds_since(group_start));
        auto cast_start = clock_now();
        device_group groups(face_groups);
        auto count_start = dexels.size();
        for (auto dim : cast_dims) {
            auto casted = groups.cast(verts, dim, 1e-3);
            for (auto &v : casted.sort_joints().get_verts(dump_mode)) {
                dexels.push_back(v);
            }
        }
        printf("PERF: on %c cast %zu in %f s\n", axis, dexels.size() - count_start, seconds_since(cast_start));
    }
    if (dexels.size()) {
        dump_gltf(dexels, "data/dump.gltf", dump_mode);
    }
    printf("PERF: all done in %f s\n", seconds_since(all_start));

    return 0;
}
