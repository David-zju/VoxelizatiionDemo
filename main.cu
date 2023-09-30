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
        auto x = verts[face.x], y = verts[face.y], z = verts[face.z];
        auto p0 = rotate(fmin(x, y, z), dir), p1 = rotate(fmax(x, y, z), dir);
        for (int i = 0, nx = xs.len; i < nx - 1; i ++) {
            auto x0 = xs[i], x1 = xs[i + 1];
            for (int j = 0, ny = ys.len; j < ny - 1; j ++) {
                auto y0 = ys[j], y1 = ys[j + 1];
                if (p0.x < x1 && p0.y < y1 &&
                    p1.x > x0 && p1.y > y0) {
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
    vector<double> xs, ys;
    vector<int> offset;
    vector<int4> faces;
};

auto get_faces_in_cells(device_vector<double3> &verts, device_vector<int4> &faces, grid_t &grid, int dir) {
    auto &xv = dir == 0 ? grid.ys : dir == 1 ? grid.zs : grid.xs;
    auto &yv = dir == 0 ? grid.zs : dir == 1 ? grid.xs : grid.ys;
    face_groups_t ret { xv, yv };

    device_vector xs(xv), ys(yv);
    ret.offset.resize(xs.len * ys.len + 1);
    device_vector<int> offset(ret.offset);
    kernel_get_faces_in_cells CU_DIM(1024, 128) (verts, faces, xs, ys, dir, offset, { });
    CUDA_ASSERT(cudaGetLastError());

    offset.copy_to(ret.offset);
    auto sum = accumulate(ret.offset.begin(), ret.offset.end(), 0);
    device_vector<int4> out(sum);
    auto offset_vec = ret.offset;
    exclusive_scan(offset_vec.begin(), offset_vec.end(), ret.offset.begin(), 0);

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
    int solid;
    double pos;
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
                out[next] = { face.w, pos };
            }
        }
    }
}

struct cast_dexel_t {
    double tol;
    int pixels;
    vector<double> xs, ys;
    vector<int> offset;
    vector<cast_joint_t> joints;
    auto sort_joints() {
        auto ptr = joints.data();
        for (int m = 0, n = offset.size(); m < n - 1; m ++) {
            auto begin = offset[m], end = offset[m + 1];
            sort(ptr + begin, ptr + end);
        }
    }
    auto dump_gltf(string file) {
        vector<double3> bin;
        for (int u = 0; u + 1 < xs.size(); u ++) for (int v = 0; v + 1 < ys.size(); v ++) {
            int w = u + v * xs.size();
            for (int i = 0; i < pixels; i ++) for (int j = 0; j < pixels; j ++) {
                int k = i + j * pixels;
                double2 p = {
                    lerp(xs[u] + tol, xs[u + 1] - tol, 1. * i / (pixels - 1)),
                    lerp(ys[v] + tol, ys[v + 1] - tol, 1. * j / (pixels - 1)),
                };
                auto m = w * pixels * pixels + k;
                auto begin = offset[m], end = offset[m + 1];
                if (0) {
                    for (int q = begin; q < end; q ++) {
                        auto &a = joints[q];
                        bin.push_back({ p.x, p.y, a.pos + tol * 10 });
                        bin.push_back({ p.x, p.y, a.pos - tol * 10 });
                    }
                } else {
                    for (int q = begin; q < end - 1; q ++) {
                        auto &a = joints[q], &b = joints[q + 1];
                        if (a.solid == b.solid) {
                            bin.push_back({ p.x, p.y, a.pos + tol });
                            bin.push_back({ p.x, p.y, b.pos - tol });
                            q ++;
                        }
                    }
                }
            }
        }

        vector<float3> out;
        for (auto p : bin) {
            out.push_back({ (float) p.x, (float) p.y, (float) p.z });
        }
        float3 p0 = { FLT_MAX, FLT_MAX, FLT_MAX }, p1 = { FLT_MIN, FLT_MIN, FLT_MIN };
        if (out.size()) {
            p0 = out.back(); p1 = out.front();
            for (auto p : out) {
                p0.x = fmin(p0.x, p.x); p1.x = fmax(p1.x, p.x);
                p0.y = fmin(p0.y, p.y); p1.y = fmax(p1.y, p.y);
                p0.z = fmin(p0.z, p.z); p1.z = fmax(p1.z, p.z);
            }
            auto d = float3 { p1.x - p0.x, p1.y - p0.y, p1.z - p0.z };
            out.push_back({ 0, 0, p0.z -= d.z });
            out.push_back({ 0, 0, p1.z += d.z });
            out.push_back({ 0, p0.y -= d.y, 0 });
            out.push_back({ 0, p1.y += d.z, 0 });
            out.push_back({ p0.x -= d.x, 0, 0 });
            out.push_back({ p1.x += d.x, 0, 0 });
        }
        std::ofstream fn(file + ".bin", std::ios::out | std::ios::binary);
        auto byteLength = out.size() * sizeof(float3);
        fn.write((char *) out.data(), byteLength);

        json j;
        std::ifstream("tool/view.gltf") >> j;
        for (auto &item : j["accessors"]) {
            item["min"][0] = p0.x; item["min"][1] = p0.y; item["min"][2] = p0.z;
            item["max"][0] = p1.x; item["max"][1] = p1.y; item["max"][2] = p1.z;
            item["count"] = out.size();
        }
        for (auto &item : j["bufferViews"]) {
            item["byteLength"] = byteLength;
        }
        auto filename = filesystem::path(file).filename().u8string();
        for (auto &item : j["buffers"]) {
            item["byteLength"] = byteLength;
            item["uri"] = filename + ".bin";
        }
        std::ofstream(file) << j.dump(2);
    }
};
auto cast_in_cells(device_vector<double3> &verts, face_groups_t &groups, int dir, int pixels, double tol) {
    device_vector xs(groups.xs), ys(groups.ys);
    device_vector faces(groups.faces);
    device_vector offset(groups.offset);
    dim3 gridDim((int) xs.len, (int) ys.len, 1),
         blockDim(pixels, pixels, 1);

    cast_dexel_t ret { tol, pixels, groups.xs, groups.ys };
    ret.offset.resize(xs.len * ys.len * blockDim.x * blockDim.y + 1);
    device_vector len(ret.offset);
    kernel_cast_in_cells CU_DIM(gridDim, blockDim) (verts, faces, xs, ys, dir, offset, tol, len, { });
    CUDA_ASSERT(cudaGetLastError());

    len.copy_to(ret.offset);
    auto sum = accumulate(ret.offset.begin(), ret.offset.end(), 0);
    device_vector<cast_joint_t> out(sum);
    auto offset_vec = ret.offset;
    exclusive_scan(offset_vec.begin(), offset_vec.end(), ret.offset.begin(), 0);

    len.copy_from(ret.offset);
    kernel_cast_in_cells CU_DIM(gridDim, blockDim) (verts, faces, xs, ys, dir, offset, tol, len, out);
    CUDA_ASSERT(cudaGetLastError());
    out.copy_to(ret.joints);
    ret.sort_joints();

    return ret;
}

int main() {
    grid_t grid;
    grid.from_json("data\\MaterialAndGridLines.json");
    grid.xs = add_midpoint(grid.xs); //grid.xs = add_midpoint(grid.xs);
    grid.ys = add_midpoint(grid.ys); //grid.ys = add_midpoint(grid.ys);
    grid.zs = add_midpoint(grid.zs); //grid.zs = add_midpoint(grid.zs);
    printf("loaded %zu x %zu x %zu (%zu) grids\n",
        grid.xs.size(), grid.ys.size(), grid.zs.size(),
        grid.xs.size() * grid.ys.size() * grid.zs.size());

    mesh_t mesh;
    mesh.from_obj("data\\toStudent_EM.obj");
    printf("loaded %zu faces with %zu vertices\n", mesh.faces.size(), mesh.verts.size());

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

    device_vector verts(mesh.verts);
    device_vector faces(mesh.faces);
    auto all_start = clock_now();
    for (int dir = 0; dir < 3; dir ++) {
        auto start_group = clock_now();
        auto groups = get_faces_in_cells(verts, faces, grid, dir);
        printf("group %zu in %f s\n", groups.faces.size(), seconds_since(start_group));
        auto start_cast = clock_now();
        auto casted = cast_in_cells(verts, groups, dir, 32, 1e-3);
        printf("cast %zu in %f s\n", casted.joints.size(), seconds_since(start_cast));
        //casted.dump_gltf("data/dump-" + to_string(dir) + ".gltf");
    }
    printf("all done in %f s\n", seconds_since(all_start));

    return 0;
}
