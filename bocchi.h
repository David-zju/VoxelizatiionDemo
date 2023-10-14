#pragma once

#include <thread>

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "deps/lodepng/lodepng.h"
#include "geom.h"

namespace bocchi {

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
__host__ __device__ __forceinline__ auto point_in_plane(double2 p, double3 A, double3 B, double3 C, int dir, double tol) {
    double2 u = { 0 }, v = { 0 };
    auto a = rotate(A, dir), b = rotate(B, dir), c = rotate(C, dir);
    auto AB = a.x != b.x,
         BC = b.x != c.x,
         CA = c.x != a.x;
    if (CA && AB) {
        u = interp(a, b, p.x); v = interp(a, c, p.x);
    } else if (AB && BC) {
        u = interp(b, c, p.x); v = interp(b, a, p.x);
    } else if (BC && CA) {
        u = interp(c, a, p.x); v = interp(c, b, p.x);
    }
    if (u.x != v.x) {
        return interp(u, v, p.y);
    }
    return DBL_MAX;
}
__global__ void kernel_get_faces_in_cells(
    buffer<double3> verts, buffer<int4> faces,
    buffer<double> xs, buffer<double> ys, int dir,
    buffer<int> len, buffer<int4> out) {
    for (int n = cuIdx(x); n < faces.len; n += cuDim(x)) {
        auto face = faces[n];
        auto a = verts[face.x], b = verts[face.y], c = verts[face.z];
        auto p0 = rotate(fmin3(a, b, c), dir), p1 = rotate(fmax3(a, b, c), dir);
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

    thrust::exclusive_scan(thrust::device, offset.ptr, offset.ptr + offset.len, offset.ptr);
    offset.copy_to(ret.offset);
    device_vector<int4> out(ret.offset.back());
    kernel_get_faces_in_cells CU_DIM(1024, 128) (verts, faces, xs, ys, dir, offset, out);
    CUDA_ASSERT(cudaGetLastError());

    out.copy_to(ret.faces);
    return ret;
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
    buffer<int> offset, double ext, double tol, int rpt,
    buffer<int> len, buffer<cast_joint_t> out) {
    int u = blockIdx.x, v = blockIdx.y, w = u + gridDim.x * v,
        begin = offset[w], end = offset[w + 1];
    if (u + 1 >= xs.len || v + 1 >= ys.len) {
        return;
    }
    int2 dim = { (int) blockDim.x * rpt, (int) blockDim.y * rpt };
    for (int i0 = 0; i0 < rpt; i0 ++) for (int j0 = 0; j0 < rpt; j0 ++) {
        int i = threadIdx.x * rpt + i0, j = threadIdx.y * rpt + j0,
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
    auto dump(string file, int mode) {
        auto verts = get_verts(mode);
        dump_gltf(verts, file, mode);
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
        // Note: RPT IS NOT REPEAT but Ray-Per-Thread
        int rpt = dim.x * dim.y > 1024 ? 2 : 1;
        dim3 gridDim((int) xs.len, (int) ys.len, 1),
             blockDim(dim.x / rpt, dim.y / rpt, 1);

        cast_dexel_t ret { dir, ext, dim, xs.to_host(), ys.to_host() };
        ret.offset.resize(xs.len * ys.len * dim.x * dim.y + 1);
        device_vector len(ret.offset);
        kernel_cast_in_cells CU_DIM(gridDim, blockDim) (verts, faces, xs, ys, dir, offset, ext, tol, rpt, len, { });
        CUDA_ASSERT(cudaGetLastError());

        thrust::exclusive_scan(thrust::device, len.ptr, len.ptr + len.len, len.ptr);
        len.copy_to(ret.offset);
        device_vector<cast_joint_t> out(ret.offset.back());
        kernel_cast_in_cells CU_DIM(gridDim, blockDim) (verts, faces, xs, ys, dir, offset, ext, tol, rpt, len, out);
        CUDA_ASSERT(cudaGetLastError());

        out.copy_to(ret.joints);
        ret.sort_joints();
        return ret;
    }
};

struct pixel_t {
    unsigned short geom;
};
__global__ void kernel_render_dexel(
        int dir, int ns,
        int2 delta, int3 pos, int3 size, double ext,
        buffer<double> points,
        buffer<int> offset, buffer<cast_joint_t> joints,
        buffer<pixel_t> out) {
    int j = blockIdx.x, w = delta.x + j * delta.y,
        begin = w + 1 < offset.len ? offset[w] : 0,
        end   = w + 1 < offset.len ? offset[w + 1] : 0,
        pixels = dir == 0 ? size.x : size.y;
    for (int i = threadIdx.x; i < pixels; i += blockDim.x) {
        auto u = i / ns + (dir == 0 ? pos.x : pos.y), v = i % ns,
             k = dir == 0 ? i + j * size.x : i * size.x + j;
        if (u + 1 < points.len) {
            auto pos = lerp(points[u], points[u + 1], lerp(ext, 1 - ext, 1. * v / (ns - 1)));
            for (int q = begin; q < end - 1; q ++) {
                auto &a = joints[q], &b = joints[q + 1];
                if (a.geom == b.geom) {
                    if (ordered(a.pos, pos, b.pos)) {
                        out[k] = { a.geom };
                    }
                    q ++;
                }
            }
        }
    }
}

auto mkdirs_for(string file) {
    auto dirname = filesystem::path(file).parent_path();
    filesystem::create_directories(dirname);
    return file;
}

auto dump_png(vector<pixel_t> &img, int width, int height, int grid, string file, map<unsigned short, int3> &colors) {
    mkdirs_for(file);
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

struct tri_dexels_t {
    grid_t grid;
    int2 sample;
    double ext;
    cast_dexel_t dexels[3][2];
};
struct device_chunk_t {
    int3 size;
    int2 sample;
    device_vector<pixel_t> pixels[3];
    auto parse() {
        vector<buffer<pixel_t>> bx(size.x), by(size.y), bz(size.z);
        for (int i = 0; i < size.x; i ++) {
            size_t len = size.y * size.z * sample.x * sample.x;
            bx[i] = { pixels[0].ptr + i * len, len };
        }
        for (int j = 0; j < size.y; j ++) {
            size_t len = size.z * size.x * sample.x * sample.x;
            by[j] = { pixels[1].ptr + j * len, len };
        }
        for (int k = 0; k < size.z; k ++) {
            size_t len = size.x * size.y * sample.x * sample.x;
            bz[k] = { pixels[2].ptr + k * len, len };
        }
        // TODO
    }
};
struct device_tri_dexels_t {
    grid_t grid;
    int2 sample;
    double ext;
    struct device_dexel_t {
        int dir;
        double ext;
        int2 dim;
        device_vector<double> xs, ys;
        device_vector<int> offset;
        device_vector<cast_joint_t> joints;
    };
    device_dexel_t dexels[3][2];
    device_tri_dexels_t(tri_dexels_t &input) {
        grid = input.grid;
        sample = input.sample;
        ext = input.ext;
        for (int i = 0; i < 3; i ++) {
            for (int j = 0; j < 2; j ++) {
                auto &src = input.dexels[i][j];
                auto &dst = dexels[i][j];
                dst.dir = src.dir;
                dst.ext = src.ext;
                dst.dim = src.dim;
                dst.xs.copy_from(src.xs);
                dst.ys.copy_from(src.ys);
                dst.offset.copy_from(src.offset);
                dst.joints.copy_from(src.joints);
            }
        }
    }
    auto render(device_chunk_t &chunk, int3 pos, int3 size) {
        auto chunk_pos = pos, chunk_size = size;
        auto nx = grid.xs.size(), ny = grid.ys.size(), nz = grid.zs.size();
        chunk.size = size;
        chunk.sample = sample;
        auto &out = chunk.pixels;
        for (int dir = 0; dir < 3; dir ++) {
            auto gsz  = rotate(int3 { (int) nx, (int) ny, (int) nz }, dir),
                 idx  = rotate(int3 { 0, 1, 2 }, dir),
                 pos  = rotate(chunk_pos, dir),
                 dim  = rotate(chunk_size, dir),
                 size = int3 { dim.x * sample.x, dim.y * sample.x, dim.z };
            auto &xs = dexels[dir][0].xs,
                 &ys = dexels[dir][0].ys;
            auto &d0 = dexels[idx.x][0],
                 &d1 = dexels[idx.y][sample.x == sample.y ? 0 : 1];
            auto &offset0 = d0.offset, &offset1 = d1.offset;
            auto &joints0 = d0.joints, &joints1 = d1.joints;
            out[dir].resize(size.x * size.y * size.z);

            auto axis = ("xyz")[dir];
            auto render_start = clock_now();
            auto render_size = int2 { min(dim.x, gsz.x - pos.x) * sample.x, min(dim.y, gsz.y - pos.y) * sample.x };
            for (int i = pos.z, j = 0; i < gsz.z && j < dim.z; i ++, j ++) {
                buffer<pixel_t> pixels = { out[dir].ptr + j * size.x * size.y, (unsigned) size.x * size.y };
                kernel_reset CU_DIM(1024, 128) (pixels, { 0xffff });
                int2 delta = { i * sample.y * gsz.y * sample.x + pos.y * sample.x, 1 };
                kernel_render_dexel CU_DIM(render_size.y, 1024) (0, sample.x, delta, pos, size, ext, xs, offset0, joints0, pixels);
                if (delta.x -= gsz.y * sample.x > 0) {
                    kernel_render_dexel CU_DIM(render_size.y, 1024) (0, sample.x, delta, pos, size, ext, xs, offset0, joints0, pixels);
                }
                delta = { i * sample.y + pos.x * sample.y * gsz.z * sample.x, sample.y * gsz.z };
                kernel_render_dexel CU_DIM(render_size.x,  1024) (1, sample.x, delta, pos, size, ext, ys, offset1, joints1, pixels);
                if (delta.x -= 1 > 0) {
                    kernel_render_dexel CU_DIM(render_size.x,  1024) (1, sample.x, delta, pos, size, ext, ys, offset1, joints1, pixels);
                }
            }
            CUDA_ASSERT(cudaDeviceSynchronize());
            if (0) {
                printf("PERF: on %c render %d x %d x %d in %f s\n", axis, size.x, size.y, size.z, seconds_since(render_start));
            }
        }
    }
};

};
