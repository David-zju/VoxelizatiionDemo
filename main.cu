#include "geom.h"

__global__ void kernel_group_gpu(
    buffer<double3> verts, buffer<int3> faces,
    buffer<double> xs, buffer<double> ys, int dir,
    buffer<int *> out) {
    for (int i = cuIdx(x); i < faces.len; i += cuDim(x)) {
        auto face = faces[i];
        auto x = verts[face.x], y = verts[face.y], z = verts[face.z];
        auto p0 = proj(fmin(x, y, z), dir), p1 = proj(fmax(x, y, z), dir);
        for (int i = 0, nx = xs.len; i < nx - 1; i ++) {
            auto x0 = xs[i], x1 = xs[i + 1];
            for (int j = 0, ny = ys.len; j < ny - 1; j ++) {
                auto y0 = ys[j], y1 = ys[j + 1];
                if (p0.x < x1 && p0.y < y1 &&
                    p1.x > x0 && p1.y > y0) {
                    int k = i + j * nx;
                    //atomicAdd(out[k], 1);
                }
            }
        }
    }
}

auto group_gpu(mesh_t &mesh, vector<double> xv, vector<double> yv, int dir) {
    auto verts = buffer_from(mesh.verts);
    auto faces = buffer_from(mesh.faces);
    auto xs = buffer_from(xv), ys = buffer_from(yv);
    device_vector<int *> out(xs.len * ys.len);
    kernel_group_gpu CU_DIM(1024, 128) (verts, faces, xs, ys, dir, out);
}

int main() {
    grid_t grid;
    grid.from_json("data\\MaterialAndGridLines.json");
    auto nx = grid.xs.size(), ny = grid.ys.size(), nz = grid.zs.size();
    printf("loaded %zu x %zu x %zu\n", grid.xs.size(), grid.ys.size(), grid.zs.size());

    mesh_t mesh;
    mesh.from_obj("data\\toStudent_EM.obj");
    auto &verts = mesh.verts;
    printf("loaded %zu faces with %zu vertices\n", mesh.faces.size(), mesh.verts.size());

    auto start_group_gpu = clock_now();
    group_gpu(mesh, grid.xs, grid.ys, 2);
    printf("gpu group in %f s\n", seconds_since(start_group_gpu));
    return 0;
}
