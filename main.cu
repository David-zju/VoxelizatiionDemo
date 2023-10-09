#include "geom.h"

#include "bocchi.h"

using namespace bocchi;

auto add_midpoint(vector<double> &arr) {
    vector<double> ret;
    for (int i = 0, n = arr.size(); i < n - 1; i ++) {
        ret.push_back(arr[i]);
        ret.push_back((arr[i] + arr[i + 1]) / 2);
    }
    ret.push_back(arr.back());
    return ret;
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
        } else if (load.substr(load.size() - 4) == ".stl") {
            loaded.from_stl(load);
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
    cast_dexel_t dexels[3][2];

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
        auto &dexel = dexels[dir][0] = groups.cast(verts, cast_pixels, ext);
        joint_count += dexel.joints.size();
        if (dump_cast.size()) for (auto &v : dexel.get_verts(dump_mode)) {
            cast_points.push_back(v);
        }
        if (cast_pixels.x != cast_pixels.y) {
            auto &dexel = dexels[dir][1] = groups.cast(verts, { cast_pixels.y, cast_pixels.x }, ext);
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
        device_vector xs(dexels[dir][0].xs), ys(dexels[dir][0].ys);
        auto &d0 = dexels[dir == 0 ? 1 : dir == 1 ? 2 : 0][0],
             &d1 = dexels[dir == 0 ? 2 : dir == 1 ? 0 : 1][1];
        device_vector offset0(d0.offset), offset1(d1.offset);
        device_vector joints0(d0.joints), joints1(d1.joints);
        device_vector<pixel_t> pixels(width * height);

        auto axis = ("xyz")[dir];
        auto render_start = clock_now();
        for (int i = 1, start, stride; i < sz.z; i ++) {
            kernel_reset CU_DIM(1024, 128) (pixels, { 0xffff });
            start = i * cast_pixels.y * height; stride = 1;
            kernel_render_dexel CU_DIM(height, 1024) (0, start, stride, width, height, ext, xs, offset0, joints0, pixels);
            start -= height;
            kernel_render_dexel CU_DIM(height, 1024) (0, start, stride, width, height, ext, xs, offset0, joints0, pixels);
            start = i * cast_pixels.y; stride = cast_pixels.y * sz.z;
            kernel_render_dexel CU_DIM(width,  1024) (1, start, stride, width, height, ext, ys, offset1, joints1, pixels);
            start -= 1;
            kernel_render_dexel CU_DIM(width,  1024) (1, start, stride, width, height, ext, ys, offset1, joints1, pixels);
            if (dump_render.size() && dump_render_axis.find(axis) != string::npos) {
                pixels.copy_to(dump_buffer);
                auto file = dump_render + string(1, axis) + "-" + to_string(i) + ".png";
                dump_png(dump_buffer, width, height, cast_pixels.x, file, dump_colors);
                printf("INFO: dumped png to %s\n", file.c_str());
            }
        }
        CUDA_ASSERT(cudaDeviceSynchronize());
        printf("PERF: on %c render %d in %f s\n", axis, sz.z, seconds_since(render_start));
    }
    printf("PERF: render done in %f s\n", seconds_since(all_start));

    return 0;
}
