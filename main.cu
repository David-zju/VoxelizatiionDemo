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
    for (auto &[axis, ref] : map<string, reference_wrapper<vector<double>>>({
        { "x", ref(grid.xs) },
        { "y", ref(grid.ys) },
        { "z", ref(grid.zs) }
    })) {
        auto &arr = ref.get();
        auto key = "load-grid-" + axis;
        if (args_list.count(key)) {
            arr.resize(0);
            for (auto list : args_list[key]) {
                double val;
                replace(list.begin(), list.end(), ',', ' ');
                stringstream ss(list);
                while (ss >> val) arr.push_back(val);
            }
        }
    }
    for (auto _ : args_list["grid-add-midpoint"]) {
        grid.xs = add_midpoint(grid.xs);
        grid.ys = add_midpoint(grid.ys);
        grid.zs = add_midpoint(grid.zs);
    }
    auto nx = grid.xs.size(), ny = grid.ys.size(), nz = grid.zs.size();
    printf("INFO: loaded %zu x %zu x %zu (%zu) grids\n", nx, ny, nz, nx * ny * nz);
    if (nx < 2 || ny < 2 || nz < 2) {
        fprintf(stderr, "FATAL: grid line on each axis should be larger than 2\n");
    }

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
            geom_list += "," + list;
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

    // Note: disabled at present
    if (0) {
        trace_t tracer(mesh, grid);
        tracer.render("data/x.png");
        return 0;
    }

    auto all_start = clock_now();
    device_vector verts(mesh.verts);
    device_vector faces(mesh.faces);

    auto dump_cast = args["dump-cast"];
    auto ext = stod(args["ext"]);
    vector<double3> cast_points;
    tri_dexels_t casted { grid, cast_pixels, ext };
    for (int dir = 0; dir < 3; dir ++) {
        auto axis = ("xyz")[dir];

        auto group_start = clock_now();
        auto face_groups = get_faces_in_cells(verts, faces, grid, dir);
        printf("PERF: on %c group %zu in %f s\n", axis, face_groups.faces.size(), seconds_since(group_start));

        auto cast_start = clock_now();
        auto joint_count = 0;
        device_group groups(face_groups);
        auto &dexels = casted.dexels[dir];
        auto &dexel = dexels[0] = groups.cast(verts, cast_pixels, ext);
        joint_count += dexel.joints.size();
        if (dump_cast.size()) for (auto &v : dexel.get_verts(dump_mode)) {
            cast_points.push_back(v);
        }
        if (cast_pixels.x != cast_pixels.y) {
            auto &dexel = dexels[1] = groups.cast(verts, { cast_pixels.y, cast_pixels.x }, ext);
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

    auto dump_render = args["dump-render"];
    replace(dump_render.begin(), dump_render.end(), '\\', '/');
    if (dump_render.size() && dump_render.back() != '/') {
        dump_render = dump_render + "/";
    }
    map<unsigned short, int3> dump_colors = { { 0xffff, { 0, 0, 0 } } };
    auto dump_render_axis = args["dump-render-axis"];

    int3 chunk_size = { 128, 128, 8 };
    vector<int3> chunks;
    for (int i = 0; i < nx; i += chunk_size.x) {
        for (int j = 0; j < ny; j += chunk_size.y) {
            for (int k = 0; k < nz; k += chunk_size.z) {
                chunks.push_back({ i, j, k });
            }
        }
    }

    device_vector<pixel_t> out[3];
    all_start = clock_now();
    for (auto chunk_pos : chunks) {
        auto render_start = clock_now();
        casted.render(out, chunk_pos, chunk_size);
        auto chunk_index = to_string(chunk_pos.x) + "_" + to_string(chunk_pos.y) + "_" + to_string(chunk_pos.z);
        if (dump_render.size()) for (int dir = 0; dir < 3; dir ++) {
            auto gsz  = rotate(int3 { (int) nx, (int) ny, (int) nz }, dir),
                 pos  = rotate(chunk_pos, dir),
                 dim  = rotate(chunk_size, dir),
                 size = int3 { dim.x * cast_pixels.x, dim.y * cast_pixels.x, dim.z };
            vector<pixel_t> pixels(size.x * size.y);
            auto axis = ("xyz")[dir];
            auto dump_dir = dump_render + chunk_index + "/" + string(1, axis) + "/";
            if (dump_render_axis.find(axis) != string::npos) for (int i = pos.z, j = 0; i < gsz.z && j < dim.z; i ++, j ++) {
                out[dir].copy_to(pixels.data(), (size_t) size.x * size.y, (size_t) j * size.x * size.y);
                auto file = dump_dir + to_string(j) + ".png";
                dump_png(pixels, size.x, size.y, cast_pixels.x, file, dump_colors);
                printf("INFO: dumped png to %s\n", file.c_str());
            }
        }
        printf("PERF: render chunk %s done in %f s\n", chunk_index.c_str(), seconds_since(all_start));
    }
    printf("PERF: render done in %f s\n", seconds_since(all_start));

    return 0;
}
