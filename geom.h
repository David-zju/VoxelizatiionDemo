#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "utils.h"
#include "deps/json/single_include/nlohmann/json.hpp"

namespace bocchi {

using namespace std;
using json = nlohmann::json;

struct grid_t {
    vector<double> xs, ys, zs;
    auto from_json(string file) {
        json j;
        ifstream(file) >> j;
        for (auto &c : j["grid"]["xs"]) {
            xs.push_back(c.get<double>());
        }
        for (auto &c : j["grid"]["ys"]) {
            ys.push_back(c.get<double>());
        }
        for (auto &c : j["grid"]["zs"]) {
            zs.push_back(c.get<double>());
        }
        return &*this;
    }
};

struct mesh_t {
    vector<double3> verts;
    vector<int4> faces;
    auto from_stl(string file) {
        ifstream fn(file);
        string line, head;
        int g = 0;
        while (getline(fn, line)) {
            stringstream s(line);
            auto str = line.c_str();
            while (*str == ' ') str ++;
            if (!strncmp(str, "solid ", 6)) {
                g ++;
            } else if (!strncmp(str, "vertex ", 7)) {
                double3 p;
                s >> head >> p.x >> p.y >> p.z;
                verts.push_back(p);
            } else if (!strncmp(str, "endfacet", 8)) {
                int n = verts.size();
                if (n >= 3) {
                    faces.push_back({ n - 1, n - 2, n - 3, g });
                }
            } else {
                // pass
            }
        }
    }
    auto from_obj(string file) {
        ifstream fn(file);
        string line, head;
        int g = 0;
        while (getline(fn, line)) {
            stringstream s(line);
            auto str = line.c_str();
            if (!strncmp(str, "g ", 2)) {
                g ++;
            } else if (!strncmp(str, "v ", 2)) {
                double3 p;
                s >> head >> p.x >> p.y >> p.z;
                verts.push_back(p);
            } else if (!strncmp(str, "f ", 2)) {
                string x, y, z;
                s >> head >> x >> y >> z;
                faces.push_back({ stoi(x) - 1, stoi(y) - 1, stoi(z) - 1, g });
            } else if (!strncmp(str, "vn ", 3)) {
                // pass
            } else if (!strncmp(str, "#", 1)) {
                // comments
            } else if (!strncmp(str, "usemtl ", 7)) {
                // pass
            } else if (!strncmp(str, "mtllib ", 7)) {
                // pass
            }  else {
                fprintf(stderr, "skip bad line %s in file %s\n", line.c_str(), file.c_str());
            }
        }
        return &*this;
    }
};

auto dump_gltf(vector<double3> &verts, string file, int mode) {
    vector<float3> out;
    for (auto p : verts) {
        out.push_back({ (float) p.x, (float) p.y, (float) p.z });
    }
    float3 p0 = { FLT_MAX, FLT_MAX, FLT_MAX }, p1 = { FLT_MIN, FLT_MIN, FLT_MIN };
    if (out.size()) {
        p0 = out.back(); p1 = out.front();
        for (auto p : out) {
            p0 = fmin3(p0, p); p1 = fmax3(p1, p);
        }
    }
    ofstream fn(file + ".bin", ios::out | ios::binary);
    auto byteLength = out.size() * sizeof(float3);
    fn.write((char *) out.data(), byteLength);

    json j;
    ifstream("tool/view.gltf") >> j;
    for (auto &item : j["meshes"]) {
        for (auto &prim : item["primitives"]) {
            prim["mode"] = mode;
        }
    }
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
    ofstream(file) << j.dump(2);
}

};
