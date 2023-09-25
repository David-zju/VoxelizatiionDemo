#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "utils.h"
#include "deps/json/single_include/nlohmann/json.hpp"

using namespace std;
using json = nlohmann::json;

struct grid_t {
    vector<double> xs, ys, zs;
    auto from_json(string file) {
        ifstream i(file);
        json j;
        i >> j;
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
    vector<int3> faces;
    auto from_obj(string file) {
        ifstream fn(file);
        string line;
        string head;
        while (getline(fn, line)) {
            stringstream s(line);
            auto str = line.c_str();
            if (line.size() < 1) {
                // pass
            } else if (!strncmp(str, "usemtl ", 7)) {
                // pass
            } else if (!strncmp(str, "g ", 2)) {
                // pass
            } else if (!strncmp(str, "v ", 2)) {
                double3 p;
                s >> head >> p.x >> p.y >> p.z;
                verts.push_back(p);
            } else if (!strncmp(str, "f ", 2)) {
                int3 f;
                string x, y, z;
                s >> head >> x >> y >> z;
                f.x = stoi(x);
                f.y = stoi(y);
                f.z = stoi(z);
                faces.push_back({ f.x - 1, f.y - 1, f.z - 1 });
            } else if (!strncmp(str, "vn ", 3)) {
                // pass
            } else if (!strncmp(str, "#", 1)) {
                // comments
            } else {
                fprintf(stderr, "skip bad line %s in file %s\n", line.c_str(), file.c_str());
            }
        }
        return &*this;
    }
};