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
    auto from_obj(string file) {
        ifstream fn(file);
        string line;
        string head;
        int g = 0;
        while (getline(fn, line)) {
            stringstream s(line);
            auto str = line.c_str();
            if (line.size() < 1) {
                // pass
            } else if (!strncmp(str, "usemtl ", 7)) {
                // pass
            } else if (!strncmp(str, "mtllib ", 7)) {
                // pass
            } else if (!strncmp(str, "g ", 2)) {
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
            } else {
                fprintf(stderr, "skip bad line %s in file %s\n", line.c_str(), file.c_str());
            }
        }
        return &*this;
    }
};