#include <optix.h>
#include <vector_types.h>
#include <vector_functions.h>

#include "utils.h"

struct Params {
    int dir;
    bocchi::buffer<double> xs, ys, zs;
    bocchi::buffer<uchar4> buffer;
    OptixTraversableHandle handle;
};

struct RayGenData {
    // No data needed
};

struct MissData {
    float3 bg_color;
};

struct HitGroupData {
    // No data needed
};

