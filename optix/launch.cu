#include "optix/launch.h"

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    float3 ray_origin, ray_direction;
    if (params.dir == 0) {

    } else if (params.dir == 1) {

    } else if (params.dir == 2) {
        ray_origin = { (float) params.zs[0], (float) params.xs[idx.x], (float) params.ys[idx.y] };
        ray_direction = { 0, 0, 1 };
    }

    unsigned int p0, p1, p2;
    optixTrace(
            params.handle,
            ray_origin,
            ray_direction,
            0.0f,                // Min intersection distance
            1e16f,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask( 255 ), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            1,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            p0, p1, p2 );
}

extern "C" __global__ void __miss__ms() {
}

extern "C" __global__ void __anyhit__ch() {
}
