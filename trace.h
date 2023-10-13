#pragma once

#include <map>
#include <string>
#include <fstream>
#include <sstream>

#ifdef USE_OPTIX
#include <nvrtc.h>
#include <optix.h>
#include <optix_device.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <optix_function_table_definition.h>
#endif

#include "deps/lodepng/lodepng.h"

#include "geom.h"
#include "optix/launch.h"
#include "data/launch_ptx.h"

#define OPTIX_ASSERT(call) CALL_AND_ASSERT(call, OPTIX_SUCCESS, optixGetErrorString)
#define NVRTC_ASSERT(call) CALL_AND_ASSERT(call, NVRTC_SUCCESS, nvrtcGetErrorString)
#define OPTIX_ASSERT_LOG(call) {       \
    char log_buf[1024];                \
    size_t log_size = sizeof(log_buf); \
    OPTIX_ASSERT(call);                \
}

namespace bocchi {

#ifdef USE_OPTIX

template <typename T> struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

auto compile_ptx(string file, const char *options[], size_t num) {
    string src;
    {
        ifstream fn(file);
        stringstream code;
        fn >> code.rdbuf();
        src = code.str();
    }
    nvrtcProgram prog;
    NVRTC_ASSERT(nvrtcCreateProgram(&prog, src.c_str(), "main.cu", 0, NULL, NULL));
    auto ret = nvrtcCompileProgram(prog, num, options);
    size_t log_size;
    NVRTC_ASSERT(nvrtcGetProgramLogSize(prog, &log_size));
    if (log_size > 1) {
        vector<char> log(log_size);
        NVRTC_ASSERT(nvrtcGetProgramLog(prog, log.data()));
        fprintf(stderr, log.data());
    }
    NVRTC_ASSERT(ret);
    size_t ptx_size;
    NVRTC_ASSERT(nvrtcGetPTXSize(prog, &ptx_size));
    vector<char> ptx(ptx_size);
    NVRTC_ASSERT(nvrtcGetPTX(prog, ptx.data()));
    NVRTC_ASSERT(nvrtcDestroyProgram(&prog));
    return ptx;
}

struct trace_t {
    OptixDeviceContext ctx;
    OptixModule mod;
    OptixPipeline pipeline;
    OptixShaderBindingTable sbt = { };

    device_vector<Params> dev_params;
    device_vector<RayGenSbtRecord>   dev_raygen;
    device_vector<MissSbtRecord>     dev_miss;
    device_vector<HitGroupSbtRecord> dev_hitgroup;
    device_vector<uchar4> dev_buffer;
    device_vector<char> dev_accel;
    Params params;
    CUstream stream;

    device_vector<double> dev_xs, dev_ys, dev_zs;

    static void log_cb(unsigned int level, const char *tag, const char *message, void *) {
        printf("[%d][%s] %s\n", level, tag, message);
    }

    auto create_program(OptixProgramGroupKind kind, map<string, string> &&entries) {
        OptixProgramGroupOptions opts = { };
        OptixProgramGroupDesc desc = { };
        desc.kind = kind;
        if (kind == OPTIX_PROGRAM_GROUP_KIND_RAYGEN) {
            if (entries.count("raygen")) {
                desc.raygen.module = mod;
                desc.raygen.entryFunctionName = entries["raygen"].c_str();
            }
        } else if (kind == OPTIX_PROGRAM_GROUP_KIND_MISS) {
            if (entries.count("miss")) {
                desc.miss.module = mod;
                desc.miss.entryFunctionName = entries["miss"].c_str();
            }
        } else if (kind == OPTIX_PROGRAM_GROUP_KIND_HITGROUP) {
            if (entries.count("closesthit")) {
                desc.hitgroup.moduleCH = mod;
                desc.hitgroup.entryFunctionNameCH = entries["closesthit"].c_str();
            }
            if (entries.count("anyhit")) {
                desc.hitgroup.moduleAH = mod;
                desc.hitgroup.entryFunctionNameAH = entries["anyhit"].c_str();
            }
            if (entries.count("intersection")) {
                desc.hitgroup.moduleIS = mod;
                desc.hitgroup.entryFunctionNameIS = entries["intersection"].c_str();
            }
        }
        OptixProgramGroup group;
        OPTIX_ASSERT_LOG(optixProgramGroupCreate(ctx, &desc, 1, &opts, log_buf, &log_size, &group));
        return group;
    }
    auto build_accel(mesh_t &mesh) {
        OptixTraversableHandle handle;
        vector<float3> verts_f32; for (auto &vert : mesh.verts) {
            verts_f32.push_back(float3 { (float) vert.x, (float) vert.y, (float) vert.z });
        }
        device_vector verts(verts_f32);
        vector<uint32_t> faces_u32; for (auto &face : mesh.faces) {
            faces_u32.push_back((uint32_t) face.x);
            faces_u32.push_back((uint32_t) face.y);
            faces_u32.push_back((uint32_t) face.z);
        };
        device_vector faces(faces_u32);

        OptixAccelBuildOptions opts = { };
        opts.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        opts.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixBuildInput input = { };
        input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        auto &arr = input.triangleArray;
        arr.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
        arr.vertexStrideInBytes = sizeof(float3);
        arr.numVertices   = verts.len;
        arr.vertexBuffers = (CUdeviceptr *) &verts.ptr;
        arr.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        arr.indexStrideInBytes = sizeof(uint32_t) * 3;
        arr.numIndexTriplets = faces.len / 3;
        arr.indexBuffer = (CUdeviceptr) faces.ptr;
        uint32_t flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
        arr.flags         = flags;
        arr.numSbtRecords = sizeof(flags) / sizeof(uint32_t);

        OptixAccelBufferSizes size;
        OPTIX_ASSERT(optixAccelComputeMemoryUsage(ctx, &opts, &input, 1, &size));
        device_vector<char> tmpBuf(size.tempSizeInBytes), outBuf(size.outputSizeInBytes);

        OptixAccelEmitDesc emitDesc = { };
        device_vector<uint64_t> compactedSize(1);
        emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitDesc.result = (CUdeviceptr) compactedSize.ptr;
        OPTIX_ASSERT(optixAccelBuild(ctx, 0, &opts, &input, 1,
            (CUdeviceptr) tmpBuf.ptr, tmpBuf.len,
            (CUdeviceptr) outBuf.ptr, outBuf.len,
            &handle, &emitDesc, 1));
        CUDA_ASSERT(cudaDeviceSynchronize());

        auto compactedSizeVec = compactedSize.to_host();
        dev_accel.resize(compactedSizeVec[0]);
        OPTIX_ASSERT(optixAccelCompact(ctx, 0, handle, (CUdeviceptr) dev_accel.ptr, dev_accel.len, &handle));
        CUDA_ASSERT(cudaDeviceSynchronize());
        return handle;
    }
    template <class T>
    auto update_record(device_vector<SbtRecord<T>> &vec, OptixProgramGroup &pg, T &&data = { }) {
        SbtRecord<T> rec;
        rec.data = data;
        OPTIX_ASSERT(optixSbtRecordPackHeader(pg, &rec));
        vec.copy_from(&rec);
        return (CUdeviceptr) vec.ptr;
    }
    trace_t(mesh_t &mesh, grid_t &grid) :
            dev_params(1), dev_buffer(0), dev_accel(1),
            dev_xs(0), dev_ys(0), dev_zs(0),
            dev_raygen(1), dev_miss(1), dev_hitgroup(1) {
        CUDA_ASSERT(cudaFree(NULL));
        OPTIX_ASSERT(optixInit());

        CUcontext cuCtx = 0;
        OptixDeviceContextOptions opts = { };
        opts.logCallbackFunction = &log_cb;
        opts.logCallbackLevel = 4;
        OPTIX_ASSERT(optixDeviceContextCreate(cuCtx, &opts, &ctx));

        OptixModuleCompileOptions modCompileOpts = { };
        modCompileOpts.optLevel          = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        modCompileOpts.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

        OptixPipelineCompileOptions pipCompileOpts = { };
        pipCompileOpts.usesMotionBlur                   = false;
        pipCompileOpts.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipCompileOpts.numPayloadValues                 = 3;
        pipCompileOpts.numAttributeValues               = 3;
        pipCompileOpts.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
        pipCompileOpts.pipelineLaunchParamsVariableName = "params";
        pipCompileOpts.usesPrimitiveTypeFlags           = (unsigned int) OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

#ifdef COMPILE_PRX_FROM_SOURCE
        const char* ptxOptions[] = {
            "-IC:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 7.6.0\\include",
            "-IC:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\include",
            "-IC:\\Projects\\bocchi",
            "--std=c++17",
        };
        auto ptx = compile_ptx("C:\\Projects\\bocchi\\optix\\launch.cu", ptxOptions, sizeof(ptxOptions) / sizeof(char *));
        auto ptx_ptr = ptx.data();
        auto ptx_len = ptx.size();
#else
        auto ptx_ptr = (const char *) data_launch_ptx;
        auto ptx_len = data_launch_ptx_len;
#endif
        OPTIX_ASSERT_LOG(optixModuleCreateFromPTX(ctx, &modCompileOpts, &pipCompileOpts, ptx_ptr, ptx_len, log_buf, &log_size, &mod));

        auto raygenPg  = create_program(OPTIX_PROGRAM_GROUP_KIND_RAYGEN,   {{ "raygen", "__raygen__rg" }}),
            missPg     = create_program(OPTIX_PROGRAM_GROUP_KIND_MISS,     {{ "miss", "__miss__ms" }}),
            hitgroupPg = create_program(OPTIX_PROGRAM_GROUP_KIND_HITGROUP, {{ "closesthit", "__closesthit__ch" }});

        OptixPipelineLinkOptions pipLinkOpts = { };
        pipLinkOpts.maxTraceDepth = 2;
        pipLinkOpts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        vector<OptixProgramGroup> group({ raygenPg, missPg, hitgroupPg });
        OPTIX_ASSERT_LOG(optixPipelineCreate(ctx, &pipCompileOpts, &pipLinkOpts,
            group.data(), group.size(), log_buf, &log_size, &pipeline));

        OptixStackSizes stackSizes = { };
        for (auto &item : group) {
            OPTIX_ASSERT(optixUtilAccumulateStackSizes(item, &stackSizes));
        }
        uint32_t sizeFromTraversal, sizeFromState, sizeForContinuation;
        OPTIX_ASSERT(optixUtilComputeStackSizes(&stackSizes, pipLinkOpts.maxTraceDepth, 0, 0,
            &sizeFromTraversal, &sizeFromState, &sizeForContinuation));
        OPTIX_ASSERT(optixPipelineSetStackSize(pipeline, sizeFromTraversal, sizeFromState, sizeForContinuation, 1));;

        sbt.raygenRecord        = update_record(dev_raygen,   raygenPg);
        sbt.missRecordBase      = update_record(dev_miss,     missPg, { .0, .0, .5 });
        sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
        sbt.missRecordCount     = 1;
        sbt.hitgroupRecordBase  = update_record(dev_hitgroup, hitgroupPg);
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        sbt.hitgroupRecordCount = 1;

        CUDA_ASSERT(cudaStreamCreate(&stream));
        dev_xs.copy_from(grid.xs); params.xs = dev_xs;
        dev_ys.copy_from(grid.ys); params.ys = dev_ys;
        dev_zs.copy_from(grid.zs); params.zs = dev_zs;
        params.handle = build_accel(mesh);
    }
    auto render(string file) {
        params.dir = 2;
        int2 size { (int) dev_xs.len, (int) dev_ys.len };
        dev_buffer.resize(size.x * size.y);
        params.buffer = dev_buffer;
        dev_params.copy_from(&params);
        OPTIX_ASSERT(optixLaunch(pipeline, stream,
            (CUdeviceptr) dev_params.ptr, sizeof(Params), &sbt, size.x, size.y, 1));
        CUDA_ASSERT(cudaDeviceSynchronize());
        auto pixels = dev_buffer.to_host();
        lodepng::encode(file, (unsigned char *) pixels.data(), size.x, size.y);
    }
    ~trace_t() {
        OPTIX_ASSERT(optixModuleDestroy(mod));
        OPTIX_ASSERT(optixDeviceContextDestroy(ctx));
    }
};

#endif

};
