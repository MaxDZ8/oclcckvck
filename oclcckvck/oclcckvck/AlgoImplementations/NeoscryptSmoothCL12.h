/*
 * Copyright (C) 2015 Massimo Del Zotto
 * This code is released under the MIT license.
 * For conditions of distribution and use, see the LICENSE or hit the web.
 */
#pragma once
#include "../StopWaitAlgorithm.h"

namespace algoImplementations {

class NeoscryptSmoothCL12 : public StopWaitAlgorithm {
public:
    NeoscryptSmoothCL12(cl_context ctx, cl_device_id dev, asizei concurrency)
        : StopWaitAlgorithm(ctx, dev, concurrency, "Neoscrypt", "smooth", "v1", false) {
        ResourceRequest resources[] = {
            ResourceRequest("buffA", CL_MEM_HOST_NO_ACCESS, (256 + 64) * concurrency),
            ResourceRequest("buffB", CL_MEM_HOST_NO_ACCESS, (256 + 32) * concurrency),
            ResourceRequest("kdfResult", CL_MEM_HOST_NO_ACCESS, 256 * concurrency),
            ResourceRequest("pad", CL_MEM_HOST_NO_ACCESS, 32 * 1024 * concurrency),
            ResourceRequest("xo", CL_MEM_HOST_NO_ACCESS, 256 * concurrency),
            ResourceRequest("xi", CL_MEM_HOST_NO_ACCESS, 256 * concurrency),
            Immediate<cl_uint>("LOOP_ITERATIONS", 128),
            Immediate<cl_uint>("KDF_CONST_N", 32),
            Immediate<cl_uint>("STATE_SLICES", 4),
            Immediate<cl_uint>("MIX_ROUNDS", 10),
            Immediate<cl_uint>("KDF_SIZE", 256)
        };
        resources[0].presentationName = "buff<sub>a</sub>";
        resources[1].presentationName = "buff<sub>b</sub>";
        resources[2].presentationName = "KDF result";
        resources[3].presentationName = "X values buffer";
        resources[4].presentationName = "Salsa results";
        resources[5].presentationName = "Chacha results";
        PrepareResources(resources, sizeof(resources) / sizeof(resources[0]), concurrency);

        typedef WorkGroupDimensionality WGD;
        KernelRequest kernels[] = {
            {
                "ns_KDF_4W.cl", "firstKDF_4way", "",
                WGD(4, 16),
                "$wuData, kdfResult, KDF_CONST_N, buffA, buffB"
            },
            {
                "ns_coreLoop_1W.cl", "sequentialWrite_1way", "-D BLOCKMIX_SALSA",
                WGD(64),
                "kdfResult, pad, LOOP_ITERATIONS, STATE_SLICES, MIX_ROUNDS, xo"
            },
            {
                "ns_coreLoop_1W.cl", "indirectedRead_1way", "-D BLOCKMIX_SALSA",
                WGD(64),
                "xo, pad, LOOP_ITERATIONS, STATE_SLICES, MIX_ROUNDS"
            },
            {
                "ns_coreLoop_1W.cl", "sequentialWrite_1way", "-D BLOCKMIX_CHACHA",
                WGD(64),
                "kdfResult, pad, LOOP_ITERATIONS, STATE_SLICES, MIX_ROUNDS, xi"
            },
            {
                "ns_coreLoop_1W.cl", "indirectedRead_1way", "-D BLOCKMIX_CHACHA",
                WGD(64),
                "xi, pad, LOOP_ITERATIONS, STATE_SLICES, MIX_ROUNDS"
            },
            {
                "ns_KDF_4W.cl", "lastKDF_4way", "",
                WGD(4, 16),
                "$candidates, $dispatchData, xo, xi, KDF_CONST_N, buffA, buffB, pad"
            }
        };
        PrepareKernels(kernels, sizeof(kernels) / sizeof(kernels[0]), dev);
    }
};

}
