/*
 * Copyright (C) 2015 Massimo Del Zotto
 * This code is released under the MIT license.
 * For conditions of distribution and use, see the LICENSE or hit the web.
 */
#pragma once
#include "../StopWaitAlgorithm.h"

namespace algoImplementations {

class MYRGRSMonolithicCL12 : public StopWaitAlgorithm {
public:
    MYRGRSMonolithicCL12(cl_context ctx, cl_device_id dev, asizei concurrency)
        : StopWaitAlgorithm(ctx, dev, concurrency, "GRSMYR", "monolithic", "v1", true) {
        auint roundCount[5] = {
            14, 14, 14, // groestl rounds
            2, 3 // SHA rounds
        };
        ResourceRequest resources[] = {
            ResourceRequest("roundCount", CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(roundCount), roundCount)
        };
        resources[0].presentationName = "Round iterations";
        PrepareResources(resources, sizeof(resources) / sizeof(resources[0]), concurrency);

        typedef WorkGroupDimensionality WGD;
        KernelRequest kernels[] = {
            {
                "grsmyr_monolithic.cl", "grsmyr_monolithic", "",
                WGD(256),
                "$candidates, $wuData, $dispatchData, roundCount"
            }
        };
        PrepareKernels(kernels, sizeof(kernels) / sizeof(kernels[0]), dev);
    }
};

}
