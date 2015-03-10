/*
 * Copyright (C) 2015 Massimo Del Zotto
 * This code is released under the MIT license.
 * For conditions of distribution and use, see the LICENSE or hit the web.
 */
#pragma once
#include "../../Common/AREN/ArenDataTypes.h"
#include <CL/cl.h>
#include "../AbstractAlgorithm.h"
#include "../StopWaitDispatcher.h"
#include <random>

extern "C" {
#include "../../SPH/sph_SIMD.h"
}


namespace stepTest {


struct SIMD_16W : public AbstractAlgorithm {
    typedef Hash512Mismatch ValidationMismatch;
    typedef BadResultsList<Hash512Mismatch> BadResults;

    cl_uint *blob = 0;
    std::vector<auint> dummyPrevious;

    SIMD_16W(std::mt19937 &random, cl_context ctx, cl_device_id dev, asizei concurrency)
        : AbstractAlgorithm(concurrency, ctx, dev, "SIMD", "16-way", "v1", 0) {
        dummyPrevious.resize(hashCount * 16);
        for(auto &i : dummyPrevious) i = random();
    }

    std::vector<std::string> Init(AbstractSpecialValuesProvider &specials) {
        const asizei passingBytes = dummyPrevious.size() * sizeof(cl_uint);
        KnownConstantProvider K; // all the constants are fairly nimble so I don't save nor optimize this in any way!
        auto SIMD_ALPHA(K[CryptoConstant::SIMD_alpha]);
        auto SIMD_BETA(K[CryptoConstant::SIMD_beta]);
        ResourceRequest resources[] = {
            ResourceRequest("io0", CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_ONLY, passingBytes, dummyPrevious.data()),
            ResourceRequest("io1", CL_MEM_HOST_READ_ONLY | CL_MEM_WRITE_ONLY, passingBytes),
            ResourceRequest("SIMD_ALPHA", CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, SIMD_ALPHA.second, SIMD_ALPHA.first),
            ResourceRequest("SIMD_BETA", CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, SIMD_BETA.second, SIMD_BETA.first),
        };
        std::vector<std::string> errors(PrepareResources(resources, sizeof(resources) / sizeof(resources[0]), hashCount, specials));
        if (errors.size()) return errors;

        typedef WorkGroupDimensionality WGD;
        KernelRequest kernels[] = {
            {
                "SIMD_16W.cl", "SIMD_16way", "",
                WGD(16, 4),
                "io0, io1, io0, SIMD_ALPHA, SIMD_BETA"
            }
        };
        return PrepareKernels(kernels, sizeof(kernels) / sizeof(kernels[0]), specials);
    }
    bool BigEndian() const { return true; }

    bool Mismatch(ValidationMismatch &bad, auint nonce) const {
        std::array<auint, 16> reference;
        {
            sph_simd512_context head;
            sph_simd512_init(&head);
            sph_simd512(&head, dummyPrevious.data() + nonce * 16, sizeof(auint) * 16);
            sph_simd512_close(&head, reference.data());
        }
        if(memcmp(blob + nonce * 16, reference.data(), sizeof(reference))) {
            std::array<cl_uint, 16> computed;
            for(asizei cp = 0; cp < 16; cp++) computed[cp] = blob[nonce * 16 + cp];
            bad = ValidationMismatch(computed, reference, nonce);
            return true;
        }
        return false;
    }

    void MapResults(cl_command_queue cq) {
        cl_mem resBuffer = resHandles.find("io1")->second;
        cl_int err = 0;
        blob = reinterpret_cast<cl_uint*>(clEnqueueMapBuffer(cq, resBuffer, CL_TRUE, CL_MAP_READ, 0, dummyPrevious.size() * sizeof(cl_uint), 0, NULL, NULL, &err));
        if(err != CL_SUCCESS) throw std::string("Failed mapping results with error ") + std::to_string(err);
    }
    void UnmapResults(cl_command_queue cq) {
        cl_mem resBuffer = resHandles.find("io1")->second;
        clEnqueueUnmapMemObject(cq, resBuffer, blob, 0, NULL, NULL);
        blob = 0;
    }
};


}
