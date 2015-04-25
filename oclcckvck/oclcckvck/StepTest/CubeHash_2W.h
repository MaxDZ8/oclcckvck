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
#include "../../SPH/sph_cubehash.h"
}


namespace stepTest {


struct CubeHash_2W : public AbstractAlgorithm {
    typedef Hash512Mismatch ValidationMismatch;
    typedef BadResultsList<Hash512Mismatch> BadResults;

    cl_uint *blob = 0;
    std::vector<auint> dummyPrevious;

    CubeHash_2W(std::mt19937 &random, cl_context ctx, cl_device_id dev, asizei concurrency)
        : AbstractAlgorithm(concurrency, ctx, dev, "CUBEHASH", "2-way", "v1", 0) {
        dummyPrevious.resize(hashCount * 16);
        for(auto &i : dummyPrevious) i = random();
    }

    std::vector<std::string> Init(ConfigDesc *desc, AbstractSpecialValuesProvider &specials, const std::string &loadPathPrefix) {
        const asizei passingBytes = this->hashCount * 16 * sizeof(cl_uint);
        ResourceRequest resources[] = {
            ResourceRequest("io0", CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_ONLY, passingBytes, dummyPrevious.data()),
            ResourceRequest("io1", CL_MEM_HOST_READ_ONLY | CL_MEM_WRITE_ONLY, passingBytes),
        };
        std::vector<std::string> errors(PrepareResources(resources, sizeof(resources) / sizeof(resources[0]), specials));
        if(errors.size()) return errors;

        typedef WorkGroupDimensionality WGD;
        KernelRequest kernels[] = {
            {
                "CubeHash_2W.cl", "CubeHash_2way", "",
                WGD(2, 32),
                "io0, io1"
            }
        };
        return PrepareKernels(kernels, sizeof(kernels) / sizeof(kernels[0]), specials, loadPathPrefix);
    }
    bool BigEndian() const { return true; }

    bool Mismatch(ValidationMismatch &bad, auint nonce) const {
        std::array<auint, 16> reference;
        std::array<auint, 16> input; // Cubehash apparently needs bytes in different order to match
        for(asizei cp = 0; cp < input.size(); cp++) input[cp] = HTOBE(dummyPrevious[nonce * 16 + cp]);
        for(asizei cp = 0; cp < input.size(); cp += 2) std::swap(input[cp], input[cp + 1]);
		{
			sph_cubehash512_context ctx;
			sph_cubehash512_init(&ctx);
			sph_cubehash512(&ctx, input.data(), sizeof(input));
			sph_cubehash512_close(&ctx, reference.data());
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
    aulong GetDifficultyNumerator() const { return 0; } // unused, not a real mining algo
};


}
