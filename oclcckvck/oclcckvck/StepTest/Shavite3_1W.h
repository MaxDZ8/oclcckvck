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
#include "../../SPH/sph_shavite.h"
}


namespace stepTest {


struct ShaVite3_1W : public AbstractAlgorithm {
    typedef Hash512Mismatch ValidationMismatch;
    typedef BadResultsList<Hash512Mismatch> BadResults;

    cl_uint *blob = 0;
    std::vector<auint> dummyPrevious;

    ShaVite3_1W(std::mt19937 &random, cl_context ctx, cl_device_id dev, asizei concurrency)
        : AbstractAlgorithm(concurrency, ctx, dev, "SHAVITE3", "1-way", "v1", 0) {
        dummyPrevious.resize(hashCount * 16);
        for(auto &i : dummyPrevious) i = random();
    }

    std::vector<std::string> Init(ConfigDesc *desc, AbstractSpecialValuesProvider &specials, const std::string &loadPathPrefix) {
        const asizei passingBytes = this->hashCount * 16 * sizeof(cl_uint);
        KnownConstantProvider K; // all the constants are fairly nimble so I don't save nor optimize this in any way!
        auto AES_T_TABLES(K[CryptoConstant::AES_T]);
        ResourceRequest resources[] = {
            ResourceRequest("io1", CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_ONLY, passingBytes, dummyPrevious.data()),
            ResourceRequest("io0", CL_MEM_HOST_READ_ONLY | CL_MEM_WRITE_ONLY, passingBytes),
            ResourceRequest("AES_T_TABLES", CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, AES_T_TABLES.second, AES_T_TABLES.first),
            Immediate<cl_uint>("sh3_roundCount", 14),
        };
        std::vector<std::string> errors(PrepareResources(resources, sizeof(resources) / sizeof(resources[0]), specials));
        if(errors.size()) return errors;

        typedef WorkGroupDimensionality WGD;
        KernelRequest kernels[] = {
            {
                "SHAvite3_1W.cl", "SHAvite3_1way", "",
                WGD(64),
                "io1, io0, AES_T_TABLES, sh3_roundCount"
            }
        };
        return PrepareKernels(kernels, sizeof(kernels) / sizeof(kernels[0]), specials, loadPathPrefix);
    }
    bool BigEndian() const { return true; }

    bool Mismatch(ValidationMismatch &bad, auint nonce) const {
        std::array<auint, 16> reference;
        {
            sph_shavite512_context head;
            sph_shavite512_init(&head);
            sph_shavite512(&head, dummyPrevious.data() + nonce * 16, sizeof(auint) * 16);
            sph_shavite512_close(&head, reference.data());
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
        cl_mem resBuffer = resHandles.find("io0")->second;
        cl_int err = 0;
        blob = reinterpret_cast<cl_uint*>(clEnqueueMapBuffer(cq, resBuffer, CL_TRUE, CL_MAP_READ, 0, dummyPrevious.size() * sizeof(cl_uint), 0, NULL, NULL, &err));
        if(err != CL_SUCCESS) throw std::string("Failed mapping results with error ") + std::to_string(err);
    }
    void UnmapResults(cl_command_queue cq) {
        cl_mem resBuffer = resHandles.find("io0")->second;
        clEnqueueUnmapMemObject(cq, resBuffer, blob, 0, NULL, NULL);
        blob = 0;
    }
    aulong GetDifficultyNumerator() const { return 0; } // unused, not a real mining algo
};


}
