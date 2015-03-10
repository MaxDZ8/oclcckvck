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
#include "../misc.h"
#include "misc.h"

extern "C" {
#include "../../SPH/sph_luffa.h"
}


namespace stepTest {


struct Luffa_1W : public AbstractAlgorithm {
    typedef Hash512Mismatch ValidationMismatch;
    typedef BadResultsList<Hash512Mismatch> BadResults;

    const asizei passingBytes;
    cl_uint *blob = 0;

    Luffa_1W(cl_context ctx, cl_device_id dev, asizei concurrency)
        : AbstractAlgorithm(concurrency, ctx, dev, "LUFFA", "1-way", "v1", 0), passingBytes(concurrency * 16 * sizeof(cl_uint)) { }

    std::vector<std::string> Init(AbstractSpecialValuesProvider &specials) {
        ResourceRequest resources[] = {
            ResourceRequest("io0", CL_MEM_HOST_READ_ONLY, passingBytes),
        };
        std::vector<std::string> errors(PrepareResources(resources, sizeof(resources) / sizeof(resources[0]), hashCount, specials));
        if(errors.size()) return errors;

        typedef WorkGroupDimensionality WGD;
        KernelRequest kernels[] = {
            {
                "Luffa_1W.cl", "Luffa_1way", "-D LUFFA_HEAD",
                WGD(256),
                "$wuData, io0"
            }
        };
        return PrepareKernels(kernels, sizeof(kernels) / sizeof(kernels[0]), specials);
    }
    bool BigEndian() const { return true; }

    bool Mismatch(ValidationMismatch &bad, std::array<auint, 20> &block, auint nonce) const {
        block[19] = nonce;
        std::array<auint, 16> reference;
        {
            sph_luffa512_context head;
            sph_luffa512_init(&head);
            sph_luffa512(&head, block.data(), sizeof(block));
            sph_luffa512_close(&head, reference.data());
        }

        for(asizei i = 0; i < reference.size(); i++) reference[i] = SWAP_BYTES(reference[i]);
        for(asizei i = 0; i < reference.size(); i += 2) std::swap(reference[i], reference[i + 1]);

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
        blob = reinterpret_cast<cl_uint*>(clEnqueueMapBuffer(cq, resBuffer, CL_TRUE, CL_MAP_READ, 0, passingBytes, 0, NULL, NULL, &err));
        if(err != CL_SUCCESS) throw std::string("Failed mapping results with error ") + std::to_string(err);
    }
    void UnmapResults(cl_command_queue cq) {
        cl_mem resBuffer = resHandles.find("io0")->second;
        clEnqueueUnmapMemObject(cq, resBuffer, blob, 0, NULL, NULL);
        blob = 0;
    }
};


}
