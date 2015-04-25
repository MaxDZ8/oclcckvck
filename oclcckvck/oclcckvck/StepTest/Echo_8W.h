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
#include "../../SPH/sph_echo.h"
}


namespace stepTest {


struct ECHO_8W : public AbstractAlgorithm {
    std::vector<auint> dummyPrevious;
    cl_mem candidates = 0;
    const aulong target = 0x9FFFFFF000000ull;


    ECHO_8W(std::mt19937 &random, cl_context ctx, cl_device_id dev, asizei concurrency)
        : AbstractAlgorithm(concurrency, ctx, dev, "ECHO", "8-way", "v1", 8 * 2) {
        dummyPrevious.resize(hashCount * 16);
        for(auto &i : dummyPrevious) i = random();
    }

    std::vector<std::string> Init(ConfigDesc *desc, AbstractSpecialValuesProvider &specials, const std::string &loadPathPrefix) {
        const asizei passingBytes = this->hashCount * 16 * sizeof(cl_uint);
        KnownConstantProvider K; // all the constants are fairly nimble so I don't save nor optimize this in any way!
        auto AES_T_TABLES(K[CryptoConstant::AES_T]);
        ResourceRequest resources[] = {
            ResourceRequest("io1", CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_ONLY, passingBytes, dummyPrevious.data()),
            ResourceRequest("AES_T_TABLES", CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, AES_T_TABLES.second, AES_T_TABLES.first)
        };
        std::vector<std::string> errors(PrepareResources(resources, sizeof(resources) / sizeof(resources[0]), specials));
        if(errors.size()) return errors;

        typedef WorkGroupDimensionality WGD;
        KernelRequest kernels[] = {
            {
                "Echo_8W.cl", "Echo_8way", "-D AES_TABLE_ROW_1 -D AES_TABLE_ROW_2 -D AES_TABLE_ROW_3 -D ECHO_IS_LAST",
                WGD(8, 8),
                "io1, $candidates, $dispatchData, AES_T_TABLES"
            }
        };
        errors = PrepareKernels(kernels, sizeof(kernels) / sizeof(kernels[0]), specials, loadPathPrefix);
        if(errors.size() == 0) {
            SpecialValueBinding desc;
            if(specials.SpecialValue(desc, "$candidates") == false) throw "Impossible: $candidates not found, but binding successful.";
            if(desc.earlyBound == false) throw "Step validity tests require output buffer to be early bound!"; // impossible with StopWaitDispatcher
            candidates = desc.resource.buff;
        }
        return errors;
    }
    bool BigEndian() const { return true; }

    aulong GetMagic(asizei hash) const {
        const auint *input = dummyPrevious.data() + hash * 16;
        std::array<auint, 16> refHash;
        {
            sph_echo512_context ctx;
            sph_echo512_init(&ctx);
            sph_echo512(&ctx, input, sizeof(auint) * 16);
            sph_echo512_close(&ctx, refHash.data());
        }
        return (aulong(refHash[7]) << 32) | refHash[6];
    }
    aulong GetDifficultyNumerator() const { return 0; } // unused, not a real mining algo
};


}
