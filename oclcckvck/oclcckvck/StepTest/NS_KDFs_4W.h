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


namespace stepTest {

//! Stuff mostly copied from M8M BlockVerifier for Neoscrypt
struct NS_KDFHelper {
    static const auint KDF_CONST_N;
    static const auint KDF_SIZE;
	static const auint MIX_ROUNDS;
	static const auint blake2S_IV[8];
	static const aubyte blake2S_sigma[10][16];
    

    std::array<auint, 64> FirstKDF(const aubyte *block, aubyte *buff_a, aubyte *buff_b);
	std::array<aubyte, 32> LastKDF(const std::array<auint, 64> &state, const aubyte *buff_a, aubyte *buff_b);

private:
    auint FastKDFIteration(auint buffStart, const aubyte *buff_a, aubyte *buff_b);
    void FillInitialBuffer(aubyte *target, auint extraBytes, const aubyte *pattern, auint blockLen);
    void Blake2S_64_32(auint *output, auint *input, auint *key, const auint numRounds);
    std::array<auint, 8> Blake2SBlockXForm(const std::array<auint, 8> hash, const std::array<auint, 4> &counter, const auint numRounds, const std::array<auint, 16> &msg);
};


struct NS_FirstKDF_4W : public AbstractAlgorithm {
    struct FKDFMismatch {
        struct {
            std::array<auint, (256 + 64) / 4> buffA;
            std::array<auint, (256 + 32) / 4> buffB;
            std::array<auint, 256 / 4> startX;
        } computed, reference;
        auint nonce;

        template<typename POD, asizei COUNT>
        static std::string Output(const std::array<POD, COUNT> &what) {
            return Hex(reinterpret_cast<const aubyte*>(what.data()), what.size() * sizeof(POD));
        }

        explicit FKDFMismatch() = default;
        void Describe(std::stringstream &conc) const {
            conc<<'['<<nonce<<"]"<<std::endl;
            conc<<"  buffA="<<Output(computed.buffA)<<" should be "<<Output(reference.buffA)<<std::endl;
            conc<<"  buffB="<<Output(computed.buffB)<<" should be "<<Output(reference.buffB)<<std::endl;
            conc<<"  startX="<<Output(computed.startX)<<" should be "<<Output(reference.startX)<<std::endl;
        }
    };

    typedef FKDFMismatch ValidationMismatch;
    typedef BadResultsList<FKDFMismatch> BadResults;

    cl_uint *blobA = nullptr, *blobB = nullptr, *blobStartX = nullptr;

    NS_FirstKDF_4W(cl_context ctx, cl_device_id dev, asizei concurrency)
        : AbstractAlgorithm(concurrency, ctx, dev, "NS_FirstKDF", "4-way", "v1", 0) { }

    std::vector<std::string> Init(ConfigDesc *desc, AbstractSpecialValuesProvider &specials, const std::string &loadPathPrefix) {
        ResourceRequest resources[] = {
            ResourceRequest("buffA", CL_MEM_HOST_READ_ONLY, (256 + 64) * hashCount),
            ResourceRequest("buffB", CL_MEM_HOST_READ_ONLY, (256 + 32) * hashCount),
            ResourceRequest("kdfResult", CL_MEM_HOST_READ_ONLY, 256 * hashCount),
            Immediate<cl_uint>("KDF_CONST_N", 32),
        };
        auto errors(PrepareResources(resources, sizeof(resources) / sizeof(resources[0]), specials));
        if(errors.size()) return errors;

        typedef WorkGroupDimensionality WGD;
        KernelRequest kernels[] = {
            {
                "ns_KDF_4W.cl", "firstKDF_4way", "",
                WGD(4, 16),
                "$wuData, kdfResult, KDF_CONST_N, buffA, buffB"
            }
        };
        return PrepareKernels(kernels, sizeof(kernels) / sizeof(kernels[0]), specials, loadPathPrefix);
    }
    bool BigEndian() const { return false; }

    bool Mismatch(FKDFMismatch &bad, std::array<auint, 20> block, auint nonce) const {
        block[19] = nonce;
	    auint buff_a[(256 + 64) / 4], buff_b[(256 + 32) / 4];
        const aubyte *baseBlockHeader = reinterpret_cast<aubyte*>(block.data());
        NS_KDFHelper helper;
	    auto initial(helper.FirstKDF(baseBlockHeader, reinterpret_cast<aubyte*>(buff_a), reinterpret_cast<aubyte*>(buff_b)));
        bool badA = memcmp(blobA + ((256 + 64) / 4) * nonce, buff_a, 256 + 64) != 0;
        bool badB = memcmp(blobB + ((256 + 32) / 4) * nonce, buff_b, 256 + 32) != 0;

        const auint outLen = 256;
        const auint sliceStride = sizeof(auint);
        const auint hashGroup = nonce / 64;
        const auint hashEntry = nonce % 64;
        aubyte *statex = reinterpret_cast<aubyte*>(blobStartX) + hashGroup * 64 * outLen + hashEntry * sliceStride;
        aubyte *refx = reinterpret_cast<aubyte*>(initial.data());
        bool badX = false;
        for(asizei set = 0; set < outLen; set++) {
            asizei dsti = (set / sliceStride) * 64 * sliceStride + set % sliceStride;
            badX |= statex[dsti] != refx[set];
        }
        if(badA || badB || badX) {
            bad.nonce = nonce;
            memcpy_s(bad.reference.buffA.data(), sizeof(bad.reference.buffA), buff_a, sizeof(buff_a));
            memcpy_s(bad.reference.buffB.data(), sizeof(bad.reference.buffB), buff_b, sizeof(buff_b));
            memcpy_s(bad.reference.startX.data(), sizeof(bad.reference.startX), initial.data(), sizeof(initial));
            memcpy_s(bad.computed.buffA.data(), sizeof(bad.computed.buffA), blobA + ((256 + 64) / 4) * nonce, 256 + 64);
            memcpy_s(bad.computed.buffB.data(), sizeof(bad.computed.buffB), blobB + ((256 + 32) / 4) * nonce, 256 + 32);
            memcpy_s(bad.computed.startX.data(), sizeof(bad.computed.startX), blobStartX + 256 * nonce, 256);
            return true;
        }
        return false;
    }

    void MapResults(cl_command_queue cq) {
        cl_mem buffA = resHandles.find("buffA")->second;
        cl_mem buffB = resHandles.find("buffB")->second;
        cl_mem startX = resHandles.find("kdfResult")->second;
        cl_int err = 0;
        asizei bytes = (256 + 64) * hashCount;
        blobA = reinterpret_cast<cl_uint*>(clEnqueueMapBuffer(cq, buffA, CL_TRUE, CL_MAP_READ, 0, bytes, 0, NULL, NULL, &err));
        if(err != CL_SUCCESS) throw std::string("Failed mapping results with error ") + std::to_string(err);
        ScopedFuncCall unmapa([this, cq, buffA]() { clEnqueueUnmapMemObject(cq, buffA, blobA, 0, NULL, NULL); });

        bytes = (256 + 32) * hashCount;
        blobB = reinterpret_cast<cl_uint*>(clEnqueueMapBuffer(cq, buffB, CL_TRUE, CL_MAP_READ, 0, bytes, 0, NULL, NULL, &err));
        if(err != CL_SUCCESS) throw std::string("Failed mapping results with error ") + std::to_string(err);
        ScopedFuncCall unmapb([this, cq, buffB]() { clEnqueueUnmapMemObject(cq, buffB, blobB, 0, NULL, NULL); });

        bytes = (256) * hashCount;
        blobStartX = reinterpret_cast<cl_uint*>(clEnqueueMapBuffer(cq, startX, CL_TRUE, CL_MAP_READ, 0, bytes, 0, NULL, NULL, &err));
        if(err != CL_SUCCESS) throw std::string("Failed mapping results with error ") + std::to_string(err);

        unmapa.Dont();
        unmapb.Dont();
    }
    void UnmapResults(cl_command_queue cq) {
        cl_mem buffA = resHandles.find("buffA")->second;
        cl_mem buffB = resHandles.find("buffB")->second;
        cl_mem startX = resHandles.find("kdfResult")->second;
        if(blobA) clEnqueueUnmapMemObject(cq, buffA, blobA, 0, NULL, NULL);
        if(buffB) clEnqueueUnmapMemObject(cq, buffB, buffB, 0, NULL, NULL);
        if(startX) clEnqueueUnmapMemObject(cq, startX, startX, 0, NULL, NULL);
        blobA = blobB = blobStartX = nullptr;
    }
    aulong GetDifficultyNumerator() const { return 0; } // unused, not a real mining algo
};


struct NS_LastKDF_4W : public AbstractAlgorithm {
    std::vector<auint> buffA, buffB, xo, xi;
    cl_mem candidates = 0;
    const aulong target = 0x9FFFFFF000000ull;


    NS_LastKDF_4W(std::mt19937 &random, cl_context ctx, cl_device_id dev, asizei concurrency)
        : AbstractAlgorithm(concurrency, ctx, dev, "NS_LastKDF", "4-way", "v1", 32 / 4) {
        buffA.resize(hashCount * (256 + 64) / 4);
        buffB.resize(hashCount * (256 + 32) / 4);
        xo.resize(hashCount * 256 / 4);
        xi.resize(hashCount * 256 / 4);
        for(auto &i : buffA) i = random();
        for(auto &i : buffB) i = random();
        for(auto &i : xo) i = random();
        for(auto &i : xi) i = random();
    }

    std::vector<std::string> Init(ConfigDesc *desc, AbstractSpecialValuesProvider &specials, const std::string &loadPathPrefix) {
        ResourceRequest resources[] = {
            ResourceRequest("buffA", CL_MEM_HOST_NO_ACCESS, (256 + 64) * hashCount, buffA.data()),
            ResourceRequest("buffB", CL_MEM_HOST_NO_ACCESS, (256 + 32) * hashCount, buffB.data()),
            ResourceRequest("xo", CL_MEM_HOST_NO_ACCESS, 256 * hashCount, xo.data()),
            ResourceRequest("xi", CL_MEM_HOST_NO_ACCESS, 256 * hashCount, xi.data()),
            ResourceRequest("pad", CL_MEM_HOST_NO_ACCESS, 256 * hashCount),
            Immediate<cl_uint>("KDF_CONST_N", 32),
        };
        auto errors(PrepareResources(resources, sizeof(resources) / sizeof(resources[0]), specials));
        if(errors.size()) return errors;

        typedef WorkGroupDimensionality WGD;
        KernelRequest kernels[] = {
            {
                "ns_KDF_4W.cl", "lastKDF_4way", "",
                WGD(4, 16),
                "$candidates, $dispatchData, xo, xi, KDF_CONST_N, buffA, buffB, pad"
            }
        };
        return PrepareKernels(kernels, sizeof(kernels) / sizeof(kernels[0]), specials, loadPathPrefix);
    }
    bool BigEndian() const { return true; }

    aulong GetMagic(asizei hash) const {
        // Again, the shaders fetch data in "block coherent order" but only for inputs. BuffAB are strided as usual.
        std::array<auint, 80> ha;
        std::array<auint, 72> hb;
        for(asizei cp = 0; cp < ha.size(); cp++) ha[cp] = buffA[hash * 80 + cp];
        for(asizei cp = 0; cp < hb.size(); cp++) hb[cp] = buffB[hash * 72 + cp];

        std::array<auint, 64> state;
        const auint stride = 1 * 64; //4*64 in shader as we pull up from 4 diffent rows
        const asizei base = (hash / 64) * 64 * 64 + 0 * 64 + (hash % 64); // get_local_id(0) * 64
        for(asizei i = 0; i < state.size(); i++) {
            const asizei srci = base + i * stride;
            state[i] = xo[srci] ^ xi[srci];
        }
        NS_KDFHelper help;
        auto final(help.LastKDF(state, reinterpret_cast<aubyte*>(ha.data()), reinterpret_cast<aubyte*>(hb.data())));
        const auint *dword = reinterpret_cast<const auint*>(final.data());
        return (aulong(dword[7]) << 32) | dword[6];
    }
    aulong GetDifficultyNumerator() const { return 0; } // unused, not a real mining algo
};


}
