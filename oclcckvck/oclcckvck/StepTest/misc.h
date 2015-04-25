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
#include <string>
#include <sstream>


namespace stepTest {


template<typename DetailedMismatch>
struct BadResultsList {
    std::vector<DetailedMismatch> mismatch; //!< first few are collected completely
    std::vector<auint> more; //!< others are partially collected, only nonce
    asizei count; //!< might be > mismatch.size() + more.size as not everything is collected even though everything is counted
    explicit BadResultsList() : count(0) { }
    bool Failed() { return count != 0; }
    std::string Describe(asizei totalTests) {
        std::stringstream conc;
        conc<<"Results differ\n";
        for(const DetailedMismatch &big : mismatch) { big.Describe(conc);    conc<<std::endl; }
        conc<<std::endl<<more.size()<<" other wrong hashes:"<<std::endl;
        bool first = true;
        asizei output = 0;
        for(auto small : more) {
            if(!first) conc<<", ";
            if(output++ > 64) {
                output -= 65;
                conc<<std::endl;
            }
            first = false;
            conc<<small;
        }
        if(count > mismatch.size() + more.size()) conc<<"...";
        const adouble ratio = adouble(count) / totalTests;
        conc<<std::endl<<std::endl<<count<<" errors total ("<<auint(ratio * 100)<<"%)\n";
        return conc.str();
    }
};


/* Nonce mismatches are returned in a different way: when two algorithms do not converge to the same results it's most likely
they will produce different nonces so it's all about detecting if we found everything we expect and nothing we don't.
Nonces are also way better defined: they're uints! */
struct BadNonces {
    std::vector<auint> badFound; //!< those nonces have been found, but they are invalid and shouldn't exist
    std::vector<auint> badMissing; //!< those are supposed to be found but they are not.
    explicit BadNonces() { }
    bool Failed() { return badFound.size() + badMissing.size() != 0; }
    std::string Describe(asizei totalTests) {
        std::stringstream conc;
        conc << "Results differ";
        if (badMissing.size()) {
            conc << "\nExpected, but not found: ";
            bool first = true;
            for (auto nonce : badMissing) {
                if (!first) conc << ", ";
                first = false;
                conc << nonce;
            }
        }
        if (badFound.size()) {
            conc << "\nFound but not supposed to be there: ";
            bool first = true;
            for (auto nonce : badFound) {
                if (!first) conc << ", ";
                first = false;
                conc << nonce;
            }
        }
        return conc.str();
    }
};


/*! An "head test" is meant to test the first step of an algorithm. The head mangles an 80-bytes block to all hashes.
There can be multiple outputs. */
template<typename AlgoHeadValidator>
class HeadTest {
    std::mt19937 random;
    std::array<aubyte, 80> dummyHeader;

public:
    AlgoHeadValidator algo;
    HeadTest(cl_context ctx, cl_device_id dev, asizei concurrency) : algo(ctx, dev, concurrency) { }
    
    //! Fetch the dispatcher with the data it will pass to the algorithm, called once immediately after both
    //! this and the dispatcher has been called.
    void MakeInputData(StopWaitDispatcher &disp) {
        for(auto &i : dummyHeader) i = random();
        if(disp.algo.BigEndian()) {
            auto endianess(dummyHeader);
            for(auto i = 0; i < endianess.size(); i += 4) {
                std::swap(endianess[i + 0], endianess[i + 3]);
                std::swap(endianess[i + 1], endianess[i + 2]);
            }
            disp.BlockHeader(endianess);
        }
        else disp.BlockHeader(dummyHeader);
        disp.TargetBits(0ull); // not used for me anyway!
    }

    typename AlgoHeadValidator::BadResults Check(StopWaitDispatcher &disp, asizei maxBadStuff = 128) {
        AlgoHeadValidator::BadResults ret;
        auto cq(disp.GetQueue());
        algo.MapResults(cq);
        ScopedFuncCall unmapAlgo([this, cq]() { algo.UnmapResults(cq); });

        std::array<auint, 20> block;
        memcpy_s(block.data(), sizeof(block), dummyHeader.data(), sizeof(dummyHeader));

        for(auint hash = 0; hash < algo.hashCount; hash++) {
            AlgoHeadValidator::ValidationMismatch bad;
            if(algo.Mismatch(bad, block, hash)) {
                if(ret.count < maxBadStuff) ret.mismatch.push_back(bad);
                else if(ret.count < maxBadStuff * 4) ret.more.push_back(hash);
                ret.count++;
            }
        }
        return ret;
    }
};


/* The "tail" of a chained-hash algorithm consumes multiple input buffers to produce a set of nonces. */
template<typename AlgoTailValidator>
class TailTest {
    std::mt19937 random; // this must be built before algo as it's needed to fetch the internal buffers.

public:
    AlgoTailValidator algo;
    TailTest(cl_context ctx, cl_device_id dev, asizei concurrency) : algo(random, ctx, dev, concurrency) { }
    void MakeInputData(StopWaitDispatcher &disp) {
        std::array<aubyte, 80> dummyHeader;
        disp.BlockHeader(dummyHeader); // unused for chained steps
        disp.TargetBits(algo.target);
    }
    BadNonces Check(StopWaitDispatcher &disp, asizei maxBadStuff = 512) const {
        BadNonces ret;
        const auto candidates(disp.GetResults());
        MinedNonces sph;
        for(auint hash = 0; hash < algo.hashCount; hash++) {
            const aulong magic = algo.GetMagic(hash);
            if(magic <= algo.target) sph.nonces.push_back(HTOBE(hash));
        }
        for(auto gpu : candidates.nonces) {
            if (std::find(sph.nonces.cbegin(), sph.nonces.cend(), gpu) == sph.nonces.cend()) {
                ret.badFound.push_back(gpu);
            }
        }
        for(auto sph : sph.nonces) {
            if (std::find(candidates.nonces.cbegin(), candidates.nonces.cend(), sph) == candidates.nonces.cend()) {
                ret.badMissing.push_back(sph);
            }
        }
        return ret;
    }
};


/* Besides head and tail, steps of an algorithm are just ... well, steps. 
This takes the flexibility of of tail (multiple inputs) but the regularity of Head, no nonce filtering but rather
regurarly tested blobs. */
template<typename AlgoStepValidator>
class StepTest {
    std::mt19937 random; // this must be built before algo as it's needed to fetch the internal buffers.

public:
    AlgoStepValidator algo;
    StepTest(cl_context ctx, cl_device_id dev, asizei concurrency) : algo(random, ctx, dev, concurrency) { }
    void MakeInputData(StopWaitDispatcher &disp) {
        std::array<aubyte, 80> dummyHeader;
        disp.BlockHeader(dummyHeader); // unused for chained steps
        disp.TargetBits(0ull); // assumed unused
    }
    typename AlgoStepValidator::BadResults Check(StopWaitDispatcher &disp, asizei maxBadStuff = 128) {
        AlgoStepValidator::BadResults ret;
        auto cq(disp.GetQueue());
        algo.MapResults(cq);
        ScopedFuncCall unmapAlgo([this, cq]() { algo.UnmapResults(cq); });
        for(auint hash = 0; hash < algo.hashCount; hash++) {
            AlgoStepValidator::ValidationMismatch bad;
            if(algo.Mismatch(bad, hash)) {
                if(ret.count < maxBadStuff) ret.mismatch.push_back(bad);
                else if(ret.count < maxBadStuff * 4) ret.more.push_back(hash);
                ret.count++;
            }
        }
        return ret;
    }
};


struct Hash512Mismatch {
    std::array<aubyte, 64> computed;
    std::array<aubyte, 64> reference;
    asizei nonce;
    explicit Hash512Mismatch() = default;
    Hash512Mismatch(std::array<aubyte, 64> &gpu, std::array<aubyte, 64> &cpu, asizei hash) : computed(gpu), reference(cpu), nonce(hash) {}
    Hash512Mismatch(std::array<auint, 16> &gpu, std::array<auint, 16> &cpu, asizei hash) : nonce(hash) {
        memcpy_s(computed.data(), sizeof(computed), gpu.data(), sizeof(gpu));
        memcpy_s(reference.data(), sizeof(reference), cpu.data(), sizeof(cpu));
    }
    void Describe(std::stringstream &conc) const {
        conc<<'['<<nonce<<"] is "<<Hex(computed.data(), sizeof(computed));
        conc<<", should be "<<Hex(reference.data(), sizeof(reference));
    }
};

}
