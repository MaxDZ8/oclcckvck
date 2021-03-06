/*
 * Copyright (C) 2015 Massimo Del Zotto
 * This code is released under the MIT license.
 * For conditions of distribution and use, see the LICENSE or hit the web.
 */
#pragma once
#include "StopWaitDispatcher.h"
#include <functional>


/*! A class derived from there groups a set of reference data inputs, expected outputs and CL algos to evaluate input at runtime.
This involves compiling a set of kernels, allocating a set of resources and binding them to the kernels.
Also, have a taste at algorithm setup mangling that will be introduced in M8M sometime in the future. */
class AlgoTest {
public:
    const aulong nominalHashCount; //!< One TestRun::iteration counts this amount of hashes.
    //! A function called each time a block is tested with the index of the completed block.
    std::function<void(asizei)> onBlockHashed;

    virtual ~AlgoTest() { }

    asizei GetNumTests() const { return GetHeaders().second; }

    //! Tests must consume exact amounts of hashes at each step or run the risk of missing nonces or placing them in the wrong bucket.
    //! This returns true if the hashes can be divided correctly. If so, it's worth calling RunTests on the algorithm.
    bool CanRunTests(const asizei concurrency) const {
        auto blocks(GetHeaders());
        for(asizei b = 0; b < blocks.second; b++) {
            asizei rem = (blocks.first[b].iterations * nominalHashCount) % concurrency;
            if(rem) return false;
        }
        return true;
    }

    //! Run validity tests on selected device. Returns a list of errors.
    std::vector<std::string> RunTests(StopWaitDispatcher &dispatch) const {
        using namespace std;
        vector<string> errorMessages;
        auto headers(GetHeaders());
        auto found(GetFound());
        const cl_uint *nextNonces = found.first;
        for(asizei bindex = 0; bindex < headers.second; bindex++) {
            const TestRun &block(headers.first[bindex]);
            std::array<aubyte, 80> header;
            for(asizei cp = 0; cp < sizeof(header); cp++) header[cp] = block.clData[cp];
            if(dispatch.algo.BigEndian()) {
                for(asizei i = 0; i < sizeof(header); i += 4) {
                    std::swap(header[i + 0], header[i + 3]);
                    std::swap(header[i + 1], header[i + 2]);
                }
            }
            dispatch.BlockHeader(header);
            dispatch.TargetBits(block.targetBits);
            dispatch.algo.Restart();

            aulong remHashes = block.iterations * nominalHashCount;
            vector<auint> candidates;
            while(remHashes) {
                const cl_uint thisScan = cl_uint(min(remHashes, aulong(dispatch.algo.hashCount)));
                Mangle(candidates, dispatch, thisScan);
                remHashes -= thisScan;
            }
            if(onBlockHashed) onBlockHashed(bindex);
            if(candidates.size() != block.numResults) {
                string msg("BAD RESULT COUNT for test block [");
                msg += to_string(bindex) + "]: " + to_string(block.numResults) + " expected, got " + to_string(candidates.size());
                errorMessages.push_back(msg);
                continue;
            }
            auto unniq(candidates);
            std::sort(unniq.begin(), unniq.end());
            std::unique(unniq.begin(), unniq.end());
            if(unniq.size() != candidates.size()) {
                string msg("BAD RESULTS for test block [");
                msg += to_string(bindex) + "]: nonces are not unique, something is going VERY WRONG!";
                errorMessages.push_back(msg);
                continue;
            }
            asizei mismatch = 0;
            for(auto nonce : unniq) {
                bool found = false;
                for(asizei check = 0; check < block.numResults; check++) {
                    if(nonce == nextNonces[check]) {
                        found = true;
                        break;
                    }
                }
                if(found == false) mismatch++;
            }
            nextNonces += block.numResults;
            if(mismatch) {
                string msg("BAD RESULTS for test block [");
                msg += to_string(bindex) + "]: " + to_string(mismatch) + " nonce values not matched.";
                errorMessages.push_back(msg);
            }
        }
        return errorMessages;
    }

protected:
    struct TestRun {
        unsigned char clData[80];
        unsigned long long targetBits;
        unsigned iterations;
        unsigned numResults;
    };
    AlgoTest(aulong referenceConcurrency) : nominalHashCount(referenceConcurrency) {  }
    virtual std::pair<const TestRun*, asizei> GetHeaders() const = 0;
    virtual std::pair<const auint*, asizei> GetFound() const = 0;

    //! Testing StopWaitAlgorithm is very easy: we just have to keep going until results are poured out, then exit.
    //! Note this is already called as long as we need iterating so that's really as simple.
    static void Mangle(std::vector<auint> &candidates, StopWaitDispatcher &dispatch, asizei hashCount) {
        if(hashCount != dispatch.algo.hashCount) {
            //! \todo
            //! if(algo.dynamicIntensity) everything is fine, algo can adapt
            throw std::string("Probably forgot to call CanRunTests first!");
        }
        bool completed = false;
        cl_int ret = 0;
        std::set<cl_event> trigger;
        while(!completed) {
            auto ev(dispatch.Tick(trigger));
            switch(ev) {
            case AlgoEvent::dispatched: break; // how can I use this?
            case AlgoEvent::exhausted: throw "Impossible! Test data inconsistent!"; // Test data enumerates all the headers so this cannot happen
            case AlgoEvent::working: {
                std::vector<cl_event> blockers;
                dispatch.GetEvents(blockers); // in this case I can just wait here. This is not always possible.
                ret = clWaitForEvents(cl_uint(blockers.size()), blockers.data());
                for(auto ev : blockers) {
                    if(trigger.find(ev) == trigger.cend()) trigger.insert(ev);
                }
            } break;
            case AlgoEvent::results:
                completed = true;
                break;
            }
        }
        auto produced(dispatch.GetResults()); // we know header already!
        for(auto el : produced.nonces) candidates.push_back(el);
    }
};
