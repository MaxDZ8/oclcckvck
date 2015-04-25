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
#include <memory>


namespace stepTest {


namespace nsHelp {
    static const auint MIX_ROUNDS = 10;

	// The usage of salsa or chacha also mandates use of a different shader compile define so and algorithm names so this has to be slightly more modular
    // It is made quite more complicated by the fact AbstractAlgorithm identifier must be currently constructed by const char* and it's const.
    enum class Pass {
        sequentialWrite,
        indirectedRead
    };

    struct Salsa {
        static const char* GetDefineName() { return "SALSA"; }
        static const char* GetAlgoName(Pass p) { return p == Pass::sequentialWrite? "SequentialWrite_salsa" : "IndirectedRead_salsa"; }
        void operator()(auint state[16]);
    };
    struct Chacha {
        static const char* GetDefineName() { return "CHACHA"; }
        static const char* GetAlgoName(Pass p) { return p == Pass::sequentialWrite? "SequentialWrite_chacha" : "IndirectedRead_chacha"; }
        void operator()(auint state[16]);
    };

	// As checking isn't considered a performance path I could avoid using a template here: they are still a bit ugly to debuggers and messages.
	template<typename MixFunc>
	static void SequentialWrite(auint iterations, auint *pad, auint *state, MixFunc &&mix) {
		static const auint perm[2][4] = {
			{0, 1, 2, 3},
			{0, 2, 1, 3}
		};
		for(auint loop = 0; loop < iterations; loop++) {
			for(auint slice = 0; slice < 4; slice++) {
				auint *one = state + perm[loop % 2][slice] * 16;
				auint *two = state + perm[loop % 2][(slice + 3) % 4] * 16;
				auint prev[16];
				for(auint el = 0; el < 16; el++) {
					pad[el] = one[el];
					one[el] ^= two[el];
					prev[el] = one[el];
				}
				pad += 16;
				mix(one);
				for(auint el = 0; el < 16; el++) one[el] += prev[el];
			}
		}
	}
	template<typename MixFunc>
	static void IndirectedRead(auint iterations, auint *state, auint *pad, MixFunc &&mix) {
		static const auint perm[2][4] = {
			{0, 1, 2, 3},
			{0, 2, 1, 3}
		};
		for(auint loop = 0; loop < iterations; loop++) {
			const auint indirected = state[48] % 128;
			for(auint slice = 0; slice < 4; slice++) {
				auint *one = state + perm[loop % 2][slice] * 16;
				for(auint el = 0; el < 16; el++) one[el] ^= pad[indirected * 64 + slice * 16 + el];
			}
			for(auint slice = 0; slice < 4; slice++) {
				auint *one = state + perm[loop % 2][slice] * 16;
				auint *two = state + perm[loop % 2][(slice + 3) % 4] * 16;
				auint prev[16];
				for(auint el = 0; el < 16; el++) {
					one[el] ^= two[el];
					prev[el] = one[el];
				}
				mix(one);
				for(auint el = 0; el < 16; el++) one[el] += prev[el];
			}
		}
	}
    

    struct NSCoreMismatch {
        auint nonce;
        std::array<auint, 64> stateGPU, stateHost;
        asizei padDifference; //!< index of first uint in pad buffer being different, if >= 128*64 pad is the same

        void Describe(std::stringstream &conc) const {
            conc<<'['<<nonce<<"] is "<<Hex(reinterpret_cast<const aubyte*>(stateGPU.data()), sizeof(stateGPU));
            conc<<", should be "<<Hex(reinterpret_cast<const aubyte*>(stateHost.data()), sizeof(stateHost));
            if(padDifference < 64 * 128) conc<<", first pad difference uint index="<<padDifference;
        }
    };
}


template<typename MixFunc>
class NS_SW : public AbstractAlgorithm {
    std::vector<auint> dummyPrevious;
    mutable std::array<auint, 64> hostState; //!< slice of dummyPrevious used as helper to input hash to CPU validation, updated verification
    mutable std::array<auint, 128 * 64> hostPad;
    mutable std::array<auint, 64> gpuStateOut; //!< helper buffer taken from GPU buffer with same layout as CPU output
    mutable std::array<auint, 128 * 64> gpuPad; //!< pad buffer values, but with same layout as CPU, for a single hash
    cl_uint *padMap = nullptr, *xoMap = nullptr;

public:
    NS_SW(std::mt19937 &random, cl_context ctx, cl_device_id dev, asizei concurrency)
        : AbstractAlgorithm(concurrency, ctx, dev, "Neoscrypt", MixFunc::GetAlgoName(nsHelp::Pass::sequentialWrite), "v1", 0) {
        dummyPrevious.resize(hashCount * 64);
        for(auto &i : dummyPrevious) i = random();
    }

    std::vector<std::string> Init(ConfigDesc *desc, AbstractSpecialValuesProvider &specials, const std::string &loadPathPrefix) {
        ResourceRequest resources[] = {
            ResourceRequest("kdfResult", CL_MEM_HOST_NO_ACCESS, 256 * hashCount, dummyPrevious.data()),
            ResourceRequest("pad", CL_MEM_HOST_READ_ONLY | CL_MEM_WRITE_ONLY, 32 * 1024 * hashCount),
            ResourceRequest("xo", CL_MEM_HOST_READ_ONLY | CL_MEM_WRITE_ONLY, 256 * hashCount),
            Immediate<cl_uint>("LOOP_ITERATIONS", 128),
            Immediate<cl_uint>("STATE_SLICES", 4),
            Immediate<cl_uint>("MIX_ROUNDS", 10),
        };
        auto errors(PrepareResources(resources, sizeof(resources) / sizeof(resources[0]), specials));
        if(errors.size()) return errors;

        typedef WorkGroupDimensionality WGD;
        const std::string blockMix("-D BLOCKMIX_" + std::string(MixFunc::GetDefineName()));
        KernelRequest kernels[] = {
            {
                "ns_coreLoop_1W.cl", "sequentialWrite_1way", blockMix.c_str(),
                WGD(64),
                "kdfResult, pad, LOOP_ITERATIONS, STATE_SLICES, MIX_ROUNDS, xo"
            }
        };
        return PrepareKernels(kernels, sizeof(kernels) / sizeof(kernels[0]), specials, loadPathPrefix);
    }
    bool BigEndian() const { return false; }

    typedef nsHelp::NSCoreMismatch ValidationMismatch;
    typedef BadResultsList<nsHelp::NSCoreMismatch> BadResults;

    bool Mismatch(ValidationMismatch &bad, auint nonce) const {
        const auint iterations = 128;
        const asizei get_local_size = 64; // number of hashes per work group
        const asizei get_group_id = nonce / get_local_size;
        const asizei get_local_id = nonce % get_local_size;
        const asizei get_global_size = hashCount;
        {
            const auint *xin = dummyPrevious.data() + get_group_id * get_local_size * 64;
            xin += get_local_id;
            for(asizei slice = 0; slice < 4; slice++) {
                const auint *currSlice = xin + slice * 16 * get_local_size;
                for(asizei el = 0; el < 16; el++) hostState[slice * 16 + el] = currSlice[el * get_local_size];
            }
            //SequentialWrite(iterations, hostPad.data(), hostState.data(), std::function<void(auint*)>(SALSA? Salsa : Chacha));
            nsHelp::SequentialWrite(iterations, hostPad.data(), hostState.data(), MixFunc());
        }

        // Now take care of the output. 
        { // the pad buffer comes in sequences of 16 uints, staggered by local id. Slices have same offset as previously
            const auint *pad = padMap + get_group_id * get_local_size * 16;
            pad += get_local_id * 16;
            asizei dsti = 0;
            for(asizei it = 0; it < iterations; it++) {
                for(asizei slice = 0; slice < 4; slice++) {
                    for(asizei el = 0; el < 16; el++) gpuPad[dsti++] = pad[(el + get_local_id) % 16];
                    pad += 16 * get_global_size;
                }
            }
        }
        { // the output value is more or less the same as input, just comes from different buffer
            const auint *xout = xoMap + get_group_id * get_local_size * 64;
            xout += get_local_id;
            for(asizei slice = 0; slice < 4; slice++) {
                const auint *currSlice = xout + slice * 16 * get_local_size;
                for(asizei el = 0; el < 16; el++) gpuStateOut[slice * 16 + el] = currSlice[el * get_local_size];
            }
        }

        const bool goodState = memcmp(gpuStateOut.data(), hostState.data(), sizeof(hostState)) == 0;
        const bool goodPad = memcmp(gpuPad.data(), hostPad.data(), sizeof(hostPad)) == 0;
        if(goodState && goodPad) return false;
        // :(
        bad.nonce = nonce;
        bad.stateGPU = gpuStateOut;
        bad.stateHost = hostState;
        bad.padDifference = 0;
        while(bad.padDifference < hostPad.size()) {
            if(hostPad[bad.padDifference] != gpuPad[bad.padDifference]) break;
            bad.padDifference++;
        }
        return true;
    }
    void MapResults(cl_command_queue cq){
        cl_mem padBuff = resHandles.find("pad")->second;
        cl_mem xoBuff = resHandles.find("xo")->second;
        cl_int err = 0;
        padMap = reinterpret_cast<cl_uint*>(clEnqueueMapBuffer(cq, padBuff, CL_TRUE, CL_MAP_READ, 0, 32 * 1024 * hashCount, 0, NULL, NULL, &err));
        if(err != CL_SUCCESS) throw std::string("Failed mapping pad buffer results with error ") + std::to_string(err);

        xoMap = reinterpret_cast<cl_uint*>(clEnqueueMapBuffer(cq, xoBuff, CL_TRUE, CL_MAP_READ, 0, 256 * hashCount, 0, NULL, NULL, &err));
        if(err != CL_SUCCESS) throw std::string("Failed mapping result buffer results with error ") + std::to_string(err);
    }

    void UnmapResults(cl_command_queue cq) {
        cl_mem padBuff = resHandles.find("pad")->second;
        cl_mem xoBuff = resHandles.find("xo")->second;
        clEnqueueUnmapMemObject(cq, padBuff, padMap, 0, NULL, NULL);
        clEnqueueUnmapMemObject(cq, xoBuff, xoMap, 0, NULL, NULL);
        padMap = xoMap = nullptr;
    }
    aulong GetDifficultyNumerator() const { return 0; } // unused, not a real mining algo
};


template<typename MixFunc>
class NS_IR : public AbstractAlgorithm {
    std::vector<auint> bigState;
    std::vector<auint> bigPad;

    mutable std::array<auint, 64> hostState; //!< slice of dummyPrevious used as helper to input hash to CPU validation, updated verification
    mutable std::array<auint, 64> gpuStateOut; //!< helper buffer taken from GPU buffer with same layout as CPU output
    mutable std::array<auint, 128 * 64> hostPad;

    cl_uint *xoMap = nullptr;

public:
    NS_IR(std::mt19937 &random, cl_context ctx, cl_device_id dev, asizei concurrency)
        : AbstractAlgorithm(concurrency, ctx, dev, "Neoscrypt", MixFunc::GetAlgoName(nsHelp::Pass::indirectedRead), "v1", 0) {
        bigState.resize(hashCount * 64);
        bigPad.resize(hashCount * 128 * 64);
        for(auto &i : bigState) i = random();
        for(auto &i : bigPad) i = random();
    }

    std::vector<std::string> Init(ConfigDesc *desc, AbstractSpecialValuesProvider &specials, const std::string &loadPathPrefix) {
        ResourceRequest resources[] = {
            ResourceRequest("pad", CL_MEM_HOST_NO_ACCESS, 32 * 1024 * hashCount, bigPad.data()),
            ResourceRequest("xo", CL_MEM_HOST_READ_ONLY, 256 * hashCount, bigState.data()),
            Immediate<cl_uint>("LOOP_ITERATIONS", 128),
            Immediate<cl_uint>("STATE_SLICES", 4),
            Immediate<cl_uint>("MIX_ROUNDS", 10)
        };
        auto errors(PrepareResources(resources, sizeof(resources) / sizeof(resources[0]), specials));
        if(errors.size()) return errors;

        typedef WorkGroupDimensionality WGD;
        const std::string blockMix("-D BLOCKMIX_" + std::string(MixFunc::GetDefineName()));
        KernelRequest kernels[] = {
            {
                "ns_coreLoop_1W.cl", "indirectedRead_1way", blockMix.c_str(),
                WGD(64),
                "xo, pad, LOOP_ITERATIONS, STATE_SLICES, MIX_ROUNDS"
            }
        };
        return PrepareKernels(kernels, sizeof(kernels) / sizeof(kernels[0]), specials, loadPathPrefix);
    }
    bool BigEndian() const { return false; }

    typedef nsHelp::NSCoreMismatch ValidationMismatch;
    typedef BadResultsList<nsHelp::NSCoreMismatch> BadResults;

    bool Mismatch(ValidationMismatch &bad, auint nonce) const {
        const auint iterations = 128;
        const asizei get_local_size = 64; // number of hashes per work group
        const asizei get_group_id = nonce / get_local_size;
        const asizei get_local_id = nonce % get_local_size;
        const asizei get_global_size = hashCount;
        {
            const auint *xio = bigState.data() + get_group_id * get_local_size * 64;
            xio += get_local_id;
            for(asizei slice = 0; slice < 4; slice++) {
                const auint *currSlice = xio + slice * 16 * get_local_size;
                for(asizei el = 0; el < 16; el++) hostState[slice * 16 + el] = currSlice[el * get_local_size];
            }

            const auint *pad = bigPad.data() + nonce * 16;
            asizei dsti = 0;
            for(asizei it = 0; it < iterations; it++) {
                for(asizei slice = 0; slice < 4; slice++) {
                    for(asizei el = 0; el < 16; el++) hostPad[dsti++] = pad[(el + get_local_id) % 16];
                    pad += 16 * get_global_size;
                }
            }
            nsHelp::IndirectedRead(iterations, hostState.data(), hostPad.data(), MixFunc());
        }

        // Now take care of the output. 
        { // the output value is more or less the same as input, just comes from different buffer
            const auint *xout = xoMap + get_group_id * get_local_size * 64;
            xout += get_local_id;
            for(asizei slice = 0; slice < 4; slice++) {
                const auint *currSlice = xout + slice * 16 * get_local_size;
                for(asizei el = 0; el < 16; el++) gpuStateOut[slice * 16 + el] = currSlice[el * get_local_size];
            }
        }

        const bool goodState = memcmp(gpuStateOut.data(), hostState.data(), sizeof(hostState)) == 0;
        if(goodState) return false;
        // :(
        bad.nonce = nonce;
        bad.stateGPU = gpuStateOut;
        bad.stateHost = hostState;
        bad.padDifference = auint(hostPad.size());
        return true;
    }
    void MapResults(cl_command_queue cq){
        cl_mem xoBuff = resHandles.find("xo")->second;
        cl_int err = 0;
        xoMap = reinterpret_cast<cl_uint*>(clEnqueueMapBuffer(cq, xoBuff, CL_TRUE, CL_MAP_READ, 0, 256 * hashCount, 0, NULL, NULL, &err));
        if(err != CL_SUCCESS) throw std::string("Failed mapping result buffer results with error ") + std::to_string(err);
    }

    void UnmapResults(cl_command_queue cq) {
        cl_mem xoBuff = resHandles.find("xo")->second;
        clEnqueueUnmapMemObject(cq, xoBuff, xoMap, 0, NULL, NULL);
        xoMap = nullptr;
    }
    aulong GetDifficultyNumerator() const { return 0; } // unused, not a real mining algo
};


}
