/*
 * Copyright (C) 2014 Massimo Del Zotto
 * This code is released under the MIT license.
 * For conditions of distribution and use, see the LICENSE or hit the web.
 */
#pragma once
#include "../Common/aes.h"
#include <CL/cl.h>
#include <vector>
#include <map>
#include "../Common/AREN/ScopedFuncCall.h"
#include <fstream>
#include <string>
#include <algorithm>
#include <array>
#include "../Common/hashing.h"

/*! A class derived from there groups a set of reference data inputs, expected outputs and CL algos to evaluate input at runtime.
This involves compiling a set of kernels, allocating a set of resources and binding them to the kernels. 
Also, have a taste at algorithm setup mangling that will be introduced in M8M sometime in the future. */
class AbstractTest {
public:
    const std::string algoName;
    const std::string impName;
    const std::string iversion;
    const aulong nominalHashCount; //!< One TestRun::iteration counts this amount of hashes.
    const aulong concurrency;      //!< Amount of hashes to be dispatched at each clEnqueueNDRangeKernel call.

    virtual ~AbstractTest() {
        if(wuData) clReleaseMemObject(wuData);
        if(dispatchData) clReleaseMemObject(dispatchData);
        if(candidates) clReleaseMemObject(candidates);
        for(auto el : kernels) { if(el.clk) clReleaseKernel(el.clk); }
        for(auto el : resHandles) { if(el.second) clReleaseMemObject(el.second); } // this always happens, put there completely constructed
    }

    //! Run validity tests on selected device. Returns a list of errors.
    std::vector<std::string> RunTests(cl_device_id device) const;

    //! This is only valid after PrepareKernels has been called.
    //! \todo Maybe this should be setup by ctor.
    aulong GetVersioningHash() const { return aiSignature; }


protected:
    struct TestRun {
        unsigned char clData[80];
        unsigned long long targetBits;
        unsigned iterations;
        unsigned numResults;
    };
    struct WorkGroupDimensionality {
        auint dimensionality;
        asizei wgs[3]; //!< short for work group size
        WorkGroupDimensionality(auint x)                   : dimensionality(1) { wgs[0] = x;    wgs[1] = 0;    wgs[2] = 0; }
        WorkGroupDimensionality(auint x, auint y)          : dimensionality(2) { wgs[0] = x;    wgs[1] = y;    wgs[2] = 0; }
        WorkGroupDimensionality(auint x, auint y, auint z) : dimensionality(3) { wgs[0] = x;    wgs[1] = y;    wgs[2] = z; }
        explicit WorkGroupDimensionality() = default;
    };
    struct KernelRequest {
        std::string fileName;
        std::string entryPoint;
        std::string compileFlags;
        WorkGroupDimensionality groupSize;
        std::string params;
    };

    struct ResourceRequest {
        std::string name;
        asizei bytes;
        cl_mem_flags memFlags;
        const aubyte *initialData; //!< must provide this->bytes amount of bytes, not owned
        bool immediate; /*!< if this is true, then this does not allocate a buffer, useful to give uints to kernels for example.
                        Immediates are not counted in memory footprint, even though they are most likely pushed to a cbuffer anyway! */
        aubyte imValue[8]; //!< used when immediate is true to store the data

        cl_image_format channels; //!< Only used if image. Parameter is considered image if imageDesc.width != 0
        cl_image_desc imageDesc;

        std::string presentationName; //!< only used when not empty, overrides name for user-presentation purposes

        ResourceRequest() { }
        ResourceRequest(const char *name, cl_mem_flags allocationFlags, asizei footprint, const aubyte *initialize = nullptr) {
            this->name = name;
            memFlags = allocationFlags;
            bytes = footprint;
            initialData = initialize;
            immediate = false;
            memset(&channels, 0, sizeof(channels));
            memset(&imageDesc, 0, sizeof(imageDesc));
        }
    };
    //! This is syntactic sugar only. It is imperative this does not add subfields as ResourceRequests are often passed around by copy.
    template<typename scalar>
    struct Immediate : ResourceRequest {
        Immediate(const char *name, const scalar &value) : ResourceRequest(name, 0, sizeof(value)) {
            immediate = true;
            memcpy_s(imValue, sizeof(imValue), &value, sizeof(value));
            initialData = imValue;
        }
    };

    AbstractTest(aulong runtimeConcurrency, aulong referenceConcurrency, cl_context ctx, const char *aname, const char *iname, const char *version)
        : nominalHashCount(referenceConcurrency), concurrency(std::min(runtimeConcurrency, referenceConcurrency)), context(ctx), algoName(aname), impName(iname), iversion(version) { 
        runtimeConcurrency = concurrency;
        if(referenceConcurrency % concurrency) throw std::string("Runtime concurrency must be a divisor of ") + std::to_string(nominalHashCount);
        wuData = dispatchData = candidates = 0;
    }
    void PrepareResources(ResourceRequest *resources, asizei numResources, asizei hashCount);
    void PrepareKernels(KernelRequest *kernels, asizei numKernels);

    std::vector<TestRun> headers;
    std::vector<auint> found;

    struct KernelDriver : WorkGroupDimensionality {
        cl_kernel clk;
        KernelDriver() = default;
        KernelDriver(const WorkGroupDimensionality &wgd, cl_kernel k) : WorkGroupDimensionality(wgd) { clk = k; }
    };

    std::vector<KernelDriver> kernels;
    std::vector<ResourceRequest> resRequests;
    std::map<std::string, cl_mem> resHandles;

    //! Special magic values common to various kernels.
    enum PrecomputedConstant {
        pc_AES_T,
        pc_SIMD_alpha,
        pc_SIMD_beta
    };
    //! The first call to those will prime an internal buffer. This buffer is guaranteed to be persistent as long as this object exists.
    //! Thereby, this is not const... \todo perhaps those buffers should be mutable?
    std::pair<const aubyte*, asizei> GetPrecomputedConstant(PrecomputedConstant what);

    std::vector<ashort> simd_alpha;
    std::vector<aushort> simd_beta;
    std::vector<auint> aes_t_tables;

    const cl_context context;

private:
    cl_mem wuData, dispatchData, candidates;
    asizei nonceBufferSize;

    aulong aiSignature; //!< represents the specific algorithm-implementation and version. Computed as a side effect of PrepareKernels

    //! Called at the end of PrepareKernels. Given a cl_kernel and its originating KernelRequest object, generates a stream of clSetKernelArg according
    //! to its internal bindings, resHandles and resRequests (for immediates).
    void BindParameters(cl_kernel kern, const KernelRequest &bindings);

    //! Maps the nonce buffer (assumed valid), blocking. Pull out nonces.
    std::vector<cl_uint> FromNonceBuffer(cl_command_queue q) const;

    /*! Using the provided command-queue/device assume all input buffers have been correctly setup and run a whole algorithm iteration (all involved steps).
    Compute exactly amount hashes, each one being i+base-th. That is, base is the global work offset.
    It is guaranteed hashCount <= this->concurrency.
    \note Some kernels have requirements on workgroup size and thus put a requirement on amount being a multiple of WG size.
    Of course this base class does not care; be careful with setup. */
    void RunAlgorithm(cl_command_queue q, asizei base, asizei hashCount) const;

    /*! Called at the end of PrepareKernels as an aid. Combines kernel file names, entrypoints, compile flags algo name and everything
    required to uniquely identify what's going to be run. */
    aulong ComputeVersionedHash(const KernelRequest *kerns, asizei numKernels, const std::map<std::string, std::string> &src) const;
};
