/*
 * Copyright (C) 2014 Massimo Del Zotto
 * This code is released under the MIT license.
 * For conditions of distribution and use, see the LICENSE or hit the web.
 */
#pragma once
#include "../Common/AREN/ArenDataTypes.h"
#include <string>
#include <CL/cl.h>
#include <vector>
#include <map>
#include "KnownConstantsProvider.h"
#include "NonceStructs.h"
#include <array>
#include "../Common/AREN/ScopedFuncCall.h"
#include <fstream>
#include "../Common/hashing.h"

//! This enumeration is used by AbstractAlgorithm::Tick to represent the internal state and tell the outer code what to do.
enum class AlgoEvent {
    dispatched, //!< some nonces have been scheduled to be tested, everything is fine
    exhausted, //!< 4Gi nonces have been consumed, it is impossible to schedule additional work, provide a new Header() asap.
    working, //!< waiting for an async operation to complete. Call GetEvents to retrieve a list of all events we're waiting on.
    results //!< at least one algorithm iteration has completed and can be restarted... as soon as you pull out results calling GetResults.
};


/*! An algorithm can be implemented in multiple ways. Each implementation might be iterated giving different versions.
This allows quite some flexibility. The 'algorithm' was called 'algorithm family' in the first M8M architecture, it was unnecessarily complicating
things in a non-performance path.
It is strongly suggest to think at those strings like they should be valid C identifiers. */
struct AlgoIdentifier {
    std::string algorithm;
    std::string implementation;
    std::string version;
    explicit AlgoIdentifier() = default;
    AlgoIdentifier(const char *a, const char *i, const char *v) : algorithm(a), implementation(i), version(v) { }
    std::string Presentation() const { return algorithm + '.' + implementation; }
    // including the version is typically not very useful for presentation purposes as algorithm versions relate to host code, not necessarily
    // to kernel versions. Better to use algorithm signature instead!
};


/*! Because of a variety of reasons M8M had a fairly complicated algorithm management. There were 'algorithm families', 'algorithm implementations' and
implementations had a set of settings. Because a friend told me I should practice templates, I went with templates but it's obvious M8M consumes very
little CPU performance so there's no real reason to work that way.

Another thing about the previous architecture is that algorithms were supposed to be dispatched independantly across devices one step each.
This has proven to be unnecessary.

With the new design, there are no more settings objects nor 'algorithm instances': they are just an instance of some class implementing AbstractAlgorithm.

Also, they're now dumb and set up at build time with full type information. External logic gets to select settings and device to use. 

The way work is dispatched is also changed but that's better discussed in a derived class. At last, as OpenCL is very convincing I have decided to drop
support for other APIs and eventually think at it again in the future: this is really AbstractCLAlgorithm. */
class AbstractAlgorithm {
public:
    const AlgoIdentifier identifier;

    /*! This is computed as a side-effect of PrepareKernels and not much of a performance path.
    Represents the specific algorithm-implementation and version. Computed as a side effect of PrepareKernels */
    aulong GetVersioningHash() const { return aiSignature; }

    /*! Algorithms continuously mangle data coming out of a certain header. The outer program pumps here header data every time needed.
    It is also necessary to identify header originator. Every time the algorithm is Tick()-ed, the algorithm will attempt to dispatch a
    new set of work consuming nonce values until exhausting. Giving the algo a new header is the only way to reset the nonce count.
    \note Setting a new header <b>does not</b> cancel work being carried out, which will be completed late. Therefore, it is still possible
    to get values from the previous job/pool setting.
    \note Mining algorithms don't care from where an header comes. All they care is the headers only. When nonces are produced, they will
    associate them to the header used to produce them. Outer code must recostruct the association. */
    void Header(const std::array<aubyte, 80> &header) {
        hashing.header = header;
        hashing.nonceBase = 0;
    }

    /*! Another thing algorithms deal with are (at least) the target bits, which are a function of "difficulty" which in fact does
    not exist. At network level only target bits exists. Stratum tries to "compress" them in difficulty (which is a "human-readable" number)
    and them maps it back to target bits. See https://bitcointalk.org/index.php?topic=957516. */
    void TargetBits(aulong reference) { hashing.target = reference; }

    /*! This is conceptually similar to "scan hash" operation in legacy miners BUT it allows to mix CPU-GPU processing.
    The list to pass here is a list of waiting events belonging to this algorithm instance (that is, returned by this->GetEvents) which
    triggered (woke up thread by clWaitForEvents). Implementations can assume those events belong to them and signal complete operation.
    Implementations are free to avoid checking the event status by GL so it is imperative triggered events are kept!
    Implementations are NOT required to remove "consumed" events and can also generate new events so the outer code must drop the list
    on return as it's no longer meaningful. */
    virtual AlgoEvent Tick(const std::vector<cl_event> &blockers) = 0;

    /*! At least one of the returned events must complete before the algorithm can continue. In the case of GPU-only, single-phase algos
    those will be wait on buffer maps for results but this does not have to be the case for all others. Implementations are required to ADD
    their events to the provided vector leaving all the other elements untouched. This allows to build a single list to wait. */
    virtual void GetEvents(std::vector<cl_event> &events) const = 0;

    /*! Return a set of found nonces, making the algorithm able to dispatch some work again. Note this does not return any other data,
    Nonces structure will have to be built by outer code. */
    virtual MinedNonces GetResults() = 0;

protected:
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

        explicit ResourceRequest() { }
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

    /*! \param ctx OpenCL context used for creating kernels and resources. Kernels take a while to build and are very small so they can be shared
                   across devices... but they currently don't.
        \param dev This is the device this algorithm is going to use for the bulk of processing. Note complicated algos might be hybrid GPU-CPU
                   and thus require multiple devices. This extension is really only meaningful for a derived class.
    */
    AbstractAlgorithm(cl_context ctx, cl_device_id dev, const char *algo, const char *imp, const char *ver)
        : identifier(algo, imp, ver), context(ctx), device(dev) { 
    }

    /*! Derived classes are expected to call this somewhere in their ctor. It deals with allocating memory and eventually initializing it in a
    data-driven way. Note special resources cannot be created using this, at least in theory. Just create them in the ctor before PrepareKernels. */
    void PrepareResources(ResourceRequest *resources, asizei numResources, asizei hashCount);

    //! Similarly, kernels are described by data and built by resolving the previously declared resources.
    void PrepareKernels(KernelRequest *kernels, asizei numKernels);

    struct KernelDriver : WorkGroupDimensionality {
        cl_kernel clk;
        std::vector< std::pair<cl_uint, cl_uint> > dtBindings; //!< dispatch time bindings. .first is algo index, .second is resource index to be remapped.
        explicit KernelDriver() = default;
        KernelDriver(const WorkGroupDimensionality &wgd, cl_kernel k) : WorkGroupDimensionality(wgd) { clk = k; }
    };

    std::vector<KernelDriver> kernels;
    std::vector<ResourceRequest> resRequests;
    std::map<std::string, cl_mem> resHandles;
    
    struct {
        std::array<aubyte, 80> header;
        aulong nonceBase; //!< this is 64 bit and considered "exhausted" when over 32 bit range.
        aulong target;
    } hashing;

    /*! Using the provided command-queue/device assume all input buffers have been correctly setup and run a whole algorithm iteration (all involved steps).
    Compute exactly amount hashes, each one being i+base-th. That is, base is the global work offset (from hashing.nonceBase).
    It is guaranteed hashCount <= this->concurrency.
    \note Some kernels have requirements on workgroup size and thus put a requirement on amount being a multiple of WG size.
    Of course this base class does not care; derived classes must be careful with setup, including rebinding special resources. */
    void RunAlgorithm(cl_command_queue q, asizei hashCount) const;

    /*! Some values are special as they need to be bound on a per-dispatch basis... in certain cases.
    This structure allows this base class to understand if a value is "special". Do not confuse those with "known constant" values, which are special
    but in a different way. They are, by definition, constant and thus not "special" in this sense. */
    struct SpecialValueBinding {
        bool earlyBound; //!< true if this parameter can be bound to the kernel once and left alone forever. Otherwise, call DispatchBinding.
        union {
            cl_mem buff;   //!< buffer holding the specific resource. Use when earlyBound.
            cl_uint index; //!< resource index (to be mapped to a specific buffer) for dispatch-time binding.
        } resource;
    };

    /*! This function returns false if the given name does not identify something recognized.
    Otherwise, it will set desc. You can be sure at this point desc.resource will always be != 0. 
    While there are no requirements on the special value names, please follow these guidelines:
    - All special value names start with '$'.
    - "$wuData" is the 80-bytes block header to hash. Yes, 80 bytes, even though we overwrite the last 4 (most of the time).
    - "$dispatchData" contains "other stuff" including targetbits... note those are probably going to be refactored as well.
    - "$candidates" is the resulting nonce buffer. 
    Those can be bound early or dinamically, there's no requirement. */
    virtual bool SpecialValue(SpecialValueBinding &desc, const std::string &name) = 0;

    /*! Late-bound buffers. This is indexed by SpecialValueBinding::index.
    Derived classes shall update this before allowing this class RunAlgorithm to run as it needs to access this to map to the correct values. */
    std::vector<cl_mem> lbBuffers;

    //! The concurrency (aka hashCount) is the maximum amount of work items which can be dispatched in a single call.
    //! For the time being, it's also the only amount which can be dispatched!
    virtual asizei GetConcurrency() const = 0;

private:
    const cl_context context;
    const cl_device_id device;
    aulong aiSignature; //!< \sa GetVersioningHash()

    //! Called at the end of PrepareKernels. Given a cl_kernel and its originating KernelRequest object, generates a stream of clSetKernelArg according
    //! to its internal bindings, resHandles and resRequests (for immediates).
    void BindParameters(KernelDriver &kd, const KernelRequest &bindings);

    /*! Called at the end of PrepareKernels as an aid. Combines kernel file names, entrypoints, compile flags algo name and everything
    required to uniquely identify what's going to be run. */
    aulong ComputeVersionedHash(const KernelRequest *kerns, asizei numKernels, const std::map<std::string, std::string> &src) const;
};
