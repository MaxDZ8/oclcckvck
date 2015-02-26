/*
 * Copyright (C) 2014 Massimo Del Zotto
 * This code is released under the MIT license.
 * For conditions of distribution and use, see the LICENSE or hit the web.
 */
#pragma once
#include "AbstractAlgorithm.h"

/*! A stop-n-wait algorithm is an algorithm which always gives the GPU 1 unit of work at time.
It dispatches data and waits for result. It is basically the same thing M8M always did, which is very similar to legacy miners.
WRT legacy miners two differences come to mind.

Legacy miners dispatch all the work, force it to finish, then ask for a buffer map and wait for it again! This is due to their use
of out-of-order queues which is not really needed, especially as many algos are single step.
M8M dispatches all the work, including the map request and then **waits for it until finished**.
An initial version of Qubit also tried to dispatch one step at time but it was nonsensically overcomplicated for no benefit.
So in short I avoid a Finish (1) and a blocking read (2). Apparently this produces better interactivity. */
class StopWaitAlgorithm : public AbstractAlgorithm {
public:
    const bool bigEndian;
    const asizei hashCount;
    StopWaitAlgorithm(cl_context ctx, cl_device_id dev, asizei concurrency, const char *algo, const char *imp, const char *ver, bool be)
        : AbstractAlgorithm(ctx, dev, algo, imp, ver), hashCount(concurrency), bigEndian(be) {
        PrepareIOBuffers(ctx, concurrency);
        
        cl_int err = 0;
        queue = clCreateCommandQueue(ctx, dev, 0, &err);
        if(!queue || err != CL_SUCCESS) throw "Could not create command queue for device!";
    }
    ~StopWaitAlgorithm() {
        if(mapping) clReleaseEvent(mapping);
        if(nonces) clEnqueueUnmapMemObject(queue, candidates, nonces, 0, NULL, NULL);
        if(queue) clReleaseCommandQueue(queue);
    }

    AlgoEvent Tick(const std::vector<cl_event> &blockers) {
        // The first, most important thing to do is to free results so I can start again.
        if(mapping) {
            if(std::find(blockers.cbegin(), blockers.cend(), mapping) == blockers.cend()) return AlgoEvent::working;
            return AlgoEvent::results;
        }
        if(hashing.nonceBase + hashCount > std::numeric_limits<auint>::max()) return AlgoEvent::exhausted; // nothing to do

        cl_int err = 0;
        if(bigEndian) {
            aubyte be[80];
            for(asizei i = 0; i < sizeof(be); i += 4) {
                for(asizei cp = 0; cp < 4; cp++) be[i + cp] = hashing.header[i + 3 - cp];
            }
            err = clEnqueueWriteBuffer(queue, wuData, CL_TRUE, 0, sizeof(be), be, 0, NULL, NULL);
        }
        else {
            err = clEnqueueWriteBuffer(queue, wuData, CL_TRUE, 0, sizeof(hashing.header), hashing.header.data(), 0, NULL, NULL);
        }
        if(err != CL_SUCCESS) throw std::string("CL error ") + std::to_string(err) + " while attempting to update $wuData";

		cl_uint buffer[5]; // taken as is from M8M FillDispatchData... how ugly!
		buffer[0] = 0;
		buffer[1] = static_cast<cl_uint>(hashing.target >> 32);
		buffer[2] = static_cast<cl_uint>(hashing.target);
		buffer[3] = 0;
		buffer[4] = 0;
        err = clEnqueueWriteBuffer(queue, dispatchData, CL_TRUE, 0, sizeof(buffer), buffer, 0, NULL, NULL);
        if(err != CL_SUCCESS) throw std::string("CL error ") + std::to_string(err) + " while attempting to update $dispatchData";

        cl_uint zero = 0;
        clEnqueueWriteBuffer(queue, candidates, true, 0, sizeof(cl_uint), &zero, 0, NULL, NULL);

        RunAlgorithm(queue, hashCount);
        hashing.nonceBase += hashCount;
        dispatchedHeader = hashing.header;

        nonces = reinterpret_cast<cl_uint*>(clEnqueueMapBuffer(queue, candidates, CL_FALSE, CL_MAP_READ, 0, nonceBufferSize, 0, NULL, &mapping, &err));
        if(err != CL_SUCCESS) throw std::string("CL error ") + std::to_string(err) + " attempting to map nonce buffers.";

        return AlgoEvent::dispatched; // this could be ae_working as well but returning ae_dispatched at least once sounds good.
    }


    void GetEvents(std::vector<cl_event> &events) const {
        if(mapping) events.push_back(mapping);
    }


    MinedNonces GetResults() {
        MinedNonces ret(dispatchedHeader);
        const asizei count = *nonces;
        nonces++;
        for(asizei cp = 0; cp < count; cp++) {
            ret.nonces.push_back(nonces[cp]);
            //! \todo migrate to hash-out kernels
        }
        nonces = nullptr;
        clReleaseEvent(mapping);
        mapping = 0;
        return ret;
    }

private:
    bool SpecialValue(SpecialValueBinding &desc, const std::string &name) {
        // This is very easy for stop-n-wait as everything can be bound statically.
        cl_mem core = 0;
        if(name == "$wuData") core = wuData;
        else if(name == "$dispatchData") core = dispatchData;
        else if(name == "$candidates") core = candidates;
        else return false;
        desc.earlyBound = true;
        desc.resource.buff = core;
        return true;
    }

    cl_mem wuData = 0, dispatchData = 0;
    cl_mem candidates = 0;
    asizei nonceBufferSize = 0;
    cl_event mapping = 0;
    cl_command_queue queue = 0;
    auint *nonces = nullptr;
    std::array<aubyte, 80> dispatchedHeader;

    void PrepareIOBuffers(cl_context context, asizei hashCount){
        cl_int error;
	    asizei byteCount = 80;
	    wuData = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, byteCount, NULL, &error);
	    if(error != CL_SUCCESS) throw std::string("OpenCL error ") + std::to_string(error) + " while trying to create wuData buffer.";
	    byteCount = 5 * sizeof(cl_uint);
	    dispatchData = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, byteCount, NULL, &error);
	    if(error != CL_SUCCESS) throw std::string("OpenCL error ") + std::to_string(error) + " while trying to create dispatchData buffer.";
        // The candidate buffer should really be dependant on difficulty setting but I take it easy.
        byteCount = 1 + hashCount / (32 * 1024);
        //! \todo pull the whole hash down so I can check mismatches
        if(byteCount < 33) byteCount = 33;
        byteCount *= sizeof(cl_uint);
        nonceBufferSize = byteCount;
        candidates = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, byteCount, NULL, &error);
	    if(error) throw std::string("OpenCL error ") + std::to_string(error) + " while trying to resulting nonces buffer.";
    }
};
