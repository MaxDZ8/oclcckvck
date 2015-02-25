/*
 * Copyright (C) 2014 Massimo Del Zotto
 * This code is released under the MIT license.
 * For conditions of distribution and use, see the LICENSE or hit the web.
 */
/* OpenCL Crypto Currency Kernel Validity ChecK.

This project is a spin-off of M8M (https://github.com/MaxDZ8/M8M).
What happened is that many people couldn't get M8M to hash correctly and I couldn't isolate pool issues from driver issues.
The current (yet-to-be-published) M8M source allows me to detect HW errors, but this does not solve the problem of giving users a tool to assess compatibility.

This program runs M8M kernels with a known input obtained from a legacy miner. The results are checked against reference.
This program only goal is to assess validity/compatibility, not performance. It is in that regard much more focused than the tool used internally during M8M development,
which (arguably) does both and will hopefully be superceded by this one. */
#include <CL/cl.h>
#include "../Common/aes.h"
#include <chrono>
#include <strstream>
#include <vector>
#include <iostream>
#include <string>


#define TEST_QUBIT 1

#if TEST_QUBIT
#include "QubitTest.h"
#endif


struct Device {
    unsigned clPlatIndex;
    cl_device_id clid;
};

struct Platform {
    unsigned clIndex;
    cl_platform_id clid;
    std::vector<Device> devices;
};


std::vector<Platform> EnumeratePlatforms() {
    std::vector<cl_platform_id> plats;
    cl_uint avail = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &avail);
    if(err != CL_SUCCESS) throw "Could not count available platforms.";
    plats.resize(avail);
    err = clGetPlatformIDs(avail, plats.data(), &avail);
    if(err != CL_SUCCESS) throw "Could not enumerate platforms.";

    std::vector<Platform> ret;
    std::vector<char> stringBuffer(64);
    for(unsigned test = 0; test < plats.size(); test++) {
        size_t chars = 0;
        err = clGetPlatformInfo(plats[test], CL_PLATFORM_PROFILE, stringBuffer.size(), stringBuffer.data(), &chars);
        if(err != CL_SUCCESS) throw "Could not probe platform profile.";
        if(memcmp(stringBuffer.data(), "FULL_PROFILE", strlen("FULL_PROFILE"))) continue; // only interested in full profile
        // All other platform data such as version or extensions is not considered relevant at this point.
        ret.push_back({ test, plats[test] });
    }
    return ret;
}


bool EnumerateGPUs(Platform &plat) {
    std::vector<cl_device_id> devs;
    cl_uint avail = 0;
    cl_int err = clGetDeviceIDs(plat.clid, CL_DEVICE_TYPE_GPU, 0, NULL, &avail);
    if(err == CL_DEVICE_NOT_FOUND) return false;
    if(err != CL_SUCCESS) throw "Error counting platform GPUs";
    devs.resize(avail);
    err = clGetDeviceIDs(plat.clid, CL_DEVICE_TYPE_GPU, cl_uint(devs.size()), devs.data(), NULL);
    if(err != CL_SUCCESS) throw "Error enumerating platform GPUs";
    std::vector<char> stringBuffer(64);
    for(unsigned test = 0; test < devs.size(); test++) {
        size_t chars = 0;
        err = clGetDeviceInfo(devs[test], CL_DEVICE_PROFILE, stringBuffer.size(), stringBuffer.data(), &chars);
        if(err != CL_SUCCESS) throw "Could not probe device profile.";
        if(memcmp(stringBuffer.data(), "FULL_PROFILE", strlen("FULL_PROFILE"))) continue; // only interested in full profile
        // Maybe CL version could be interesting...
        plat.devices.push_back({ test, devs[test] });
    }
    return plat.devices.size() != 0;
}


std::string Hex(const aubyte *blob, asizei count) {
    const char *hex = "0123456789abcdef";
    std::string build;
    for(asizei i = 0; i < count; i++) {
        auto c = blob[i];
        build += hex[c >> 4];
        build += hex[c & 0x0F];
    }
    return build;
}


template<typename POD>
std::string Hex(const POD &blob) {
    return Hex(reinterpret_cast<const aubyte*>(&blob), sizeof(blob));
}


void Dispatch(const AbstractTest &algo, const std::vector<Device> &devices) {
    // In theory I could test all devices concurrently but I'm currently using a single set of resources.
    // As I'm already requesting resource migration all performance estimations are off, this will have to be fixed in the future.
    std::ofstream errorLog;
    for(auto &device : devices) {
        auto errors(algo.RunTests(device.clid));
        if(errors.size()) {
            auto signature(Hex(algo.GetVersioningHash()));
            if(errorLog.is_open() == false) {
                std::string fname(algo.algoName + '_' + algo.impName + '_' + signature + ".txt");
                errorLog.open(fname.c_str());
                if(errorLog.is_open() == false) throw "Could not open error log file.";
                errorLog<<"Algorithm:      "<<algo.algoName<<std::endl;
                errorLog<<"Implementation: "<<algo.impName<<std::endl;
                errorLog<<"Signature:      "<<signature<<std::endl<<std::endl;
            }
            std::cout<<"Algorithm:      "<<algo.algoName<<std::endl;
            std::cout<<"Implementation: "<<algo.impName<<std::endl;
            std::cout<<"Signature:      "<<signature<<std::endl<<std::endl;
            for(auto err : errors) {
                errorLog<<err<<std::endl;
                std::cout<<err<<std::endl;
            }
        }
    }
}


int main() {
    try {
        std::vector<Platform> plats(EnumeratePlatforms());
        std::cout<<"Found "<<plats.size()<<" OpenCL platform"<<(plats.size() > 1? "s" : "")<<" for processing."<<std::endl;
        for(unsigned p = 0; p < plats.size(); p++) {
            if(!EnumerateGPUs(plats[p])) {
                plats.erase(plats.begin() + p);
                p--;
            }
            else std::cout<<"Platform "<<plats[p].clIndex<<" counts "<<plats[p].devices.size()<<" device"<<(plats[p].devices.size() > 1? "s" : "")<<" to test."<<std::endl;
        }
        std::cout<<"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - "<<std::endl;
        std::vector<cl_context> platContext;
        platContext.reserve(plats.size());
        ScopedFuncCall relAllContext([&platContext]() { for(auto el = 0; el < platContext.size(); el++) clReleaseContext(platContext[el]); });
        auto errorFunc = [](const char *errinfo, const void *private_info, size_t cb, void *user_data) { // not thread safe.
            const Platform *plat = reinterpret_cast<Platform*>(user_data);
            std::cout<<"Error from context ["<<plat->clIndex<<"]"<<std::endl
                     <<"    "<<errinfo<<std::endl;
        };
        for(unsigned p = 0; p < plats.size(); p++) {
            cl_context_properties ctxprops[] = {
                CL_CONTEXT_PLATFORM, cl_context_properties(plats[p].clid),
                0, 0 // two zeros just in case
            };
            std::vector<cl_device_id> devs(plats[p].devices.size());
            for(asizei cp = 0; cp < devs.size(); cp++) devs[cp] = plats[p].devices[cp].clid;
            cl_int err = 0;
            cl_context ctx = clCreateContext(ctxprops, cl_uint(devs.size()), devs.data(), errorFunc, plats.data() + p, &err);
            platContext.push_back(ctx); // reserved, cannot fail
        }
    #if TEST_QUBIT
        for(unsigned p = 0; p < plats.size(); p++) {
            const asizei concurrency = 1024 * 16;
            Dispatch(QubitTest(concurrency, platContext[p]), plats[p].devices);
        }
    #endif
    } catch(const char *msg) { std::cout<<msg<<std::endl; }
    catch(const std::string &msg) { std::cout<<msg<<std::endl; }
    return 0;
}
