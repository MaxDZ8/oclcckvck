/*
 * Copyright (C) 2015 Massimo Del Zotto
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


#define TEST_QUBIT_FIVESTEPS 1
#define TEST_MYRGRS_MONOLITHIC 1
#define TEST_FRESH_WARM 1
#define TEST_NEOSCRYPT_SMOOTH 1

#if TEST_QUBIT_FIVESTEPS
#include "TestData/Qubit.h"
#include "AlgoImplementations/QubitFiveStepsCL12.h"
#endif

#if TEST_MYRGRS_MONOLITHIC
#include "TestData/MYRGRS.h"
#include "AlgoImplementations/MYRGRSMonolithicCL12.h"
#endif

#if TEST_FRESH_WARM
#include "TestData/Fresh.h"
#include "AlgoImplementations/FreshWarmCL12.h"
#endif

#if TEST_NEOSCRYPT_SMOOTH
#include "TestData/Neoscrypt.h"
#include "AlgoImplementations/NeoscryptSmoothCL12.h"
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


std::string Header(const AlgoIdentifier &id, aulong signature) {
    std::string val;
    val += "Algorithm:      " + id.algorithm + '\n';
    val += "Implementation: " + id.implementation + '\n';
    val += "Signature:      " + Hex(signature) + '\n';
    return val;
}


template<typename Type>
Type GetCLDevProp(cl_device_info what, cl_device_id device) {
    Type buff;
    asizei required = 0;
    cl_int err = clGetDeviceInfo(device, what, sizeof(buff), &buff, &required);
    if(required > sizeof(buff)) return Type(-2);
    else if(err != CL_SUCCESS) return Type(-1);
    return buff;
};


std::string Header(const std::vector<Platform> &plats, asizei plat, asizei dev) {
    std::vector<char> buff(256);
    auto platform(plats[plat].clid);
    auto device(plats[plat].devices[dev].clid);
    auto getCLPlatProp = [&buff, platform](cl_platform_info what) -> std::string {
        asizei required = 0;
        clGetPlatformInfo(platform, what, 0, NULL, &required);
        buff.resize(required);
        cl_int err = clGetPlatformInfo(platform, what, buff.size(), buff.data(), &required);
        if(err != CL_SUCCESS) return "<ERROR>";
        return std::string(buff.data(), required) + '\n';
    };
    auto getCLDevPropUINT  = [device](cl_device_info what) -> std::string { return std::to_string(GetCLDevProp<cl_uint>(what, device)) + '\n'; };
    auto getCLDevPropULONG = [device](cl_device_info what) -> std::string { return std::to_string(GetCLDevProp<cl_ulong>(what, device)) + '\n'; };
    auto getCLDevPropBOOL = [device](cl_device_info what) -> std::string { return GetCLDevProp<cl_bool>(what, device)? "true\n" : "false\n"; };
    auto getCLDevPropSTRING = [&buff, device](cl_device_info what) -> std::string {
        asizei required = 0;
        clGetDeviceInfo(device, what, 0, NULL, &required);
        buff.resize(required);
        cl_int err = clGetDeviceInfo(device, what, buff.size(), buff.data(), &required);
        if(err != CL_SUCCESS) return "<ERROR>";
        return std::string(buff.data(), required) + '\n';
    };
    std::string val("Device platform [" + std::to_string(plat) + "]\n");
    val += "  Name:       " + getCLPlatProp(CL_PLATFORM_NAME);
    val += "  Version:    " + getCLPlatProp(CL_PLATFORM_VERSION);
    val += "  Vendor:     " + getCLPlatProp(CL_PLATFORM_VENDOR);
    val += "  Profile:    " + getCLPlatProp(CL_PLATFORM_PROFILE);
    val += "  Extensions: " + getCLPlatProp(CL_PLATFORM_EXTENSIONS);
    val += '\n';
    val += "Device [" + std::to_string(dev) + "]\n";
    val += "  ID:             " + getCLDevPropUINT(CL_DEVICE_VENDOR_ID);
    val += "  Chip name:      " + getCLDevPropSTRING(CL_DEVICE_NAME);
    val += "  Cores:          " + getCLDevPropUINT(CL_DEVICE_MAX_COMPUTE_UNITS);
    val += "  Nominal clock:  " + getCLDevPropUINT(CL_DEVICE_MAX_CLOCK_FREQUENCY);
    val += "  Max alloc: . .  " + getCLDevPropULONG(CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    val += "  Base alignment: " + getCLDevPropULONG(CL_DEVICE_MEM_BASE_ADDR_ALIGN);
    val += "  Unified memory: " + getCLDevPropBOOL(CL_DEVICE_HOST_UNIFIED_MEMORY);
    val += "  Little endian:  " + getCLDevPropBOOL(CL_DEVICE_ENDIAN_LITTLE);
    val += "  Driver version: " + getCLDevPropSTRING(CL_DRIVER_VERSION);
    val += "  Device version: " + getCLDevPropSTRING(CL_DEVICE_VERSION);
    val += "  CL-C version:   " + getCLDevPropSTRING(CL_DEVICE_OPENCL_C_VERSION);
    val += "  Extensions:     " + getCLDevPropSTRING(CL_DEVICE_EXTENSIONS);
    return val + '\n';
}


template<typename TestData, typename TestSubject>
void Dispatch(const std::vector<Platform> &plats, const std::vector<cl_context> &platContext, asizei concurrency) {
    std::ofstream errorLog;
    for(unsigned p = 0; p < plats.size(); p++) {
        for(unsigned d = 0; d < plats[p].devices.size(); d++) {
            TestSubject imp(platContext[p], plats[p].devices[d].clid, concurrency);
            TestData test;
            if(!test.CanRunTests(concurrency)) {
                std::string msg(imp.identifier.Presentation());
                msg += " cannot be tested with concurrency " + std::to_string(concurrency);
                msg += ", not currently supposed to happen.";
                throw msg;
            }
            auto errors(test.RunTests(imp));
            auto signature(imp.GetVersioningHash());
            const std::string errorHeader(Header(imp.identifier, signature) + Header(plats, p, d));
            if(errors.size()) {
                if(errorLog.is_open() == false) {
                    std::string fname(imp.identifier.Presentation() + '.' + Hex(signature) + ".txt");
                    errorLog.open(fname.c_str());
                    if(errorLog.is_open() == false) throw "Could not open error log file.";
                    errorLog<<errorHeader<<std::endl;
                }
                std::cout<<errorHeader<<std::endl;
                for(auto err : errors) {
                    errorLog<<err<<std::endl;
                    std::cout<<err<<std::endl;
                }
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
    #if TEST_QUBIT_FIVESTEPS
        {
            const asizei concurrency = 1024 * 16;
            Dispatch<testData::Qubit, algoImplementations::QubitFiveStepsCL12>(plats, platContext, concurrency);
        }
    #endif
    #if TEST_MYRGRS_MONOLITHIC
        {
            const asizei concurrency = 1024 * 16;
            Dispatch<testData::MYRGRS, algoImplementations::MYRGRSMonolithicCL12>(plats, platContext, concurrency);
        }
    #endif
    #if TEST_FRESH_WARM
        {
            const asizei concurrency = 1024 * 16;
            Dispatch<testData::Fresh, algoImplementations::FreshWarmCL12>(plats, platContext, concurrency);
        }
    #endif
    #if TEST_NEOSCRYPT_SMOOTH
        {
            const asizei concurrency = 1024 * 4;
            Dispatch<testData::Neoscrypt, algoImplementations::NeoscryptSmoothCL12>(plats, platContext, concurrency);
        }
    #endif
    } catch(const char *msg) { std::cout<<msg<<std::endl; }
    catch(const std::string &msg) { std::cout<<msg<<std::endl; }
    return 0;
}
