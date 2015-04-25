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
#include <chrono>
#include <sstream>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include "AbstractAlgorithm.h"
#include "StopWaitDispatcher.h"
#include "misc.h"
#include "StepTest/misc.h"


#if defined(TEST_QUBIT_FIVESTEPS)
#include "TestData/Qubit.h"
#include "AlgoImplementations/QubitFiveStepsCL12.h"
#endif

#if defined(TEST_MYRGRS_MONOLITHIC)
#include "TestData/MYRGRS.h"
#include "AlgoImplementations/MYRGRSMonolithicCL12.h"
#endif

#if defined(TEST_FRESH_WARM)
#include "TestData/Fresh.h"
#include "AlgoImplementations/FreshWarmCL12.h"
#endif

#if defined(TEST_NEOSCRYPT_SMOOTH)
#include "TestData/Neoscrypt.h"
#include "AlgoImplementations/NeoscryptSmoothCL12.h"
#endif

#if defined(TEST_LUFFA_1W_HEAD)
#include "StepTest/Luffa_1W.h"
#endif

#if defined(TEST_CUBEHASH_2W_CHAINED)
#include "StepTest/CubeHash_2W.h"
#endif

#if defined(TEST_SHAVITE3_1W_CHAINED)
#include "StepTest/Shavite3_1W.h"
#endif

#if defined(TEST_SIMD_16W_CHAINED)
#include "StepTest/SIMD_16W.h"
#endif

#if defined(TEST_ECHO_8W_TAIL)
#include "StepTest/ECHO_8W.h"
#endif

#if defined(TEST_NS_FIRSTKDF_4W_HEAD) || defined(TEST_NS_LASTKDF_4W_TAIL)
#include "StepTest/NS_KDFs_4W.h"
#endif

#if defined(TEST_NS_SW_SALSA) || defined(TEST_NS_IR_SALSA) || defined(TEST_NS_SW_CHACHA) || defined(TEST_NS_IR_CHACHA)
#include "StepTest/NS_CoreLoops.h"
#endif

bool opt_verbose = true;
bool opt_showTestTime = true;


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


std::string Header(const AlgoIdentifier &id, const std::string &signature) {
    std::string val;
    val += "Algorithm:      " + id.algorithm + '\n';
    val += "Implementation: " + id.implementation + '\n';
    val += "Signature:      " + signature + '\n';
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
        return std::string(buff.data(), required - 1) + '\n';
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
        return std::string(buff.data(), required - 1) + '\n';
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
    val += "  Max alloc:      " + getCLDevPropULONG(CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    val += "  Base alignment: " + getCLDevPropULONG(CL_DEVICE_MEM_BASE_ADDR_ALIGN);
    val += "  Unified memory: " + getCLDevPropBOOL(CL_DEVICE_HOST_UNIFIED_MEMORY);
    val += "  Little endian:  " + getCLDevPropBOOL(CL_DEVICE_ENDIAN_LITTLE);
    val += "  Driver version: " + getCLDevPropSTRING(CL_DRIVER_VERSION);
    val += "  Device version: " + getCLDevPropSTRING(CL_DEVICE_VERSION);
    val += "  CL-C version:   " + getCLDevPropSTRING(CL_DEVICE_OPENCL_C_VERSION);
    val += "  Extensions:     " + getCLDevPropSTRING(CL_DEVICE_EXTENSIONS);
    return val + '\n';
}


void Whoops(std::ofstream &errorLog, const std::string &filename, const char *msg) {
    if(errorLog.is_open() == false) {
        std::string fname(filename.c_str());
        errorLog.open(fname.c_str());
        if(errorLog.is_open() == false) throw "Could not open error log file.";
    }
    errorLog<<msg<<std::endl;
    throw std::string(msg);
}


template<typename TestData, typename TestSubject>
void Dispatch(const std::vector<Platform> &plats, const std::vector<cl_context> &platContext, asizei concurrency) {
    std::ofstream errorLog;
    for(unsigned p = 0; p < plats.size(); p++) {
        for(unsigned d = 0; d < plats[p].devices.size(); d++) {
            TestSubject imp(platContext[p], plats[p].devices[d].clid, concurrency);
            StopWaitDispatcher dispatcher(imp);
            auto presentation(imp.identifier.Presentation());
            std::string hexSign;
            auto filename = [p, d, &presentation, &hexSign]() -> std::string {
                std::string ret('p' + std::to_string(p) + 'd' + std::to_string(d) + '-' + presentation);
                if(hexSign.length()) ret += hexSign;
                else ret += "failed initialization";
                return ret + ".txt";
            };
            try {
                auto errors(imp.Init(nullptr, dispatcher.AsValueProvider(), ""));
                hexSign = imp.GetVersioningHash()? Hex(imp.GetVersioningHash()) : std::string("-failed_to_init");
                if(errors.size()) {
                    std::string meh;
                    for(auto err : errors) meh += err + "\n\n";
                    throw meh;
                }
                TestData test;
                if(opt_verbose) std::cout<<"Testing "<<presentation<<" ("<<hexSign<<") on plat"<<p<<".dev"<<d<<"\n";
                if(!test.CanRunTests(concurrency)) {
                    std::string msg(presentation);
                    msg += " cannot be tested with concurrency " + std::to_string(concurrency);
                    msg += ", not currently supposed to happen.";
                    throw msg;
                }
                if(opt_verbose) {
                    for(asizei t = 0; t < test.GetNumTests(); t++) std::cout<<char('0' + t % 10);
                    std::cout<<std::endl;
                    test.onBlockHashed = [](asizei progress) { std::cout<<'.'; };
                }
                const auto start(std::chrono::system_clock::now());
                try {
                    errors = test.RunTests(dispatcher);
                } catch(const std::string &msg) {
                    errors.push_back(msg);
                } catch(const char *msg) {
                    errors.push_back(msg);
                }
                const auto finished(std::chrono::system_clock::now());
                if(errors.size()) {
                    std::string allErrors(Header(imp.identifier, hexSign) + Header(plats, p, d));
                    for(auto err : errors) allErrors += err + '\n';
                    throw allErrors;
                }
                if(opt_verbose) std::cout<<std::endl;
                if(opt_showTestTime) std::cout<<"t="<<std::chrono::duration_cast<std::chrono::milliseconds>(finished - start).count()<<" ms"<<std::endl;
            } catch(const std::string &msg) {
                Whoops(errorLog, filename(), msg.c_str());
            } catch(const char *msg) {
                Whoops(errorLog, filename(), msg);
            }
        }
    }
}


template<typename StepComparator>
void Compare(const std::vector<Platform> &plats, const std::vector<cl_context> &platContext, asizei concurrency) {
    std::ofstream errorLog;
    for(unsigned p = 0; p < plats.size(); p++) {
        for(unsigned d = 0; d < plats[p].devices.size(); d++) {
            StepComparator test(platContext[p], plats[p].devices[d].clid, concurrency);
            StopWaitDispatcher dispatcher(test.algo);
            test.MakeInputData(dispatcher);
            auto presentation(test.algo.identifier.Presentation());
            std::string hexSign;
            auto filename = [p, d, &presentation, &hexSign]() -> std::string {
                std::string ret('p' + std::to_string(p) + 'd' + std::to_string(d) + '-' + presentation);
                if(hexSign.length()) ret += hexSign;
                else ret += "failed initialization";
                return ret + ".txt";
            };
            try {
                auto errors(test.algo.Init(nullptr, dispatcher.AsValueProvider(), ""));
                hexSign = test.algo.GetVersioningHash()? Hex(test.algo.GetVersioningHash()) : std::string("-failed_to_init");
                if(errors.size()) {
                    std::string meh;
                    for(auto err : errors) meh += err + "\n\n";
                    throw meh;
                }
                if(opt_verbose) std::cout<<"Testing "<<presentation<<" ("<<hexSign<<") on plat"<<p<<".dev"<<d<<"\n";
                // It is assumed steps complete in a single dispatch so no need to iterate up to producing results!
                std::set<cl_event> triggered;
                while(dispatcher.Tick(triggered) != AlgoEvent::working) { }
                std::vector<cl_event> blockers;
                dispatcher.GetEvents(blockers);
                if(blockers.size()) { // step tests can still blocking map.
                    clWaitForEvents(cl_uint(blockers.size()), blockers.data());
                }
                typedef std::array<aubyte, 64> Hash;
                auto bad(test.Check(dispatcher));
                if(bad.Failed()) throw bad.Describe(test.algo.hashCount);
            } catch(const std::string &msg) {
                Whoops(errorLog, filename(), msg.c_str());
            } catch(const char *msg) {
                Whoops(errorLog, filename(), msg);
            }
        }
    }
}


void AlgoTests(const std::vector<Platform> &plats, const std::vector<cl_context> &platContext) {
#if defined(TEST_QUBIT_FIVESTEPS)
    try {
        const asizei concurrency = 1024 * 16;
        Dispatch<testData::Qubit, algoImplementations::QubitFiveStepsCL12>(plats, platContext, concurrency);
    } catch(const std::string &what) { std::cout<<what<<std::endl; }
#endif
#if defined(TEST_MYRGRS_MONOLITHIC)
    try {
        const asizei concurrency = 1024 * 16;
        Dispatch<testData::MYRGRS, algoImplementations::MYRGRSMonolithicCL12>(plats, platContext, concurrency);
    } catch(const std::string &what) { std::cout<<what<<std::endl; }
#endif
#if defined(TEST_FRESH_WARM)
    try {
        const asizei concurrency = 1024 * 16;
        Dispatch<testData::Fresh, algoImplementations::FreshWarmCL12>(plats, platContext, concurrency);
    } catch(const std::string &what) { std::cout<<what<<std::endl; }
#endif
#if defined(TEST_NEOSCRYPT_SMOOTH)
    try {
        const asizei concurrency = 1024 * 4;
        Dispatch<testData::Neoscrypt, algoImplementations::NeoscryptSmoothCL12>(plats, platContext, concurrency);
    } catch(const std::string &what) { std::cout<<what<<std::endl; }
#endif
}



bool StepTests(const std::vector<Platform> &plats, const std::vector<cl_context> &platContext) {
    bool failed = false;
    using namespace stepTest;
#if defined(TEST_LUFFA_1W_HEAD)
    try {
        const asizei concurrency = 1024 * 16 * 4;
        Compare< HeadTest<Luffa_1W> >(plats, platContext, concurrency);
    } catch(const std::string &what) { std::cout<<what<<std::endl;    failed = true; }
#endif
#if defined(TEST_CUBEHASH_2W_CHAINED)
    try {
        const asizei concurrency = 1024 * 16 * 4;
        Compare< StepTest< CubeHash_2W> >(plats, platContext, concurrency);
    } catch(const std::string &what) { std::cout<<what<<std::endl;    failed = true; }
#endif
#if defined(TEST_SHAVITE3_1W_CHAINED)
    try {
        const asizei concurrency = 1024 * 16 * 4;
        Compare< StepTest< ShaVite3_1W> >(plats, platContext, concurrency);
    } catch(const std::string &what) { std::cout<<what<<std::endl;    failed = true; }
#endif
#if defined(TEST_SIMD_16W_CHAINED)
    try {
        const asizei concurrency = 1024 * 16 * 4;
        Compare< StepTest<SIMD_16W> >(plats, platContext, concurrency);
    } catch(const std::string &what) { std::cout<<what<<std::endl;    failed = true; }
#endif
#if defined(TEST_ECHO_8W_TAIL)
    try {
        const asizei concurrency = 1024 * 16 * 4;
        Compare< TailTest<ECHO_8W> >(plats, platContext, concurrency);
    } catch(const std::string &what) { std::cout<<what<<std::endl;    failed = true; }
#endif
#if defined(TEST_NS_FIRSTKDF_4W_HEAD)
    try {
        const asizei concurrency = 1024 * 16 * 4;
        Compare< HeadTest<NS_FirstKDF_4W> >(plats, platContext, concurrency);
    } catch(const std::string &what) { std::cout<<what<<std::endl;    failed = true; }
#endif
#if defined(TEST_NS_SW_SALSA)
    try {
        const asizei concurrency = 1024 * 8;
        Compare< StepTest< NS_SW<nsHelp::Salsa> > >(plats, platContext, concurrency);
    } catch(const std::string &what) { std::cout<<what<<std::endl;    failed = true; }
#endif
#if defined(TEST_NS_IR_SALSA)
    try {
        const asizei concurrency = 1024 * 8;
        Compare< StepTest< NS_IR<nsHelp::Salsa> > >(plats, platContext, concurrency);
    } catch(const std::string &what) { std::cout<<what<<std::endl;    failed = true; }
#endif
#if defined(TEST_NS_SW_CHACHA)
    try {
        const asizei concurrency = 1024 * 8;
        Compare< StepTest< NS_SW<nsHelp::Chacha> > >(plats, platContext, concurrency);
    } catch(const std::string &what) { std::cout<<what<<std::endl;    failed = true; }
#endif
#if defined(TEST_NS_IR_CHACHA)
    try {
        const asizei concurrency = 1024 * 8;
        Compare< StepTest< NS_IR<nsHelp::Chacha> > >(plats, platContext, concurrency);
    } catch(const std::string &what) { std::cout<<what<<std::endl;    failed = true; }
#endif
#if defined(TEST_NS_FASTKDF_4W_TAIL)
    try {
        const asizei concurrency = 1024 * 16 * 4;
        Compare< TailTest<NS_LastKDF_4W> >(plats, platContext, concurrency);
    } catch(const std::string &what) { std::cout<<what<<std::endl;    failed = true; }
#endif
    return !failed;
}


int main() {
    try {
        std::vector<Platform> plats(EnumeratePlatforms());
        if(opt_verbose) std::cout<<"Found "<<plats.size()<<" OpenCL platform"<<(plats.size() > 1? "s" : "")<<" for processing."<<std::endl;
        for(unsigned p = 0; p < plats.size(); p++) {
            if(!EnumerateGPUs(plats[p])) {
                plats.erase(plats.begin() + p);
                p--;
            }
            else if(opt_verbose) std::cout<<"Platform "<<plats[p].clIndex<<" counts "<<plats[p].devices.size()<<" device"<<(plats[p].devices.size() > 1? "s" : "")<<" to test."<<std::endl;
        }
        if(opt_verbose) std::cout<<"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - "<<std::endl;
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
        bool algoTests = true;
        const bool stepTests = true;
        if(stepTests) algoTests &= StepTests(plats, platContext);
        if(algoTests) AlgoTests(plats, platContext);
    } catch(const char *msg) { std::cout<<msg<<std::endl; }
    catch(const std::string &msg) { std::cout<<msg<<std::endl; }
    return 0;
}
