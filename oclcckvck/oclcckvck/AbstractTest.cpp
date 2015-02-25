/*
 * Copyright (C) 2014 Massimo Del Zotto
 * This code is released under the MIT license.
 * For conditions of distribution and use, see the LICENSE or hit the web.
 */
#include "AbstractTest.h"


std::pair<const aubyte*, asizei> AbstractTest::GetPrecomputedConstant(PrecomputedConstant pc) {
    const aubyte *data = nullptr;
    asizei footprint = 0;
    switch(pc) {
    case pc_AES_T: {
        if(this->aes_t_tables.size() == 0) {
            aes_t_tables.resize(4 * 256);
            auint *lut = aes_t_tables.data();
            aes::RoundTableRowZero(lut);
            for(asizei i = 0; i < 256; i++) lut[1 * 256 + i] = _rotl(lut[i],  8);
            for(asizei i = 0; i < 256; i++) lut[2 * 256 + i] = _rotl(lut[i], 16);
            for(asizei i = 0; i < 256; i++) lut[3 * 256 + i] = _rotl(lut[i], 24);
        }
        data = reinterpret_cast<const aubyte*>(aes_t_tables.data());
        footprint = sizeof(auint) * aes_t_tables.size();
    } break;
    case pc_SIMD_alpha: {
        /* The ALPHA table contains (41^n) % 257, with n [0..255]. Due to large powers, you might thing this is a huge mess but it really isn't
        due to modulo properties. More information can be found in SIMD documentation from Ecole Normale Superieure, webpage of Gaetan Laurent,
        you need to look for the "Full Submission Package", will end up with a file SIMD.zip, containing reference.c which explains what to do at LN121.
        Anyway, the results of the above operations are MOSTLY 8-bit numbers. There's an exception however: alphaValue[128] is 0x0100.
        I cut it easy and make everything a short. */
        if(simd_alpha.size() == 0) {
            simd_alpha.resize(256);
            const int base = 41; 
            int power = 1; // base^n
            for(int loop = 0; loop < 256; loop++) {
                simd_alpha[loop] = ashort(power);
                power = (power * base) % 257;
            }
        }
        data = reinterpret_cast<const aubyte*>(simd_alpha.data());
        footprint = sizeof(ashort) * simd_alpha.size();
    } break;
    case pc_SIMD_beta: {
        // The BETA table is very similar to ALPHA. It is built in two steps. In the first, it is basically an alpha table with a different base...
        if(simd_beta.size() == 0) {
            simd_beta.resize(256);
            int base = 163;  // according to documentation, this should be "alpha^127 (respectively alpha^255 for SIMD-256)" which is not 0xA3 to me but I don't really care.
            int power = 1; // base^n
            for(int loop = 0; loop < 256; loop++) {
                simd_beta[loop] = static_cast<aushort>(power);
                power = (power * base) % 257;
            }
            // Now reference implementation mangles it again adding the powers of 40^n,
            // but only in the "final" message expansion. So we need to do nothing more.
            // For some reason the beta value table is called "yoff_b_n" in legacy kernels by lib-SPH... 
        }
        data = reinterpret_cast<const aubyte*>(simd_beta.data());
        footprint = sizeof(aushort) * simd_beta.size();
    } break;
    }
    if(!data) throw "Unknown precomputed constant requested.";
    return std::make_pair(data, footprint);
}


void AbstractTest::PrepareResources(ResourceRequest *resources, asizei numResources, asizei hashCount) {
    for(auto res = resources; res != resources + numResources; res++) {
        if(resHandles.find(res->name) != resHandles.cend()) throw std::string("Duplicated resource name \"" + res->name + '"');
        if(res->name[0] == '$') throw "Trying to allocate a special resource, not supported.";
        resRequests.push_back(*res);
        if(res->immediate) continue; // nothing to allocate here
        ScopedFuncCall popLast([this]() { resRequests.pop_back(); });
        cl_mem build = 0;
        cl_int err = 0;
        if(res->imageDesc.image_width) {
            build = clCreateImage(context, res->memFlags, &res->channels, &res->imageDesc, &res->initialData, &err);
            if(err == CL_INVALID_VALUE) throw std::string("Invalid flags specified for \"") + res->name + '"';
            else if(err == CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)  throw std::string("Invalid image format descriptor for \"") + res->name + '"';
            else if(err == CL_INVALID_IMAGE_DESCRIPTOR) throw std::string("Invalid image descriptor for \"") + res->name + '"';
            else if(err == CL_INVALID_IMAGE_SIZE) throw std::string("Image \"") + res->name + "\" is too big!";
            else if(err == CL_INVALID_HOST_PTR) throw std::string("Invalid host data for \"") + res->name + '"';
            else if(err == CL_IMAGE_FORMAT_NOT_SUPPORTED) throw std::string("Invalid image format for \"") + res->name + '"';
            else if(err != CL_SUCCESS) throw std::string("Some error while creating \"") + res->name + '"';
        }
        else {
            build = clCreateBuffer(context, res->memFlags, res->bytes, const_cast<aubyte*>(res->initialData), &err);
            if(err == CL_INVALID_VALUE) throw std::string("Invalid flags specified for \"") + res->name + '"';
            else if(err == CL_INVALID_BUFFER_SIZE) throw std::string("Buffer size for \"") + res->name + "\" is zero";
            else if(err == CL_INVALID_HOST_PTR) throw std::string("Invalid host data for \"") + res->name + '"';
            else if(err != CL_SUCCESS) throw std::string("Some error while creating \"") + res->name + '"';
        }
        ScopedFuncCall relMem([build]() { clReleaseMemObject(build); });
        resHandles.insert(std::make_pair(res->name, build));
        relMem.Dont();
        popLast.Dont();
    }

    // Generate the special buffers. Those should really be per-device as well!
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


void AbstractTest::PrepareKernels(KernelRequest *kernels, asizei numKernels) {
    // First of all, let's build a set of unique file names. Some algorithms load up the same file more than once.
    // Those are usually very few entries so it's probably faster using an array but set is easier.
    std::map<std::string, std::string> load;
    std::vector<char> source;
    for(auto k = kernels; k < kernels + numKernels; k++) {
        if(load.find(k->fileName) != load.end()) continue;
        auto newKern = load.insert(std::make_pair(k->fileName, std::string())).first;

        std::ifstream disk(newKern->first, std::ios::binary);
        if(disk.is_open() == false) throw std::string("Could not open \"") + newKern->first + '"';
        disk.seekg(0, std::ios::end);
        auto size = disk.tellg();
        if(size >= 1024 * 1024 * 8) throw std::string("Kernel source in \"") + newKern->first + "\" is too big, measures " + std::to_string(size) + " bytes!";
        source.resize(asizei(size) + 1);
        disk.seekg(0, std::ios::beg);
        disk.read(source.data(), size);
        source[asizei(size)] = 0; // not required by specification, but some older drivers are stupid
        newKern->second = source.data();
    }
    // Run all the compile calls. One program must be built for each requested kernel as it will go with different compile options but they have the same source.
    // OpenCL is reference counted (bleargh) so programs can go at the end of this function.
    // I wanted to do this asyncronously but BuildProgram goes with notification functions instead of events (?) so I would have to do that multithreaded.
    // Or, I might just Sleep one second. Not bad either but I cannot be bothered in getting a sleep call here.
    // By the way, CL spec reads as error: "CL_INVALID_OPERATION if the build of a program executable for any of the devices listed in device_list by a previous
    // call to clBuildProgram for program has not completed." So this is really non concurrent?
    std::vector<cl_program> progs(numKernels);
    ScopedFuncCall clearProgs([&progs]() { for(auto el : progs) { if(el) clReleaseProgram(el); } });
    for(asizei loop = 0; loop < numKernels; loop++) {
        const char *str = load.find(kernels[loop].fileName)->second.c_str();
        const asizei len = strlen(str);
        cl_int err = 0;
        cl_program created = clCreateProgramWithSource(context, 1, &str, &len, &err);
        if(err != CL_SUCCESS) throw std::string("Failed to create program \"") + kernels[loop].fileName + '"';
        progs[loop] = created;

        err = clBuildProgram(created, NULL, 0, kernels[loop].compileFlags.c_str(), NULL, NULL);
        std::string errString;
        if(err == CL_INVALID_BUILD_OPTIONS) errString = std::string("Bad build options for \"") + kernels[loop].fileName + '"';
        else if(err != CL_SUCCESS) errString = std::string("OpenCL error ") + std::to_string(err) + "for \"" + kernels[loop].fileName + '"';
        if(errString.length()) {
            cl_device_id sample;
            err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 1, &sample, NULL);
            if(err != CL_SUCCESS) throw errString + "(also failed to get a sample device for context)";
            std::vector<char> log(256);
            asizei requiredChars;
            err = clGetProgramBuildInfo(created, sample, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), &requiredChars);
            if(requiredChars > log.size()) {
                log.resize(requiredChars);
                err = clGetProgramBuildInfo(created, sample, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), &requiredChars);
            }
            if(err != CL_SUCCESS) throw errString + "(also failed to get build error log)";
            throw errString + '\n' + "ERROR LOG:\n" + std::string(log.data(), requiredChars);
        }
    }
    this->kernels.reserve(numKernels);
    for(asizei loop = 0; loop < numKernels; loop++) {
        cl_int err;
        cl_kernel kern = clCreateKernel(progs[loop], kernels[loop].entryPoint.c_str(), &err);
        if(err != CL_SUCCESS) throw std::string("Could not create kernel \"") + kernels[loop].fileName + ':' + kernels[loop].entryPoint + "\", error " + std::to_string(err);
        this->kernels.push_back(KernelDriver(kernels[loop].groupSize, kern));
    }
    for(asizei loop = 0; loop < numKernels; loop++) BindParameters(this->kernels[loop].clk, kernels[loop]);
    aiSignature = ComputeVersionedHash(kernels, numKernels, load);
}


void AbstractTest::BindParameters(cl_kernel kern, const KernelRequest &bindings) {
    // First split out the bindings.
    std::vector<std::string> params;
    asizei comma = 0, prev = 0;
    while((comma = bindings.params.find(',', comma)) != std::string::npos) {
        params.push_back(std::string(bindings.params.cbegin() + prev, bindings.params.cbegin() + comma));
        comma++;
        prev = comma;
    }
    params.push_back(std::string(bindings.params.cbegin() + prev, bindings.params.cend()));
    for(auto &name : params) {
        const char *begin = name.c_str();
        const char *end = name.c_str() + name.length() - 1;
        while(begin < end && *begin == ' ') begin++;
        while(end > begin && *end == ' ') end--;
        end++;
        if(begin != name.c_str() || end != name.c_str() + name.length()) name.assign(begin, end - begin);
        if(name.length() == 0) throw "Kernel binding has empty name.";
    }
    // Now look em up, some are special and perhaps they might need an unified way of mangling (?)
    for(cl_uint loop = 0; loop < params.size(); loop++) {
        const auto &name(params[loop]);
        if(name[0] == '$') {
            cl_mem core = 0;
            if(name == "$wuData") core = wuData;
            else if(name == "$dispatchData") core = dispatchData;
            else if(name == "$candidates") core = candidates;
            else throw std::string("No such core resource \"") + name + '"';
            clSetKernelArg(kern, loop, sizeof(core), &core);
            continue;
        }
        auto bound = resHandles.find(name);
        if(bound != resHandles.cend()) {
            clSetKernelArg(kern, loop, sizeof(cl_mem), &bound->second);
            continue;
        }
        // immediate, maybe
        auto imm = std::find_if(resRequests.cbegin(), resRequests.cend(), [&name](const ResourceRequest &rr) {
            return rr.immediate && rr.name == name;
        });
        if(imm == resRequests.cend()) throw std::string("Could not find parameter \"") + name + '"';
        clSetKernelArg(kern, loop, imm->bytes, imm->initialData);
    }
}


std::vector<std::string> AbstractTest::RunTests(cl_device_id device) const {
    using namespace std;
    cl_int err = 0;
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if(!queue || err != CL_SUCCESS) throw "Could not create command queue for device!";
    ScopedFuncCall clearQueue([queue]() { clReleaseCommandQueue(queue); });
    vector<string> errorMessages;
    const cl_uint *nextNonces = found.data();
    for(asizei bindex = 0; bindex < headers.size(); bindex++) {
        const TestRun &block(headers[bindex]);
        aubyte be[80];
        for(asizei i = 0; i < sizeof(be); i += 4) {
            for(asizei cp = 0; cp < 4; cp++) be[i + cp] = block.clData[i + 3 - cp];
        }
        err = clEnqueueWriteBuffer(queue, wuData, CL_TRUE, 0, sizeof(be), be, 0, NULL, NULL);
        if(err != CL_SUCCESS) throw string("CL error ") + to_string(err) + " while attempting to update $wuData";
        
		cl_uint buffer[5]; // taken as is from M8M FillDispatchData... how ugly!
		buffer[0] = 0;
		buffer[1] = static_cast<cl_uint>(block.targetBits >> 32);
		buffer[2] = static_cast<cl_uint>(block.targetBits);
		buffer[3] = 0;
		buffer[4] = 0;
        err = clEnqueueWriteBuffer(queue, dispatchData, CL_TRUE, 0, sizeof(buffer), buffer, 0, NULL, NULL);
        if(err != CL_SUCCESS) throw string("CL error ") + to_string(err) + " while attempting to update $dispatchData";

        aulong remHashes = block.iterations * nominalHashCount;
        cl_uint base = 0;
        vector<auint> candidates;
        while(remHashes) {
            const cl_uint thisScan = cl_uint(min(remHashes, aulong(concurrency)));
		    cl_uint zero = 0;
		    clEnqueueWriteBuffer(queue, this->candidates, CL_TRUE, 0, sizeof(cl_uint), &zero, 0, NULL, NULL);
            RunAlgorithm(queue, base, thisScan);
            vector<auint> results(FromNonceBuffer(queue));
            base += thisScan;
            remHashes -= thisScan;
            for(auto nonce : results) candidates.push_back(nonce);
        }
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


std::vector<cl_uint> AbstractTest::FromNonceBuffer(cl_command_queue q) const {
    cl_int err;
    cl_uint *nonces = reinterpret_cast<cl_uint*>(clEnqueueMapBuffer(q, candidates, CL_TRUE, CL_MAP_READ, 0, nonceBufferSize, 0, NULL, NULL, &err));
    if(err != CL_SUCCESS) throw std::string("CL error ") + std::to_string(err) + " attempting to map nonce buffers.";
    ScopedFuncCall unmap([q, nonces, this]() { clEnqueueUnmapMemObject(q, candidates, nonces, 0, NULL, NULL); });
    const cl_uint numNonces = *nonces;
    nonces++;
    std::vector<cl_uint> ret;
    for(asizei cp = 0; cp < numNonces; cp++) ret.push_back(nonces[cp]);
    return ret;
}


void AbstractTest::RunAlgorithm(cl_command_queue q, asizei base, asizei amount) const {
    for(asizei loop = 0; loop < kernels.size(); loop++) {
        const auto &kern(kernels[loop]);
        asizei woff[3], wsize[3];
        memset(woff, 0, sizeof(woff));
        memset(wsize, 0, sizeof(wsize));
        /* Kernels used by M8M always have the same group format layout: given an N-dimensional kernel,
        - (N-1)th dimension is the hash being computed in global work -> number of hashes computed per workgroup in local work declaration
        - All previous dimensions are the "team" and can be easily pulled from declaration.
        Work offset leaves "team players" untouched while global work size is always <team size><total hashes>. */
        woff[kern.dimensionality - 1] = base;
        for(auto cp = 0u; cp < kern.dimensionality - 1; cp++) wsize[cp] = kern.wgs[cp];
        wsize[kern.dimensionality - 1] = amount;

        cl_int error = clEnqueueNDRangeKernel(q, kernels[loop].clk, kernels[loop].dimensionality, woff, wsize, kernels[loop].wgs, 0, NULL, NULL);
        if(error != CL_SUCCESS) {
            std::string ret("OpenCL error " + std::to_string(error) + " returned by clEnqueueNDRangeKernel(");
            ret += algoName;
            ret += '[' + std::to_string(loop) + ']';
            throw ret;
        }
    }
}


aulong AbstractTest::ComputeVersionedHash(const KernelRequest *kerns, asizei numKernels, const std::map<std::string, std::string> &src) const {
	std::string sign(algoName + '.' + impName + '.' + iversion + '\n');
    for(auto kern = kerns; kern < kerns + numKernels; kern++) {
        sign += ">>>>" + kern->fileName + ':' + kern->entryPoint + '(' + kern->compileFlags + ')' + '\n';
        // groupSize is most likely not to be put there...
        // are param bindings to be put there?
        sign += src.find(kern->fileName)->second + "<<<<\n";
    }
	hashing::SHA256 blah(reinterpret_cast<const aubyte*>(sign.c_str()), sign.length());
	hashing::SHA256::Digest blobby;
	blah.GetHash(blobby);
	aulong ret = 0; // ignore endianess here so we get to know host endianess by algo signature
	for(asizei loop = 0; loop < blobby.size(); loop += 8) {
		aulong temp;
		memcpy_s(&temp, sizeof(temp), blobby.data() + loop, sizeof(temp));
		ret ^= temp;
	}
	return ret;
}
