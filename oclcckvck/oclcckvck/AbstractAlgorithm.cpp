/*
 * Copyright (C) 2015 Massimo Del Zotto
 * This code is released under the MIT license.
 * For conditions of distribution and use, see the LICENSE or hit the web.
 */
#include "AbstractAlgorithm.h"


void AbstractAlgorithm::PrepareResources(ResourceRequest *resources, asizei numResources, asizei hashCount) {
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
}


void AbstractAlgorithm::PrepareKernels(KernelRequest *kernels, asizei numKernels) {
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
    for(asizei loop = 0; loop < numKernels; loop++) BindParameters(this->kernels[loop], kernels[loop]);
    aiSignature = ComputeVersionedHash(kernels, numKernels, load);
}


void AbstractAlgorithm::BindParameters(KernelDriver &kdesc, const KernelRequest &bindings) {
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
        SpecialValueBinding desc;
        if(SpecialValue(desc, name)) {
            if(desc.earlyBound) clSetKernelArg(kdesc.clk, loop, sizeof(desc.resource.buff), &desc.resource.buff);
            else kdesc.dtBindings.push_back(std::make_pair(loop, desc.resource.index));
            continue;
        }
        auto bound = resHandles.find(name);
        if(bound != resHandles.cend()) {
            clSetKernelArg(kdesc.clk, loop, sizeof(cl_mem), &bound->second);
            continue;
        }
        // immediate, maybe
        auto imm = std::find_if(resRequests.cbegin(), resRequests.cend(), [&name](const ResourceRequest &rr) {
            return rr.immediate && rr.name == name;
        });
        if(imm == resRequests.cend()) throw std::string("Could not find parameter \"") + name + '"';
        clSetKernelArg(kdesc.clk, loop, imm->bytes, imm->initialData);
    }
}


void AbstractAlgorithm::RunAlgorithm(cl_command_queue q, asizei amount) const {
    for(asizei loop = 0; loop < kernels.size(); loop++) {
        const auto &kern(kernels[loop]);
        for(auto param : kern.dtBindings) {
            cl_mem buff(lbBuffers[param.second]);
            clSetKernelArg(kern.clk, param.first, sizeof(buff), &buff);
        }
        asizei woff[3], wsize[3];
        memset(woff, 0, sizeof(woff));
        memset(wsize, 0, sizeof(wsize));
        /* Kernels used by M8M always have the same group format layout: given an N-dimensional kernel,
        - (N-1)th dimension is the hash being computed in global work -> number of hashes computed per workgroup in local work declaration
        - All previous dimensions are the "team" and can be easily pulled from declaration.
        Work offset leaves "team players" untouched while global work size is always <team size><total hashes>. */
        woff[kern.dimensionality - 1] = hashing.nonceBase;
        for(auto cp = 0u; cp < kern.dimensionality - 1; cp++) wsize[cp] = kern.wgs[cp];
        wsize[kern.dimensionality - 1] = amount;

        cl_int error = clEnqueueNDRangeKernel(q, kernels[loop].clk, kernels[loop].dimensionality, woff, wsize, kernels[loop].wgs, 0, NULL, NULL);
        if(error != CL_SUCCESS) {
            std::string ret("OpenCL error " + std::to_string(error) + " returned by clEnqueueNDRangeKernel(");
            ret += identifier.algorithm + '.' + identifier.implementation;
            ret += '[' + std::to_string(loop) + "])";
            throw ret;
        }
    }
}


aulong AbstractAlgorithm::ComputeVersionedHash(const KernelRequest *kerns, asizei numKernels, const std::map<std::string, std::string> &src) const {
	std::string sign(identifier.algorithm + '.' + identifier.implementation + '.' + identifier.version + '\n');
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
