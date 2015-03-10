/*
 * Copyright (C) 2015 Massimo Del Zotto
 * This code is released under the MIT license.
 * For conditions of distribution and use, see the LICENSE or hit the web.
 */
#include "NS_KDFs_4W.h"

namespace stepTest {

    
const auint NS_KDFHelper::KDF_CONST_N = 32;
const auint NS_KDFHelper::KDF_SIZE = 256;
const auint NS_KDFHelper::MIX_ROUNDS = 10;

const auint NS_KDFHelper::blake2S_IV[8]= {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

const aubyte NS_KDFHelper::blake2S_sigma[10][16] = {
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 } ,
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 } ,
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 } ,
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 } ,
    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 } ,
    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 } ,
    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 } ,
    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 } ,
    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 } ,
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13 , 0 } ,
};


std::array<auint, 64> NS_KDFHelper::FirstKDF(const aubyte *block, aubyte *buff_a, aubyte *buff_b) {
	// Just look at CL kernels for some extra documentation. They're structured 4-way currently and they append the appropriate nonce.
	// It doesn't need to do that here.
	FillInitialBuffer(buff_a, 64, block, 20);
	FillInitialBuffer(buff_b, 32, block, 20);

	auint buffStart = 0;
	for(auint loop = 0; loop < KDF_CONST_N; loop++) buffStart = FastKDFIteration(buffStart, buff_a, buff_b);
	const auint outLen = 256;
	auint remaining = KDF_SIZE - buffStart;
	auint valid = remaining < outLen? remaining : outLen;
	std::array<auint, 64> retval;
	aubyte *output = reinterpret_cast<aubyte*>(retval.data());
	for(auint set = 0; set < valid; set++) output[set] = buff_b[set + buffStart] ^ buff_a[set];
	for(auint set = valid; set < outLen; set++) {
		auint srci = set - valid;
		output[set] = buff_b[srci] ^ buff_a[srci + remaining];
	}
	return retval;
}


std::array<aubyte, 32> NS_KDFHelper::LastKDF(const std::array<auint, 64> &state, const aubyte *buff_a, aubyte *buff_b) {
	// Just look at CL kernels for some extra documentation. They're structured 4-way currently and they append the appropriate nonce.
	// It doesn't need to do that here.
	FillInitialBuffer(buff_b, 32, reinterpret_cast<const aubyte*>(state.data()), 64);
	auint buffStart = 0;
	for(auint loop = 0; loop < KDF_CONST_N; loop++) buffStart = FastKDFIteration(buffStart, buff_a, buff_b);
	const auint outLen = 32;
	auint remaining = KDF_SIZE - buffStart;
	auint valid = remaining < outLen? remaining : outLen;
	std::array<aubyte, 32> retval;
	aubyte *output = retval.data();
	for(auint set = 0; set < valid; set++) output[set] = buff_b[set + buffStart] ^ buff_a[set];
	for(auint set = valid; set < outLen; set++) {
		auint srci = set - valid;
		output[set] = buff_b[srci] ^ buff_a[srci + remaining];
	}
	return retval;
}


auint NS_KDFHelper::FastKDFIteration(auint buffStart, const aubyte *buff_a, aubyte *buff_b) {
	auint input[16], key[8];
	memcpy_s(input, sizeof(input), buff_a + buffStart, sizeof(input));
	memcpy_s(key, sizeof(key), buff_b + buffStart, sizeof(key));

	auint prf_output[8];
	Blake2S_64_32(prf_output, input, key, MIX_ROUNDS);
	auint sum = 0;
	for(auint el = 0; el < 8; el++) {
		sum += (prf_output[el]      ) & 0xFF;
		sum += (prf_output[el] >>  8) & 0xFF;
		sum += (prf_output[el] >> 16) & 0xFF;
		sum += (prf_output[el] >> 24) & 0xFF;
	}
	buffStart = sum % KDF_SIZE; // or &= (256 - 1), the same
	for(auint cp = 0; cp < 8; cp++) {
		auint bval;
		memcpy_s(&bval, sizeof(bval), buff_b + buffStart + cp * 4, sizeof(bval));
		bval ^= prf_output[cp];
		memcpy_s(buff_b + buffStart + cp * 4, sizeof(bval), &bval, sizeof(bval));
	}
	if(buffStart < sizeof(key)) {
		// Forward what I just wrote. The GPU kernel keeps this in registers, in CPU I can just take it easy.
		asizei rem = sizeof(key) - buffStart;
		auint count = auint(sizeof(prf_output) < rem? sizeof(prf_output) : rem);
		const aubyte *src = buff_b + buffStart;
		aubyte *dst = buff_b + buffStart + KDF_SIZE;
		const asizei trailing = (256 + 32) - (buffStart + KDF_SIZE);
		memcpy_s(dst, trailing, src, count);
	}
	auint rem = KDF_SIZE - buffStart;
	if(rem < sizeof(prf_output)) {
		auint count = sizeof(prf_output) - rem;
		aubyte *src = buff_b + KDF_SIZE;
		aubyte *dst = buff_b;
		for(auint cp = 0; cp < count; cp++) dst[cp] = src[cp];
	}
	return buffStart;
}


void NS_KDFHelper::FillInitialBuffer(aubyte *target, auint extraBytes, const aubyte *pattern, auint blockLen) {
	blockLen *= 4;
	const auint fullBlocks = KDF_SIZE / blockLen;
	// First, repeat the passed block an integral amount of times.
	aubyte *dst = target;
	asizei rem = KDF_SIZE;
	for(auint block = 0; block < fullBlocks; block++) {
		memcpy_s(dst, rem, pattern, blockLen);
		dst += blockLen;
		rem -= blockLen;
	}
	// Fill remaining bytes by wrapping-around.
	memcpy_s(dst, rem, pattern, rem);
	memcpy_s(dst + rem, extraBytes, pattern, extraBytes);
}


void NS_KDFHelper::Blake2S_64_32(auint *output, auint *input, auint *key, const auint numRounds) {
	std::array<auint, 8> hash;
	memcpy_s(hash.data(), sizeof(hash), blake2S_IV, sizeof(blake2S_IV));
	hash[0] ^= sizeof(auint) * 8;
	hash[0] ^= (sizeof(auint) * 8) << 8;
	hash[0] ^= (1 << 16) | (1 << 24);
	std::array<auint, 16> block;
	for(auint cp = 0; cp <  8; cp++) block[cp] = key[cp];
	for(auint cp = 8; cp < 16; cp++) block[cp] = 0;
	std::array<auint, 4> counter;
	counter[0] = 64;
	counter[1] = counter[2] = counter[3] = 0;
	hash = Blake2SBlockXForm(hash, counter, numRounds, block);
	counter[0] = 128;
	counter[2] = ~0;
	memcpy_s(block.data(), sizeof(block), input, sizeof(input[0]) * 16);
	hash = Blake2SBlockXForm(hash, counter, numRounds, block);
	for(asizei cp = 0; cp < hash.size(); cp++) output[cp] = hash[cp];
}


std::array<auint, 8> NS_KDFHelper::Blake2SBlockXForm(const std::array<auint, 8> hash, const std::array<auint, 4> &counter, const auint numRounds, const std::array<auint, 16> &msg) {
	std::array<auint, 16> val;
	for(auint el =  0; el <  8; el++) val[el] = hash[el];
	for(auint el =  8; el < 12; el++) val[el] = 0;
	for(auint el = 12; el < 16; el++) val[el] = counter[el - 12];

	for(auint el = 0; el < 8; el++) val[el + 8] ^= blake2S_IV[el];
	aubyte a, b, c, d;
	auto roundMix = [&a, &b, &c, &d, &val](const std::array<auint, 16> &msg, const aubyte *perm) {
		val[a] = val[a] + val[b] + msg[perm[0]];
		val[d] = _rotr(val[d] ^ val[a], 16);
		val[c] = val[c] + val[d];
		val[b] = _rotr(val[b] ^ val[c], 12);
		val[a] = val[a] + val[b] + msg[perm[1]];
		val[d] = _rotr(val[d] ^ val[a], 8);
		val[c] = val[c] + val[d];
		val[b] = _rotr(val[b] ^ val[c], 7);
	};
	for(auint round = 0; round < numRounds; round++) {
		const aubyte *perm = blake2S_sigma[round];
		for(auint col = 0; col < 4; col++) {
			a = col + 0 * 4;
			b = col + 1 * 4;
			c = col + 2 * 4;
			d = col + 3 * 4;
			roundMix(msg, perm);
			perm += 2;
		}
		for(auint diag = 0; diag < 4; diag++) {
			a = (diag + 0) % 4 + 0 * 4;
			b = (diag + 1) % 4 + 1 * 4;
			c = (diag + 2) % 4 + 2 * 4;
			d = (diag + 3) % 4 + 3 * 4;
			roundMix(msg, perm);
			perm += 2;
		}
	}
	std::array<auint, 8> retVal(hash);
	for(auint i = 0; i < 8; i++) retVal[i] ^= val[i] ^ val[i + 8];
	return retVal;
}


}
