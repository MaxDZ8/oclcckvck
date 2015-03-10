/*
 * Copyright (C) 2015 Massimo Del Zotto
 * This code is released under the MIT license.
 * For conditions of distribution and use, see the LICENSE or hit the web.
 */
#include "NS_CoreLoops.h"

namespace stepTest {

namespace nsHelp {


void Salsa::operator()(auint state[16]) {
	for(auint loop = 0; loop < MIX_ROUNDS; loop++) {
		// First we mangle 4 independant columns. Each column starts on a diagonal cell so they are "rotated up" somehow.
		state[ 4] ^= _rotl(state[ 0] + state[12], 7u);
		state[ 8] ^= _rotl(state[ 4] + state[ 0], 9u);
		state[12] ^= _rotl(state[ 8] + state[ 4], 13u);
		state[ 0] ^= _rotl(state[12] + state[ 8], 18u);

		state[ 9] ^= _rotl(state[ 5] + state[ 1], 7u);
		state[13] ^= _rotl(state[ 9] + state[ 5], 9u);
		state[ 1] ^= _rotl(state[13] + state[ 9], 13u);
		state[ 5] ^= _rotl(state[ 1] + state[13], 18u);

		state[14] ^= _rotl(state[10] + state[ 6], 7u);
		state[ 2] ^= _rotl(state[14] + state[10], 9u);
		state[ 6] ^= _rotl(state[ 2] + state[14], 13u);
		state[10] ^= _rotl(state[ 6] + state[ 2], 18u);

		state[ 3] ^= _rotl(state[15] + state[11], 7u);
		state[ 7] ^= _rotl(state[ 3] + state[15], 9u);
		state[11] ^= _rotl(state[ 7] + state[ 3], 13u);
		state[15] ^= _rotl(state[11] + state[ 7], 18u);

		// Then we mangle rows, again those are rotated. First is rotated 3, others are rotated less.
		// It would be easier to visualize that the other way around.
		state[ 1] ^= _rotl(state[ 0] + state[ 3], 7u);
		state[ 2] ^= _rotl(state[ 1] + state[ 0], 9u);
		state[ 3] ^= _rotl(state[ 2] + state[ 1], 13u);
		state[ 0] ^= _rotl(state[ 3] + state[ 2], 18u);

		state[ 6] ^= _rotl(state[ 5] + state[ 4], 7u);
		state[ 7] ^= _rotl(state[ 6] + state[ 5], 9u);
		state[ 4] ^= _rotl(state[ 7] + state[ 6], 13u);
		state[ 5] ^= _rotl(state[ 4] + state[ 7], 18u);

		state[11] ^= _rotl(state[10] + state[ 9], 7u);
		state[ 8] ^= _rotl(state[11] + state[10], 9u);
		state[ 9] ^= _rotl(state[ 8] + state[11], 13u);
		state[10] ^= _rotl(state[ 9] + state[ 8], 18u);

		state[12] ^= _rotl(state[15] + state[14], 7u);
		state[13] ^= _rotl(state[12] + state[15], 9u);
		state[14] ^= _rotl(state[13] + state[12], 13u);
		state[15] ^= _rotl(state[14] + state[13], 18u);
	}
}


void Chacha::operator()(auint state[16]) {
	for(auint loop = 0; loop < MIX_ROUNDS; loop++) {
		// Here we have some mangling "by column".
		state[ 0] += state[ 4];    state[12] = _rotl(state[12] ^ state[ 0], 16u);
		state[ 8] += state[12];    state[ 4] = _rotl(state[ 4] ^ state[ 8], 12u);
		state[ 0] += state[ 4];    state[12] = _rotl(state[12] ^ state[ 0], 8u);
		state[ 8] += state[12];    state[ 4] = _rotl(state[ 4] ^ state[ 8], 7u);

		state[ 1] += state[ 5];    state[13] = _rotl(state[13] ^ state[ 1], 16u);
		state[ 9] += state[13];    state[ 5] = _rotl(state[ 5] ^ state[ 9], 12u);
		state[ 1] += state[ 5];    state[13] = _rotl(state[13] ^ state[ 1], 8u);
		state[ 9] += state[13];    state[ 5] = _rotl(state[ 5] ^ state[ 9], 7u);

		state[ 2] += state[ 6];    state[14] = _rotl(state[14] ^ state[ 2], 16u);
		state[10] += state[14];    state[ 6] = _rotl(state[ 6] ^ state[10], 12u);
		state[ 2] += state[ 6];    state[14] = _rotl(state[14] ^ state[ 2], 8u);
		state[10] += state[14];    state[ 6] = _rotl(state[ 6] ^ state[10], 7u);

		state[ 3] += state[ 7];    state[15] = _rotl(state[15] ^ state[ 3], 16u);
		state[11] += state[15];    state[ 7] = _rotl(state[ 7] ^ state[11], 12u);
		state[ 3] += state[ 7];    state[15] = _rotl(state[15] ^ state[ 3], 8u);
		state[11] += state[15];    state[ 7] = _rotl(state[ 7] ^ state[11], 7u);

		// Then we mix by diagonal.
		state[ 0] += state[ 5];    state[15] = _rotl(state[15] ^ state[ 0], 16u);
		state[10] += state[15];    state[ 5] = _rotl(state[ 5] ^ state[10], 12u);
		state[ 0] += state[ 5];    state[15] = _rotl(state[15] ^ state[ 0], 8u);
		state[10] += state[15];    state[ 5] = _rotl(state[ 5] ^ state[10], 7u);

		state[ 1] += state[ 6];    state[12] = _rotl(state[12] ^ state[ 1], 16u);
		state[11] += state[12];    state[ 6] = _rotl(state[ 6] ^ state[11], 12u);
		state[ 1] += state[ 6];    state[12] = _rotl(state[12] ^ state[ 1], 8u);
		state[11] += state[12];    state[ 6] = _rotl(state[ 6] ^ state[11], 7u);

		state[ 2] += state[ 7];    state[13] = _rotl(state[13] ^ state[ 2], 16u);
		state[ 8] += state[13];    state[ 7] = _rotl(state[ 7] ^ state[ 8], 12u);
		state[ 2] += state[ 7];    state[13] = _rotl(state[13] ^ state[ 2], 8u);
		state[ 8] += state[13];    state[ 7] = _rotl(state[ 7] ^ state[ 8], 7u);

		state[ 3] += state[ 4];    state[14] = _rotl(state[14] ^ state[ 3], 16u);
		state[ 9] += state[14];    state[ 4] = _rotl(state[ 4] ^ state[ 9], 12u);
		state[ 3] += state[ 4];    state[14] = _rotl(state[14] ^ state[ 3], 8u);
		state[ 9] += state[14];    state[ 4] = _rotl(state[ 4] ^ state[ 9], 7u);
	}
}


}

}
