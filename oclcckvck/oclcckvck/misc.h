/*
 * Copyright (C) 2015 Massimo Del Zotto
 * This code is released under the MIT license.
 * For conditions of distribution and use, see the LICENSE or hit the web.
 */
#pragma once
#include <string>
#include "../Common/AREN/ArenDataTypes.h"


static std::string Hex(const aubyte *blob, asizei count) {
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
