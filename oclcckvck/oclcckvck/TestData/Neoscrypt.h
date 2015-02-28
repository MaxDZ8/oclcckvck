/*
 * Copyright (C) 2015 Massimo Del Zotto
 * This code is released under the MIT license.
 * For conditions of distribution and use, see the LICENSE or hit the web.
 */
#pragma once
#include "../AlgoTest.h"

namespace testData {

class Neoscrypt : public AlgoTest {
public:
    Neoscrypt() : AlgoTest(8 * 1024) { }
    std::pair<const TestRun*, asizei> GetHeaders() const { return std::make_pair(deterministic, sizeof(deterministic) / sizeof(deterministic[0])); }
    std::pair<const auint*, asizei> GetFound() const { return std::make_pair(found_candidates, sizeof(found_candidates) / sizeof(found_candidates[0])); }

private:
    const static TestRun deterministic[21];
    const static unsigned int found_candidates[25];
};

}
