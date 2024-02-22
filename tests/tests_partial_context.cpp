#include <iostream>
#include <unordered_map>
#include <functional>
#include "PartialContext.h"
#include "Bitset.h"

using namespace mlconcepts;

std::unordered_map<std::string, std::function<int(int,char**)>> tests = 
{
    {
        "small_context",
        [](int argc, char** argv) {
            PartialContext<uint64_t> c(4, 12); 
            c.SetIncidence(0, 2);
            c.SetIncidence(0, 4);
            c.SetIncidence(0, 11);
            c.SetIncidence(1, 5);
            c.SetIncidence(1, 7);
            c.SetIncidence(1, 8);
            c.SetIncidence(1, 10);
            c.SetIncidence(2, 1);
            c.SetIncidence(2, 3);
            c.SetIncidence(2, 5);
            c.SetIncidence(2, 6);
            c.SetIncidence(2, 9);
            c.SetIncidence(3, 5);
            c.SetIncidence(3, 9);
            return c.ComputeClosureSize(3) == 2 && c.ComputeClosureSize(0) == 1 ? 0 : 1;
        }
    }
};

int main(int argc, char** argv) {
    if (argc < 2) return -1;
    if (tests.count(std::string(argv[1])) == 0) return -2;
    return tests[std::string(argv[1])](argc, argv);
}