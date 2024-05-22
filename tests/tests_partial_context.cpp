#include <iostream>
#include <unordered_map>
#include <functional>
#include <sstream>
#include "PartialContext.h"
#include "Bitset.h"

using namespace mlconcepts;

std::unordered_map<std::string, std::function<int(int,char**)>> tests = 
{
    {
        "bitset64_serialization",
        [](int argc, char** argv) {
            Bitset<std::uint64_t> set(134);
            set.Add(15); set.Add(16); set.Add(17); set.Add(18); set.Add(19);
            set.Add(21); set.Add(75); set.Add(111);
            std::ostringstream oss(std::ios::binary);
            set.Serialize(oss);
            auto s = oss.str();
            std::istringstream iss(s, std::ios::binary);
            Bitset<std::uint64_t> set2(134);
            set2.Deserialize(iss);
            return set == set2 ? 0 : 1;
        }
    },
    {
        "bitset64_serialization_longsequences",
        [](int argc, char** argv) {
            Bitset<std::uint64_t> set({
                0ull, 0ull,0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF
            });
            std::ostringstream oss(std::ios::binary);
            set.Serialize(oss);
            auto s = oss.str();
            std::istringstream iss(s, std::ios::binary);
            Bitset<std::uint64_t> set2(256);
            set2.Deserialize(iss);
            return set == set2 ? 0 : 1;
        }
    },
    {
        "context64_serialization",
        [](int argc, char** argv) {
            PartialContext<std::uint64_t> context(130, 120);
            context.SetIncidence(1, 63); context.SetIncidence(1, 88);
            context.SetIncidence(1, 111); context.SetIncidence(2, 63);
            context.SetIncidence(63, 76); context.SetIncidence(87, 110);
            context.SetIncidence(6, 68); context.SetIncidence(35, 35);
            context.SetIncidence(35, 53); context.SetIncidence(12, 63);
            context.SetIncidence(66, 65); context.SetIncidence(1, 10);
            std::ostringstream oss(std::ios::binary);
            context.Serialize(oss);
            auto s = oss.str();
            std::istringstream iss(s, std::ios::binary);
            PartialContext<std::uint64_t> context2(iss);
            return context == context2 ? 0 : 1;
        }
    },
    {
        "small_context",
        [](int argc, char** argv) {
            PartialContext<std::uint64_t> c(4, 12); 
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
            return c.ComputeClosureSize(3) == 2 && 
                   c.ComputeClosureSize(0) == 1 ? 0 : 1;
        }
    },
    {
        "bitstreams1",
        [](int argc, char** argv) {
            std::ostringstream s1(std::ios::binary);
            OutputBitstream b1(s1);
            b1.WriteBits<3>(0x2);  //010                   010
            b1.WriteBits<6>(0x3C); //1 1110 0010           111100 010
            b1.WriteBits<5>(0xD);  //01 1011 1110 0010     01101 111100 010
            b1.WriteBits<2>(0x1);  //0101 1011 1110 0010   01 01101 111100 010
            b1.Close();
            std::string s(s1.str());
            if ((std::uint8_t)s[0] != 0xE2 || (std::uint8_t)s[1] != 0x5B) return 1;
            std::istringstream s2(s, std::ios::binary);
            InputBitstream b2(s2);
            if (b2.ReadBits<3>() != 0x2 || b2.ReadBits<6>() != 0x3C || 
                b2.ReadBits<5>() != 0xD || b2.ReadBits<2>() != 0x1) {
                return 2;
            }
            return 0;
        }
    }
};

int main(int argc, char** argv) {
    if (argc < 2) return -1;
    if (tests.count(std::string(argv[1])) == 0) return -2;
    return tests[std::string(argv[1])](argc, argv);
}