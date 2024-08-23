#include "Bisimulation.h"
#include <string_view>

using namespace mlconcepts;
typedef LTS<std::uint32_t> LTS32;

bool check_bisim(const std::vector<std::uint32_t>& blocks,
                 const std::vector<std::uint32_t>& expected) 
{
    if (blocks.size() != expected.size())
        return false;
    auto nblocks = *std::max_element(blocks.begin(), blocks.end()) + 1;
    std::vector<std::uint32_t> mapping(nblocks);
    for (auto& x : mapping) x = std::numeric_limits<std::uint32_t>::max();
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        if (mapping[blocks[i]] != std::numeric_limits<std::uint32_t>::max() &&
            mapping[blocks[i]] != expected[i]) {
            return false;
        }
        mapping[blocks[i]] = expected[i];
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) return -1;
    std::string_view test_name(argv[1]);

    if (test_name == "pt_edges") {
        std::vector<std::uint32_t> sources({ 0, 0, 1, 1, 2, 2, 2, 3 });
        std::vector<std::uint32_t> targets({ 2, 1, 0, 1, 1, 2, 3, 0 });
        
        mlconcepts::internal::_pt_edges edges(sources, targets, 4);
        
        auto v0 = edges.preimage(0);
        auto v1 = edges.preimage(1);
        auto v2 = edges.preimage(2);
        auto v3 = edges.preimage(3);
        
        if (v0.size() != 2 || v1.size() != 3 || v2.size() != 2 || v3.size()!=1)
            return 1;
        if (v0[0] != 1 || v0[1] != 3)
            return 2;
        if (v1[0] != 0 || v1[1] != 1 || v1[2] != 2)
            return 3;
        if (v2[0] != 0 || v2[1] != 2)
            return 4;
        if (v3[0] != 2)
            return 5;
        return 0;
    } else if (test_name == "bisimulations") {
        auto blocks = ComputeBisimulation(LTS32::CompleteTree(2, 4));
        std::vector<std::uint32_t> expected({
            0, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1
        });
        if (!check_bisim(blocks, expected)) return 1;

        blocks = ComputeBisimulation(LTS32::Loop(5));
        expected = std::vector<std::uint32_t>({
            0, 0, 0, 0, 0
        });
        if (!check_bisim(blocks, expected)) return 2;

        blocks = ComputeBisimulation(LTS32::List(5));
        expected = std::vector<std::uint32_t>({
            0, 4, 3, 2, 1
        });
        if (!check_bisim(blocks, expected)) return 3;

        blocks = ComputeBisimulation(LTS32::CompleteTree(3, 3));
        expected = std::vector<std::uint32_t>({
            0, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1
        });
        if (!check_bisim(blocks, expected)) return 4;

        blocks = ComputeBisimulation(LTS32(6, {0, 0, 1, 3, 3, 4},
                                                 {1, 2, 2, 4, 5, 5}));
        expected = std::vector<std::uint32_t>({
            0, 2, 1, 0, 2, 1
        });
        if (!check_bisim(blocks, expected)) return 5;

        blocks = ComputeBisimulation(LTS32(5, {0, 0, 0, 1, 2, 3},
                                                 {1, 2, 3, 2, 4, 4}));
        expected = std::vector<std::uint32_t>({
            0, 3, 2, 2, 1
        });
        if (!check_bisim(blocks, expected)) return 6;

        return 0;
    } else if (test_name == "bisimulations_cpu") {
        auto blocks = ComputeBisimulationCPU(LTS32::CompleteTree(2, 4));
        std::vector<std::uint32_t> expected({
            0, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1
        });
        if (!check_bisim(blocks, expected)) return 1;

        blocks = ComputeBisimulationCPU(LTS32::Loop(5));
        expected = std::vector<std::uint32_t>({
            0, 0, 0, 0, 0
        });
        if (!check_bisim(blocks, expected)) return 2;

        blocks = ComputeBisimulationCPU(LTS32::List(5));
        expected = std::vector<std::uint32_t>({
            0, 4, 3, 2, 1
        });
        if (!check_bisim(blocks, expected)) return 3;

        blocks = ComputeBisimulationCPU(LTS32::CompleteTree(3, 3));
        expected = std::vector<std::uint32_t>({
            0, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1
        });
        if (!check_bisim(blocks, expected)) return 4;

        blocks = ComputeBisimulationCPU(LTS32(6, {0, 0, 1, 3, 3, 4},
                                                 {1, 2, 2, 4, 5, 5}));
        expected = std::vector<std::uint32_t>({
            0, 2, 1, 0, 2, 1
        });
        if (!check_bisim(blocks, expected)) return 5;

        blocks = ComputeBisimulationCPU(LTS32(5, {0, 0, 0, 1, 2, 3},
                                                 {1, 2, 3, 2, 4, 4}));
        expected = std::vector<std::uint32_t>({
            0, 3, 2, 2, 1
        });
        if (!check_bisim(blocks, expected)) return 6;

        return 0;
    }
    return -2;
}