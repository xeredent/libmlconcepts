#pragma once
#include <cstdint>
#include <utility>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <ranges>
#include <list>
#include <vector>

#include "Bitset.h"

namespace mlconcepts {

/// @brief A labelled transition system stored using a block encoding. This
///    encoding makes use of three arrays: offsets, labels, and targets.
///    The first one contains contains #(number of nodes) + 1 elements, and, 
///    for each node in the graph, it contains the offset to the information
///    on its outgoing edges contained in the other two arrays. The offsets 
///    array is increasing, and for each node at index i, its associated edges
///    go from index offsets[i] and offsets[i+1], and hence are found in a
///    contiguous block of memory, which can be efficiently accessed by a GPU.
///    The arrays labels and targets, contain information on the label and the
///    end of each edge (the source of the edge is derived from its position in
///    the array, as edges with source i are found between offsets[i] and 
///    offsets[i+1]).
///    This is encoding is currently not in use by the algorithms in the
///    library, but is kept for future implementations.
/// @tparam NodeID The type of the indices of the nodes.
/// @tparam EdgeID The type of the indices of the edges.
/// @tparam LabelID The type of the indices of the labels.
template<typename NodeID = std::uint32_t, 
         typename EdgeID = std::uint32_t,
         typename LabelID = std::uint32_t>
class BlockLTS {
public:
    /// @brief Constructs an empty labelled transition system.
    BlockLTS() {
        offsets = { 0 };
    }

    /// @brief Constructs a labelled transition system.
    /// @param offsets An array of size #(number of nodes) + 1, containing for
    ///     each node the indices that indicate where to find information on
    ///     its outgoing edges in the other two vectors. Edges for the node at
    ///     index i are found between offsets[i] and offsets[i+1].
    /// @param labels A vector containing the labels of each transition (edge)
    ///     in the system. If the vector is empty, the LTS is assumed to be a
    ///     digraph.
    /// @param targets A vector containing the end points of each transition
    ///     (edge). The source of each edge is inferred by its index in the
    ///     array using the information in the offsets array.
    BlockLTS(std::vector<EdgeID>&& offsets,
             std::vector<LabelID>&& labels,
             std::vector<NodeID>&& targets) {
        this->offsets = std::move(offsets);
        this->labels = std::move(labels);
        this->targets = std::move(targets);
    }

    /// @brief Adds an edge to the graph without adding information on the
    ///     labels of the graph. If the LTS already contains labels for the
    ///     transition, you should not use this method, but its overload
    ///     with an additional label parameter.
    /// @param src The source of the edge.
    /// @param dst The destination of the edge.
    void AddEdge(NodeID src, NodeID dst) {
        while (src >= (NodeID)offsets.size()) 
            offsets.push_back(targets.size());
        for (NodeID i = src; i < (NodeID)offsets.size(); ++i) 
            ++offsets[i];
        targets.insert(targets.begin() + offsets[src], dst);
    }

    /// @brief Adds a labelled edge to the graph. It should not be used in
    ///     graph if it does not contain label information.
    /// @param src The source of the edge.
    /// @param dst The destination of the edge.
    /// @param label The label of the edge.
    void AddEdge(NodeID src, NodeID dst, LabelID label) {
        AddEdge(src, dst);
        labels.insert(labels.begin() + offsets[src], label);
    }

    /// @brief Gets the number of edges in the graph.
    /// @return The number of edges in the graph.
    std::size_t GetEdgesSize() const {
        return targets.size();
    }

    /// @brief Gets the number of nodes in the graph.
    /// @return The number of nodes in the graph.
    std::size_t GetNodesSize() const {
        return offsets.size() - 1;
    }

    /// @brief Gets the offsets of the LTS.
    /// @return A const reference to the offsets of the LTS.
    const std::vector<EdgeID>& GetOffsets() const {
        return offsets;
    }

    /// @brief Gets the labels of the edges of the LTS.
    /// @return A const reference to the labels of the LTS.
    const std::vector<LabelID>& GetLabels() const {
        return labels;
    }

    /// @brief Gets the targets of the edges of the LTS.
    /// @return A const reference to the targets of the LTS.
    const std::vector<NodeID>& GetTargets() const {
        return targets;
    }

    /// @brief Checks whether the graph contains labelled edges.
    /// @return Whether the graph is labelled.
    bool IsLabelled() const {
        return labels.size() != 0;
    }

private:
    std::vector<EdgeID> offsets;
    std::vector<LabelID> labels;
    std::vector<NodeID> targets;
};

/// @brief A labelled transition system encoded by storing the sources,
///     targets, and labels of the edges in three separate vectors.
/// @tparam EdgeID The type of the indices of the nodes.
/// @tparam LabelID The type of the indices of the labels.
template<typename NodeID = std::uint32_t,
         typename LabelID = std::uint32_t>
class LTS {
public:
    /// @brief Constructs an empty labelled transition system.
    /// @param numberOfNodes The number of nodes of the LTS.
    LTS(std::size_t numberOfNodes = 0) : nodesSize(numberOfNodes) { }

    /// @brief Constructs a labelled transition system without labels, i.e. a
    ///     directed graph.
    /// @param numberOfNodes The number of nodes of the graph.
    /// @param _sources For each edge, contains the source node of the edge.
    /// @param _targets For each edge, contains the target node of the edge.
    LTS(std::size_t numberOfNodes, std::vector<NodeID>&& _sources, 
        std::vector<NodeID>&& _targets) : 
        nodesSize(numberOfNodes), sources(std::move(_sources)), 
        targets(std::move(_targets)) {}

    /// @brief Constructs a labelled transition system.
    /// @param numberOfNodes The number of nodes of the graph.
    /// @param _sources For each edge, contains the source node of the edge.
    /// @param _targets For each edge, contains the target node of the edge.
    /// @param _labels For each edge, contains the label of the edge.
    LTS(std::size_t numberOfNodes, std::vector<NodeID>&& _sources, 
        std::vector<NodeID>&& _targets, std::vector<LabelID>&& _labels) : 
        nodesSize(numberOfNodes), sources(std::move(_sources)), 
        targets(std::move(_targets)), labels(std::move(_labels)) {}
                                         
    /// @brief Adds an edge to the graph without adding information on the
    ///     labels of the graph. If the LTS already contains labels for the
    ///     transition, you should not use this method, but its overload
    ///     with an additional label parameter.
    /// @param src The source of the edge.
    /// @param dst The destination of the edge.
    void AddEdge(std::size_t src, std::size_t dst) {
        if (src >= nodesSize) nodesSize = src + 1;
        if (dst >= nodesSize) nodesSize = dst + 1;
        sources.push_back(src);
        targets.push_back(dst);
    }

    /// @brief Adds a labelled edge to the graph. It should not be used in
    ///     graph if it does not contain label information.
    /// @param src The source of the edge.
    /// @param dst The destination of the edge.
    /// @param label The label of the edge.
    void AddEdge(NodeID src, NodeID dst, LabelID label) {
        AddEdge(src, dst);
        labels.push_back(label);
    }

    /// @brief Gets the sources of the edges of of the LTS.
    /// @return A const reference to the sources of the LTS.
    const std::vector<NodeID>& GetSources() const {
        return sources;
    }

    /// @brief Gets the targets of the edges of of the LTS.
    /// @return A const reference to the targets of the LTS.
    const std::vector<NodeID>& GetTargets() const {
        return targets;
    }

    /// @brief Gets the labels of the edges of the LTS.
    /// @return A const reference to the labels of the LTS.
    const std::vector<LabelID>& GetLabels() const {
        return labels;
    }

        /// @brief Gets the number of edges in the graph.
    /// @return The number of edges in the graph.
    std::size_t GetEdgesSize() const {
        return targets.size();
    }

    /// @brief Gets the number of nodes in the graph.
    /// @return The number of nodes in the graph.
    std::size_t GetNodesSize() const {
        return nodesSize;
    }

    /// @brief Returns the number of labels in the LTS.
    /// @return The number of labels used for the edges.
    std::size_t GetLabelsCount() const {
        if (!IsLabelled()) return 0;
        return *std::max_element(labels.begin(), labels.end()) + 1;
    }

    /// @brief Checks whether the graph contains labelled edges.
    /// @return Whether the graph is labelled.
    bool IsLabelled() const {
        return labels.size() != 0;
    }

    /// @brief Generates an unlabelled complete tree of a given height.
    /// @param branching How many children each internal node has.
    /// @param height The height of the tree.
    /// @return A complete unlabelled directed tree with root at index 0.
    static LTS<NodeID, LabelID> CompleteTree(std::size_t branching, 
                                             std::size_t height) {
        std::size_t nodeCount = (std::pow(branching, height) - 1) / (branching - 1); 
        std::vector<NodeID> src(nodeCount - 1);
        std::vector<NodeID> dst(nodeCount - 1);
        for (std::size_t i = 0; i < src.size(); ++i) {
            src[i] = i / branching;
            dst[i] = i + 1;
        }
        return LTS<NodeID, LabelID>(nodeCount,
                                    std::move(src), std::move(dst));
    }

    /// @brief Generates an unlabelled loop of a given length.
    /// @param length The length of the loop.
    /// @return A loop of a given size.
    static LTS<NodeID, LabelID> Loop(std::size_t length) {
        std::vector<NodeID> src(length);
        std::vector<NodeID> dst(length);
        for (std::size_t i = 0; i < src.size(); ++i) {
            src[i] = i;
            dst[i] = i == src.size() - 1 ? 0 : i + 1;
        }
        return LTS<NodeID, LabelID>(length, std::move(src), std::move(dst));
    }

    /// @brief Generates an unlabelled loop of a given length.
    /// @param length The length of the list.
    /// @return A list of a given size.
    static LTS<NodeID, LabelID> List(std::size_t length) {
        std::vector<NodeID> src(length - 1);
        std::vector<NodeID> dst(length - 1);
        for (std::size_t i = 0; i < src.size(); ++i) {
            src[i] = i;
            dst[i] = i + 1;
        }
        return LTS<NodeID, LabelID>(length, std::move(src), std::move(dst));
    }

private:
    std::size_t nodesSize;
    std::vector<NodeID> sources;
    std::vector<NodeID> targets;
    std::vector<LabelID> labels;
};

/// @brief Returns a vector representing the partition induced by the top of
///     the lattice of equivalence relations on a finite set, i.e., the 
///     partition containing a single block.
/// @tparam NodeID The type of the nodes in the set (an integer big enough to
///     represent the elements of the set).
/// @param n The number of elements of the set.
/// @return A vector indicating for each element its partition (every element)
///     lies in a different class. Practically, it returns a vector of size n
///     whose elements are all 0.
template<typename NodeID = std::uint32_t>
inline std::vector<NodeID> MaximumEquivalence(std::size_t n) {
    std::vector<NodeID> ret;
    for (std::size_t i = 0; i < n; ++i) {
        ret.push_back(0);
    }
    return ret;
}

namespace internal {

/// @brief Re-adapts data in the format used by the CUDA bisimulation algorithm
///     so as to make it easy to compute pre-images for the Paige-Tarjan CPU
///     version.
/// @tparam NodeID The integer type representing nodes. 
template<typename NodeID = std::uint32_t>
class _pt_edges {
public:
    /// @brief Readapts a set of edges of expressed as two separated vectors
    ///     indicating the sources and targets of each edge.
    /// @param sources The source node of each edge.
    /// @param targets The target node of each edge.
    /// @param nodeCount The number of nodes in the graph.
    _pt_edges(const std::vector<NodeID>& sources, 
              const std::vector<NodeID>& targets,
              std::size_t nodeCount) : sortedSources(sources.size()),
                                       starts(nodeCount + 1)
    {
        // Start by sorting the edges by the target nodes.
        std::vector<NodeID> indices(sources.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                [&targets](NodeID a, NodeID b) {
                    return targets[a] < targets[b];
                }
        );
        for (std::size_t i = 0; i < sources.size(); ++i) 
            sortedSources[i] = sources[indices[i]];
        // For each node, find the index where the edges ending in that node
        // start and put them in the vector `starts`.
        NodeID currentIndex = 0;
        auto sortedTargets = indices | std::views::transform(
                                   [&targets](NodeID i) { return targets[i];});
        starts[0] = 0;
        for (NodeID node = 0; node < nodeCount; ++node) {
            auto last = std::upper_bound(sortedTargets.begin() + currentIndex,
                                         sortedTargets.end(), node);
            starts[node + 1] = last - (sortedTargets.begin() + currentIndex);
        }
    }

    /// @brief Returns the preimage of a node.
    /// @param node The node the preimage of which is fetched.
    /// @return A span over where the preimage is stored. Its life depends on
    ///     this object.
    std::span<const NodeID> preimage(NodeID node) const {
        return std::span(sortedSources.begin() + starts[node],
                         sortedSources.begin() + starts[node + 1]);
    }

private:
    std::vector<NodeID> sortedSources;
    std::vector<NodeID> starts;
};

/// @brief Holds data about a partition to be used in the Paige-Tarjan
///     algorithm.
/// @tparam NodeID The integer type representing nodes. 
template<typename NodeID = std::uint32_t>
class _pt_partition {
public:
    /// @brief Restructures a blocks vector indicating a block ID for each node
    ///     to efficiently perform the operations required by Paige-Tarjan.
    /// @param v The blocks vector: a block index for each node.
    /// @param nblocks The number of blocks in the partition.
    _pt_partition(std::vector<NodeID> v, std::size_t nblocks) : sizes(nblocks) { 
        for (std::size_t i = 0; i < nblocks; ++i) {
            blocks.push_back(Bitset<std::uint64_t>(v.size()));
            sizes[i] = 0;
        }
        for (std::size_t i = 0; i < v.size(); ++i) {
            blocks[v[i]].Add(i);
            sizes[v[i]]++;
        }
    }

    /// @brief Splits each block of the partition using a splitter set.
    /// @param set The splitter.
    /// @return A vector containing for each new block the ID of the block it
    ///     originated from in a split.
    std::vector<NodeID> split(const Bitset<std::uint64_t>& set) {
        std::vector<NodeID> newBlocksOrigin;
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            if (!preprocessedBlock(i) && !blocks[i].Disjoint(set) && 
                !blocks[i].SubsetOf(set)) 
            {
                auto prevSize = sizes[i];
                blocks.push_back(blocks[i].Split(set));
                sizes[i] = blocks[i].Size();
                sizes.push_back(prevSize - sizes[i]);
                newBlocksOrigin.push_back(i);
            }
        }
        return newBlocksOrigin;
    }

    /// @brief Splits each block of the partition using the preimage of a set.
    /// @param set The set the preimage of which is computed.
    /// @param edges The edges of the graph.
    /// @return A vector containing for each new block the ID of the block it
    ///     originated from in a split.
    std::vector<NodeID> split(const Bitset<std::uint64_t>& set, 
                              const _pt_edges<NodeID>& edges) {
        Bitset<std::uint64_t> preimage(set.MaxSize());
        for (auto x : set) {
            for (auto n : edges.preimage(x)) {
                preimage.Add(n);
            }
        }
        return split(preimage);
    }

    /// @brief Preprocesses the partition by splitting each block by the
    ///     the preimage of the set of nodes. Keeps track of the blocks 
    ///     generated during preprocessing: as they cannot generate any further
    ///     splits, they will be ignored by the next steps of the algorithms.
    /// @param edges The edges of the graph.
    /// @param nodeCount The number of nodes in the graph.
    /// @note This implementation of the preprocessing is pretty badly
    ///     performing in some edge cases (e.g. graphs which are lists), as
    ///     it is really fast with graphs with a lot of loops, but keeps a
    ///     :math:`\\Theta(n^2)` worst case.
    void preprocess(const _pt_edges<NodeID>& edges, std::size_t nodeCount) {
        preprocessIndexStart = blocks.size();
        Bitset<std::uint64_t> preimage(nodeCount);
        for (std::size_t x = 0; x < nodeCount; ++x) {
            for (auto n : edges.preimage(x)) {
                preimage.Add(n);
            }
        }
        bool continuePreprocessing = split(preimage).size() > 0;
        if (!continuePreprocessing) return;
        preprocessIndexEnd = blocks.size();
        std::size_t lastPreprocessed = preprocessIndexStart;
        Bitset<std::uint64_t> excludedVertices(nodeCount);
        while (continuePreprocessing) {
            for(; lastPreprocessed < preprocessIndexEnd; ++lastPreprocessed)
                blocks[lastPreprocessed].Union(excludedVertices);
            preimage.Clear();
            for (std::size_t x = 0; x < nodeCount; ++x) {
                if (!excludedVertices.Contains(x)) {
                    for (auto n : edges.preimage(x)) {
                        preimage.Add(n);
                    }
                }
            } 
            continuePreprocessing = split(preimage).size() > 0;
            preprocessIndexEnd = blocks.size();
        }
    }

    /// @brief Checks whether a block was added via preprocessing, and hence
    ///     whether it should be ignored by the Paige-Tarjan algorithm.
    /// @param i The index of the block.
    /// @return Whether the block was added during preprocessing.
    bool preprocessedBlock(std::size_t i) {
        return i >= preprocessIndexStart && i < preprocessIndexEnd;
    }

    /// @brief Converts the partition into a vector listing the index of the
    ///     block to which each node belongs.
    /// @param v The vector containing the blocks information.
    void toVector(std::vector<NodeID>& v) {
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            for (auto x : blocks[i]) {
                v[x] = i;
            }
        }
    }

    /// @brief Returns the size of some block.
    /// @param index The index of the block.
    /// @return The size of the block at the specified index.
    std::size_t sizeAt(std::size_t index) const {
        return sizes[index];
    }

    /// @brief Returns the number of blocks in the partition.
    /// @return The number of blocks in the partition.
    std::size_t size() const {
        return blocks.size();
    }

    Bitset<std::uint64_t>& operator[](std::size_t i) { return blocks[i]; }
    auto begin() { return blocks.begin(); }
    auto end() { return blocks.end(); }
    auto cbegin() { return blocks.cbegin(); }
    auto cend() { return blocks.cend(); }

private:
    std::vector<Bitset<std::uint64_t>> blocks;
    std::vector<NodeID> sizes;
    std::size_t preprocessIndexStart = 0;
    std::size_t preprocessIndexEnd = 0;
};

/// @brief Represents a utility partition used to compute the coarsest stable
///     refinement. It is called X in:
///     R. Paige and R.E. Tarjan. Three partition refinement algorithms. 
///     SIAM J. Comput., 16(6):973-989, 198.
/// @tparam NodeID The integer type representing nodes. 
template<typename NodeID = std::uint32_t>
class _pt_x_partition {
public:
    /// @brief Initializes a support partition with a single block (a compound
    ///     block containing all the blocks in a )
    /// @param _Q 
    _pt_x_partition(_pt_partition<NodeID>& _Q) : Q(_Q), qBlocksMap(_Q.size()) {
        blocks.push_back(std::list<NodeID>());
        for (std::size_t i = 0; i < _Q.size(); ++i) {
            if (!_Q.preprocessedBlock(i))
                blocks[0].push_front(i);
            qBlocksMap[i] = 0;
        }
        if (blocks.size() > 0 && blocks[0].size() > 1)
            compoundBlocks.push_front(0);
    }

    /// @brief Executes the Paige-Tarjan algorithm on this partition X, and its
    ///     underlying partition Q.
    /// @param edges The edges of the graph.
    void paige_tarjan(const _pt_edges<NodeID>& edges) {
        while (!compoundBlocks.empty()) 
            iteration(edges);
    }

private:
    /// @brief The partition Q as mentioned in the original paper by Paige and
    ///     Tarjan. Q is a refinement of X.
    _pt_partition<NodeID>& Q;

    /// @brief Maps blocks of Q to the block of X that contains them.
    std::vector<NodeID> qBlocksMap;

    /// @brief A list of the compound blocks of X (blocks which are the union
    ///     of more than one block of Q).
    std::list<NodeID> compoundBlocks;

    /// @brief Vector of the blocks of X. Each block of X is represented as the
    ///     list of blocks of Q it is union of.
    std::vector<std::list<NodeID>> blocks;

    /// @brief Firstly, pick a compound block S. Then, select a block B in its
    ///     refinement in Q which has less than half of the elements of S.
    ///     Before returning, it removes B from the compound block S, and
    ///     possibly demotes the compound block to simple (singleton) block.
    /// @return The block B.
    NodeID selectBlock() {
        auto& S = blocks[compoundBlocks.front()];
        auto it = S.begin();
        auto S0 = *it;
        std::advance(it, 1);
        auto S1 = *it;
        auto B = Q.sizeAt(S0) < Q.sizeAt(S1) ? S0 : S1;
        S.erase(Q.sizeAt(S0) < Q.sizeAt(S1) ? S.begin() : it);
        if (S.size() < 2) compoundBlocks.pop_front();
        return B;
    }

    void iteration(const _pt_edges<NodeID>& edges) {
        // Select a splitting block in Q and perform the split.
        auto B = selectBlock();
        auto newBlocksStart = Q.size();
        auto newBlocksOrigin = Q.split(Q[B], edges);
        // After performing the splits, track down to which blocks of X the
        // blocks of Q belong.
        for (std::size_t i = 0; i < Q.size() - newBlocksStart; ++i) {
            auto newXBlock = qBlocksMap[newBlocksOrigin[i]];
            qBlocksMap.push_back(newXBlock);
            blocks[newXBlock].push_front(newBlocksStart + i);
            // If a split made a block in X compound, track it down.
            if (blocks[newXBlock].size() == 2) {
                compoundBlocks.push_front(newXBlock);
            }
        }
    }
};

}

/// @brief Computes the coarsest bisimulation refining an equivalence
///     relation for a given labelled transition system. Uses CPU resources.
///     Implemented by using a naive implementation of the Paige-Tarjan
///     algorithm. This algortihm is described in:
///     R. Paige and R.E. Tarjan. Three partition refinement algorithms. 
///     SIAM J. Comput., 16(6):973-989, 198.
/// @tparam EdgeID The type of the indices of the nodes.
/// @tparam LabelID The type of the indices of the labels.
/// @param lts A labelled transition system.
/// @param blocks Input-output parameter. As an input, it is the starting
///     partition which is refined to find the coarsest bisimulation refining
///     it. As an output parameter, it contains the output bisimulation.
///     For each node, this vector contains a representative of its class.
/// @note In future versions of this library, this procedure should be
///     optimized.
template<typename NodeID = std::uint32_t,
         typename LabelID = std::uint32_t>
inline void ComputeBisimulationCPU(const LTS<NodeID, LabelID>& lts, 
                                   std::vector<NodeID>& blocks) {
    mlconcepts::internal::_pt_edges edges(lts.GetSources(), lts.GetTargets(),
                                          lts.GetNodesSize());
    auto nblocks = *std::max_element(blocks.begin(), blocks.end()) + 1;
    mlconcepts::internal::_pt_partition<NodeID> Q(blocks, nblocks);
    Q.preprocess(edges, lts.GetNodesSize());
    mlconcepts::internal::_pt_x_partition<NodeID> X(Q);
    X.paige_tarjan(edges);
    Q.toVector(blocks);
}

/// @brief Computes the coarsest bisimulation  for a given labelled transition
///     system. Uses CPU resources.
///     Implemented by using a naive implementation of the Paige-Tarjan
///     algorithm. This algortihm is described in:
///     R. Paige and R.E. Tarjan. Three partition refinement algorithms. 
///     SIAM J. Comput., 16(6):973-989, 198.
/// @tparam EdgeID The type of the indices of the nodes.
/// @tparam LabelID The type of the indices of the labels.
/// @param lts A labelled transition system.
/// @return The coarsest bisimulation.
///     For each node, this vector contains a representative of its class.
/// @note In future versions of this library, this procedure should be
///     optimized.
template<typename NodeID = std::uint32_t,
         typename LabelID = std::uint32_t>
inline std::vector<NodeID> 
ComputeBisimulationCPU(const LTS<NodeID, LabelID>& lts) {
    std::vector<NodeID> blocks(MaximumEquivalence(lts.GetNodesSize()));
    ComputeBisimulationCPU(lts, blocks);
    return blocks;
}

#ifdef CUDA_ENABLED
/// @brief Computes the coarsest bisimulation included in an equivalence
///     relation for a given labelled transition system. Uses Cuda.
/// @tparam EdgeID The type of the indices of the nodes.
/// @tparam LabelID The type of the indices of the labels.
/// @param lts A labelled transition system.
/// @param blocks Input-output parameter. As an input, it is the starting
///     partition which is refined to find the coarsest bisimulation refining
///     it. As an output parameter, it contains the output bisimulation.
///     For each node, this vector contains a representative of its class.
template<typename NodeID = std::uint32_t,
         typename LabelID = std::uint32_t>
void ComputeBisimulationCuda(const LTS<NodeID, LabelID>& lts,
                             std::vector<NodeID>& blocks);

/// @brief Computes the coarsest bisimulation for a given labelled transition
///     system. Uses Cuda.
/// @tparam EdgeID The type of the indices of the nodes.
/// @tparam LabelID The type of the indices of the labels.
/// @param lts A labelled transition system.
/// @return The coarsest bisimulation.
///     For each node, this vector contains a representative of its class.
template<typename NodeID = std::uint32_t,
         typename LabelID = std::uint32_t>
inline std::vector<NodeID> 
ComputeBisimulationCuda(const LTS<NodeID, LabelID>& lts) {
    std::vector<NodeID> blocks(MaximumEquivalence(lts.GetNodesSize()));
    ComputeBisimulationCuda(lts, blocks);
    return blocks;
}

/// @brief Computes the coarsest bisimulation included in an equivalence
///     relation for a given labelled transition system. Uses Cuda.
/// @tparam EdgeID The type of the indices of the nodes.
/// @tparam LabelID The type of the indices of the labels.
/// @param lts A labelled transition system.
/// @param blocks Input-output parameter. As an input, it is the starting
///     partition which is refined to find the coarsest bisimulation refining
///     it. As an output parameter, it contains the output bisimulation.
///     For each node, this vector contains a representative of its class.
template<typename NodeID = std::uint32_t,
         typename LabelID = std::uint32_t>
inline void ComputeBisimulation(const LTS<NodeID, LabelID>& lts,
                                std::vector<NodeID>& blocks) {
    ComputeBisimulationCuda(lts, blocks);
}

/// @brief Computes the coarsest bisimulation for a given labelled transition
///     system. Uses Cuda.
/// @tparam EdgeID The type of the indices of the nodes.
/// @tparam LabelID The type of the indices of the labels.
/// @param lts A labelled transition system.
/// @return The coarsest bisimulation.
///     For each node, this vector contains a representative of its class.
template<typename NodeID = std::uint32_t,
         typename LabelID = std::uint32_t>
inline std::vector<NodeID> 
ComputeBisimulation(const LTS<NodeID, LabelID>& lts) {
    return ComputeBisimulationCuda(lts);
}
#else 
/// @brief Computes the coarsest bisimulation included in an equivalence
///     relation for a given labelled transition system. Uses CPU resources.
/// @tparam EdgeID The type of the indices of the nodes.
/// @tparam LabelID The type of the indices of the labels.
/// @param lts A labelled transition system.
/// @param blocks Input-output parameter. As an input, it is the starting
///     partition which is refined to find the coarsest bisimulation refining
///     it. As an output parameter, it contains the output bisimulation.
///     For each node, this vector contains a representative of its class.
template<typename NodeID = std::uint32_t,
         typename LabelID = std::uint32_t>
inline void ComputeBisimulation(const LTS<NodeID, LabelID>& lts,
                                std::vector<NodeID>& blocks) {
    ComputeBisimulationCPU(lts, blocks);
}

/// @brief Computes the coarsest bisimulation for a given labelled transition
///     system. Uses CPU resources.
/// @tparam EdgeID The type of the indices of the nodes.
/// @tparam LabelID The type of the indices of the labels.
/// @param lts A labelled transition system.
/// @return The coarsest bisimulation.
///     For each node, this vector contains a representative of its class.
template<typename NodeID = std::uint32_t,
         typename LabelID = std::uint32_t>
inline std::vector<NodeID> 
ComputeBisimulation(const LTS<NodeID, LabelID>& lts) {
    return ComputeBisimulationCPU(lts);
}

#endif


}