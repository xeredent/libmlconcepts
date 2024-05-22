#pragma once
#include <cstddef>
#include "Dataset.h"
#include "Settings.h"

namespace mlconcepts
{

/// @brief Transforms datasets into formal contexts and tracks information on
///        how the attributes in the context map to the features in the dataset.
/// @tparam CtxSelector The type of the object that decides what contexts to
///                     generate given a dataset.
/// @tparam RealQuantizer The type of the object that performs quantization
///                       on real data.
/// @tparam CategoricalQuantizer The type of the object that performs
///                              quantization on categorical data.
template <class CtxSelector, class RealQuantizer, class CategoricalQuantizer>
class Conceptifier {
public:
    typedef typename CtxSelector::FeatureSet FeatureSet;

private:

    /// @brief An object which provides methods to generate feature sets used
    ///        to create contexts.
    CtxSelector selector;

    /// @brief For each context, its associated feature set is stored here.
    std::vector<FeatureSet> featureSets;

    /// @brief For each context, the number of attributes in the context is
    ///        stored here.
    std::vector<std::size_t> contextAttributeCount;

    /// @brief For each context, a vector containing a pair mapping a feature
    ///        assigner (via the id of the feature) to its offset in the
    ///        context is stored here.
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> contextOffsets;

    /// @brief A quantizer for the real-valued part of the analyzed datasets.
    RealQuantizer realQuantizer;

    /// @brief A quantizer for the categorical data in the analyzed datasets.
    CategoricalQuantizer categoricalQuantizer;


    /// @brief Initializes the quantizers to operate on a given dataset.
    /// @param dataset The dataset the quantizers will work on.
    /// @param settings Settings.
    void InitializeQuantizers(const Dataset& dataset, 
                             const ModelSettings& settings) {
        realQuantizer = RealQuantizer(); 
        realQuantizer.Initialize(dataset, settings);
        categoricalQuantizer = CategoricalQuantizer(); 
        categoricalQuantizer.Initialize(dataset, settings);
    }

    /// @brief Clears the vectors storing data on the offsets of the
    ///        feature assigners for each context.
    void InitializeOffsetData() {
        contextAttributeCount.clear();
        contextOffsets.clear();
    }

    /// @brief Fills a formal context given a set of features and a dataset.
    ///        Assumes that offsets information for the context has already 
    ///        been pushed into contextOffsets.
    /// @tparam Context The type implementing the formal context.
    /// @param set The set of features considered in the context.
    /// @param dataset The dataset used to generate the context.
    /// @param ctx The context which is filled.
    /// @param ctxID The ID of the context.
    template<class Context>
    void FillContext(const FeatureSet& set,
                     const Dataset& dataset,
                     Context& ctx,
                     std::size_t ctxID) {
        // This loops over the pairs (f, offset) for each feature in the
        // context. Hence, by extracting the first projection, the loop
        // iterates over all the features. This little dirty trick saves
        // time with respect to the iterator of FeatureSet,
        // which will likely be Bitset<T>.
        for (const auto& p : contextOffsets[ctxID]) { 
            auto f = p.first; 
            if (f < dataset.RealFeatureCount()) {
                auto& assigner = realQuantizer.GetAssigner(f);
                assigner.SetOffset(p.second);
                dataset.ForEachRealColumn(
                    f, 
                    [&assigner,&ctx](std::size_t obj, 
                                     std::size_t feature, 
                                     double v) {  
                        ctx.SetIncidence(obj, assigner.Assign(v));
                    }
                );
            } else {
                auto& assigner = categoricalQuantizer.GetAssigner(
                    f - dataset.RealFeatureCount()
                );
                assigner.SetOffset(p.second);
                dataset.ForEachCategoricalColumn(f, 
                    [&assigner,&ctx](std::size_t obj,
                                     std::size_t feature,
                                     std::size_t v) { 
                        ctx.SetIncidence(obj, assigner.Assign(v));
                    }
                );
            }
        }
    }

    /// @brief Creates a context from a feature set and adds it in a vector.
    /// @tparam Context The type implementing the formal context.
    /// @param set The set of features considered in the context.
    /// @param dataset The dataset used to generate the context.
    /// @param contexts The vector where the context is pushed.
    template<class Context>
    void AddContext(const FeatureSet& set, 
                    const Dataset& dataset, 
                    std::vector<Context>& contexts) {
        // The offsets for the context are computed
        std::size_t offset = 0;
        auto offsets = std::vector<std::pair<std::size_t, std::size_t>>();
        for (auto f : set) {
            offsets.push_back(std::make_pair(f, offset));
            offset += f < dataset.RealFeatureCount() ? 
                      realQuantizer.GetAssigner(f).BinsCount() : 
                      categoricalQuantizer.GetAssigner(
                          f - dataset.RealFeatureCount()
                      ).BinsCount();
        }
        contextOffsets.push_back(std::move(offsets));
        contextAttributeCount.push_back(offset);
        // And then the context is created, filled, and pushed into the vector
        // of contexts
        Context ctx(dataset.Size(), offset);
        FillContext(set, dataset, ctx, contexts.size());
        contexts.push_back(std::move(ctx));
    }

    public:

    /// @brief Initializes a vector of formal contexts with the data contained
    ///        in a dataset, by generating the contexts suggested by the
    ///        context selector.
    /// @tparam Context The type implementing the formal context.
    /// @param dataset The dataset.
    /// @param settings The feature assigners may pull some settings data.
    /// @param contexts The vector of contexts which is filled.
    template<class Context>
    void Initialize(const Dataset& dataset, 
                    const ModelSettings& settings,
                    std::vector<Context>& contexts) {
        InitializeQuantizers(dataset, settings);
        InitializeOffsetData();
        selector.SetSettings(settings);
        featureSets = std::move(
            selector.GenerateStartingContexts(dataset.FeatureCount())
        );
        for (const auto& set : featureSets) {
            AddContext(set, dataset, contexts);
        }
    }

    bool UpdateModel(std::size_t newContextStartingID) {
        //TODO IMPLEMENT
        return false;
    }


    /// @brief Processes a set of elements for prediction. Extracts attribute
    ///        sets (sets of intensions) for every element in the dataset,
    ///        according to one of the contexts learned via Initialize and
    ///        UpdateModel.
    /// @tparam AttrSet The type of the attribute set.
    /// @param dataset The dataset.
    /// @param ctxID The index of the context.
    /// @return The vector of attributes of all the objects in dataset in the
    ///         context of index ctxID.
    template <class AttrSet>
    std::vector<AttrSet> ProcessData(const Dataset& dataset, 
                                     std::size_t ctxID) {
        // Initializes a vector of intensions, one for each element in the
        // dataset. This consumes a bit of memory, but is in general much 
        // faster than looping over the objects and computing for each of
        // them their outlier degree.
        std::vector<AttrSet> incidences;
        for (std::size_t i = 0; i < dataset.Size(); ++i) 
            incidences.push_back(AttrSet(contextAttributeCount[ctxID]));
        // Sets the proper offsets in the quantizers for the given contextID
        for (const auto& p : contextOffsets[ctxID]) {
            if (p.first < dataset.RealFeatureCount()) 
                realQuantizer.GetAssigner(p.first).SetOffset(p.second);
            else {
                categoricalQuantizer.GetAssigner(
                    p.first - dataset.RealFeatureCount()
                ).SetOffset(p.second);
            }
        }
        // The loops over the features in the context (the first projection of
        // contextOffsets[ctxID])
        for (const auto& p : contextOffsets[ctxID]) {
            auto f = p.first;
            if (f < dataset.RealFeatureCount()) {
                const auto& assigner = realQuantizer.GetAssigner(f);
                dataset.ForEachRealColumn(
                    f, 
                    [&assigner,&incidences](std::size_t obj, 
                                            std::size_t feature, 
                                            double v) { 
                        incidences[obj].Add(assigner.Assign(v));
                    }
                );
            } else {
                const auto& assigner = categoricalQuantizer.GetAssigner(
                    f - dataset.RealFeatureCount()
                );
                dataset.ForEachCategoricalColumn(
                    f, 
                    [&assigner,&incidences](std::size_t obj,
                                            std::size_t feature,
                                            std::size_t v) { 
                        incidences[obj].Add(assigner.Assign(v));
                    }
                );
            }
        }
        return incidences;
    }

    /// @brief Estimates the size in bytes occupied by this object.
    /// @return The estimated size in bytes.
    std::size_t EstimateSize() const {
        std::size_t sz = sizeof(this);
        for (const auto& b : featureSets) sz += b.EstimateSize();
        sz += sizeof(FeatureSet) * 
              (featureSets.capacity() - featureSets.size());
        sz += contextAttributeCount.capacity() * sizeof(std::size_t);
        for (const auto& v : contextOffsets) 
            sz += sizeof(v) + v.capacity() * 
                  sizeof(std::pair<std::size_t, std::size_t>);
        return sz;
    }

    /// @brief Retrieves the number of feature sets stored in the conceptifier.
    /// @return The number of feature sets used to generate formal contexts.
    std::size_t GetFeatureSetsCount() const {
        return featureSets.size();
    }

    /// @brief Returns a copy of the feature sets vector.
    /// @return A copy of the feature sets vector.
    std::vector<FeatureSet> GetFeatureSets() const {
        return featureSets;
    }

protected:
    /// @brief Generates context offset data. To be used after both the feature
    ///        sets and the quantizers are initialized (for instance, in
    ///        deserialization).
    void GenerateContextOffsetsFromQuantizers() {
        contextAttributeCount.clear();
        contextOffsets.clear(); 
        std::size_t realFeatures = realQuantizer.GetAssignersCount();
        for (std::size_t fsID = 0; fsID < featureSets.size(); ++fsID) {
            std::size_t offset = 0;
            contextOffsets.push_back(
                std::vector<std::pair<std::size_t,std::size_t>>()
            );
            for (auto f : featureSets[fsID]) {
                contextOffsets[fsID].push_back(
                    std::make_pair(f, offset)
                );
                offset += f < realFeatures ? 
                          realQuantizer.GetAssigner(f).BinsCount() :
                          (realFeatures + 
                           categoricalQuantizer.GetAssigner(
                               f - realFeatures
                           ).BinsCount());
            }
            contextAttributeCount.push_back(offset);
        }
    }

    static constexpr const std::uint32_t encodingMagicNumber = 0x73746463;

public:
    /// @brief Encodes the conceptifier into a stream as binary data.
    /// @param stream The stream where to write the conceptifier.
    void Serialize(std::ostream& stream) const {
        io::LittleEndianWrite(stream, encodingMagicNumber);
        realQuantizer.Serialize(stream);
        categoricalQuantizer.Serialize(stream);
        io::LittleEndianWrite(stream, (std::uint32_t)featureSets.size());
        for (const auto& x : featureSets) x.Serialize(stream);
    }

    /// @brief Decodes the conceptifier from a binary stream.
    /// @param stream The stream where to read the conceptifier from.
    void Deserialize(std::istream& stream) {
        if (io::LittleEndianRead<std::uint32_t>(stream) != encodingMagicNumber) {
            throw std::runtime_error(
                "Parsing error. Invalid format: wrong "
                " magic number for standard conceptifier."
            );
        }
        realQuantizer.Deserialize(stream);
        categoricalQuantizer.Deserialize(stream);
        std::uint32_t count = io::LittleEndianRead<std::uint32_t>(stream);
        std::size_t featureCount = realQuantizer.GetAssignersCount() + 
                                   categoricalQuantizer.GetAssignersCount();
        featureSets.clear();
        for (std::size_t i = 0; i < count; ++i) {
            FeatureSet newSet(featureCount);
            newSet.Deserialize(stream);
            featureSets.push_back(std::move(newSet));
        }
        GenerateContextOffsetsFromQuantizers();
    }
    
};

}