#pragma once
#include <vector>
#include <utility>
#include "ContextConcepts.h"
#include "Dataset.h"
#include "StandardConceptifier.h"
#include "Bitset.h"
#include "Model.h"
#include "Settings.h"
#include "ContextSelector.h"
#include "FeatureAssigner.h"
#include "GradientDescent.h"

namespace mlconcepts 
{

/// @brief Implements an unsupervised outlier detection model.
/// @tparam Context The type implementing formal contexts.
/// @tparam Conceptifier The conceptifier which generates the formal contexts from a given dataset.
template <class Context,
          class Conceptifier = Conceptifier<ConfigurableContextSelector<uint32_t>,
                                            AllUniformQuantizer<double>, AllValuesQuantizer<size_t>>
         >
class UnsupervisedODModel : public SupervisedModel {
protected:
    static constexpr const uint32_t encodingMagicNumber = 0x756e6f64;
    typedef typename Context::AttributeSet AttSet;
    std::vector<Context> contexts;
    Conceptifier conceptifier;
    size_t realFeatureCount; 
    size_t categoricalFeatureCount;

public:

    void Fit(const Dataset& X) override {
        realFeatureCount = X.RealFeatureCount(); categoricalFeatureCount = X.CategoricalFeatureCount();
        conceptifier.Initialize(X, settings, contexts);
    }

    Eigen::VectorXd Predict(const Dataset& X) override {
        if (realFeatureCount != X.RealFeatureCount() || categoricalFeatureCount != X.CategoricalFeatureCount())
            throw std::runtime_error("Prediction data column types do not match those of the training set");
        Eigen::VectorXd predictions = Eigen::VectorXd::Zero(X.Size());
        for (size_t ctxID = 0; ctxID < contexts.size(); ++ctxID) {
            auto intensions = conceptifier.template ProcessData<AttSet>(X, ctxID);
            for (size_t obj = 0; obj < X.Size(); ++obj) {
                predictions(obj) += std::exp(-std::pow(contexts[ctxID].ComputeClosureSize(intensions[obj]), 2));
            }
        }
        predictions /= contexts.size();
        return predictions;
    }


    size_t EstimateSize() const override {
        size_t sz = sizeof(*this) + conceptifier.EstimateSize() - sizeof(Conceptifier);
        for (const auto& c : contexts) sz += c.EstimateSize();
        return sz;
    }

    const std::vector<Context>& GetContexts() { return contexts; }

    void Serialize(std::ostream& stream) const {
        io::LittleEndianWrite(stream, encodingMagicNumber);
        io::LittleEndianWrite(stream, (uint32_t)realFeatureCount);
        io::LittleEndianWrite(stream, (uint32_t)categoricalFeatureCount);
        conceptifier.Serialize(stream);
        for (const auto& ctx : contexts) ctx.Serialize(stream);
    }

    void Deserialize(std::istream& stream) {
        if (io::LittleEndianRead<uint32_t>(stream) != encodingMagicNumber)
            throw std::runtime_error("Parsing error. Invalid format for unsupervised outlier detection model.");
        realFeatureCount = io::LittleEndianRead<uint32_t>(stream);
        categoricalFeatureCount = io::LittleEndianRead<uint32_t>(stream);
        conceptifier.Deserialize(stream);
        contexts.clear();
        for (size_t i = 0; i < conceptifier.GetFeatureSetsCount(); ++i)
            contexts.push_back(Context(stream));
    }
};

}