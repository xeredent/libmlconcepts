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
/// @tparam Conceptifier The conceptifier which generates the formal contexts
///         from a given dataset.
template <class Context,
          class Conceptifier = Conceptifier<
              ConfigurableContextSelector<std::uint32_t>,
              AllUniformQuantizer<double>, 
              AllValuesQuantizer<std::size_t>
          >
> class UnsupervisedODModel : public SupervisedModel {
protected:
    static constexpr const std::uint32_t encodingMagicNumber = 0x756e6f64;
    typedef typename Context::AttributeSet AttSet;
    std::vector<Context> contexts;
    Conceptifier conceptifier;
    std::size_t realFeatureCount; 
    std::size_t categoricalFeatureCount;

public:

    /// @brief Trains the model so as to fit a dataset.
    /// @param X The dataset used for training.
    void Fit(const Dataset& X) override {
        realFeatureCount = X.RealFeatureCount(); 
        categoricalFeatureCount = X.CategoricalFeatureCount();
        conceptifier.Initialize(X, settings, contexts);
    }

    /// @brief Computes prediction scores for a given dataset.
    /// @param X The dataset for which to compute predictions.
    /// @return A vector containing a prediction for each entry in the dataset.
    Eigen::VectorXd Predict(const Dataset& X) override {
        if (realFeatureCount != X.RealFeatureCount() || 
            categoricalFeatureCount != X.CategoricalFeatureCount()) {
            throw std::runtime_error("Prediction data column types do not " 
                                     " match those of the training set");
        }
        Eigen::VectorXd predictions = Eigen::VectorXd::Zero(X.Size());
        for (std::size_t ctxID = 0; ctxID < contexts.size(); ++ctxID) {
            auto intensions = conceptifier.template ProcessData<AttSet>(X,
                                                                        ctxID);
            for (std::size_t obj = 0; obj < X.Size(); ++obj) {
                predictions(obj) += std::exp(
                    -std::pow(
                        contexts[ctxID].ComputeClosureSize(intensions[obj]), 
                        2
                    )
                );
            }
        }
        predictions /= contexts.size();
        return predictions;
    }


    /// @brief Estimates the size of the model.
    /// @return Lower bound of the size of the model in bytes.
    std::size_t EstimateSize() const override {
        std::size_t sz = sizeof(*this) + conceptifier.EstimateSize() -
                         sizeof(Conceptifier);
        for (const auto& c : contexts) sz += c.EstimateSize();
        return sz;
    }

    /// @brief Returns the formal contexts generated during training.
    /// @return The formal contexts generated during training.
    const std::vector<Context>& GetContexts() { return contexts; }

    /// @brief Writes the model state to a stream.
    /// @param stream The stream the model state is written to.
    void Serialize(std::ostream& stream) const {
        io::LittleEndianWrite(stream, encodingMagicNumber);
        io::LittleEndianWrite(stream, (std::uint32_t)realFeatureCount);
        io::LittleEndianWrite(stream, (std::uint32_t)categoricalFeatureCount);
        conceptifier.Serialize(stream);
        for (const auto& ctx : contexts) ctx.Serialize(stream);
    }

    /// @brief Loads the model state from a stream.
    /// @param stream The stream the model state is read from.
    void Deserialize(std::istream& stream) {
        if (io::LittleEndianRead<std::uint32_t>(stream) != encodingMagicNumber) {
            throw std::runtime_error("Parsing error. Invalid format for " 
                                     " unsupervised outlier detection model.");
        }
        realFeatureCount = io::LittleEndianRead<std::uint32_t>(stream);
        categoricalFeatureCount = io::LittleEndianRead<std::uint32_t>(stream);
        conceptifier.Deserialize(stream);
        contexts.clear();
        for (std::size_t i = 0; i < conceptifier.GetFeatureSetsCount(); ++i)
            contexts.push_back(Context(stream));
    }
};

}