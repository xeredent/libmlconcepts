#pragma once
#include <vector>
#include "ContextConcepts.h"
#include "Dataset.h"
#include "StandardConceptifier.h"
#include "Bitset.h"
#include "Model.h"
#include "Settings.h"
#include "ContextSelector.h"
#include "FeatureAssigner.h"

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
};

template <class Context,
          class Conceptifier = Conceptifier<SimpleDoubletonContextSelector<uint32_t>,
                                            AllUniformQuantizer<double>, AllValuesQuantizer<size_t>>
         >
class SupervisedODModel : public SupervisedModel {
protected:
    std::vector<Context> contexts;
    Context::ObjectSet inliers;
    Conceptifier conceptifier;
    size_t realFeatureCount; 
    size_t categoricalFeatureCount;

public:
    void Fit(const Dataset& Xy) override {
        realFeatureCount = X.RealFeatureCount(); categoricalFeatureCount = X.CategoricalFeatureCount();
        conceptifier.Initialize(Xy, settings, contexts);
        Xy.ForEachLabel([this](size_t obj, size_t value) { if (value == 0) inliers.Add(obj); });
    }

    Eigen::VectorXd Predict(const Dataset& X) override {
        if (realFeatureCount != X.RealFeatureCount() || categoricalFeatureCount != X.CategoricalFeatureCount())
            throw std::runtime_error("Prediction data column types do not match those of the training set");
        Eigen::VectorXd predictions = Eigen::VectorXd::Zero(X.Size());
        for (size_t ctxID = 0; ctxID < contexts.size(); ++ctxID) {
            auto intensions = conceptifier.template ProcessData<AttSet>(X, ctxID);
            for (size_t obj = 0; obj < X.Size(); ++obj) {
                predictions(obj) += std::exp(-std::pow(contexts[ctxID].ComputeFilteredClosureSize(intensions[obj], inliers), 2));
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
};

}