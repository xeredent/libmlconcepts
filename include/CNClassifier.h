#pragma once
#include "ContextConcepts.h"
#include "Dataset.h"
#include "StandardConceptifier.h"
#include "Bitset.h"
#include "Model.h"
#include "Settings.h"
#include "PartialContext.h"
#include "ContextSelector.h"
#include "FeatureAssigner.h"
#include "GradientDescent.h"
#include "BasicMath.h"

namespace mlconcepts {

template <class ClassLabel = std::uint16_t,
          class Context = PartialContext<std::uint64_t>,
          class Conceptifier = Conceptifier<SimpleDoubletonContextSelector<std::uint32_t>,
                                            AllUniformQuantizer<double>, AllValuesQuantizer<std::size_t>>
         >
class CNClassifier : public ClassificationModel {
    static constexpr const std::uint32_t encodingMagicNumber = 0x73636e63;
    typedef typename Context::AttributeSet AttSet;
    std::vector<Context> contexts;
    std::vector<ClassLabel> trainSetLabels;
    std::size_t classLabelsCount;
    Conceptifier conceptifier;
    std::size_t realFeatureCount; 
    std::size_t categoricalFeatureCount;
    GradientDescent<Eigen::MatrixXd> gradientDescent;
    
    /// @brief A matrix (nclasses * ncontexts) \times nobjects, which contains for each object
    /// and each context a vector containing for each class the percentage of the elements
    /// in the closure of the object (according to the context) which lies in the class.
    /// that belong to that class.
    Eigen::MatrixXd classScores;

    /// @brief Initializes the labels vector based on a training set.
    /// @param Xy The dataset.
    void InitializeLabels(const Dataset& Xy) {
        classLabelsCount = Xy.CountLabels();
        assert(classLabelsCount < std::numeric_limits<ClassLabel>::max()); //Consider using bigger class labels
        trainSetLabels.resize(Xy.Size()); trainSetLabels.shrink_to_fit();
        Xy.ForEachLabel([this](std::size_t obj, std::size_t value) { trainSetLabels[obj] = value; });
    }

    /// @brief Computes class scores for a dataset (usually a set you want to compute predictions for).
    /// @param Xy The dataset used to compute the class scores.
    Eigen::MatrixXd ComputeClassScores(const Dataset& Xy) {
        Eigen::MatrixXd scores(classLabelsCount * contexts.size(), Xy.Size()); 
        for (std::size_t obj = 0; obj < Xy.Size(); ++obj) {
            auto rowID = 0;
            for (std::size_t ctxID = 0; ctxID < contexts.size(); ++ctxID) {
                auto closure = contexts[ctxID].ComputeClosure(obj);
                for (std::size_t cl = 0; cl < classLabelsCount; ++cl) scores(rowID + cl, obj) = 0;
                for (auto o : closure) {
                    ++scores(rowID + trainSetLabels[o], obj);
                }
                const auto& slice = scores(Eigen::seq(rowID, rowID + classLabelsCount - 1), obj);
                auto maxValue = *std::max_element(slice.begin(), slice.end());
                scores(Eigen::seq(rowID, rowID + classLabelsCount - 1), obj) /= maxValue;
                rowID += classLabelsCount;
            }
        }
        return scores;
    }

    /// @brief Computes class scores for every context for which they have not been computed yet.
    /// It is supposed to be used between training rounds.
    /// @param Xy The dataset used to compute the class scores
    /// @param firstMissingContextID The ID of the first missing context from the class scores matrix
    void ComputeMissingClassScores(const Dataset& Xy, std::size_t firstMissingContextID = 0) {
        classScores.conservativeResize(classLabelsCount * contexts.size(), Xy.Size()); 
        for (std::size_t obj = 0; obj < Xy.Size(); ++obj) {
            auto rowID = firstMissingContextID * classLabelsCount;
            for (std::size_t ctxID = firstMissingContextID; ctxID < contexts.size(); ++ctxID) {
                auto closure = contexts[ctxID].ComputeClosure(obj);
                for (std::size_t cl = 0; cl < classLabelsCount; ++cl) classScores(rowID + cl, obj) = 0;
                for (auto o : closure) {
                    ++classScores(rowID + trainSetLabels[o], obj);
                }
                const auto& slice = classScores(Eigen::seq(rowID, rowID + classLabelsCount - 1), obj);
                auto maxValue = *std::max_element(slice.begin(), slice.end());
                classScores(Eigen::seq(rowID, rowID + classLabelsCount - 1), obj) /= maxValue;
                rowID += classLabelsCount;
            }
        }
    }

    /// @brief Initializes a gradient descent object based on the input parameters.
    void InitializeGradientDescent(double learningRate, double momentum, double stopThreshold) {
        gradientDescent = GradientDescent<Eigen::MatrixXd>(
            classLabelsCount, classLabelsCount * contexts.size(),
            [&classScores = std::as_const(classScores), &labels = std::as_const(trainSetLabels)]
            (const Eigen::MatrixXd& W) -> double { //loss function 
                return math::SoftmaxCrossEntropy(W * classScores, labels);
            },
            [&classScores = std::as_const(classScores), &labels = std::as_const(trainSetLabels)]
            (const Eigen::MatrixXd& W) -> Eigen::MatrixXd { //gradient
                auto gradient = math::SoftmaxColumnwise(W * classScores);
                gradient(labels, Eigen::all).array() -= 1;
                return (gradient * classScores.transpose()) / classScores.cols();
            },
            learningRate, momentum, stopThreshold);
    }

    /// @brief Initializes a gradient descent object based on the current settings and the given dataset.
    void InitializeGradientDescent(const Dataset& Xy) {
        InitializeGradientDescent(settings.Get("LearningRate", 0.01), 
                                  settings.Get("Momentum", 0.01), 
                                  settings.Get("StopThreshold", 0.01));
    }

public:
    
    void Fit(const Dataset& Xy) override {
        realFeatureCount = Xy.RealFeatureCount(); categoricalFeatureCount = Xy.CategoricalFeatureCount();
        conceptifier.Initialize(Xy, settings, contexts);
        InitializeLabels(Xy);
        ComputeMissingClassScores(Xy);
        InitializeGradientDescent(Xy);
        Train(settings.Get("TrainEpochs", (std::size_t)1000));
    }

    void Train(std::size_t epochsCount, std::ostream& outputStream = std::cout) {
        gradientDescent.Train(epochsCount, settings.Get("ShowTraining", true), 
                              settings.Get("ShowTrainingDelay", 100.0), outputStream);
    }

    Eigen::VectorXi Predict(const Dataset& X) override {
        if (realFeatureCount != X.RealFeatureCount() || categoricalFeatureCount != X.CategoricalFeatureCount())
            throw std::runtime_error("Prediction data column types do not match those of the training set");
        auto probabilities = PredictProba(X);
        Eigen::VectorXi ret = Eigen::VectorXi::Zero(X.Size());
        for (std::size_t obj = 0; obj < X.Size(); ++obj) {
            for (std::size_t label = 1; label < classLabelsCount; ++label) {
                if (probabilities(label, obj) > probabilities(ret(obj), obj))
                    ret(obj) = label;
            }
        }
        return ret;
    }

    Eigen::MatrixXd PredictProba(const Dataset& X) override {
        if (realFeatureCount != X.RealFeatureCount() || categoricalFeatureCount != X.CategoricalFeatureCount())
            throw std::runtime_error("Prediction data column types do not match those of the training set");
        auto scores = ComputeClassScores(X);
        return math::SoftmaxColumnwise(gradientDescent.GetWeights() * scores);
    }

    std::size_t EstimateSize() const override {
        std::size_t sz = sizeof(*this) + conceptifier.EstimateSize() - sizeof(Conceptifier);
        for (const auto& c : contexts) sz += c.EstimateSize();
        // This is a very much lower approximation for Eigen objects
        sz += classScores.size() * sizeof(double);
        sz += (classLabelsCount * classLabelsCount * contexts.size()) * sizeof(double); //the weights vector in the GradientDescent object
        sz += trainSetLabels.capacity() * sizeof(ClassLabel);
        return sz;
    }

    void Serialize(std::ostream& stream) const {
    }

    void Deserialize(std::istream& stream) {
    }
};

}