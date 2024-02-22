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

template <class Context,
          class Conceptifier = Conceptifier<SimpleDoubletonContextSelector<uint32_t>,
                                            AllUniformQuantizer<double>, AllValuesQuantizer<size_t>>
         >
class SupervisedODModel : public SupervisedModel {
protected:
    typedef typename Context::AttributeSet AttSet;
    std::vector<Context> contexts;
    Context::ObjectSet inliers;
    Conceptifier conceptifier;
    size_t realFeatureCount; 
    size_t categoricalFeatureCount;
    Eigen::MatrixXd outlierDegrees;
    GradientDescent<Eigen::VectorXd> gradientDescent;

    /// @brief Computes the outlier degrees vector according to each context for a given dataset.
    /// It is supposed to be used during prediction computations.
    /// @param data The dataset.
    /// @return A matrix of outlier degrees containing a value for each context and each object in the dataset.
    Eigen::MatrixXd ComputeOutlierDegrees(const Dataset& data) {
        Eigen::MatrixXd outdegsVector(contexts.size(), data.Size());
        for (size_t ctxID = 0; ctxID < contexts.size(); ++ctxID) {
            auto intensions = conceptifier.template ProcessData<AttSet>(data, ctxID);
            for (size_t obj = 0; obj < data.Size(); ++obj) {
                outdegsVector(ctxID, obj) = 
                        std::exp(-std::pow(contexts[ctxID].ComputeFilteredClosureSize(intensions[obj], inliers), 2));
            }
        }
        return outdegsVector;
    }

    /// @brief Computes outlier degrees for every context for which they have not been computed yet.
    /// It is supposed to be used between training rounds.
    /// @param Xy The dataset used to compute the outlier degrees
    /// @param firstMissingContextID The ID of the first missing context from the outlier degrees matrix
    void ComputeMissingOutlierDegrees(const Dataset& Xy, size_t firstMissingContextID = 0) {
        outlierDegrees.conservativeResize(contexts.size(), Xy.Size()); 
        for (size_t obj = 0; obj < Xy.Size(); ++obj) {
            for (size_t ctxID = firstMissingContextID; ctxID < contexts.size(); ++ctxID) {
                outlierDegrees(ctxID, obj) = std::exp(-std::pow(contexts[ctxID].ComputeFilteredClosureSize(obj, inliers), 2));
            }
        }
    }

    /// @brief Initializes the inliers vector based on a training set.
    /// @param Xy The dataset.
    void InitializeInliers(const Dataset& Xy) {
        inliers = typename Context::ObjectSet(Xy.Size());
        Xy.ForEachLabel([this](size_t obj, size_t value) { if (value == 0) inliers.Add(obj); });
    }

    /// @brief Initializes the gradient descent object to train on the current contexts.
    void InitializeGradientDescent(const Dataset& Xy) {
        double learningRate = this->settings.Get("LearningRate", 0.01);
        double momentum = this->settings.Get("Momentum", 0.01);
        double stopThreshold = this->settings.Get("StopThreshold", 0.01);
        double balance = (double)Xy.Size() / (double)(Xy.Size() - inliers.Size());
        //Constructs the gradient descent object by passing its parameters, and
        //two lambdas for the loss and the gradient.
        gradientDescent = GradientDescent<Eigen::VectorXd>(
            contexts.size(), //weightsCount
            [balance, inliers = std::as_const(inliers), outdegs = std::as_const(outlierDegrees), nobjs = Xy.Size()]
            (const Eigen::VectorXd& w) -> double { //loss
                double Wp = std::accumulate(w.begin(), w.end(), 0.0, [](double a, double b){ return b > 0 ? a + b : a; });
                double Wm = std::accumulate(w.begin(), w.end(), 0.0, [](double a, double b){ return b < 0 ? a + b : a; });
                double sum = 0;
                for (size_t i = 0; i < nobjs; ++i) {
                    double ycap = (w.dot(outdegs(Eigen::all, i)) - Wm) / (Wp + Wm);
                    sum += inliers.Contains(i) ? std::pow(ycap,2)/balance : std::pow((1 - ycap), 2); 
                }
                return sum;
            },
            [balance, inliers = std::as_const(inliers), outdegs = std::as_const(outlierDegrees), nobjs = Xy.Size()]
            (const Eigen::VectorXd& w) -> Eigen::VectorXd { //gradient
                double Wp = std::accumulate(w.begin(), w.end(), 0.0, [](double a, double b){ return b > 0 ? a + b : a; });
                double Wm = std::accumulate(w.begin(), w.end(), 0.0, [](double a, double b){ return b < 0 ? a + b : a; });
                Eigen::VectorXd output = Eigen::VectorXd::Zero(w.size());
                for (size_t k = 0; k < (size_t)w.size(); ++k) {
                    for (size_t i = 0; i < nobjs; ++i) {
                        double dot = w.dot(outdegs(Eigen::all,i));
                        double ycap = (dot - Wm) / (Wp + Wm);
                        double dycap = w(k) < 0 ? ((outdegs(k,i) - 1) * (Wp + Wm) - dot) / std::pow(Wp + Wm, 2.0) :
                                                  ((outdegs(k,i) - 1) * (Wp + Wm) - dot) / std::pow(Wp + Wm, 2.0);
                        output(k) += inliers.Contains(i) ?  2*(ycap)*1/balance*dycap : (2*(1 - ycap))*(-dycap);
                    }
                }
                return output;
            },
            learningRate,
            momentum,
            stopThreshold
        );
    }

public:
    void Fit(const Dataset& Xy) override {
        realFeatureCount = Xy.RealFeatureCount(); categoricalFeatureCount = Xy.CategoricalFeatureCount();
        conceptifier.Initialize(Xy, settings, contexts);
        InitializeInliers(Xy);
        ComputeMissingOutlierDegrees(Xy);
        InitializeGradientDescent(Xy);
        Train(settings.Get("TrainEpochs", (size_t)1000));
    }

    void Train(size_t epochsCount, std::ostream& outputStream = std::cout) {
        bool showTraining = settings.Get("ShowTraining", true);
        size_t showTrainingDelay = settings.Get("ShowTrainingDelay", 100.0);
        gradientDescent.Train(epochsCount, showTraining, showTrainingDelay, outputStream);
    }

    Eigen::VectorXd Predict(const Dataset& X) override {
        if (realFeatureCount != X.RealFeatureCount() || categoricalFeatureCount != X.CategoricalFeatureCount())
            throw std::runtime_error("Prediction data column types do not match those of the training set");
        Eigen::VectorXd predictions(X.Size());
        auto outdegs = ComputeOutlierDegrees(X);
        double Wp = gradientDescent.SumOfPositiveWeights(), Wm = gradientDescent.SumOfNegativeWeights();
        for (size_t obj = 0; obj < X.Size(); ++obj) {
            predictions(obj) = (gradientDescent.GetWeights().dot(outdegs(Eigen::all, obj)) - Wm) / (Wp + Wm);
        }
        return predictions;
    }

    size_t EstimateSize() const override {
        size_t sz = sizeof(*this) + conceptifier.EstimateSize() - sizeof(Conceptifier);
        for (const auto& c : contexts) sz += c.EstimateSize();
        // This is a very much lower approximation for Eigen objects
        sz += outlierDegrees.size() * sizeof(double);
        sz += contexts.size() * sizeof(double); //the weights vector in the GradientDescent object
        return sz;
    }

    const std::vector<Context>& GetContexts() { return contexts; }
};

}