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

/// @brief Implements a supervised outlier detection model.
/// @tparam Context The type implementing formal contexts.
/// @tparam Conceptifier The conceptifier which generates the formal contexts
///         from a given dataset.
template <class Context,
          class Conceptifier = Conceptifier<
            SimpleDoubletonContextSelector<std::uint32_t>,
            AllUniformQuantizer<double>, AllValuesQuantizer<std::size_t>
          > 
> class SupervisedODModel : public SupervisedModel {
protected:
    static constexpr const std::uint32_t encodingMagicNumber = 0x756e6f64;
    typedef typename Context::AttributeSet AttSet;
    std::vector<Context> contexts;
    Context::ObjectSet inliers;
    Conceptifier conceptifier;
    std::size_t realFeatureCount; 
    std::size_t categoricalFeatureCount;
    Eigen::MatrixXd outlierDegrees;
    GradientDescent<Eigen::VectorXd> gradientDescent;
    double gradientDescentBalance;

    /// @brief Computes the outlier degrees vector according to each context
    ///        for a given dataset. It is supposed to be used during prediction
    ///        computations.
    /// @param data The dataset.
    /// @return A matrix of outlier degrees containing a value for each context
    ///         and each object in the dataset.
    Eigen::MatrixXd ComputeOutlierDegrees(const Dataset& data) {
        Eigen::MatrixXd outdegsVector(contexts.size(), data.Size());
        for (std::size_t ctxID = 0; ctxID < contexts.size(); ++ctxID) {
            auto intensions = conceptifier.template ProcessData<AttSet>(data, ctxID);
            for (std::size_t obj = 0; obj < data.Size(); ++obj) {
                outdegsVector(ctxID, obj) = std::exp(
                    -std::pow(
                        contexts[ctxID].ComputeFilteredClosureSize(
                            intensions[obj],
                            inliers), 
                        2
                    )
                );
            }
        }
        return outdegsVector;
    }

    /// @brief Computes outlier degrees for every context for which they have
    ///        not been computed yet. It is supposed to be used between training
    ///        rounds.
    /// @param Xy The dataset used to compute the outlier degrees
    /// @param missingCID The ID of the first missing context from the outlier
    ///                   degrees matrix.
    void ComputeMissingOutlierDegrees(const Dataset& Xy, 
                                      std::size_t missingCID = 0) {
        outlierDegrees.conservativeResize(contexts.size(), Xy.Size()); 
        for (std::size_t obj = 0; obj < Xy.Size(); ++obj) {
            for (auto ctxID = missingCID; ctxID < contexts.size(); ++ctxID) {
                outlierDegrees(ctxID, obj) = std::exp(
                    -std::pow(
                        contexts[ctxID].ComputeFilteredClosureSize(
                            obj,
                            inliers), 
                        2
                    )
                );
            }
        }
    }

    /// @brief Initializes the inliers vector based on a training set.
    /// @param Xy The dataset.
    void InitializeInliers(const Dataset& Xy) {
        inliers = typename Context::ObjectSet(Xy.Size());
        Xy.ForEachLabel([this](std::size_t obj, std::size_t value) { 
            if (value == 0) inliers.Add(obj); }
        );
    }

    /// @brief Initializes the gradient descent object to train on the current
    ///        contexts.
    void InitializeGradientDescent(double learningRate, 
                                   double momentum,
                                   double stopThreshold,
                                   double balance,
                                   std::size_t nobjs) {
        // Constructs the gradient descent object by passing its parameters,
        // and two lambdas for the loss and the gradient.
        gradientDescentBalance = balance;
        gradientDescent = GradientDescent<Eigen::VectorXd>(
            contexts.size(), // weightsCount
            // Loss function
            [balance, inliers = std::as_const(inliers), 
             outdegs = std::as_const(outlierDegrees), nobjs]
            (const Eigen::VectorXd& w) -> double {
                double Wp = std::accumulate(
                    w.begin(), w.end(), 0.0, 
                    [](double a, double b){ return b > 0 ? a + b : a; }
                );
                double Wm = std::accumulate(
                    w.begin(), w.end(), 0.0,
                    [](double a, double b){ return b < 0 ? a + b : a; }
                );
                double sum = 0;
                for (std::size_t i = 0; i < nobjs; ++i) {
                    double ycap = (w.dot(outdegs(Eigen::all, i)) - Wm) / 
                                  (Wp + Wm);
                    sum += inliers.Contains(i) ? 
                           std::pow(ycap,2)/balance : 
                           std::pow((1 - ycap), 2); 
                }
                return sum;
            },
            // Gradient function
            [balance, inliers = std::as_const(inliers), 
             outdegs = std::as_const(outlierDegrees), nobjs]
            (const Eigen::VectorXd& w) -> Eigen::VectorXd {
                double Wp = std::accumulate(
                    w.begin(), w.end(), 0.0, 
                    [](double a, double b){ return b > 0 ? a + b : a; }
                );
                double Wm = std::accumulate(
                    w.begin(), w.end(), 0.0,
                    [](double a, double b){ return b < 0 ? a + b : a; }
                );
                Eigen::VectorXd output = Eigen::VectorXd::Zero(w.size());
                for (std::size_t k = 0; k < (std::size_t)w.size(); ++k) {
                    for (std::size_t i = 0; i < nobjs; ++i) {
                        double dot = w.dot(outdegs(Eigen::all,i));
                        double ycap = (dot - Wm) / (Wp + Wm);
                        double dycap = w(k) < 0 ? 
                            ((outdegs(k,i) - 1) * (Wp + Wm) - dot) / 
                                std::pow(Wp + Wm, 2.0) :
                            ((outdegs(k,i) - 1) * (Wp + Wm) - dot) / 
                                std::pow(Wp + Wm, 2.0);
                        output(k) += inliers.Contains(i) ?  
                            2 * (ycap) * 1 / balance * dycap : 
                            (2 * (1 - ycap)) * (-dycap);
                    }
                }
                return output;
            },
            learningRate,
            momentum,
            stopThreshold
        );
    }

    /// @brief Initializes a gradient descent object based on the current
    ///        settings and the given dataset.
    void InitializeGradientDescent(const Dataset& Xy) {
        double learningRate = this->settings.Get("LearningRate", 0.01);
        double momentum = this->settings.Get("Momentum", 0.01);
        double stopThreshold = this->settings.Get("StopThreshold", 0.01);
        double balance = (double)Xy.Size() / 
                         (double)(Xy.Size() - inliers.Size());
        InitializeGradientDescent(learningRate, momentum, 
                                  stopThreshold, balance, Xy.Size());
    }

public:

    /// @brief Trains the model so as to fit a dataset. The dataset is assumed
    ///        to have its labels defined.
    /// @param X The dataset used for training.
    void Fit(const Dataset& Xy) override {
        realFeatureCount = Xy.RealFeatureCount(); 
        categoricalFeatureCount = Xy.CategoricalFeatureCount();
        conceptifier.Initialize(Xy, settings, contexts);
        InitializeInliers(Xy);
        ComputeMissingOutlierDegrees(Xy);
        InitializeGradientDescent(Xy);
        Train(settings.Get("TrainEpochs", (std::size_t)1000));
    }

    void Train(std::size_t epochsCount, 
               std::ostream& outputStream = std::cout) {
        bool showTraining = settings.Get("ShowTraining", true);
        std::size_t showTrainingDelay = settings.Get("ShowTrainingDelay", 
                                                     100.0);
        gradientDescent.Train(epochsCount, showTraining, 
                              showTrainingDelay, outputStream);
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
        Eigen::VectorXd predictions(X.Size());
        auto outdegs = ComputeOutlierDegrees(X);
        double Wp = gradientDescent.SumOfPositiveWeights();
        double Wm = gradientDescent.SumOfNegativeWeights();
        for (std::size_t obj = 0; obj < X.Size(); ++obj) {
            predictions(obj) = (
                gradientDescent.GetWeights().dot(outdegs(Eigen::all, obj)) - Wm
            ) / (Wp + Wm);
        }
        return predictions;
    }

    /// @brief Computes prediction scores for a given dataset and output the
    ///        weighted outlier degrees scores for each context and object.
    /// @param X The dataset for which to compute predictions.
    /// @return A pair containing the vector of all the predictions, and a 
    ///         matrix containing for each context (row-wise) and each object
    ///         in the dataset (column-wise) its weighted outlier degree score.
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> 
    PredictExplain(const Dataset& X) {
        if (realFeatureCount != X.RealFeatureCount() || 
            categoricalFeatureCount != X.CategoricalFeatureCount()) {
            throw std::runtime_error("Prediction data column types do not " 
                                     " match those of the training set");
        }
        Eigen::VectorXd predictions(X.Size());
        Eigen::MatrixXd outdegs = ComputeOutlierDegrees(X);
        double Wp = gradientDescent.SumOfPositiveWeights();
        double Wm = gradientDescent.SumOfNegativeWeights();
        for (std::size_t obj = 0; obj < X.Size(); ++obj) {
            outdegs(Eigen::all, obj).array() *= gradientDescent.GetWeights().array();
            predictions(obj) = (outdegs(Eigen::all, obj).sum() - Wm) / 
                               (Wp + Wm);
        }
        return std::pair(predictions, outdegs);
    }

    /// @brief Estimates the size of the model.
    /// @return Lower bound of the size of the model in bytes.
    std::size_t EstimateSize() const override {
        std::size_t sz = sizeof(*this) + conceptifier.EstimateSize() -
                         sizeof(Conceptifier);
        for (const auto& c : contexts) sz += c.EstimateSize();
        // This is a very much lower approximation for Eigen objects
        sz += outlierDegrees.size() * sizeof(double);
        //the weights vector in the GradientDescent object
        sz += contexts.size() * sizeof(double);
        return sz;
    }

    /// @brief Returns the formal contexts generated by the training.
    /// @return The formal contexts generated by the training.
    const std::vector<Context>& GetContexts() { return contexts; }

    /// @brief Returns a copy of the feature sets vector.
    /// @return A copy of the feature sets vector.
    std::vector<typename Conceptifier::FeatureSet> GetFeatureSets() const {
        return conceptifier.GetFeatureSets();
    }

    /// @brief Returns a copy of the weights vector.
    /// @return A copy of the weights vector.
    Eigen::VectorXd GetWeights() {
        return gradientDescent.GetWeights();
    }

    /// @brief Writes the model state to a stream.
    /// @param stream The stream the model state is written to.
    void Serialize(std::ostream& stream) const {
        io::LittleEndianWrite(stream, encodingMagicNumber);
        io::LittleEndianWrite(stream, (std::uint32_t)realFeatureCount);
        io::LittleEndianWrite(stream, (std::uint32_t)categoricalFeatureCount);
        io::LittleEndianWrite(stream, (std::uint64_t)outlierDegrees.cols());
        conceptifier.Serialize(stream);
        for (const auto& ctx : contexts) ctx.Serialize(stream);
        inliers.template SerializeRLE<2,5>(stream);
        for (int obj = 0; obj < outlierDegrees.cols(); ++obj) {
            for (std::size_t ctxID = 0; ctxID < contexts.size(); ++ctxID) {
                io::LittleEndianWrite(stream, 
                                      (double)outlierDegrees(ctxID, obj));
            }
        }
        io::LittleEndianWrite(stream, gradientDescent.GetLearningRate());
        io::LittleEndianWrite(stream, gradientDescent.GetMomentum());
        io::LittleEndianWrite(stream, gradientDescent.GetStopThreshold());
        io::LittleEndianWrite(stream, gradientDescentBalance);
        for (std::size_t i = 0; i < contexts.size(); ++i) 
            io::LittleEndianWrite(stream, gradientDescent.GetWeight(i));
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
        std::uint64_t objectCount = io::LittleEndianRead<std::uint64_t>(stream);
        conceptifier.Deserialize(stream);
        contexts.clear();
        for (std::size_t i = 0; i < conceptifier.GetFeatureSetsCount(); ++i)
            contexts.push_back(Context(stream));
        inliers = typename Context::ObjectSet(objectCount);
        inliers.template DeserializeRLE<2,5>(stream);
        outlierDegrees = Eigen::MatrixXd(contexts.size(), objectCount);
        for (std::size_t obj = 0; obj < objectCount; ++obj) {
            for (std::size_t ctxID = 0; ctxID < contexts.size(); ++ctxID) {
                outlierDegrees(ctxID, obj) = io::LittleEndianRead<double>(stream);
            }
        }
        double learningRate = io::LittleEndianRead<double>(stream);
        double momentum = io::LittleEndianRead<double>(stream);
        double stopThreshold = io::LittleEndianRead<double>(stream);
        double balance = io::LittleEndianRead<double>(stream);
        InitializeGradientDescent(learningRate, momentum, stopThreshold,
                                  balance, objectCount);
        for (std::size_t i = 0; i < contexts.size(); ++i) 
            gradientDescent.SetWeight(i, io::LittleEndianRead<double>(stream));
    }
};

}