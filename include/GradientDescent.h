#pragma once
#include <functional>
#include <vector>
#include <chrono>
#include <Eigen/Core>

namespace mlconcepts {

/// @brief A simple implementation of the gradient descent algorithm.
///        Meant to be used with extremely simple functions.
/// @tparam Vector The type that represents vectors.
template <class Vector = Eigen::VectorXd>
class GradientDescent
{
    std::function<double(const Vector&)> loss;
    std::function<Vector(const Vector&)> gradient;
    Vector weights;
    Vector prevStep;
    double learningRate;
    double momentum;
    double improvementStopThreshold;
    
public:
    /// @brief Constructs an empty gradient descent object.
    GradientDescent() {}

    /// @brief Constructs a gradient descent object.
    /// @param weightCount The number of weights that are trained.
    /// @param _loss The loss function.
    /// @param _gradient The gradient of the loss function.
    /// @param _learningRate The learning rate.
    /// @param _momentum The momentum.
    /// @param _improvementStopThreshold Threshold of loss improvement under
    ///                                  which training is halted. 
    GradientDescent(std::size_t weightCount, 
                    std::function<double(const Vector&)> _loss =
                        [](const Vector&){ return 0.0; },
                    std::function<Vector(const Vector&)> _gradient =
                        [](const Vector&) { return Vector::Zero(0); }, 
                    double _learningRate = 0.01,
                    double _momentum = 0.01,
                    double _improvementStopThreshold = 0.0) : 
                    loss(_loss), 
                    gradient(_gradient),
                    weights(Vector::Random(weightCount)),
                    prevStep(Vector::Zero(weightCount)), 
                    learningRate(_learningRate),
                    momentum(_momentum),
                    improvementStopThreshold(_improvementStopThreshold) {
        
    }

    /// @brief Constructs a gradient descent object.
    /// @param weightRows The number of rows in the trained weight matrix.
    /// @param weightCols The number of columns in the trained weight matrix.
    /// @param _loss The loss function.
    /// @param _gradient The gradient of the loss function.
    /// @param _learningRate The learning rate.
    /// @param _momentum The momentum.
    /// @param _improvementStopThreshold Threshold of loss improvement under
    ///                                  which training is halted. 
    GradientDescent(std::size_t weightRows, 
                    std::size_t weightCols, 
                    std::function<double(const Vector&)> _loss = 
                        [](const Vector&){ return 0.0; },
                    std::function<Vector(const Vector&)> _gradient = 
                        [](const Vector&) { return Vector::Zero(0); }, 
                    double _learningRate = 0.01,
                    double _momentum = 0.01,
                    double _improvementStopThreshold = 0.0) : 
                    loss(_loss),
                    gradient(_gradient),
                    weights(Vector::Random(weightRows, weightCols)),
                    prevStep(Vector::Zero(weightRows, weightCols)),
                    learningRate(_learningRate),
                    momentum(_momentum),
                    improvementStopThreshold(_improvementStopThreshold) {
        
    }

    /// @brief Randomizes the trained weights.
    void Reinitialize() {
        if (Vector::NumDimensions == 1)
            weights = Vector::Random(weights.rows());
        else 
            weights = Vector::Random(weights.rows(), weights.cols());
    }
    
    /// @brief Computes an iteration of the gradient descent.
    void ComputeIteration() {
        prevStep *= momentum;
        prevStep -= learningRate * gradient(weights);
        weights += prevStep;
    }

    /// @brief Returns the learning rate of the gradient descent object.
    /// @return The learning rate.
    double GetLearningRate() const { return learningRate; }

    /// @brief Returns the momentum of the gradient descent object.
    /// @return The momentum.
    double GetMomentum() const { return momentum; }

    /// @brief Returns the stop threshold of the gradient descent object.
    /// @return The loss variation stop threshold.
    double GetStopThreshold() const { return improvementStopThreshold; }

    /// @brief Manually sets one of the trained weight.
    /// @param id The index of the weight.
    /// @param v The value assigned to the weight.
    void SetWeight(std::size_t id, double v) { weights(id) = v; }

    /// @brief Sets the weights matrix.
    /// @param v The weights matrix to copy in the gradient descent object.
    void SetWeights(const Eigen::Ref<const Vector>& v) { weights = v; }

    /// @brief Computes the loss w.r.t. the current weights.
    /// @return The current loss.
    double GetLoss() { return loss(weights); }

    /// @brief Gets the value of one of the weights.
    /// @param id The id of the weight.
    /// @return The value of the indicated weight.
    double GetWeight(std::size_t id) const { return weights(id); }

    /// @brief Gets the value of one of the weights.
    /// @param rowID The row of the weights matrix to read.
    /// @param colID The column of the weights matrix to read.
    /// @return The value of the indicated weight.
    double GetWeight(std::size_t rowID, std::size_t colID) const { 
        return weights(rowID, colID); 
    }

    /// @brief Retrieves the weights matrix.
    /// @return The weights of the gradient descent object.
    const Vector& GetWeights() const { return weights; }
    
    /// @brief Trains the weights using the stored loss function.
    /// @param maxEpochs The maximum number of epochs after which the training
    ///                  is stopped.
    /// @param writeLog Whether to write training information in an output log.
    /// @param writeLogMinInterval Minimum time interval (in milliseconds)
    ///                            between log writes.
    /// @param stream The logging stream.
    /// @return Returns the number of epochs for which the weights have been
    ///         trained.
    std::size_t Train(std::size_t maxEpochs, 
                      bool writeLog = false, 
                      double writeLogMinInterval = 100.0,
                      std::ostream& stream = std::cout) {
        double prevLoss = GetLoss();
        auto prevClock = std::chrono::high_resolution_clock::now();
        if (writeLog) 
            stream << "Train epoch 0, loss " << prevLoss;
        for (std::size_t epoch = 0; epoch < maxEpochs; ++epoch) { 
            ComputeIteration();
            double newLoss = GetLoss();
            if (writeLog) { 
                auto nowClock = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> timeDistance = 
                    (nowClock - prevClock);
                if (epoch == maxEpochs - 1 || 
                    writeLogMinInterval <= timeDistance.count()) {
                    prevClock = nowClock;
                    stream << "Train epoch " << (epoch + 1) << 
                              ", loss " << newLoss << "         " <<
                              "                   \r";
                }
            }
            if (prevLoss - newLoss < improvementStopThreshold && 
                prevLoss - newLoss >= 0.0) {
                if (writeLog) {
                    stream << "Train epoch " << maxEpochs << 
                              ", loss " << prevLoss << "            "
                              "              \n";
                }
                return epoch + 1;
            }
            prevLoss = newLoss;
        }
        return maxEpochs;
    }

    /// @brief Returns the sum of the positive weights.
    /// @return The sum of the positive weights.
    double SumOfPositiveWeights() const {
        return std::accumulate(
            weights.begin(),
            weights.end(),
            0.0,
            [](double a, double b){ return b > 0 ? a + b : a; }
        );
    }


    /// @brief Returns the sum of the negative weights.
    /// @return The sum of the negative weights.
    double SumOfNegativeWeights() const {
        return std::accumulate(
            weights.begin(),
            weights.end(),
            0.0,
            [](double a, double b){ return b < 0 ? a + b : a; }
        );
    }
};

}