#pragma once
#include <functional>
#include <vector>
#include <chrono>
#include <format>
#include <Eigen/Core>

namespace mlconcepts {

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
    GradientDescent(size_t weightCount = 0, 
                    std::function<double(const Vector&)> _loss = [](const Vector&){ return 0.0; }, 
                    std::function<Vector(const Vector&)> _gradient = [](const Vector&) { return Vector::Zero(0); }, 
                    double _learningRate = 0.01,
                    double _momentum = 0.01, double _improvementStopThreshold = 0.0) : 
                    loss(_loss), gradient(_gradient), weights(Vector::Random(weightCount)),
                    prevStep(Vector::Zero(weightCount)), learningRate(_learningRate),
                    momentum(_momentum), improvementStopThreshold(_improvementStopThreshold) {
        
    }

    void Reinitialize() {
        weights = Vector::Random(weights.rows());
    }
    
    void ComputeIteration() {
        prevStep *= momentum;
        prevStep -= learningRate * gradient(weights);
        weights += prevStep;
    }

    void SetWeights(const Eigen::Ref<const Vector>& v) { weights = v; }
    double GetLoss() { return loss(weights); } 
    const Vector& GetWeights() { return weights; }
    
    size_t Train(size_t maxEpochs, bool writeLog = false, double writeLogMinInterval = 100.0, std::ostream& stream = std::cout) {
        double prevLoss = GetLoss();
        auto prevClock = std::chrono::high_resolution_clock::now();
        if (writeLog) stream << std::format("Train epoch 0, loss {:g}", prevLoss);
        for (size_t epoch = 0; epoch < maxEpochs; ++epoch) { 
            ComputeIteration();
            double newLoss = GetLoss();
            if (writeLog) { 
                auto nowClock = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> timeDistance = (nowClock - prevClock);
                if ( epoch == maxEpochs - 1 || writeLogMinInterval <= timeDistance.count()) {
                    prevClock = nowClock;
                    stream << std::format("Train epoch {:d}, loss {:g}                            \r", epoch + 1, newLoss);
                }
            }
            if (prevLoss - newLoss < improvementStopThreshold && prevLoss - newLoss >= 0.0) {
                if (writeLog) stream << std::format("Train epoch {:d}, loss {:g}                          \n", maxEpochs, prevLoss);
                return epoch + 1;
            }
            prevLoss = newLoss;
        }
        return maxEpochs;
    }

    double SumOfPositiveWeights() const {
        return std::accumulate(weights.begin(), weights.end(), 0.0, [](double a, double b){ return b > 0 ? a + b : a; } );
    }

    double SumOfNegativeWeights() const {
        return std::accumulate(weights.begin(), weights.end(), 0.0, [](double a, double b){ return b < 0 ? a + b : a; } );
    }
};

}