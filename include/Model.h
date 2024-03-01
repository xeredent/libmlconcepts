#pragma once
#include "Dataset.h"
#include "Settings.h"

namespace mlconcepts
{

class UnsupervisedModel {
protected:
    ModelSettings settings;
public:
    virtual void Fit(const Dataset& X) = 0;
    virtual Eigen::VectorXd Predict(const Dataset& X) = 0;
    virtual size_t EstimateSize() const = 0;

    template <class T>
    void Set(const std::string& name, const T& v) { settings.Set(name, v); }
};

class SupervisedModel {
protected:
    ModelSettings settings;
public:
    virtual void Fit(const Dataset& Xy) = 0;
    virtual Eigen::VectorXd Predict(const Dataset& X) = 0;
    virtual size_t EstimateSize() const = 0;
    
    template <class T>
    void Set(const std::string& name, const T& v) { settings.Set(name, v); }
};

class ClassificationModel {
protected:
    ModelSettings settings;
public:
    virtual void Fit(const Dataset& Xy) = 0;
    virtual Eigen::VectorXi Predict(const Dataset& X) = 0;
    virtual Eigen::MatrixXd PredictProba(const Dataset& X) = 0;
    virtual size_t EstimateSize() const = 0;
    
    template <class T>
    void Set(const std::string& name, const T& v) { settings.Set(name, v); }
};

}