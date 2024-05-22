#pragma once
#include "Dataset.h"
#include "Settings.h"

namespace mlconcepts
{

/// @brief Simple abstract interface for unsupervised outlier detection models.
class UnsupervisedModel {
protected:
    ModelSettings settings;
public:
    virtual ~UnsupervisedModel() { }

    /// @brief Trains the model so as to fit a dataset.
    /// @param X The dataset used for training.
    virtual void Fit(const Dataset& X) = 0;

    /// @brief Computes prediction scores for a given dataset.
    /// @param X The dataset for which to compute predictions.
    /// @return A vector containing a prediction for each entry in the dataset.
    virtual Eigen::VectorXd Predict(const Dataset& X) = 0;

    /// @brief Estimates the size of the model.
    /// @return Lower bound of the size of the model in bytes.
    virtual std::size_t EstimateSize() const = 0;

    /// @brief Sets one of the settings/parameters in the model.
    /// @tparam T The type of the parameter.
    /// @param name The name of the parameter.
    /// @param v The value the parameters is setted to.
    template <class T>
    void Set(const std::string& name, const T& v) { 
        settings.Set(name, v); 
    }
};

/// @brief Simple abstract interface for supervised outlier detection models.
class SupervisedModel {
protected:
    ModelSettings settings;
public:
    virtual ~SupervisedModel() { }

    /// @brief Trains the model so as to fit a dataset. The dataset is assumed
    ///        to have its labels defined.
    /// @param X The dataset used for training.
    virtual void Fit(const Dataset& Xy) = 0;

    /// @brief Computes prediction scores for a given dataset.
    /// @param X The dataset for which to compute predictions.
    /// @return A vector containing a prediction for each entry in the dataset.
    virtual Eigen::VectorXd Predict(const Dataset& X) = 0;

    /// @brief Estimates the size of the model.
    /// @return Lower bound of the size of the model in bytes.
    virtual std::size_t EstimateSize() const = 0;

    /// @brief Sets one of the settings/parameters in the model.
    /// @tparam T The type of the parameter.
    /// @param name The name of the parameter.
    /// @param v The value the parameters is setted to.
    template <class T>
    void Set(const std::string& name, const T& v) {
        settings.Set(name, v); 
    }
};

class ClassificationModel {
protected:
    ModelSettings settings;
public:
    virtual ~ClassificationModel() { }

    /// @brief Trains the model so as to fit a dataset. The dataset is assumed
    ///        to have its labels defined.
    /// @param X The dataset used for training.
    virtual void Fit(const Dataset& Xy) = 0;

    /// @brief Computes predictions for a given dataset.
    /// @param X The dataset for which to compute predictions.
    /// @return A vector containing a prediction (a class) for each entry in
    ///         the dataset.
    virtual Eigen::VectorXi Predict(const Dataset& X) = 0;

    /// @brief Computes predictions for a given dataset.
    /// @param X The dataset for which to compute predictions.
    /// @return A matrix containing for each item in the dataset a prediction
    ///         score for each class.
    virtual Eigen::MatrixXd PredictProba(const Dataset& X) = 0;

    /// @brief Estimates the size of the model.
    /// @return Lower bound of the size of the model in bytes.
    virtual std::size_t EstimateSize() const = 0;
    

    /// @brief Sets one of the settings/parameters in the model.
    /// @tparam T The type of the parameter.
    /// @param name The name of the parameter.
    /// @param v The value the parameters is setted to.
    template <class T>
    void Set(const std::string& name, const T& v) {
        settings.Set(name, v);
    }
};

}