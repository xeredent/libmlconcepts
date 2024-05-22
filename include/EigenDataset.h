#pragma once
#include <algorithm>
#include <Eigen/Core>
#include "Dataset.h"

namespace mlconcepts 
{

/// @brief A dataset encoded using Eigen matrices.
template<class RealMat = Eigen::MatrixXd, 
         class CatMat = Eigen::MatrixXi, 
         class Labels = Eigen::VectorXi>
class EigenDataset : public Dataset {
    Eigen::Ref<RealMat> realData;
    Eigen::Ref<CatMat> categoricalData;
    Labels emptyLabels;
    Eigen::Ref<Labels> labelData;

public:
    EigenDataset(Eigen::Ref<RealMat> realdata, Eigen::Ref<CatMat> catdata) : 
                realData(realdata), 
                categoricalData(catdata), 
                emptyLabels(Labels::Zero(0)), 
                labelData(emptyLabels) {
        if (realData.rows() != 0 && categoricalData.rows() != 0 && 
            realData.rows() != categoricalData.rows())
            throw std::runtime_error(
                "The categorical part and the real part of a dataset must "
                " have the same size (number of rows)"
            );
    }
    EigenDataset(Eigen::Ref<RealMat> realdata, 
                 Eigen::Ref<CatMat> catdata, 
                 Eigen::Ref<Labels> labels) : 
                realData(realdata), 
                categoricalData(catdata), 
                labelData(labels) {
        if (realData.rows() != 0 && categoricalData.rows() != 0 && 
            realData.rows() != categoricalData.rows())
            throw std::runtime_error(
                "The categorical part and the real part of a dataset must " 
                " have the same size (number of rows)"
            );
    }

    /// @brief Executes a function on every element of the real-valued part of
    //         the dataset.
    /// @param f function of type void(std::size_t,std::size_t,double), where:
    ///          - the first parameter is the index of the row (object id),
    ///          - the second parameter is the index of the column (feature id),
    ///          - the third parameter is the value of element.
    void ForEachReal(
        const std::function<void(std::size_t, std::size_t, double)>& label
    ) const override {
        std::size_t col = 0;
        for (const auto& c : realData.colwise()) {
            std::size_t row = 0;
            for (const auto& x : c) {
                label(row, col, x);
                ++row;
            }
            ++col;
        }
    }

    /// @brief Executes a function on every element of the categorical part of
    ///        the dataset.
    /// @param f function of type void(std::size_t,std::size_t,double), where:
    ///          - the first parameter is the index of the row (object id),
    ///          - the second parameter is the index of the column (feature id),
    ///          - the third parameter is the value of element.
    void ForEachCategorical(
        const std::function<void(std::size_t, std::size_t, std::size_t)>& label
    ) const override {
        std::size_t col = realData.cols();
        for (const auto& c : categoricalData.colwise()) {
            std::size_t row = 0;
            for (auto x : c) {
                label(row, col, x);
                ++row;
            }
            ++col;
        }
    }

    /// @brief Executes a function on every element of a column in the
    ///        real-valued part of the dataset.
    /// @param col The column to operate on.
    /// @param f function of type void(std::size_t,std::size_t,double), where:
    ///          - the first parameter is the index of the row (object id),
    ///          - the second parameter is the index of the column (feature id),
    ///          - the third parameter is the value of element.
    virtual void ForEachRealColumn(
        std::size_t col,
        const std::function<void(std::size_t,std::size_t,double)>& f
    ) const override {
        std::size_t row = 0;
        for (const auto& x : realData(Eigen::all, col)) {
            f(row, col, x);
            ++row;
        }
    }

    /// @brief Executes a function on every element in a column of the
    ///        categorical part of the dataset.
    /// @param col The column to operate on.
    /// @param f function of type void(std::size_t,std::size_t,double), where:
    ///          - the first parameter is the index of the row (object id),
    ///          - the second parameter is the index of the column (feature id),
    ///          - the third parameter is the value of element.
    virtual void ForEachCategoricalColumn(
        std::size_t col, 
        const std::function<void(std::size_t,std::size_t,std::size_t)>& f
    ) const override {
        std::size_t row = 0;
        for (const auto& x : categoricalData(Eigen::all, col - realData.cols())) {
            f(row, col, x);
            ++row;
        }
    }

    /// @brief Executes a function on every element label in the dataset.
    /// @param f function of type void(std::size_t,std::size_t,double), where:
    ///          - the first parameter is the index of the row (object id),
    ///          - the second parameter is the label of element.
    virtual void ForEachLabel(
        const std::function<void(std::size_t,std::size_t)>& f
    ) const override{
        std::size_t obj = 0;
        for (auto x : labelData) {
            f(obj, x);
            ++obj;
        }
    }

    /// @brief Retrieves the minimum value for a given real-valued feature.
    /// @param feature The ID of the feature.
    /// @return The minimum value for the feature.
    virtual double MinReal(std::size_t feature) const override {
        const auto& slice = realData(Eigen::all, feature);
        return *std::min_element(slice.begin(), slice.end());
    }

    /// @brief Retrieves the maximum value for a given real-valued feature.
    /// @param feature The ID of the feature.
    /// @return The maximum value for the feature.
    virtual double MaxReal(std::size_t feature) const override {
        const auto& slice = realData(Eigen::all, feature);
        return *std::max_element(slice.begin(), slice.end());
    }
    
    /// @brief Counts the number of different values that a categorical feature
    ///        can take.
    /// @param feature The ID of the feature.
    /// @return The cardinality of the range of the feature.
    virtual std::size_t CountCategorical(std::size_t feature) const override {
        if (feature < (std::size_t)realData.cols()) 
            throw std::runtime_error("Invalid index of categorical data");
        const auto& slice = realData(Eigen::all, feature - realData.cols());
        return *std::max_element(slice.begin(), slice.end()) + 1;
    }

    /// @brief Retrieves the number of values that labels can have. 
    /// @return The number of labels in the dataset.
    virtual std::size_t CountLabels() const override {
        return *std::max_element(labelData.begin(), labelData.end()) + 1;
    }

    /// @brief Retrieves the size of the dataset, i.e., the number of entries
    ///        it contains.
    /// @return The size of the dataset.
    virtual std::size_t Size() const override {
        return realData.rows();
    }

    /// @brief Retrieves the number of features of the dataset. 
    /// @return The size of the dataset.
    virtual std::size_t FeatureCount() const override {
        return realData.cols() + categoricalData.cols();
    }

    /// @brief Retrieves the number of real features of the dataset. 
    /// @return The size of the dataset.
    virtual std::size_t RealFeatureCount() const override {
        return realData.cols();
    }

    /// @brief Retrieves the number of categorical features of the dataset. 
    /// @return The size of the dataset.
    virtual std::size_t CategoricalFeatureCount() const override {
        return categoricalData.cols();
    }

    /// @brief Converts the labels to an Eigen::Vector an returns them.
    /// @return A vector containing the labels.
    virtual const Eigen::Ref<Eigen::VectorXi>& 
    GetLabelsAsEigen() const override {
        return labelData;
    }

    /// @brief Returns a cell of real type in the dataset.
    /// @param obj The object id.
    /// @param feature The feature id.
    /// @return The cell value.
    virtual double GetReal(std::size_t obj, 
                           std::size_t feature) const override {
        return realData(obj, feature);
    }
    
    /// @brief Returns a cell of categorical in the dataset.
    /// @param obj The object id.
    /// @param feature The feature id.
    /// @return The cell value.
    virtual std::size_t GetCategorical(std::size_t obj, 
                                       std::size_t feature) const override {
        return categoricalData(obj, feature - realData.cols());
    }
};

}