#pragma once
#include <functional>
#include <Eigen/Core>

namespace mlconcepts 
{

/// @brief Abstract view on a datasets that exposes only the functions needed
///        for conceptification. A dataset contains a real-valued part, a 
///        categorical part, and a label part.
class Dataset {
public:
    /// @brief Executes a function on every element of the real-valued part of
    ///        the dataset.
    /// @param f function of type void(std::size_t,std::size_t,double), where:
    ///          - the first parameter is the index of the row (object id),
    ///          - the second parameter is the index of the column (feature id),
    ///          - the third parameter is the value of element.
    virtual void ForEachReal(
        const std::function<void(std::size_t,std::size_t,double)>& f
    ) const = 0;

    /// @brief Executes a function on every element of the categorical part of
    ///        the dataset.
    /// @param f function of type void(std::size_t,std::size_t,double), where:
    ///          - the first parameter is the index of the row (object id),
    ///          - the second parameter is the index of the column (feature id),
    ///          - the third parameter is the value of element.
    virtual void ForEachCategorical(
        const std::function<void(std::size_t,std::size_t,std::size_t)>& f
    ) const = 0;

    /// @brief Executes a function on every element of a column in the
    ///        real-valued part of the dataset.
    /// @param col The column to operate on.
    /// @param f function of type void(std::size_t,std::size_t,double), where:
    /// - the first parameter is the index of the row (object id),
    /// - the second parameter is the index of the column (feature id),
    /// - the third parameter is the value of element.
    virtual void ForEachRealColumn(
        std::size_t col, 
        const std::function<void(std::size_t,std::size_t,double)>& f
    ) const = 0;

    /// @brief Executes a function on every element in a column of the
    ///        categorical part of the dataset.
    /// @param col The column to operate on.
    /// @param f function of type void(std::size_t,std::size_t,double), where:
    /// - the first parameter is the index of the row (object id),
    /// - the second parameter is the index of the column (feature id),
    /// - the third parameter is the value of element.
    virtual void ForEachCategoricalColumn(
        std::size_t col,
        const std::function<void(std::size_t,std::size_t,std::size_t)>& f
    ) const = 0;

    /// @brief Executes a function on every element label in the dataset.
    /// @param f function of type void(std::size_t,std::size_t,double), where:
    /// - the first parameter is the index of the row (object id),
    /// - the second parameter is the label of element.
    virtual void ForEachLabel(
        const std::function<void(std::size_t,std::size_t)>& f
    ) const = 0;

    /// @brief Retrieves the minimum value for a given real-valued feature.
    /// @param feature The ID of the feature.
    /// @return The minimum value for the feature.
    virtual double MinReal(std::size_t feature) const = 0;

    /// @brief Retrieves the maximum value for a given real-valued feature.
    /// @param feature The ID of the feature.
    /// @return The maximum value for the feature.
    virtual double MaxReal(std::size_t feature) const = 0;

    /// @brief Counts the number of different values that a categorical feature
    ///        can take.
    /// @param feature The ID of the feature.
    /// @return The cardinality of the range of the feature.
    virtual std::size_t CountCategorical(std::size_t feature) const = 0;

    /// @brief Retrieves the number of values that labels can have. 
    /// @return The number of labels in the dataset.
    virtual std::size_t CountLabels() const = 0;

    /// @brief Retrieves the size of the dataset, i.e., the number of entries
    ///        it contains.
    /// @return The size of the dataset.
    virtual std::size_t Size() const = 0;

    /// @brief Retrieves the number of features of the dataset. 
    /// @return The size of the dataset.
    virtual std::size_t FeatureCount() const = 0;

    /// @brief Retrieves the number of real features of the dataset. 
    /// @return The size of the dataset.
    virtual std::size_t RealFeatureCount() const = 0;

    /// @brief Retrieves the number of categorical features of the dataset. 
    /// @return The size of the dataset.
    virtual std::size_t CategoricalFeatureCount() const = 0;

    /// @brief Converts the labels to an Eigen::Vector an returns them.
    /// @return A vector containing the labels.
    virtual const Eigen::Ref<Eigen::VectorXi>& GetLabelsAsEigen() const = 0;

    /// @brief Returns a cell of real type in the dataset.
    /// @param obj The object id.
    /// @param feature The feature id.
    /// @return The cell value.
    virtual double GetReal(std::size_t obj, std::size_t feature) const = 0;

    /// @brief Returns a cell of categorical in the dataset.
    /// @param obj The object id.
    /// @param feature The feature id.
    /// @return The cell value.
    virtual std::size_t GetCategorical(std::size_t obj, 
                                       std::size_t feature) const = 0;
};

}