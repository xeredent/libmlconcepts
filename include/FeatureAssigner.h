#pragma once
#include <Dataset.h>
#include <memory>
#include "Settings.h"

namespace mlconcepts
{

/// @brief Quantizes a feature into bins.
/// @tparam T The type of the feature.
template <class T>
class FeatureAssigner {
protected:
    size_t offset;
public:
    /// @brief Sets an offset applied to each bin produced by the quantizer.
    /// @param off The offset to set.
    void SetOffset(size_t off) { offset = off; }

    /// @brief Retrieves the amount of bins the assigner uses.
    /// @return The number of bins.
    virtual size_t BinsCount() const = 0;

    /// @brief Assigns an element of the dataset to a bin.
    /// @param v The value of the element.
    /// @return The bin assigned to the value.
    virtual int Assign(T v) const = 0;

    /// @brief Assigns an element of the dataset to a bin, and makes sure that the bin is in the range
    /// [offset, BinsCount()[.
    /// @param v The value of the element.
    /// @return The bin assigned to the value.
    int SafeAssign(T v) const {
        auto b = Assign(v);
        if (b < offset) return offset;
        if (b > offset + BinsCount() - 1) return offset + BinsCount() - 1;
        return b;
    }
};

/// @brief Divides a feature into bins distributed uniformly.
/// @tparam T The type of the feature.
template <class T>
class UniformAssigner : public FeatureAssigner<T> {
    T min;
    T max;
    int nbins;
public:
    UniformAssigner(const Dataset& dataset, size_t feature, int nbins) {
        min = dataset.MinReal(feature);
        max = dataset.MaxReal(feature);
        this->nbins = nbins;
    }

    /// @brief Retrieves the amount of bins the assigner uses.
    /// @return The number of bins.
    size_t BinsCount() const override { return (size_t)nbins; }

    /// @brief Assigns an element of the dataset to a bin.
    /// @param v The value of the element.
    /// @return The bin assigned to the value.
    int Assign(T v) const override {
        int id = ((v - min) / (max - min)) * nbins;
        if (id >= nbins) id = nbins - 1;
        if (id < 0) return this->offset;
        return this->offset + id;
    }

};

/// @brief Quantizes by creating a bin for each value that a feature can take.
/// @tparam T The type of the feature.
template <class T>
class ValueAssigner : public FeatureAssigner<T> {
    size_t attrCount;
public:
    ValueAssigner(const Dataset& dataset, size_t feature) { 
        attrCount = dataset.CountCategorical(feature);
    }

    /// @brief Retrieves the amount of bins the assigner uses.
    /// @return The number of bins.
    size_t BinsCount() const override { return attrCount; }

    /// @brief Assigns an element of the dataset to a bin.
    /// @param v The value of the element.
    /// @return The bin assigned to the value.
    int Assign(T v) const override { return this->offset + v; }
};

/// @brief Given a dataset, creates feature assigners, so as to be able to quantize any entry in the dataset.
/// @tparam T The type of the features it supports.
template<class T>
class Quantizer {
protected:
    std::vector<std::unique_ptr<FeatureAssigner<T>>> assigners;

public:

    virtual void Initialize(const Dataset& dataset, const ModelSettings& settings) = 0;

    FeatureAssigner<T>& GetAssigner(size_t id) {
        return *assigners[id].get();
    }

    size_t GetTotalBins() const {
        size_t sz = 0;
        for (const auto& a : assigners) {
            sz += a->BinsCount();
        }
        return sz;
    }
};

template<class T>
class AllUniformQuantizer : public Quantizer<T> {
public:
    void Initialize(const Dataset& dataset, const ModelSettings& settings) override {
        size_t generalBins = settings.Get<size_t>("UniformBins", 32);
        for(size_t i = 0; i < dataset.RealFeatureCount(); ++i) {
            size_t bins = settings.Get<size_t>(std::format("UniformBins_{:d}", i), generalBins);;
            this->assigners.push_back(std::make_unique<UniformAssigner<T>>(dataset, i, bins));
        }
    }
};

template<class T>
class AllValuesQuantizer : public Quantizer<T> {
public:
    void Initialize(const Dataset& dataset, const ModelSettings& settings) override {
        for(size_t i = 0; i < dataset.CategoricalFeatureCount(); ++i) {
            this->assigners.push_back(std::make_unique<ValueAssigner<T>>(dataset, i + dataset.RealFeatureCount()));
        }
    }
};


}