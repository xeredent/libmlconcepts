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
    UniformAssigner(T _min, T _max, int _nbins) : min(_min), max(_max), nbins(_nbins) {}
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

    T GetMin() const { return min; }
    T GetMax() const { return max; }
};

/// @brief Quantizes by creating a bin for each value that a feature can take.
/// @tparam T The type of the feature.
template <class T>
class ValueAssigner : public FeatureAssigner<T> {
    size_t attrCount;
public:
    ValueAssigner(size_t _attrCount) : attrCount(_attrCount) {}
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

    /// @brief Initializes the quantizer so as to translate a train set into formal contexts.
    /// @param dataset The trainset.
    /// @param settings The settings used to generate the quantizer.
    virtual void Initialize(const Dataset& dataset, const ModelSettings& settings) = 0;
    virtual void Serialize(std::ostream& stream) const = 0;
    virtual void Deserialize(std::istream& stream) = 0;

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

    size_t GetAssignersCount() { return assigners.size(); }
};

template<class T>
class AllUniformQuantizer : public Quantizer<T> {
    static constexpr const uint64_t encodingMagicNumber = 0x71616c6c756e6966;
public:
    
    /// @brief Initializes the quantizer so as to translate a train set into formal contexts.
    /// @param dataset The trainset.
    /// @param settings The settings used to generate the quantizer.
    void Initialize(const Dataset& dataset, const ModelSettings& settings) override {
        size_t generalBins = settings.Get<size_t>("UniformBins", 32);
        for(size_t i = 0; i < dataset.RealFeatureCount(); ++i) {
            size_t bins = settings.Get<size_t>(std::format("UniformBins_{:d}", i), generalBins);;
            this->assigners.push_back(std::make_unique<UniformAssigner<T>>(dataset, i, bins));
        }
    }
    
    virtual void Serialize(std::ostream& stream) const override {
        io::LittleEndianWrite(stream, encodingMagicNumber);
        io::LittleEndianWrite(stream, (uint32_t)this->assigners.size());
        for (const auto& a : this->assigners) {
            auto assigner = static_cast<UniformAssigner<T>*>(a.get());
            io::LittleEndianWrite(stream, assigner->GetMin());
            io::LittleEndianWrite(stream, assigner->GetMax());
            io::LittleEndianWrite(stream, (uint32_t)assigner->BinsCount());
        }
    }

    virtual void Deserialize(std::istream& stream) override {
        if (io::LittleEndianRead<uint64_t>(stream) != encodingMagicNumber) 
            throw std::runtime_error("Parsing error. Invalid format: invalid magic number for all uniform quantizer.");
        uint32_t count = io::LittleEndianRead<uint32_t>(stream);
        for (size_t i = 0; i < count; ++i) {
            T min = io::LittleEndianRead<T>(stream); T max = io::LittleEndianRead<T>(stream); 
            uint32_t nbins = io::LittleEndianRead<uint32_t>(stream);
            this->assigners.push_back(std::make_unique<UniformAssigner<T>>(min, max, nbins));
        }
    }
};

template<class T>
class AllValuesQuantizer : public Quantizer<T> {
    static constexpr const uint64_t encodingMagicNumber = 0x71616c6c76616c75;
public:

    /// @brief Initializes the quantizer so as to translate a train set into formal contexts.
    /// @param dataset The trainset.
    /// @param settings The settings used to generate the quantizer.
    void Initialize(const Dataset& dataset, const ModelSettings& settings) override {
        for(size_t i = 0; i < dataset.CategoricalFeatureCount(); ++i) {
            this->assigners.push_back(std::make_unique<ValueAssigner<T>>(dataset, i + dataset.RealFeatureCount()));
        }
    }

    virtual void Serialize(std::ostream& stream) const override {
        io::LittleEndianWrite(stream, encodingMagicNumber);
        io::LittleEndianWrite(stream, (uint32_t)this->assigners.size());
        for (const auto& a : this->assigners) io::LittleEndianWrite(stream, (uint32_t)a->BinsCount());
    }

    virtual void Deserialize(std::istream& stream) override {
        if (io::LittleEndianRead<uint64_t>(stream) != encodingMagicNumber) 
            throw std::runtime_error("Parsing error. Invalid format: invalid magic number for all uniform quantizer.");
        uint32_t count = io::LittleEndianRead<uint32_t>(stream);
        for (size_t i = 0; i < count; ++i) {
            uint32_t nbins = io::LittleEndianRead<uint32_t>(stream);
            this->assigners.push_back(std::make_unique<ValueAssigner<T>>(nbins));
        }
    }
};


}