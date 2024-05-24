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
    std::size_t offset;
public:
    virtual ~FeatureAssigner() { }

    /// @brief Sets an offset applied to each bin produced by the quantizer.
    /// @param off The offset to set.
    void SetOffset(std::size_t off) { offset = off; }

    /// @brief Retrieves the amount of bins the assigner uses.
    /// @return The number of bins.
    virtual std::size_t BinsCount() const = 0;

    /// @brief Assigns an element of the dataset to a bin.
    /// @param v The value of the element.
    /// @return The bin assigned to the value.
    virtual int Assign(T v) const = 0;

    /// @brief Assigns an element of the dataset to a bin, and makes sure that
    ///        the bin is in the range [offset, BinsCount()[.
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
    /// @brief Constructs a feature assigner that uniformly breaks a range into
    ///        equally spaced intervals.
    /// @param _min The infimum of the range.
    /// @param _max The supremum of the range.
    /// @param _nbins The number of bins the range is split into.
    UniformAssigner(T _min, T _max, int _nbins) : 
                   min(_min), max(_max), nbins(_nbins) {}
    
    /// @brief Constructs a feature assigner that uniformly breaks a range into
    ///        equally spaced intervals.
    /// @param dataset The dataset the data is read from.
    /// @param feature The feature which is used to compute the splitted range.
    /// @param nbins The number of bins the range is split into.
    UniformAssigner(const Dataset& dataset, 
                    std::size_t feature, 
                    int nbins) {
        min = dataset.MinReal(feature);
        max = dataset.MaxReal(feature);
        this->nbins = nbins;
    }

    /// @brief Retrieves the amount of bins the assigner uses.
    /// @return The number of bins.
    std::size_t BinsCount() const override { return (std::size_t)nbins; }

    /// @brief Assigns an element of the dataset to a bin.
    /// @param v The value of the element.
    /// @return The bin assigned to the value.
    int Assign(T v) const override {
        int id = ((v - min) / (max - min)) * nbins;
        if (id >= nbins) id = nbins - 1;
        if (id < 0) return this->offset;
        return this->offset + id;
    }

    /// @brief Returns the minimum value that the feature takes.
    /// @return Returns the minimum value that the feature takes.
    T GetMin() const { return min; }

    /// @brief Returns the maximum value that the feature takes.
    /// @return Returns the maximum value that the feature takes.
    T GetMax() const { return max; }
};

/// @brief Quantizes by creating a bin for each value that a feature can take.
/// @tparam T The type of the feature.
template <class T>
class ValueAssigner : public FeatureAssigner<T> {
    std::size_t attrCount;
public:
    ValueAssigner(std::size_t _attrCount) : attrCount(_attrCount) {}
    ValueAssigner(const Dataset& dataset, std::size_t feature) { 
        attrCount = dataset.CountCategorical(feature);
    }

    /// @brief Retrieves the amount of bins the assigner uses.
    /// @return The number of bins.
    std::size_t BinsCount() const override { return attrCount; }

    /// @brief Assigns an element of the dataset to a bin.
    /// @param v The value of the element.
    /// @return The bin assigned to the value.
    int Assign(T v) const override { return this->offset + v; }
};

/// @brief Given a dataset, creates feature assigners, so as to be able to
///        quantize any entry in the dataset.
/// @tparam T The type of the features it supports.
template<class T>
class Quantizer {
protected:
    std::vector<std::unique_ptr<FeatureAssigner<T>>> assigners;

public:

    /// @brief Initializes the quantizer so as to translate a train set into
    ///        formal contexts.
    /// @param dataset The trainset.
    /// @param settings The settings used to generate the quantizer.
    virtual void Initialize(const Dataset& dataset, 
                            const ModelSettings& settings) = 0;

    /// @brief Writes the quantizer state to a stream.
    /// @param stream The stream the quantizer state is written to.
    virtual void Serialize(std::ostream& stream) const = 0;

    /// @brief Loads a quantizer state from a stream.
    /// @param stream The stream the quantizer state is read from.
    virtual void Deserialize(std::istream& stream) = 0;

    /// @brief Retrieves the feature assigner for a given feature.
    /// @param id The id of the feature.
    /// @return The feature assigner for the given feature.
    FeatureAssigner<T>& GetAssigner(std::size_t id) {
        return *assigners[id].get();
    }

    /// @brief Computes the number of bins across all the (real) features.
    /// @return The sum of the number of bins of all feature assigners.
    std::size_t GetTotalBins() const {
        std::size_t sz = 0;
        for (const auto& a : assigners) {
            sz += a->BinsCount();
        }
        return sz;
    }

    /// @brief Returns the number of feature assigners of the quantizer.
    /// @return The number of feature assigners.
    std::size_t GetAssignersCount() { return assigners.size(); }
};

/// @brief Quantizer that handles all features using uniform assigners.
/// @tparam T The type of the supported features.
template<class T>
class AllUniformQuantizer : public Quantizer<T> {

    /// @brief Used to mark serialized files.
    static constexpr const std::uint64_t 
    encodingMagicNumber = 0x71616c6c756e6966;

public:
    
    /// @brief Initializes the quantizer so as to translate a train set into
    ///        formal contexts.
    /// @param dataset The trainset.
    /// @param settings The settings used to generate the quantizer.
    void Initialize(const Dataset& dataset, 
                    const ModelSettings& settings) override {
        std::size_t generalBins = settings.Get<std::size_t>("UniformBins", 32);
        for(std::size_t i = 0; i < dataset.RealFeatureCount(); ++i) {
            std::size_t bins = settings.Get<std::size_t>(
                ("UniformBins_" + std::to_string(i)),
                generalBins
            );
            this->assigners.push_back(
                std::make_unique<UniformAssigner<T>>(dataset, i, bins)
            );
        }
    }
    
    /// @brief Writes the quantizer state to a stream.
    /// @param stream The stream the quantizer state is written to.
    virtual void Serialize(std::ostream& stream) const override {
        io::LittleEndianWrite(stream, encodingMagicNumber);
        io::LittleEndianWrite(stream, (std::uint32_t)this->assigners.size());
        for (const auto& a : this->assigners) {
            auto assigner = static_cast<UniformAssigner<T>*>(a.get());
            io::LittleEndianWrite(stream, assigner->GetMin());
            io::LittleEndianWrite(stream, assigner->GetMax());
            io::LittleEndianWrite(stream, (std::uint32_t)assigner->BinsCount());
        }
    }

    /// @brief Loads a quantizer state from a stream.
    /// @param stream The stream the quantizer state is read from.
    virtual void Deserialize(std::istream& stream) override {
        if (io::LittleEndianRead<std::uint64_t>(stream) != encodingMagicNumber)
            throw std::runtime_error(
                "Parsing error. Invalid format: invalid magic number for " 
                " all uniform quantizer."
            );
        std::uint32_t count = io::LittleEndianRead<std::uint32_t>(stream);
        for (std::size_t i = 0; i < count; ++i) {
            T min = io::LittleEndianRead<T>(stream); 
            T max = io::LittleEndianRead<T>(stream); 
            std::uint32_t nbins = io::LittleEndianRead<std::uint32_t>(stream);
            this->assigners.push_back(
                std::make_unique<UniformAssigner<T>>(min, max, nbins)
            );
        }
    }
};

template<class T>
class AllValuesQuantizer : public Quantizer<T> {

    /// @brief Used to mark serialized files.
    static constexpr const std::uint64_t 
    encodingMagicNumber = 0x71616c6c76616c75;

public:

    /// @brief Initializes the quantizer so as to translate a train set into
    ///        formal contexts.
    /// @param dataset The trainset.
    /// @param settings The settings used to generate the quantizer.
    void Initialize(const Dataset& dataset, 
                    const ModelSettings& settings) override {
        for(std::size_t i = 0; i < dataset.CategoricalFeatureCount(); ++i) {
            this->assigners.push_back(
                std::make_unique<ValueAssigner<T>>(
                    dataset, 
                    i + dataset.RealFeatureCount()
                )
            );
        }
    }

    /// @brief Writes the quantizer state to a stream.
    /// @param stream The stream the quantizer state is written to.
    virtual void Serialize(std::ostream& stream) const override {
        io::LittleEndianWrite(stream, encodingMagicNumber);
        io::LittleEndianWrite(stream, (std::uint32_t)this->assigners.size());
        for (const auto& a : this->assigners) 
            io::LittleEndianWrite(stream, (std::uint32_t)a->BinsCount());
    }

    /// @brief Loads a quantizer state from a stream.
    /// @param stream The stream the quantizer state is read from.
    virtual void Deserialize(std::istream& stream) override {
        if (io::LittleEndianRead<std::uint64_t>(stream) != encodingMagicNumber) 
            throw std::runtime_error(
                "Parsing error. Invalid format: invalid magic number for all " 
                " uniform quantizer."
            );
        std::uint32_t count = io::LittleEndianRead<std::uint32_t>(stream);
        for (std::size_t i = 0; i < count; ++i) {
            std::uint32_t nbins = io::LittleEndianRead<std::uint32_t>(stream);
            this->assigners.push_back(
                std::make_unique<ValueAssigner<T>>(nbins)
            );
        }
    }
};


}