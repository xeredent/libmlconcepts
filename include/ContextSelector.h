#pragma once
#include "Bitset.h"
#include "Settings.h"

namespace mlconcepts
{

/// @brief Selects which feature sets to use to generate formal contexts.
/// @tparam T The type of the word unit used in the bitset represeting the
///           generated feature sets.
template<class T = std::uint64_t>
class ContextSelector {
protected:
    const ModelSettings* settings = nullptr;
public:
    /// @brief Generates the starting feature sets used to build formal
    ///        contexts in a conceptifier.
    /// @param featureCount The number of features it is possible to select.
    /// @return A vector of feature sets.
    virtual std::vector<Bitset<T>> 
    GenerateStartingContexts(std::size_t featureCount) const {
        return std::vector<Bitset<T>>();
    }

    /// @brief Called by the conceptifier iteratively. The conceptifier starts
    ///        by generating formal contexts extracted by
    ///        GenerateStartingContexts(std::size_t).
    ///        Then, it trains the contexts on the train set and extracts
    ///        importance values for each feature set (context). This function
    ///        takes the importance values of the feature sets, and possibly
    ///        suggests new to contexts to the conceptifier. Finally, the
    ///        conceptifier generates the suggested contexts, and the model is
    ///        trained all over again, and the feature set importance values
    ///        are extracted again and evaluated by this function, until it no
    ///        longer suggests any feature set.
    /// @param featureCount The number of features it is possible to select.
    /// @param newStartingIndex The index of the first feature set added in the
    ///                         last iteration.
    /// @param featureSets A vector of feature sets used in training. New
    ///                    feature sets can be pushed to suggests new contexts.
    /// @param importance A vector of the same size of featureSets containing
    ///                   the importance value of each feature set.
    virtual void UpdateRound(std::size_t featureCount, 
                             std::size_t newStartingIndex,
                             std::vector<Bitset<T>>& featureSets, 
                             const std::vector<double>& importance) const {
        
    }

    void SetSettings(const ModelSettings& s) { settings = &s; }

    typedef Bitset<T> FeatureSet;
};

/// @brief Selects which feature sets to use to generate formal contexts.
///        Only generates singleton contexts, and performs no update round.
/// @tparam T The type of the word unit used in the bitset represeting the
///           generated feature sets.
template<class T = std::uint64_t>
class SimpleSingletonContextSelector : public ContextSelector<T> {
public:
    /// @brief Generates the starting feature sets used to build formal
    ///        contexts in a conceptifier.
    /// @param featureCount The number of features it is possible to select.
    /// @return A vector of feature sets.
    virtual std::vector<Bitset<T>> 
    GenerateStartingContexts(std::size_t featureCount) const override {
        std::vector<Bitset<T>> v;
        for (std::size_t i = 0; i < featureCount; ++i) 
            v.push_back(Bitset<T>(featureCount, {i}));
        return v;
    }
};

/// @brief Selects which feature sets to use to generate formal contexts.
///        Only generates doubleton contexts, and performs no update round.
/// @tparam T The type of the word unit used in the bitset represeting the
///           generated feature sets.
template<class T = std::uint64_t>
class SimpleDoubletonContextSelector : public ContextSelector<T> {
public:
    /// @brief Generates the starting feature sets used to build formal
    ///        contexts in a conceptifier.
    /// @param featureCount The number of features it is possible to select.
    /// @return A vector of feature sets.
    virtual std::vector<Bitset<T>> 
    GenerateStartingContexts(std::size_t featureCount) const override {
        std::vector<Bitset<T>> v;
        for (std::size_t i = 0; i < featureCount; ++i) 
            for (std::size_t j = i; j < featureCount; ++j)
                v.push_back(Bitset<T>(featureCount, {i, j}));
        return v;
    }
};

/// @brief Selects which feature sets to use to generate formal contexts.
///        The behaviour of this generator depends on the settings inserted
///        by the user.
/// @tparam T The type of the word unit used in the bitset represeting the
///           generated feature sets.
template<class T = std::uint64_t>
class ConfigurableContextSelector : public ContextSelector<T> {
public:
    /// @brief Generates the starting feature sets used to build formal
    ///        contexts in a conceptifier.
    /// @param featureCount The number of features it is possible to select.
    /// @return A vector of feature sets.
    virtual std::vector<Bitset<T>> 
    GenerateStartingContexts(std::size_t featureCount) const override {
        bool doubletons = this->settings->
                          template Get<bool>("GenerateDoubletons", true);
        bool singletons = this->settings->
                          template Get<bool>("GenerateSingletons", true);
        bool full = this->settings->template Get<bool>("GenerateFull", true);
        std::vector<Bitset<T>> v;
        if (doubletons) {
            for (std::size_t i = 0; i < featureCount; ++i) 
                for (std::size_t j = i; j < featureCount; ++j)
                    v.push_back(Bitset<T>(featureCount, {i, j}));
        }
        else if (singletons) {
            for (std::size_t i = 0; i < featureCount; ++i)
                v.push_back(Bitset<T>(featureCount, {i}));
        }
        if (full) {
            auto b = Bitset<T>(featureCount);
            for (std::size_t i = 0; i < featureCount; ++i) b.Add(i);
            v.push_back(std::move(b)); 
        }
        return v;
    }
};

}