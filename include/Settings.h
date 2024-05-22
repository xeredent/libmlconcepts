#pragma once
#include <sstream>
#include <unordered_map>

namespace mlconcepts
{

/// @brief Extremely simple settings class which supports all the types for
///        which the operators >> and << with std::istream and std::ostream
///        have been defined.
class ModelSettings {
    std::unordered_map<std::string, std::string> map;
public:

    /// @brief Sets a parameters in the settings.
    /// @tparam T The type of the parameter.
    /// @param name The name of the parameter.
    /// @param value The value to set.
    template <class T>
    void Set(const std::string& name, const T& value) {
        std::ostringstream ss;
        ss << value;
        map[name] = ss.str();
    }
    
    /// @brief Gets the value of one of the parameters in the settings.
    /// @tparam T The type of the parameter.
    /// @param name The name of the parameter.
    /// @param defaultValue The default value to return if the parameter is not
    ///                     found.
    /// @return The value of the parameter.
    template <class T> 
    T Get(const std::string& name, const T& defaultValue) const {
        if (map.count(name) == 0) return defaultValue;
        const auto& s = map.at(name);
        T toReturn;
        std::istringstream ss(s); 
        ss >> toReturn;
        return toReturn;
    }
};

template<>
bool ModelSettings::Get(const std::string& name, 
                        const bool& defaultValue) const {
    if (map.count(name) == 0) return defaultValue;
    const auto& s = map.at(name);
    return s == "TRUE" || s == "true" || s == "True" || s == "1";
}

}