#pragma once
#include <sstream>
#include <unordered_map>

namespace mlconcepts
{

class ModelSettings {
    std::unordered_map<std::string, std::string> map;
public:

    template <class T>
    void Set(const std::string& name, const T& value) {
        std::ostringstream ss;
        ss << value;
        map[name] = ss.str();
    }
    
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
bool ModelSettings::Get(const std::string& name, const bool& defaultValue) const {
    if (map.count(name) == 0) return defaultValue;
    const auto& s = map.at(name);
    return s == "TRUE" || s == "true" || s == "True" || s == "1";
}

}