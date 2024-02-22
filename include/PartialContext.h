#pragma once
#include <numeric>
#include "Bitset.h"

namespace mlconcepts
{

/// @brief Represents a generic formal context. Satisfies the concept ClosureProvider.
/// This class of formal contexts is not optimized for any specific case.
/// @tparam T The integer type used as basic unit to build bit-sets to represent sets of objects and attributes.
template <class T = uint64_t>
class PartialContext
{
    size_t nobjs;
    size_t nattrs;
    
    /// @brief The incidence relation. The i-th vector contains the set of attributes possessed by the i-th object.
    std::vector<Bitset<T>> incidence;
    /// @brief The inverse of the incidence relation. Maps the i-th attribute to its extension.
    std::vector<Bitset<T>> incidenceConv; 

    /// @brief Computes the meet of the extension of a sets of attributes.
    /// @param result A pointer to a contiguous region of memory containing at least as many integers 
    /// of type T to store the set of objects. The result of the meet is stored here. The initial value 
    /// of this vector is intersected with all the meet.
    /// @param attrs The set of attributes whose extension is computed.
    inline void MeetExtensions(T* result, const Bitset<T>& attrs) const {
        for (size_t x = 0; x < nattrs; ++x) {
            if (attrs.Contains(x)) incidenceConv[x].Intersect(result);
        }
    }
    
public:
    /// @brief Constructs a Partial Context.
    /// @param nobjs The number of objects in the context. Cannot be zero.
    /// @param nattrs The number of attributes in the context. Cannot be zero.
    PartialContext(size_t nobjs, size_t nattrs) {
        if (nobjs == 0 || nattrs == 0) 
            throw std::runtime_error("Partial contexts cannot have empty object/attributes sets");
        this->nobjs = nobjs;
        this->nattrs = nattrs;
        for (size_t i = 0; i < nobjs; ++i) {
            incidence.push_back(Bitset<T>(nattrs));
        }
        for (size_t i = 0; i < nattrs; ++i) {
            incidenceConv.push_back(Bitset<T>(nobjs));
        }
        incidenceConv.shrink_to_fit();
        incidence.shrink_to_fit();
    }

    /// @brief Computes the extension of an attribute set. This coincides with the closure of
    /// an object when the attribute set is its intention.
    /// @param attrs The attribute set.
    /// @return The extension of the input attributes.
    Bitset<T> ComputeClosure(const Bitset<T>& attrs) const {
        Bitset<T> closure(nobjs, (T)-1);
        MeetExtensions(closure.Data(), attrs);
        return closure;
    }

    /// @brief Computes the extension of an attribute set and intersects it with a filter. 
    /// This coincides with the closure of an object when the attribute set is its intension.
    /// @param attrs The attribute set.
    /// @param filter A set which is intersected with the computed extension.
    /// @return The filtered extension of the input attributes.
    Bitset<T> ComputeFilteredClosure(const Bitset<T>& attrs,
                                                           const Bitset<T>& filter) const {
        Bitset<T> closure = std::copy(filter);
        MeetExtensions(closure.Data(), attrs);
        return closure;
    }

    /// @brief Computes the size of the extension of an attribute set. It coincides with
    /// the size of the closure of an object when the attributes are its intension.
    /// @param attrs The set of attributes.
    /// @return The size of the extension.
    size_t ComputeClosureSize(const Bitset<T>& attrs) const { 
        T closure[incidenceConv[0].WordSize()];
        for (auto& w : closure) w = (T)-1;
        MeetExtensions(closure, attrs);
        return std::accumulate(closure, closure + incidenceConv[0].WordSize(), 0,
                              [](T x, T y) { return x + std::popcount(y); } );
    }

    /// @brief Computes the closure of an object given its ID.
    /// @param objID The ID of the object.
    /// @return The size of the closure of the input object.
    size_t ComputeClosureSize(size_t objID) const { 
        return ComputeClosureSize(incidence[objID]); 
    }
    
    /// @brief Computes the size of the extension of an attribute set, after filtering it with
    /// a set of objects.
    /// @param attrs The set of attributes whose extension is computed.
    /// @param filter A set which is intersected with the computed extension.
    /// @return The size of the filtered extension of the input attributes.
    size_t ComputeFilteredClosureSize(const Bitset<T>& attrs, 
                                      const Bitset<T>& filter) const {
        T closure[incidenceConv[0].WordSize()];
        for (size_t i = 0; i < incidenceConv[0].WordSize(); ++i)
            closure[i] = filter.GetWord(i);
        MeetExtensions(closure, attrs);
        return std::accumulate(closure, closure + incidenceConv[0].WordSize(), 0,
                              [](T x, T y) { return x + std::popcount(y); } );
    }

    /// @brief Computes the size of the closure of an object, after filtering it with
    /// a set of objects.
    /// @param objID The ID of the object.
    /// @param filter A set which is intersected with the computed extension.
    /// @return The size of the filtered extension of the input attributes.
    size_t ComputeFilteredClosureSize(size_t objID, Bitset<T>& filter) const { 
        return ComputeFilteredClosureSize(incidence[objID], filter); 
    }

    /// @brief Returns the intension of an object.
    /// @param obj The ID of the object.
    /// @return The intension of the input object.
    Bitset<T>& GetIntension(size_t obj) { 
        return incidence[obj]; 
    }

    /// @brief Returns the extension of an attribute.
    /// @param obj The ID of the attribute.
    /// @return The extension of the input attribute.
    Bitset<T>& GetExtension(size_t attr) { 
        return incidenceConv[attr]; 
    }

    /// @brief Inserts a pair in the incidence relation.
    /// @param obj The ID of the object.
    /// @param attr The ID of the attribute.
    void SetIncidence(size_t obj, size_t attr) {  
        incidence[obj].Add(attr); 
        incidenceConv[attr].Add(obj); 
    }

    /// @brief Estimates the size in bytes of this object.
    /// @return The estimated size of this object in bytes.
    size_t EstimateSize() const {
        size_t sz = sizeof(PartialContext);
        for (const auto& e : incidence) sz += e.EstimateSize();
        for (const auto& e : incidenceConv) sz += e.EstimateSize();
        return sz;
    }

    void WriteToStream(std::ostream& s = std::cout, size_t entriesLimit = 10) const {
        s << "Partial Context (A, X, I), with |A| = " << nobjs << ", |X| = " << nattrs << " and" << std::endl;
        for (size_t i = 0; i < std::min(nobjs, entriesLimit); ++i) {
            s << "\tI[" << i << "] = ";
            incidence[i].WriteToStream(s, true);
        }
        if (nobjs > entriesLimit) s << "\t...";
        s << std::endl;
    }

    typedef Bitset<T> ObjectSet;
    typedef Bitset<T> AttributeSet;
};

}