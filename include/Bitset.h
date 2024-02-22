#pragma once
#include <cstdint>
#ifdef PARTIAL_CONTEXT_INCIDENCE_ENTRY_USE_UNIQUE_PTR
#include <memory>
#else
#include <vector>
#endif
#include <iostream>
#include <sstream>
#include <format>
#include <algorithm>
#include <numeric>

namespace mlconcepts
{

/// @brief A bit vector representing a set of objects or features.
/// @tparam T The type of words forming the bit vector.
template <class T = uint64_t>
class Bitset {
    static constexpr size_t bits = 8 * sizeof(T);
    static constexpr size_t mask = bits - 1;
    #ifdef PARTIAL_CONTEXT_INCIDENCE_ENTRY_USE_UNIQUE_PTR
    std::unique_ptr<T[]> data;
    size_t dataSize;
    #else
    std::vector<T> data;
    #endif

protected:

    int GetWordIndex(int i) const noexcept { 
        return (i / bits); 
    }

    int GetBitIndex(int i) const noexcept { 
        return i & mask; 
    }

    inline size_t size() const noexcept {
        #ifdef PARTIAL_CONTEXT_INCIDENCE_ENTRY_USE_UNIQUE_PTR
        return dataSize;
        #else
        return data.size(); 
        #endif
    }

public:
    /// @brief Constructs an incidence entry, i.e. a bitset of a given size.
    /// @param n The size of the entry.
    Bitset(int n = 0, T initWord = 0) {
        static_assert(std::is_integral<T>::value, "The word type should be an integer type.");
        #ifdef PARTIAL_CONTEXT_INCIDENCE_ENTRY_USE_UNIQUE_PTR
        dataSize = n / bits + 1;
        data = std::make_unique<T[]>(dataSize);
        #else
        size_t size = n / bits + 1;
        data.reserve(size);
        data.resize(size);
        data.shrink_to_fit();
        #endif
        for (size_t i = 0; i < size; ++i)
            data[i] = initWord;
    }

    /// @brief Constructs an incidence entry and initializes it with some values.
    /// @param n The size of the entry.
    /// @param l The initialization list
    Bitset(int n, std::initializer_list<size_t> l) {
        static_assert(std::is_integral<T>::value, "The word type should be an integer type.");
        #ifdef PARTIAL_CONTEXT_INCIDENCE_ENTRY_USE_UNIQUE_PTR
        dataSize = n / bits + 1;
        data = std::make_unique<T[]>(dataSize);
        #else
        size_t size = n / bits + 1;
        data.reserve(size);
        data.resize(size);
        data.shrink_to_fit();
        #endif
        for (size_t i = 0; i < size; ++i)
            data[i] = 0;
        for (auto x : l) Add(x);
    }
    
    /// @brief Returns the size of the entry in words.
    /// @return The size of the entry.
    size_t WordSize() const noexcept { 
        return size();
    }

    /// @brief Inserts an element in the entry.
    /// @param i The id of the element.
    void Add(size_t i) { 
        data[i / bits] |= ((T)1 << (i & mask));  
    }

    /// @brief Gets whether an element is in the entry.
    /// @param i The id of the element.
    /// @return Whether the element is in the entry.
    bool Contains(size_t i) const { 
        return data[i / bits] & ((T)1 << (i & mask)); 
    }

    /// @brief Retrieves a word in the bit vector.
    /// @param i The id of the word.
    /// @return The word at the specified id.
    T GetWord(size_t i) const { 
        return data[i]; 
    }
    
    /// @brief Checks whether the entry is a subset of another entry.
    /// @param b The entry to test against.
    /// @return Whether the entry is a subset of b.
    bool SubsetOf(Bitset& b) const {
        for (size_t i = 0; i < size(); ++i) {
            if (data[i] != (data[i] & b.data[i]))
                return false;
        }
        return true;
    }
    
    /// @brief Intersects with another entry and stores the result in the argument.
    /// @param p The entry to intersect with as a T-array of size at least Size().
    void Intersect(T* p) const {
        for (size_t i = 0; i < size(); ++i) {
            p[i] &= data[i];
        }
    }

    /// @brief Intersects in place with another entry. Stores the result in the argument.
    /// @param p The entry to intersect with.
    void Intersect(Bitset& e) const {
        Intersect(e.data.data());
    }
    
    /// @brief Returns the number of elements in the entry.
    /// @return The number of elements in the entry.
    size_t Size() const {
        #ifdef PARTIAL_CONTEXT_INCIDENCE_ENTRY_USE_UNIQUE_PTR
        size_t v = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            v += std::popcount(data[i]);
        }
        return v;
        #else
        return std::accumulate(data.begin(), data.end(), 0, [](T a, T b){ return a + std::popcount(b); }); 
        #endif
    }
    
    /// @brief Writes the entry to a file as text.
    /// @param f The file where to write the entry.
    /// @param endline Adds a trailing newline if true.
    void WriteToFile(FILE* f = stdout, bool endline = false) const { 
        for (size_t i = 0; i < size(); ++i) 
            fprintf(f, "%016lX", data[i]); 
        if (endline) fprintf(f, "\n");  
    }

    /// @brief Writes the entry to a stream as text.
    /// @param f The stream where to write the entry.
    /// @param endline Adds a trailing newline if true.
    void WriteToStream(std::ostream& f = std::cout, bool endline = false) const { 
        for (size_t i = 0; i < size(); ++i) 
            f << std::format("{:016X}", data[i]); 
        if (endline) f << std::endl;
    }

    /// @brief Converts the entry to a string representation.
    /// @return A string representing the entry.
    std::string ToString() {
        std::ostringstream ss;
        WriteToStream(ss, false);
        return ss.str();
    }

    /// @brief Estimates the size of the entry in bytes.
    /// @return The estimated size of the entry in bytes.
    size_t EstimateSize() const {
        size_t sz = sizeof(*this);
        #ifdef PARTIAL_CONTEXT_INCIDENCE_ENTRY_USE_UNIQUE_PTR
        sz += dataSize * sizeof(T);
        #else
        sz += data.capacity() * sizeof(T);
        #endif
        return sz;
    }

    /// @brief Returns a pointer to the underlying data.
    /// @return A pointer to the data in the entry.
    T* Data() {
        #ifdef PARTIAL_CONTEXT_INCIDENCE_ENTRY_USE_UNIQUE_PTR
        return data.get();
        #else
        return data.data();
        #endif
    }


    class Iterator
    {
        size_t index; size_t size; const Bitset<T>& set;
        void next() { ++index; for (; index < size; ++index) if (set.Contains(index)) break; }
    public:
        using difference_type = size_t;
        using element_type = size_t;
        using pointer = const size_t *;
        using reference = const size_t &;
        explicit Iterator(const Bitset<T>& bset, size_t start = 0) : index(start - 1), set(bset) 
                                                        { size = set.size() * bits; next(); }
        Iterator& operator++() { next(); return *this; }
        Iterator operator++(int) { Iterator retval = *this; ++(*this); return retval; }
        bool operator==(Iterator other) const { return index == other.index; }
        bool operator!=(Iterator other) const { return !(*this == other); }
        reference operator*() const { return index; }   
    };

    Iterator begin() const { return Iterator(*this, 0); }
    Iterator end() const { return Iterator(*this, size() * bits); }

    size_t GetFirstElement() const { 
        for (size_t i = 0; i < bits * size(); ++i)
            if (Contains(i)) 
                return i;
        return SIZE_MAX;
    }

    size_t GetLastElement() const { 
        for (int i = bits * size(); i >= 0 ; --i)
            if (Contains(i)) 
                return i;
        return SIZE_MAX;
    }
};

}