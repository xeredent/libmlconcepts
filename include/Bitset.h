#pragma once
#include <cstdint>
#ifdef PARTIAL_CONTEXT_INCIDENCE_ENTRY_USE_UNIQUE_PTR
#include <memory>
#else
#include <vector>
#endif
#include <iostream>
#include <sstream>
#include <ranges>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include "Bitstream.h"

namespace mlconcepts
{

/// @brief A bit vector representing a set of objects or features.
/// @tparam T The type of words forming the bit vector.
template <class T = std::uint64_t>
class Bitset {
    static constexpr std::size_t bits = 8 * sizeof(T);
    static constexpr std::size_t mask = bits - 1;
    #ifdef PARTIAL_CONTEXT_INCIDENCE_ENTRY_USE_UNIQUE_PTR
    std::unique_ptr<T[]> data;
    std::size_t dataSize;
    #else
    std::vector<T> data;
    #endif

protected:

    inline int GetWordIndex(int i) const noexcept { 
        return (i / bits); 
    }

    inline int GetBitIndex(int i) const noexcept { 
        return i & mask; 
    }

    inline std::size_t size() const noexcept {
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
        static_assert(std::is_integral<T>::value, 
                      "The word type should be an integer type.");
        #ifdef PARTIAL_CONTEXT_INCIDENCE_ENTRY_USE_UNIQUE_PTR
        dataSize = n / bits + 1;
        data = std::make_unique<T[]>(dataSize);
        #else
        std::size_t size = n / bits + (n % bits != 0 ? 1 : 0);
        data.reserve(size);
        data.resize(size);
        data.shrink_to_fit();
        #endif
        for (std::size_t i = 0; i < size; ++i)
            data[i] = initWord;
    }

    /// @brief Constructs an incidence entry and initializes it with some values.
    /// @param n The size of the entry.
    /// @param l The initialization list
    Bitset(int n, std::initializer_list<std::size_t> l) {
        static_assert(std::is_integral<T>::value, 
                      "The word type should be an integer type.");
        #ifdef PARTIAL_CONTEXT_INCIDENCE_ENTRY_USE_UNIQUE_PTR
        dataSize = n / bits + 1;
        data = std::make_unique<T[]>(dataSize);
        #else
        std::size_t size = n / bits + (n % bits != 0 ? 1 : 0);
        data.reserve(size);
        data.resize(size);
        data.shrink_to_fit();
        #endif
        for (std::size_t i = 0; i < size; ++i)
            data[i] = 0;
        for (auto x : l) Add(x);
    }

    /// @brief Constructs an incidence entry given a list of words.
    /// @param l The list of words.
    Bitset(std::initializer_list<T> l) {
        #ifdef PARTIAL_CONTEXT_INCIDENCE_ENTRY_USE_UNIQUE_PTR
        throw std::error("initializer list initialization is not "
                         " allowed in unique_ptr mode")
        #else
        for (auto x : l) data.push_back(x);
        data.shrink_to_fit();
        #endif
    }
    
    /// @brief Returns the size of the entry in words.
    /// @return The size of the entry.
    std::size_t WordSize() const noexcept { 
        return size();
    }

    /// @brief Inserts an element in the entry.
    /// @param i The id of the element.
    void Add(std::size_t i) { 
        data[i / bits] |= ((T)1 << (i & mask));  
    }

    /// @brief Gets whether an element is in the entry.
    /// @param i The id of the element.
    /// @return Whether the element is in the entry.
    bool Contains(std::size_t i) const { 
        return data[i / bits] & ((T)1 << (i & mask)); 
    }

    /// @brief Retrieves a word in the bit vector.
    /// @param i The id of the word.
    /// @return The word at the specified id.
    T GetWord(std::size_t i) const { 
        return data[i]; 
    }
    
    /// @brief Checks whether the entry is a subset of another entry.
    /// @param b The entry to test against.
    /// @return Whether the entry is a subset of b.
    bool SubsetOf(Bitset& b) const {
        for (std::size_t i = 0; i < size(); ++i) {
            if (data[i] != (data[i] & b.data[i]))
                return false;
        }
        return true;
    }
    
    /// @brief Intersects with another entry and stores the result in the
    ///        argument.
    /// @param p The entry to intersect with as a T-array of size at least
    ///          Size().
    void Intersect(T* p) const {
        for (std::size_t i = 0; i < size(); ++i) {
            p[i] &= data[i];
        }
    }

    /// @brief Intersects in place with another entry.
    ///        Stores the result in the argument.
    /// @param p The entry to intersect with.
    void Intersect(Bitset& e) const {
        Intersect(e.data.data());
    }
    
    /// @brief Returns the number of elements in the entry.
    /// @return The number of elements in the entry.
    std::size_t Size() const {
        #ifdef PARTIAL_CONTEXT_INCIDENCE_ENTRY_USE_UNIQUE_PTR
        std::size_t v = 0;
        for (std::size_t i = 0; i < data.size(); ++i) {
            v += std::popcount(data[i]);
        }
        return v;
        #else
        return std::accumulate(
            data.begin(), data.end(), 0, 
            [](T a, T b) { return a + std::popcount(b); }
        );
        #endif
    }
    
    /// @brief Writes the entry to a file as text.
    /// @param f The file where to write the entry.
    /// @param endline Adds a trailing newline if true.
    void WriteToFile(FILE* f = stdout, bool endline = false) const { 
        for (auto x : std::ranges::views::reverse(data)) 
            fprintf(f, "%016X", x); 
        if (endline) fprintf(f, "\n");  
    }

    /// @brief Writes the entry to a stream as text.
    /// @param f The stream where to write the entry.
    /// @param endline Adds a trailing newline if true.
    void WriteToStream(std::ostream& f = std::cout,
                       bool endline = false) const { 
        auto prevFlags = f.flags();
        for (auto x : std::ranges::views::reverse(data)) 
            f << std::hex << std::setw(16) << std::setfill('0') << x;
        if (endline) f << std::endl;
        f.flags(prevFlags);
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
    std::size_t EstimateSize() const {
        std::size_t sz = sizeof(*this);
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

    /// @brief Returns a word in the underlying bit vector representation.
    /// @param i The index of the word.
    /// @return The word at index i.
    T WordAt(std::size_t i) const { return data[i]; }

    /// @brief Iterator that goes through the elements of a bitset.
    class Iterator
    {
        std::size_t index; std::size_t size; const Bitset<T>& set;
        void next() { 
            ++index; 
            for (; index < size; ++index) {
                if (set.Contains(index)) break; 
            }
        }
    public:
        using difference_type = std::size_t;
        using element_type = std::size_t;
        using pointer = const std::size_t *;
        using reference = const std::size_t &;
        explicit Iterator(const Bitset<T>& bset, std::size_t start = 0) : 
                         index(start - 1), set(bset) { 
            size = set.size() * bits; next(); 
        }
        Iterator& operator++() { next(); return *this; }
        Iterator operator++(int) { 
            Iterator retval = *this; ++(*this); 
            return retval; 
        }
        bool operator==(Iterator other) const { return index == other.index; }
        bool operator!=(Iterator other) const { return !(*this == other); }
        reference operator*() const { return index; }   
    };

    Iterator begin() const { return Iterator(*this, 0); }
    Iterator end() const { return Iterator(*this, size() * bits); }

    /// @brief Converts the bitset to a vector of indices.
    /// @return A vector containing the indices of all the elements in the set.
    std::vector<std::size_t> ToIndexVector() const {
        std::vector<std::size_t> ret;
        for (auto x : *this) ret.push_back(x);
        return ret;
    }

protected:

    /// @brief Counts the number of zero bits from a given bit in a given word.
    ///        The input parameters are changed to reflect the position of the
    ///        first bit containing one after the sequence.
    /// @param wordID The word where to start counting.
    /// @param startingBitID The bit where to start counting.
    /// @return The size of the sequence of bits 0.
    std::size_t CountRLEBitsZero(std::size_t* wordID,
                                 std::size_t* startingBitID) const {
        //Count the 0 bits in the current word
        std::size_t sequence = std::min(
            (int)bits - (int)*startingBitID, 
            std::countr_zero(
                static_cast<typename std::make_unsigned<T>::type>(data[*wordID]) >>
                *startingBitID
            )
        );
        //if zeroes continue up to the end of the word
        if (sequence + *startingBitID == bits) {
            //count all the zero words
            for (++*wordID; *wordID < size() && data[*wordID] == 0; ++*wordID) {
                sequence += bits;
            }
            //count bits in the last word if it contains also ones
            if (*wordID < size()) {
                // updates the starting bit to the first one and add
                // it to the sequence
                *startingBitID = std::countr_zero(data[*wordID]);
                sequence += *startingBitID;
            }
        } else *startingBitID += sequence;
        return sequence;
    }

    /// @brief Counts the number of one bits from a given bit in a given word.
    ///        The input parameters are changed to reflect the position of the
    ///        first bit containing zero after the sequence.
    /// @param wordID The word where to start counting.
    /// @param startingBitID The bit where to start counting.
    /// @return The size of the sequence of bits 1.
    std::size_t CountRLEBitsOne(std::size_t* wordID,
                                std::size_t* startingBitID) const {
        //Count the 1 bits in the current word
        std::size_t sequence = std::min(
            (int)bits - (int)*startingBitID, 
            std::countr_one(
                static_cast<typename std::make_unsigned<T>::type>(data[*wordID]) >> 
                *startingBitID
            )
        );
        //if ones continue up to the end of the word
        if (sequence + *startingBitID == bits) {
            // count all the ~0 words
            for (++*wordID; *wordID < size() && data[*wordID] == ~(T)0; ++*wordID) {
                sequence += bits;
            }
            // count bits in the last word if it contains also zeroes
            if (*wordID < size()) {
                // updates the starting bit to the first one and add
                // it to the sequence
                *startingBitID = std::countr_one(data[*wordID]);
                sequence += *startingBitID;
            }
        } else *startingBitID += sequence;
        return sequence;
    }

    /// @brief Encodes a sequence of bits using RLE. It essentially writes the
    ///        length of the sequence (as the only possible values are zero
    ///        and one). Given a bitcount for the smallest sequence encoded,
    ///        every bitcount-integer between [1,2^bitcount - 1] encodes its
    ///        value as length, and 0 encodes the length 2^bitcount - 1 plus
    ///        the value encoded by the following bitcount-integer in the
    ///        stream.
    /// @tparam bitcount The number of bits of the smallest integer encoding
    ///                  length.
    /// @param stream The stream where to write the sequence.
    /// @param sequence The size of the sequence.
    template<std::uint8_t bitcount>
    void WriteRLESequence(OutputBitstream& stream, 
                          std::size_t sequence) const {
        const int maxvalue = std::pow(2, bitcount) - 1;
        while (sequence > maxvalue) {
            sequence -= std::min(sequence, (std::size_t)maxvalue);
            stream.WriteBits<bitcount>(0);
        } 
        stream.WriteBits<bitcount>(sequence);
    }

    /// @brief Decodes a sequence of bits encoded via RLE.
    /// @tparam bitcount The number of bits of the smallest integer encoding
    ///                  length.
    /// @param stream The stream the sequence is read from.
    /// @return The length of the sequence.
    template<std::uint8_t bitcount>
    std::size_t ReadRLESequence(InputBitstream& stream) {
        const int maxvalue = std::pow(2, bitcount) - 1;
        std::size_t count = 0;
        auto data = stream.ReadBits<bitcount>();
        while (data == 0) {
            data = stream.ReadBits<bitcount>();
            count += maxvalue;
        }
        return data + count;
    }

    /// @brief Sets a sequence of bit in the set to zero.
    /// @param wordID The word where the starting bit is found. Changes the
    ///               input parameter to the word where the sequence ends.
    /// @param bitID The bit where the sequence starts. Changes the input
    ///              parameter to the bit id where the sequence ends.
    /// @param sequence The sequence length.
    void ApplyRLESequenceToDataZero(std::size_t* wordID,
                                    std::size_t* bitID,
                                    std::size_t sequence) {
        if (sequence >= bits - *bitID) {
            data[*wordID] &= ((T)1 << *bitID) - 1;
            sequence -= bits - *bitID;
            ++*wordID;
            while (sequence >= bits) {
                data[*wordID] = 0;
                ++*wordID;
                sequence -= bits;
            }
            *bitID = sequence;
            if (sequence > 0) data[*wordID] = 0;
        } else {
            data[*wordID] &= ((T)1 << *bitID) - 1;
            *bitID += sequence;
        }
    }

    /// @brief Sets a sequence of bit in the set to one.
    /// @param wordID A pointer word where the starting bit is found. Changes
    ///               the input parameter to the word where the sequence ends.
    /// @param bitID A pointer to the bit where the sequence starts. Changes
    ///              the input parameter to the bit id where the sequence ends.
    /// @param sequence The sequence length.
    void ApplyRLESequenceToDataOne(std::size_t* wordID,
                                   std::size_t* bitID,
                                   std::size_t sequence) {
        if (sequence >= bits - *bitID) {
            data[*wordID] |= ~(((T)1 << *bitID) - 1);
            sequence -= bits - *bitID;
            ++*wordID;
            while (sequence >= bits) {
                data[*wordID] = ~(T)0;
                ++*wordID;
                sequence -= bits;
            }
            *bitID = sequence;
            if (sequence > 0) data[*wordID] = ((T)1 << *bitID) - 1;
        } else {
            data[*wordID] = (~data[*wordID] & (((T)1 << *bitID) - 1)) ^ 
                            (((T)1 << (*bitID + sequence)) - 1);
            *bitID += sequence;
        }
    }

    

public:

    /// @brief Serializes the bitset using run length encoding.
    /// @param stream The stream where to write the bitset.
    /// @tparam zerobitunit The word size (in bits) that describes the smallest
    ///                     zero-sequence.
    /// @tparam onebitunit The word size (in bits) that describes the smallest
    ///                    one-sequence.
    template<int zerobitunit, int onebitunit>
    void SerializeRLE(std::ostream& stream) const {
        std::size_t wordID = 0, bitID = 0;
        OutputBitstream bstream(stream);
        //Checks whether to start with a sequence of ones
        bool phaseOne = data[0] & 1;
        //writes down whether the sequence starts with 0 or 1
        bstream.WriteBits<1>((phaseOne) ? 1 : 0);
        while (wordID < size() && bitID < bits) {
            if (phaseOne) {
                WriteRLESequence<onebitunit>(bstream, 
                                             CountRLEBitsOne(&wordID, &bitID));
            }
            else {
                WriteRLESequence<zerobitunit>(bstream,
                                              CountRLEBitsZero(&wordID, &bitID));
            }
            phaseOne = !phaseOne;
        }
        bstream.Close();
    }

    /// @brief Parses a run length encoded bitset from a stream. Assumes that
    ///        the size of the bitset is already known in advance and was 
    ///        correctly passed to the constructor of this object.
    /// @param stream The stream the bitset is decoded from.
    /// @tparam zerobitunit The word size (in bits) that describes the smallest
    ///                     zero-sequence.
    /// @tparam onebitunit The word size (in bits) that describes the smallest
    ///                    one-sequence.
    template<int zerobitunit, int onebitunit>
    void DeserializeRLE(std::istream& stream) {
        InputBitstream bstream(stream);
        //Gets whether to start with a sequence of RLE-encoded ones or zeroes
        bool phaseOne = bstream.ReadBits<1>() == 1;
        std::size_t readBits = 0, sequenceLength = 0, wordID = 0, bitID = 0;
        do {
            if (phaseOne) {
                sequenceLength = ReadRLESequence<onebitunit>(bstream);
                if (readBits + sequenceLength > bits * size()) {
                    throw std::runtime_error(
                        "RLE decoding error: the decoded sequence " 
                        " is bigger than expected"
                    );
                }
                ApplyRLESequenceToDataOne(&wordID, &bitID, sequenceLength);
            } else {
                sequenceLength = ReadRLESequence<zerobitunit>(bstream);
                if (readBits + sequenceLength > bits * size()) {
                    throw std::runtime_error(
                        "RLE decoding error: the decoded sequence "
                        " is bigger than expected"
                    );
                }
                ApplyRLESequenceToDataZero(&wordID, &bitID, sequenceLength);
            }
            phaseOne = !phaseOne;
            readBits += sequenceLength;
        } while (readBits < bits * size());
    }

    /// @brief Serializes the bitset in a stream.
    /// @param stream The stream where to write the bitset.
    void Serialize(std::ostream& stream) const { SerializeRLE<5, 2>(stream); }

    /// @brief Parses a run length encoded bitset from a stream. Assumes that
    ///        the size of the bitset is already known in advance and was
    ///        correctly passed to the constructor of this object.
    /// @param stream The stream the bitset is decoded from.
    void Deserialize(std::istream& stream) { DeserializeRLE<5, 2>(stream); }
};

template<class T>
bool operator==(const Bitset<T>& b1, const Bitset<T>& b2) {
    if (b1.WordSize() != b2.WordSize()) return false;
    for (std::size_t i = 0; i < b1.WordSize(); ++i) 
        if (b1.WordAt(i) != b2.WordAt(i)) return false;
    return true;
}

template<class T>
bool operator!=(const Bitset<T>& b1, const Bitset<T>& b2) {
    if (b1.WordSize() != b2.WordSize()) return true;
    for (std::size_t i = 0; i < b1.WordSize(); ++i) 
        if (b1.WordAt(i) != b2.WordAt(i)) return true;
    return false;
}

template<class T>
bool operator<=(const Bitset<T>& b1, const Bitset<T>& b2) {
    if (b1.WordSize() != b2.WordSize()) return false;
    return b1.SubsetOf(b2);
}

template<class T>
bool operator<(const Bitset<T>& b1, const Bitset<T>& b2) {
    return b1 <= b2 && b1 != b2;
}

template<class T>
bool operator>=(const Bitset<T>& b1, const Bitset<T>& b2) {
    return b2 <= b1;
}

template<class T>
bool operator>(const Bitset<T>& b1, const Bitset<T>& b2) {
    return b2 < b1;
}

}