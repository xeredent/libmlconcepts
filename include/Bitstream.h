#pragma once
#include <bit>
#include <iostream>
#include <cstdint>

namespace mlconcepts {

/// @brief Gives tools to write any bit-size integers in a typical binary
///        output stream.
class OutputBitstream {
    /// @brief The stream where to write data
    std::ostream& stream;

    /// @brief The bits which have not been written to the stream yet.
    std::uint8_t remainder;

    /// @brief The amount of pending bits which are waiting for more data to
    ///        come before being written.
    std::uint8_t remainderSize;
public:
    inline OutputBitstream(std::ostream& str) : stream(str), 
                                                remainder(0), 
                                                remainderSize(0) { }

    /// @brief Writes a given amount of bits into the underlying stream.
    /// Actually writes only multiples of 8 bits and stores the remainder.
    /// @tparam bitcount The amount of bits.
    /// @param bits The data to write.
    template <std::uint8_t bitcount> 
    requires (bitcount <= 8) and (bitcount != 0)
    void WriteBits(std::uint8_t bits) {
        if (bitcount + remainderSize >= 8) {
            auto data = (std::uint8_t)(remainder | (bits << remainderSize));
            stream.write(reinterpret_cast<char*>(&data), 1);
            remainder = bits >> (8 - remainderSize);
            remainderSize = bitcount + remainderSize - 8;
        } else {
            remainder = remainder | (bits << remainderSize);
            remainderSize += bitcount;
        }
    }

    /// @brief Writes any remainder data by adding trailing zeroes as to 
    ///        reach 8 bits. Does not close the underlying stream.
    inline void Close() {
        if (remainderSize != 0) 
            stream.write(reinterpret_cast<char*>(&remainder), 1);
    }
};

/// @brief Gives tools to read any bit-size integers in a typical binary
///        output stream.
class InputBitstream {
    /// @brief The stream where to write data
    std::istream& stream;

    /// @brief The bits which have not been written to the stream yet.
    std::uint8_t remainder;

    /// @brief The amount of pending bits which are waiting for more data to
    ///        come before being written.
    std::uint8_t remainderSize;

public:
    /// @brief Constructs an input bitstream.
    /// @param str The input stream the data is read from.
    inline InputBitstream(std::istream& str) : stream(str), 
                                               remainder(0), 
                                               remainderSize(0) { }

    /// @brief Reads a given amount of bits into the underlying stream.
    /// Actually reads only multiples of 8 bits and stores the remainder.
    /// @tparam bitcount The amount of bits.
    /// @param bits The data to write.
    template <std::uint8_t bitcount> 
    requires (bitcount <= 8) and (bitcount != 0)
    std::uint8_t ReadBits() {
        if (bitcount <= remainderSize) {
            std::uint8_t data = remainder & ((1 << bitcount) - 1);
            remainderSize -= bitcount;
            remainder >>= bitcount;
            return data;
        } else {
            std::uint8_t newData;
            stream.read(reinterpret_cast<char*>(&newData), 1);
            std::uint8_t data = (remainder | (newData << (remainderSize))) & 
                                ((1 << bitcount) - 1);
            remainder = newData >> (bitcount - remainderSize);
            remainderSize = 8 - (bitcount - remainderSize);
            return data;
        }
    }
};

namespace io {

/// @brief Writes an integer to a stream in little endian byte order.
/// @tparam T The type of the integer.
/// @param s The stream where to write the value.
/// @param x The value to write in the stream.
template<class T> requires (sizeof(T) <= 8)
void LittleEndianWrite(std::ostream& s, T x) {
    unsigned char buffer[8] = {0};
    if (sizeof(T) == 1) {
        buffer[0] = x & 0xFF; 
    } else if (sizeof(T) == 2) {
        buffer[0] = x & 0xFF;
        buffer[1] = (x >> 8) & 0xFF;
    } else if (sizeof(T) == 4) {
        buffer[0] = x & 0xFF;
        buffer[1] = (x >> 8) & 0xFF;
        buffer[2] = (x >> 16) & 0xFF;
        buffer[3] = (x >> 24) & 0xFF;
    } else if (sizeof(T) == 8) {
        buffer[0] = x & 0xFF;
        buffer[1] = (x >> 8) & 0xFF;
        buffer[2] = (x >> 16) & 0xFF;
        buffer[3] = (x >> 24) & 0xFF;
        buffer[4] = (x >> 32) & 0xFF;
        buffer[5] = (x >> 40) & 0xFF;
        buffer[6] = (x >> 48) & 0xFF;
        buffer[7] = (x >> 56) & 0xFF;
    }
    s.write((char*)buffer, sizeof(T));
}

/// @brief Reads an integer from a stream in little endian byte order.
/// @tparam T The type of the integer.
/// @param s The stream where to read the value from.
template<class T> requires (sizeof(T) <= 8)
T LittleEndianRead(std::istream& s) {
    unsigned char buffer[sizeof(T)];
    s.read((char*)buffer, sizeof(T));
    if (sizeof(T) == 1) {
        return buffer[0];
    } else if (sizeof(T) == 2) {
        return buffer[0] | ((T)buffer[1] << 8);
    } else if (sizeof(T) == 4) {
        return    buffer[0]        | ((T)buffer[1] << 8) | 
              ((T)buffer[2] << 16) | ((T)buffer[3] << 24);
    } else if (sizeof(T) == 8) {
        return     buffer[0]        | ((T)buffer[1] <<  8) | 
               ((T)buffer[2] << 16) | ((T)buffer[3] << 24) |
               ((T)buffer[4] << 32) | ((T)buffer[5] << 40) | 
               ((T)buffer[6] << 48) | ((T)buffer[7] << 56);
    }
    return T();
}

template<>
double LittleEndianRead(std::istream& s) {
    union {std::uint64_t v; double d;} var;
    var.v = LittleEndianRead<std::uint64_t>(s);
    return var.d;
}

template<>
void LittleEndianWrite(std::ostream& s, double x) {
    union {std::uint64_t v; double d;} var;
    var.d = x;
    LittleEndianWrite(s, var.v);
}


}

}
