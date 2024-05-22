#pragma once
#include <cstddef>
#include <concepts>

namespace mlconcepts
{

/// A ClosureSizeProvider must expose some methods to compute closure sizes.
template <class T, class A, class F>
concept ClosureSizeProvider = requires(const T c, 
                                       const A attrs, 
                                       const F filter, 
                                       std::size_t id) {
    { c.ComputeClosureSize(id) } -> std::convertible_to<std::size_t>;
    { c.ComputeClosureSize(attrs) } -> std::convertible_to<std::size_t>;
    { c.ComputeFilteredClosureSize(id, filter) } -> std::convertible_to<std::size_t>;
    { c.ComputeFilteredClosureSize(attrs, filter) } -> std::convertible_to<std::size_t>;
};


/// A ClosureProvider must expose some methods to compute closures and closure sizes.
template <class T, class A, class F>
concept ClosureProvider = ClosureSizeProvider<T, A, F> and
                          requires(const T c, const A attrs, const F filter) {
    c.ComputeClosure(attrs);
    c.ComputeFilteredClosure(attrs, filter);
};

}