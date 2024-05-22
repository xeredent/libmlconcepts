#pragma once
#include <cmath>
#include <Eigen/Core>

namespace mlconcepts{
namespace math {

/// @brief Computes the softmax function for a given vector.
/// @param x A real vector.
/// @return The result of the softmax function application.
template<typename Derived>
Eigen::MatrixBase<Derived> Softmax(const Eigen::VectorXd& x) {
    auto expX = (x.array() - x.maxCoeff()).exp();
    return expX / expX.sum();
}

/// @brief Computes the softmax function columnwise for a given matrix.
/// @param X The matrix for which the softmax function is applied.
/// @return A matrix containing the output of the columnwise softmax function
///         application.
template<typename Derived>
Eigen::MatrixXd SoftmaxColumnwise(const Eigen::MatrixBase<Derived>& X) {
    auto expX = (X.rowwise() - X.colwise().maxCoeff()).array().exp();
    return expX.array().rowwise() / expX.array().colwise().sum();
}

/// @brief Given a matrix (classes x samples), applies softmax columnwise and
///        computes cross-entropy.
/// @tparam LabelVector The type of the vector of labels. Can be Eigen::VectorXi,
///         or any std::vector with integer type.
/// @param X The input matrix.
/// @param labels The labels representing to which class each object in the
///               sample belongs.
/// @return The cross entropy loss.
template<typename Derived, class LabelVector = Eigen::Ref<Eigen::VectorXi> >
double SoftmaxCrossEntropy(const Eigen::MatrixBase<Derived>& X, LabelVector labels) {
    return (-(SoftmaxColumnwise(X)(labels, Eigen::all)).array().log()).sum() / 
           (double)X.cols();
}

}
}