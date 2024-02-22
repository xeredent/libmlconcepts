#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "PartialContext.h"
#include "EigenDataset.h"
#include "OutlierDetectionModel.h"

using namespace mlconcepts;

typedef UnsupervisedODModel<PartialContext<uint64_t>, Conceptifier<ConfigurableContextSelector<uint32_t>, 
        AllUniformQuantizer<double>, AllValuesQuantizer<size_t>>>
        UODUniform; 

typedef UnsupervisedODModel<PartialContext<uint64_t>, Conceptifier<ConfigurableContextSelector<uint32_t>, 
        AllUniformQuantizer<double>, AllValuesQuantizer<size_t>>>
        UODUniform; 

namespace mlconcepts {
    auto DefaultNullMatrixXi = Eigen::MatrixXi::Zero(0, 0);
}

PYBIND11_MODULE(mlconcepts, m) {
    m.doc() = "A module which implements outlier detection and classification algorithms based on formal concepts analysis.";
    pybind11::class_<UODUniform>(m, "UODUniform")
        .def(pybind11::init([](size_t n, bool singletons, bool doubletons, bool full){
            auto model = UODUniform();
            model.Set("UniformBins", n);
            model.Set("GenerateSingletons", singletons);
            model.Set("GenerateDoubletons", doubletons);
            model.Set("GenerateFull", full);
            return model;
        }), pybind11::arg("n") = 32, pybind11::arg("singletons") = true, pybind11::arg("doubletons") = true,
            pybind11::arg("full") = true)
        .def("fit", [](UODUniform& model, Eigen::Ref<Eigen::MatrixXd> X, Eigen::Ref<Eigen::MatrixXi> Xc) {
            EigenDataset<Eigen::MatrixXd, Eigen::MatrixXi> dataset(X, Xc);
            model.Fit(dataset);
        }, pybind11::arg("X"), pybind11::arg_v("Xc", mlconcepts::DefaultNullMatrixXi, "null matrix"),
        "Trains the model so that it fits a dataset.")
        .def("predict", [](UODUniform& model, Eigen::Ref<Eigen::MatrixXd> X, Eigen::Ref<Eigen::MatrixXi> Xc) {
            EigenDataset<Eigen::MatrixXd, Eigen::MatrixXi> dataset(X, Xc);
            return model.Predict(dataset);
        }, pybind11::arg("X"), pybind11::arg_v("Xc", mlconcepts::DefaultNullMatrixXi, "null matrix"),
           pybind11::return_value_policy::move, "Predicts")
        .def("estimate_size", &UODUniform::EstimateSize, "Estimates the size in bytes occupied by the object");

}