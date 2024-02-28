#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <fstream>
#include "PartialContext.h"
#include "EigenDataset.h"
#include "UnsupervisedOutlierDetectionModel.h"
#include "SupervisedOutlierDetectionModel.h"

using namespace mlconcepts;

typedef UnsupervisedODModel<PartialContext<uint64_t>, Conceptifier<ConfigurableContextSelector<uint32_t>, 
        AllUniformQuantizer<double>, AllValuesQuantizer<size_t>>>
        UODUniform; 

typedef SupervisedODModel<PartialContext<uint64_t>, Conceptifier<ConfigurableContextSelector<uint32_t>, 
        AllUniformQuantizer<double>, AllValuesQuantizer<size_t>>>
        SODUniform; 

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
        .def("estimate_size", &UODUniform::EstimateSize, "Estimates the size in bytes occupied by the object")
        .def("save", [](const UODUniform& m, const std::string& filename) {
            std::ofstream f(filename, std::ios::out | std::ios::binary);
            m.Serialize(f);
            f.close();
        })
        .def("load", [](UODUniform& m, const std::string& filename) {
            std::ifstream f(filename, std::ios::in | std::ios::binary);
            m.Deserialize(f);
            f.close();
        });

    pybind11::class_<SODUniform>(m, "SODUniform")
        .def(pybind11::init([](size_t n, bool singletons, bool doubletons, bool full,
                              double learningRate, double momentum, double stopThreshold,
                              size_t trainEpochs, bool showTraining, double showTrainingDelay){
            auto model = SODUniform();
            model.Set("UniformBins", n);
            model.Set("GenerateSingletons", singletons);
            model.Set("GenerateDoubletons", doubletons);
            model.Set("GenerateFull", full);
            model.Set("LearningRate", learningRate);
            model.Set("Momentum", momentum);
            model.Set("StopThreshold", stopThreshold);
            model.Set("TrainEpochs", trainEpochs);
            model.Set("ShowTraining", showTraining);
            model.Set("ShowTrainingDelay", showTrainingDelay);
            return model;
        }), pybind11::arg("n") = 32, pybind11::arg("singletons") = true, pybind11::arg("doubletons") = true,
            pybind11::arg("full") = true, pybind11::arg("learningRate") = 0.01, pybind11::arg("momentum") = 0.01,
            pybind11::arg("stopThreshold") = 0.01, pybind11::arg("trainEpochs") = 1000, pybind11::arg("showTraining") = true, 
            pybind11::arg("showTrainingDelay") = 100.0)
        .def("fit", [](SODUniform& model, Eigen::Ref<Eigen::MatrixXd> X, Eigen::Ref<Eigen::VectorXi> y, Eigen::Ref<Eigen::MatrixXi> Xc) {
            EigenDataset<Eigen::MatrixXd, Eigen::MatrixXi> dataset(X, Xc, y);
            model.Fit(dataset);
        }, pybind11::arg("X"), pybind11::arg("y"), pybind11::arg_v("Xc", mlconcepts::DefaultNullMatrixXi, "null matrix"),
        "Trains the model so that it fits a dataset.")
        .def("predict", [](SODUniform& model, Eigen::Ref<Eigen::MatrixXd> X, Eigen::Ref<Eigen::MatrixXi> Xc) {
            EigenDataset<Eigen::MatrixXd, Eigen::MatrixXi> dataset(X, Xc);
            return model.Predict(dataset);
        }, pybind11::arg("X"), pybind11::arg_v("Xc", mlconcepts::DefaultNullMatrixXi, "null matrix"),
           pybind11::return_value_policy::move, "Predicts")
        .def("estimate_size", &SODUniform::EstimateSize, "Estimates the size in bytes occupied by the object")
        .def("save", [](const SODUniform& m, const std::string& filename) {
            std::ofstream f(filename, std::ios::out | std::ios::binary);
            m.Serialize(f);
            f.close();
        })
        .def("load", [](SODUniform& m, const std::string& filename) {
            std::ifstream f(filename, std::ios::in | std::ios::binary);
            m.Deserialize(f);
            f.close();
        });

}