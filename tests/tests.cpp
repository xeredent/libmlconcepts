#include <iostream>
#include <unordered_map>
#include <functional>
#include "PartialContext.h"
#include "EigenDataset.h"
#include "OutlierDetectionModel.h"

using namespace mlconcepts;

int main(int argc, char** argv) {
    //Eigen::MatrixXd realData{ { 1.25, 2.1, 2.2 }, { 1.1, 1.0, 3.0 }, { 2.5, 3.5, 4.5 } };
    Eigen::MatrixXd realData{ { 1.25, 2.1}, { 1.1, 1.0 }, { 2.5, 3.5 } };
    Eigen::MatrixXi categoricalData{ { 1 }, { 0 }, { 0 } };
    Eigen::MatrixXi categoricalData2 = Eigen::MatrixXi::Zero(0, 0);
    Eigen::VectorXi labelData{ {0}, {0}, {1} };
    EigenDataset<Eigen::MatrixXd> d(realData, categoricalData2);

    Bitset<uint64_t> set(40);
    set.Add(5); set.Add(9); set.Add(16); set.Add(5); set.Add(0);
    for (auto x : set) std::cout << x << " ";
    std::cout << std::endl;
    std::cout << set.GetFirstElement() << " " << set.GetLastElement() << std::endl;

    UnsupervisedODModel<PartialContext<uint64_t>> model;
    model.Set("UniformBins", 3);
    model.Fit(d);
    for (const auto& c : model.GetContexts()) c.WriteToStream();
    std::cout << "Estimated consumed memory: " << model.EstimateSize() << std::endl;
    return 0;
}