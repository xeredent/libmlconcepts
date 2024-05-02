# libmlconcepts

This library collects a series of (interpretable) machine learning algorithms
which are based on FCA. It is composed of three parts:

- A c++ header only library that makes heavy use of templates to aim for flexibility, while squeezing as much performance as possible from the machine.
- The python module `mlconceptscore`, i.e., pybind bindings that expose the implementation of some classes of the c++ library.
- The python package `mlconcepts`, i.e. a light wrapper over `mlconceptscore` which also exposes some utility functions and classes.

## Installation

### Dependencies

The three components of this project have different dependencies, and `mlconcepts`
depends on `mlconceptscore`, which depends on `libmlconcepts`. The external dependecies are as follows:

- `libmlconcepts` depends on `eigen 3.4` and a `c++23` standard library.
- `mlconceptscore` requires `pybind11`.
- `mlconcepts` requires the following python libraries:
	- `numpy`
	- [Optional] `pandas` to enable pandas' DataFrame loaders, and to read excel, json, sql, and csv files.
	- [Optional] `h5py` to parse matlab files.
	- [Optional] `scipy`to parse matlab files up to version 7.3.

### Compile
To compile `mlconceptscore`, you need `cmake>=3.26`, and a c++23-enabled compiler. To compile all the components of the project run:

```bash
mkdir build
cd build
cmake ..
make all
```

## Basic Usage (mlconcepts)

As of the present version, `mlconcepts` exposes only models for outlier detection
and does not make use of all the (experimental) functionalities of libmlconcepts.
The two available models are `UODModel` and `SODModel` for unsupervised and supervised
outlier detection, respectively.

The library implements a simple and easily extensible data loader, which allows
to import data from different sources, and data frames from other libraries.

### Basic example
Assuming that a dataset containing a column `outlier` is stored in the file 
`dataset.csv`, a basic model could be trained as follows.

```python
import mlconcepts

model = mlconcepts.SODModel() #creates the model
model.fit("dataset.csv", labels = "outlier") #trains the model on the dataset
model.save("model.bin") #compresses and serializes the model to file
```

### A slightly more involved example

```python
import mlconcepts
import mlconcepts.data
from sklearn.metrics import roc_auc_score
import sklearn.model_selection

#Loads the dataset.
data = mlconcepts.data.load("dataset.csv", labels = "outlier")

#data.split takes as an input any splits generator, such as the ones of sklearn
for train, test in data.split(sklearn.model_selection.StratifiedKFold(n_splits = 4, shuffle = True)):
	model = mlconcepts.SODModel(n = 32, #number of bins for quantization
                                epochs = 1000, #number of training iterations
                                show_training = False) #whether to show training info
	model.fit(train)
	predictions = model.predict(test)
	print("AUC: ", roc_auc_score(test.y, predictions))
```

## Contributing

TODO: write something

## License
All the parts of `libmlconcepts` are licensed under the BSD3 license.

## References
[[1] Flexible categorization for auditing using formal concept analysis and 
Dempster-Shafer theory](https://arxiv.org/abs/2210.17330)

[[2] A Meta-Learning Algorithm for Interrogative Agendas](https://arxiv.org/abs/2301.01837)

[[3] Outlier detection using flexible categorisation and interrogative agendas](https://www.sciencedirect.com/science/article/pii/S0167923624000290)