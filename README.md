# libmlconcepts

This library collects a series of (interpretable) machine learning algorithms
which are based on FCA, e.g. [[3]](https://www.sciencedirect.com/science/article/pii/S0167923624000290). 
It is composed of three parts:

- A c++ header only library that makes heavy use of templates to aim for flexibility, while squeezing as much performance as possible from the machine.
- The python module `mlconceptscore`, i.e., pybind bindings that expose the implementation of some classes of the c++ library.
- The python package `mlconcepts`, i.e. a light wrapper over `mlconceptscore` which also exposes some utility functions and classes.

## Installation

The python package `mlconcepts` can be installed by running

```bash
pip install --user mlconcepts
```

Pre-compiled builds for every architecture supported by 
[cibuildwheel](https://cibuildwheel.pypa.io/en/stable/)
are available. 

If you are running on an `x86_64` or `i686` machine, the binaries are compiled for processors that support
the `AVX` instruction set. All intel CPUs except Celeron and Pentiums from Q1 2011, and all AMD cpus from
Q2 2013 are fine. Starting from Tiger Lake, also Celeron and Pentiums support it.
If your CPU does not support this instruction set, consider installing from a source distribution.

### Installing from source distribution

Installing `mlconcepts` from a source distribution only requires a `c++23` compiler, all the
other dependencies are automatically fetched. If `cmake` is not able to find the compiler
during the installation process, please set the environment variable `CXX` as follows

```bash
CXX = /path/to/c++/compiler 
```

### Dependencies

The three components of this project have different dependencies, and `mlconcepts`
depends on `mlconceptscore`, which depends on `libmlconcepts`. The external dependecies are as follows:

- `libmlconcepts` depends on `eigen 3.4` and a `c++23` standard library.
- `mlconceptscore` requires `pybind11` and `numpy`.
- `mlconcepts` requires the following python libraries:
	- [Optional] `pandas` to enable pandas' DataFrame loaders, and to read excel, json, sql, and csv files.
	- [Optional] `h5py` to parse matlab files.
	- [Optional] `scipy`to parse matlab files up to version 7.3.

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
import sklearn.metrics
import sklearn.model_selection

#Loads the dataset.
data = mlconcepts.data.load("dataset.csv", labels = "outlier")

#data.split takes as an input any splits generator, such as the ones of sklearn
skf = sklearn.model_selection.StratifiedKFold(n_splits = 4, shuffle = True)
for train, test in data.split(skf):
	model = mlconcepts.SODModel(
		    n = 32, #number of bins for quantization
            epochs = 1000, #number of training iterations
            show_training = False #whether to show training info
	)
	model.fit(train)
	predictions = model.predict(test)
	print("AUC: ", sklearn.metrics.roc_auc_score(test.y, predictions))
```

### Extracting feature-importance
Consider the following dataset `trainset.csv` containing data about some mushrooms

|feat1|feat2|feat3| color | poisonous |
|:---:|:---:|:---:|:-----:|:---------:|
| 5.3 | 3.2 | 2.4 |  red  |    no     |
| 3.1 | 8.7 | 4.2 | green |   yes     |
| 3.2 | 8.8 | 5.2 | green |   yes     |
| 1.2 | 5.3 | 8.8 |  red  |    no     |
| 3.2 | 1.2 | 1.0 |  red  |    no     |
| 9.0 | 8.9 | 2.0 | green |   yes     |
| 3.6 | 1.7 | 1.4 | green |    no     |
| 5.6 | 6.9 | 2.3 |  red  |    no     |

And consider the associated `set.csv` for which we want to make a prediction

|feat1|feat2|feat3| color | poisonous |
|:---:|:---:|:---:|:-----:|:---------:|
| 2.2 | 8.75| 6.3 | green |   yes     |
| 5.4 | 2.2 | 1.7 |  red  |    no     |

The following code extracts some predictions and generates feature importance information
for each prediction

```python
import mlconcepts

model = UODModel(n = 4)
model.fit("trainset.csv", labels="poisonous")
expl = model.predict_explain("set.csv", labels="poisonous")

print(expl[0])
print(expl[1])
```

The output containing the most important feature/feature pairs for the two samples will be

```json
Prediction: 0.9768333792608045. 
Explainers: { 
	{ att1 } : 1.0665980106418878, 
	{ att2 } : 0.8065285222496102, 
	{ att0, att2 } : 0.8019954963264249 
}

Prediction: 0.20744964895969192. 
Explainers: { 
	full : 0.1373221852146331,
	{ att1 } : 0.01953542400235907,
	{ att0 } : 0.01655167190477671 
}
```

## Limitations

The current version of `libmlconcepts` cannot work with streamed/batched datasets due to how FCA algorithms work. I have an idea on how to make batching work efficiently for outlier detection, which would also erase any memory requirement for the algorithm, and probably also speed it up a bit. The c++ library is designed in such a way that such a different internal representation of formal contexts and data would become transparent.

I do not know yet how to efficiently batch the classifiers that the `c++` library exposes (and currently are under development).

## Compile mlconceptscore on your own
To compile `mlconceptscore`, you need `cmake>=3.26`, and a c++23-enabled compiler. To compile all the components of the project run:

```bash
mkdir build
cd build
cmake ..
make all
```

## Contributing

The python code in the repository uses [Google's style guidelines](https://google.github.io/styleguide/).
Feel free to create any pull requests.

## License
All the parts of `libmlconcepts` are licensed under the BSD3 license.

## References
[[1] Flexible categorization for auditing using formal concept analysis and 
Dempster-Shafer theory](https://arxiv.org/abs/2210.17330)

[[2] A Meta-Learning Algorithm for Interrogative Agendas](https://arxiv.org/abs/2301.01837)

[[3] Outlier detection using flexible categorisation and interrogative agendas](https://www.sciencedirect.com/science/article/pii/S0167923624000290)