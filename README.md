**Libmlconcepts** is a library which collects a series of interpretable machine learning algorithms based on [formal concept analysis (FCA)](https://en.wikipedia.org/wiki/Formal_concept_analysis), a theory that develops a principled way to generate hierarchical classifications out of a dataset.

Currently, this library implements interpretable (unsupervised and supervised) outlier detection algorithms, and some experimental classification algorithms.The outlier detection algorithms are based on ideas found in [[3]](https://www.sciencedirect.com/science/article/pii/S0167923624000290), and the algorithms for classification lie their roots in the meta-algorithm in [[2]](https://arxiv.org/abs/2301.01837).

### Installation

In most architectures, the python package `mlconcepts` can be installed by just running

```bash
pip install --user mlconcepts
```

The library also uses optional dependencies to support several dataset formats and representations.
Install any of the following libraries to enable support to parse excel, json, sql, csv, and matlab files, and to seamlessly use pandas dataframes in the framework (more info in the [Dependencies](#Dependencies) section):

```bash
pip install --user pandas scipy h5py
```

In case your platform/architecture is not supported, or if you want to squeeze every drop of performance from your machine, see below how to install from a source distribution.

<details>
<summary>
I don't care about Python at all; how do I use mlconcepts in my c++ project?
</summary>
The python package <em>mlconcepts</em> is based on <em>libmlconcepts</em>, a c++20 header-only library; hence, there is no need to install anything. Even though I haven't had the time yet to add cmake files that find libmlconcepts, integrating it in your project is as simple as putting the include directory in your project (together with a copy of the very-permissive license) and telling your compiler where to find it.
</details>

<details>
<summary>How do I install from a source distribution?</summary>
Installing <em>mlconcepts</em> from a source distribution only requires a c++20 compiler, all the other dependencies are automatically fetched. To install from source distribution run

```bash
pip install --user --no-binary mlconcepts mlconcepts
```

If the c++ compiler is not found during the installation process, specify it by setting the environment variable `CXX` to the path to the c++20 compiler. In most systems this is done as follows:

```bash
CXX=/path/to/c++20/compiler pip install --user --no-binary mlconcepts mlconcepts
```

In Windows, the environment variable has to be set before running pip

```bash
set CXX=/path/to/c++20/compiler
pip install --user --no-binary mlconcepts mlconcepts
```
</details>

<details>
<summary>Are pre-built binaries distributed for my platform?</summary>
mlconcepts uses [cibuildwheel](https://cibuildwheel.pypa.io/en/stable/) to generate pre-built binaries for a wide range of architectures and platforms, and for several Python versions (from 3.6 to 3.12).

Currently, binaries are built for:
- Linux distributions (manylinux), x86_64 and i686;
- Windows, x86_64 and i686;
- MacOS 13 and 14, x86_64 and arm64, except for Python 3.8.

Upon request, I can add aarch64 for linux distributions without much trouble. For all other architectures, you can compile from a source distribution as described above.
</details>

<details>
<summary>I have a very old Intel/AMD CPU. Will mlconcepts work in my machine?</summary>
If you are running on an `x86_64` or `i686` machine, the binaries are compiled for processors that support
the `AVX` instruction set. All intel CPUs except Celeron and Pentiums from Q1 2011, and all AMD cpus from
Q2 2013 are fine. Starting from Tiger Lake, also Celeron and Pentiums support it.
If your CPU does not support this instruction set, consider installing from a source distribution.
</details>

<details>
<summary>I don't want to use pip to install mlconcepts. How can I compile it from this repository?</summary>
If for some reason you want to compile mlconcepts from source without pip, you need a c++20 compiler and cmake>=3.26.

After cloning the repository, change directory to its root and run

```bash
cmake -GNinja -B build .
cd build
ninja -j8 all
```

The build directory will contain the folder mlconcepts, which you can move to wherever you keep your python packages.
</details>


## Basic Usage (python)

This section shows some basic examples using the library. Check the generated [documentation](https://xeredent.github.io/libmlconcepts/) for more details on the functions and classes exposed by the library, and check [this jupyter notebook](docs/examples.ipynb) for more examples. 

#### Datasets

The library implements a simple and easily extensible data loader, which allows to import data from different sources, and data frames from other libraries. All these different sources are abstracted by the class `mlconcepts.data.Dataset`, which represent generic datasets with numerical and categorical data, and, possibly, some labels on the items of the dataset (e.g., indicating what elements are outliers, or the class of each element).

In general, datasets are loaded using the function `mlconcepts.load`, which accepts any type of data that it can transform into a dataset, such as `numpy` arrays, `pandas` dataframes, or paths to files with a known format (csv, xlsx, mat, json, or sql). For instance, dataset `"data.csv"` is loaded as follows

```python
import mlconcepts
dataset = mlconcepts.load("data.csv")
```

The example above loads a dataset suitable for unsupervised machine learning tasks, as no label column is specified in the dataset. To load a dataset with a some labels in the `"outlier"` column, we can run

```python
dataset = mlconcepts.load("data.csv", labels="outlier")
```

<details>
<summary>How to load from numpy arrays?</summary>
Loading a numpy array is as easy as passing it to the load function:

```python
import numpy
import mlconcepts
mat = numpy.array([[4.3, 2.4, 1.2], [3.3, 2.1, 0.6]])
dataset = mlconcepts.load(mat)
```

The dtype of the passed matrix has to be convertible to a floating point number. Categorical data can be added in a second matrix, whose elements must be (convertible to) integers representing an index to the category of each element for every column. For instance, a single categorical feature can be added as follows:

```python
num = numpy.array([[4.3, 2.4, 1.2], [3.3, 2.1, 0.6]])
cat = numpy.array([[2], [1]])
dataset = mlconcepts.load(num, Xc=cat)
```

Similarly, labels can added by setting the parameter "y" to a (numpy) vector whose dtype is convertible to integers. For instance, to say that the first element is an outlier, and the second is not:

```python
num = numpy.array([[4.3, 2.4, 1.2], [3.3, 2.1, 0.6]])
cat = numpy.array([[2], [1]])
outliers = numpy.array([1, 0])
dataset = mlconcepts.load(num, Xc=cat, y=outliers)
```

</details>

<details>
<summary>How to load from a pandas dataframe?</summary>
Pandas dataframes can be loaded by just passing them to the load function, e.g.,

```python
df = pandas.DataFrame({"a" : [4.2, 1.2], "b" : ["no", "yes"]})
data = mlconcepts.load(df)
```

The loading function automatically detects that "b" is a categorical feature and stores it as such. Labels can be set by indicating a suitable column (categorical or integer) as follows:

```python
df = pandas.DataFrame({"a" : [4.2, 1.2], "b" : ["no", "yes"]})
data = mlconcepts.load(df, labels="b")
```

Alternatively, a label column can be specified as a numpy array as follows:

```python
df = pandas.DataFrame({"a" : [4.2, 1.2], "b" : ["no", "yes"]})
outliers = numpy.array([1, 0])
data = mlconcepts.load(df, y=outliers)
```
</details>

<details>
<summary>How to load from a matlab file?</summary>
To load from a matlab file, we need to tell mlconcepts what are the names of the matlab variables containing the data. To do so, we pass a map containing the parameters "Xname", "Xcname", "yname", which default to "X", "Xc", and "y", respectively. For instance, to load from a matlab file "data.mat" where numerical data is in a matrix "X", and the labels are in a vector "y", run

```python
data = mlconcepts.load("data.mat", settings={ "Xname" : "X", "yname" : "y" })
```
</details>

<details>
<summary>How to load categorical data?</summary>
Most dataloaders handle categorical data automatically, as it is usually not hard to distinguish it from numerical data. The only exception is when you want to consider a column of integers to be categorical. In this case the parameter "categorical" can be used to specify a list of features which are in principle numerical, but should be considered as categorical, e.g.,

```python
df = pandas.DataFrame({ 'age' : [32, 86], 'has_internet_plan' : [1, 0] })
data = mlconcepts.load(df, categorical=["has_internet_plan"])
```
</details>

<details>
<summary>How to generate splits of a dataset?</summary>
mlconcept's Dataset objects support splitting using any split generator, i.e., an object with a method <em>split</em>, which, given the dataset, returns a generator yielding a set of indices used to sample the training set, and a set of indices used to sample the test set.

For example, all the split generators of sklearn are supported, as shown in the following snippet

```python
import mlconcepts
import sklearn.model_selection
data = mlconcepts.load("data.csv")

skf = sklearn.model_selection.StratifiedKFold(n_splits = 4, shuffle = True)
for train, test in data.split(skf):
	pass
	# Do something with the splits
	# Both train and test are mlconcepts.data.Dataset objects.
```
</details>

#### Models

The library currently exposes two classes for outlier detection: `UODModel` for unsupervised OD, and `SODModel` for supervised OD. Models can simply be created as follows:

```python
unsup_model = mlconcepts.UODModel()
sup_model = mlconcepts.SODModel()
```

Details on the supported parameters can be found in the drop-downs below.

 After a model is created, the methods `fit`, `predict`, and `predict_explain` train the model, compute predictions for some set, and compute predictions with accompanying explanations for a set, respectively. These three methods can either take a `mlconcepts.data.Dataset` object, or can be called with the same signature as the method `mlconcepts.load` [described above](#Datasets). In the latter case, a dataset is loaded by forwarding the arguments to the function `mlconcepts.load`, and then the task is executed on that dataset.

For instance, the following basic example trains a supervised model on a dataset `"train.json"` and computes predictions on the dataset `"data.csv"`.

```python
model = mlconcepts.SODModel(
	n=16, # hyperparameter: numerical features are quantized in n bins
	epochs=2000, # maximum number of training iterations
	show_training=False # Whether to write training information to stdout
)
model.fit("train.json", labels="outlier")
predictions = model.predict("data.csv", labels="outlier")
print(predictions)
# [0.212, 0.985326, 0.0041, ...]
```

The classes implementing the classification algorithms are experimental and currently not covered in this documentation.

<details>
<summary>What parameters are available for unsupervised models?</summary>
Unsupervised models support parameters that guide the way in which the underlying FCA algorithm works:

```python
model = mlconcepts.UODModel(
	n=32, # number of bins for numerical feature quantization
	quantizer="uniform", # numerical features are uniformly split in bins
	explorer="none", # strategy to explore different feature sets
	singletons=True, # whether to explore all single element sets
	doubletons=True, # whether to explore all two-elements sets
	full=True # whether to explorer the set of all features
)
```

The first two parameters concern the treatment of numerical features. Numerical features are indeed quantized into cateogrical ones in a first step. The only supported quantizer at the time of writing is `uniform`.

To make its predictions, the algorithm essentially creates different agents which make a prediction considering different perspectives, i.e., different feature sets they observe. At the end all the agents combine their scores into a final prediction. The `explorer` determines what agents to generate. Currently, in the stable version of the program, only `none` is supported. The last three parameters set the agents which are generated at the start. The indicated explorer is then used to generate more agents (to consider more feature sets) depending on which are more promising.
</details>

<details>
<summary>What parameters are available for supervised models?</summary>
Supervised models support all parameters of unsupervised models, plus parameters for gradient descent:

```python
model = mlconcepts.SODModel(
    epochs=1000, # maximum number of training iterations
    show_training=True, # whether to show training information in the terminal
    learning_rate=0.01, # learning rate for gradient descent
    momentum=0.01, # momentum for gradient descent
    stop_threshold=0.001 # loss change under which training is halted
)
```
</details>

<details>
<summary>What is the output of predict_explain?</summary>

The method `predict_explain` returns an `ExplanationData` object. ExplanationData overrides the subscript operator, so as to return an `ExplanationEntry` for any index of an element in the dataset the prediction was computed on. An `ExplanationEntry` is an iterable, which, when iterated over, returns pairs (set of feature, importance). The pairs are yielded in decreasing order with respect to their importance. 

Roughly speaking, each set of features is mapped to its relevance in making a prediction in the model.

The following snippet shows the relevance of every (computed) feature set of the second item in a dataset:

```python
model = mlconcepts.UODModel(n=64)
model.fit("trainset.csv")
explanations = model.predict_explain("data.csv")
for feature_set, relevance in explanations[1]:
	print(feature_set, relevance)
```
</details>

<details>
<summary>Could you show a slightly more complicated example?</summary>
The following example shows how to load a dataset, generate splits, and compute its performance (AUC score):

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
            n=32, #number of bins for quantization
            epochs=1000, #number of training iterations
            learning_rate=0.01, #learning rate for gradient descent
            show_training=False #whether to show training info
	)
	model.fit(train)
	predictions = model.predict(test)
	print("AUC: ", sklearn.metrics.roc_auc_score(test.y, predictions))
```
</details>


#### Understanding the explanations

In this section, you can find an informal explanation of how the algorithms of this library work, and a small example on how to interpret the explanation data that the library spits out.

##### How do these algorithms work concretely?

The following paragraphs explain the algorithms very informally and loosely. For a more thorough presentation see [[3]](#references).

The algorithms in this library compute explanations that resemble those of methods based on computing Shapley values (e.g. [shap](https://github.com/shap/shap)). Together with any prediction, also the *relevance* of the features in making the prediction is computed. The relevance is not computed just with respect to single features, but rather with respect to *sets of features*, as the interaction of different features could be a stronger indicator for some prediction tasks. For instance, if we want to flag suspicious hotels in a city, a hotel in a good location is not necessarily suspicious, and also a low-price hotel is not necessarily suspicious. However, the combination of the two things, a hotel in a good location and with low prices, might rise our suspiciousness.

Roughly speaking, these algorithms generate agents, and each agent looks at the data from a different perspective: each agent observes only a given set of features, and uses it to create a hierarchical categorization of the data (a formal ontology, using FCA), which they use to make a prediction. After all agents have made their predictions, their scores are suitably combined into a final prediction, and their contribution in this deliberation corresponds to the relevance of the feature set they are observing.

##### An example in practice

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

The following code computes some predictions and generates feature importance information for each prediction.

```python
import mlconcepts

model = UODModel(n = 4)
model.fit("trainset.csv", labels="poisonous")
expl = model.predict_explain("set.csv", labels="poisonous")
```

For instance, the feature-set importance data of the first two elements in `set.csv` can be shown as follows:

```python
print(expl[0])
print(expl[1])
```

which yields

```
Prediction: 0.9768333792608045. 
Explainers: { 
	{ feat2 } : 1.0665980106418878, 
	{ feat3 } : 0.8065285222496102, 
	{ feat1, feat3 } : 0.8019954963264249 
}

Prediction: 0.20744964895969192. 
Explainers: { 
	full : 0.1373221852146331,
	{ feat2 } : 0.01953542400235907,
	{ feat1 } : 0.01655167190477671 
}
```

This output lists the three most important feature sets in making a *positive* prediction, i.e., in saying that something *is* an outlier. For the first element (which is an outlier), `feat2` alone is a strong predictor (it is always very high for the outliers), but, for instance, also the interaction of `feat1` and `feat3` is. Indeed, an inspection of the training set shows that similar values of `feat1` would not be good predictors for lower values of `feat3`.

## Limitations

The current version of `libmlconcepts` cannot work with streamed/batched datasets due to how FCA algorithms work. I have an idea on how to make batching work efficiently for outlier detection, which would also erase any memory requirement for the algorithm, and probably also speed it up a bit. The c++ library is designed in such a way that such a different internal representation of formal contexts and data would become transparent.

I do not know yet how to efficiently batch the classifiers that the `c++` library exposes (and currently are under development).

## Dependencies

This project has three components: `libmlconcepts`, a c++ header only library; `mlconceptscore`, python bindings over the c++ library; `mlconcepts`, a python wrapper around `mlconceptscore` that exposes also some QOL functions/classes.
Of course, `mlconcepts` depends on `mlconceptscore`, which depends on `libmlconcepts`. The external dependecies are as follows:

- `libmlconcepts` depends on `eigen 3.4` and the `c++20` standard library.
- `mlconceptscore` requires `pybind11` and `numpy`.
- `mlconcepts` supports several optional features. It depends on:
	- [Optional] `pandas` to enable pandas' DataFrame loaders, and to read excel, json, sql, and csv files.
	- [Optional] `h5py` to parse matlab files.
	- [Optional] `scipy`to parse matlab files up to version 7.3.


## Contributing

The python code in the repository uses [Google's style guidelines](https://google.github.io/styleguide/).
Feel free to create any pull requests.

## License
All the parts of `libmlconcepts`, `mlconceptscore`, and `mlconcepts` are licensed under the BSD3 license.

## References
[[1] Flexible categorization for auditing using formal concept analysis and 
Dempster-Shafer theory](https://arxiv.org/abs/2210.17330)

[[2] A Meta-Learning Algorithm for Interrogative Agendas](https://arxiv.org/abs/2301.01837)

[[3] Outlier detection using flexible categorisation and interrogative agendas](https://www.sciencedirect.com/science/article/pii/S0167923624000290)