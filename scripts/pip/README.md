# mlconcepts

This library is a wrapper around [libmlconcepts](https://github/xeredent/libmlconcepts), namely
a `c++` library which implements a series of (interpretable) machine learning algorithms
based on FCA, e.g. [[3]](https://www.sciencedirect.com/science/article/pii/S0167923624000290).

Installing `mlconcepts` from a source distribution only requires a `c++23` compiler, all the
other dependencies are automatically fetched. If `cmake` is not able to find the compiler
during the installation process, please set the environment variable `CXX` as follows

```bash
CXX = /path/to/c++/compiler
```


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


## References
[[1] Flexible categorization for auditing using formal concept analysis and 
Dempster-Shafer theory](https://arxiv.org/abs/2210.17330)

[[2] A Meta-Learning Algorithm for Interrogative Agendas](https://arxiv.org/abs/2301.01837)

[[3] Outlier detection using flexible categorisation and interrogative agendas](https://www.sciencedirect.com/science/article/pii/S0167923624000290)