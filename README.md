# libmlconcepts

This library collects a series of (interpretable) machine learning algorithms
which are based on FCA. It is composed of three parts:

- A c++ header only library that makes heavy use of templates to aim for flexibility, while squeezing as much performance as possible from the machine.
- The python module `mlconceptscore`, i.e., pybind bindings that expose the implementation of some classes of the c++ library.
- The python module `mlconcepts`, i.e. a light wrapper over `mlconceptscore` which also exposes some utility functions and classes.

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

### 

## License
All the parts of `libmlconcepts` are licensed under the BSD3 license.

## References
[[1] Flexible categorization for auditing using formal concept analysis and 
Dempster-Shafer theory](https://arxiv.org/abs/2210.17330)

[[2] A Meta-Learning Algorithm for Interrogative Agendas](https://arxiv.org/abs/2301.01837)

[[3] Outlier detection using flexible categorisation and interrogative agendas](https://www.sciencedirect.com/science/article/pii/S0167923624000290)