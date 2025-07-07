[![PyPI Version](https://img.shields.io/pypi/v/ANFISpy)](https://pypi.org/project/ANFISpy/)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)]()

# ANFISpy
A Python implementation of **Adaptive Neuro-Fuzzy Inference Systems (ANFIS)**, combining neural networks and fuzzy logic for interpretable machine learning. The implementation is based on the original [ANFIS](https://ieeexplore.ieee.org/abstract/document/256541?casa_token=bWStLllx3e8AAAAA:Z7Tj7kk-7lHlGSIEVJZfJVtRi_IVpig2ANbVv6qou4Ok32c7X7Yfh8SsvIUUBjALl3dfHRgFRJs3) paper, adapting the model to perform both regression and classification tasks with customizable membership functions. A Recurrent ANFIS (RANFIS), a Gated Recurrent Unit ANFIS (GRU-ANFIS) and Long-Short Term memmory ANFIS (LSTM-ANFIS) are also implemented, suited for time series regression and classification.

# Key Features
- **Regression and Classification**
- **Time Series Analysis with RANFIS, GRU-ANFIS and LSTM-ANFIS**
- **Visualization and Interpretability** via `.print_rules()`, `.plot_var()`, `.plot_rules()`  
- **Various Membership Functions** (`GaussianMF`, `BellMF`, `TriangularMF`, `SigmoidMF`)  
- **PyTorch Integration** (GPU acceleration, optimizers, ...) 

# Repository Organization
The repository is organized in the following directories:
- **ANFISpy**: has the implementation of the ANFIS's based models;
- **examples**: has jupyter-notebooks with examples of how to use the models;
- **tests**: has testing files for managing the code behaviour.

# Installation
The installation of the package can be done using `pip` in a `bash` terminal:

```bash
pip install ANFISpy
```
Then, the package can be imported in Python using:

```python
from ANFISpy import ANFIS
from ANFISpy import RANFIS
```

# Quick Example
The ANFIS model can be used to perform both regression and classification, as explained in [anfis_example.ipynb](https://github.com/mZaiam/ANFISpy/blob/main/examples/anfis_example.ipynb). To instantiate a regression model, set the value of `n_classes` in the `output` to 1.

```python
from ANFISpy import ANFIS

n_vars = 3
mf_names = [['L', 'M', 'H']]

variables = {
    'inputs': {
        'n_sets': [3, 3, 3],
        'uod': n_vars * [(0, 1)],
        'var_names': ['var1', 'var2', 'var3'],
        'mf_names': n_vars * mf_names,
    },
    'output': {
        'var_names': 'out',
        'n_classes': 1,
    },
}

anfis_regression = ANFIS(variables, 'gaussian')
```

To create a clasification model, set the value of `n_classes` in the `output` to a number of classes greater or equal to 2. 

```python
from ANFISpy import ANFIS

n_vars = 3
mf_names = [['L', 'M', 'H']]

variables = {
    'inputs': {
        'n_sets': [3, 3, 3],
        'uod': n_vars * [(0, 1)],
        'var_names': ['var1', 'var2', 'var3'],
        'mf_names': n_vars * mf_names,
    },
    'output': {
        'var_names': 'out',
        'n_classes': 3,
    },
}

anfis_classification = ANFIS(variables, 'bell')
```

The notebook [anfis_example.ipynb](https://github.com/mZaiam/ANFISpy/blob/main/examples/anfis_example.ipynb) has a more detailed explanation of how to use the model, as well as the visualization methods implemented. The RANFIS, GRU-ANFIS and LSTM-ANFIS models are instantiated the same way as ANFIS, and examples can be seen in [ranfis_example.ipynb](https://github.com/mZaiam/ANFISpy/blob/main/examples/ranfis_example.ipynb), [gruanfis_example.ipynb](https://github.com/mZaiam/ANFISpy/blob/main/examples/gruanfis_example.ipynb) and [lstmanfis_example.ipynb](https://github.com/mZaiam/ANFISpy/blob/main/examples/lstmanfis_example.ipynb).

# Contact
This repository was built by Matheus Zaia Monteiro. Feel free to get in contact via the following e-mail: `matheus.z.monteiro@gmail.com`.
