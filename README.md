[![PyPI Version](https://img.shields.io/pypi/v/ANFISpy)](https://pypi.org/project/ANFISpy/)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)]()

# ANFISpy
A Python implementation of **Adaptive Neuro-Fuzzy Inference Systems (ANFIS)**, combining neural networks and fuzzy logic for interpretable machine learning. The implementation is based on the original [ANFIS](https://ieeexplore.ieee.org/abstract/document/256541?casa_token=bWStLllx3e8AAAAA:Z7Tj7kk-7lHlGSIEVJZfJVtRi_IVpig2ANbVv6qou4Ok32c7X7Yfh8SsvIUUBjALl3dfHRgFRJs3) paper, adapting the model to perform both regression and classification tasks with customizable membership functions.

# Key Features
- **Regression and Classification**  
- **Visualization and Interpretability** via `.print_rules()`, `.plot_var()`, `.plot_rules()`  
- **Various Membership Functions** (`GaussianMF`, `BellMF`, `TrinagularMF`, `SigmoidMF`)  
- **PyTorch Integration** (GPU acceleration, optimizers, ...) 

# Repository Organization
The repository is organized in the following directories:
- **ANFISpy**: has the implementation of the model, with the following subdirectories: ;
- **examples**: has jupyter-notebooks with synthetic data examples;
- **tests**: has testing files for managing the code behaviour.

# Installation
The installation of the package can be done using `pip` in a `bash` terminal:

```bash
pip install ANFISpy
