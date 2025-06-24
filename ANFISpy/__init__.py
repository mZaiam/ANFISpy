from .utils import _print_rules, _plot_var, _plot_rules
from .mfs import GaussianMF, BellMF, SigmoidMF, TriangularMF
from .layers import Antecedents, ConsequentsClassification, ConsequentsRegression, InferenceClassification, InferenceRegression
from .anfis import ANFIS, RANFIS

__version__ = "0.2.2"
