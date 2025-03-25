import pytest
import torch
from ANFISpy.layers import ConsequentsRegression, ConsequentsClassification

n_vars = 2
n_sets = [2, 3]
n_rules = 6 
n_samples = 11
x = torch.randn(n_samples, n_vars)

def test_consequentreg_initialization():
    cons = ConsequentsRegression(n_sets=n_sets)
        
def test_consequentreg_output():
    cons = ConsequentsRegression(n_sets=n_sets)
    out = cons(x)
    assert out.shape[0] == n_samples
    assert out.shape[1] == n_rules
