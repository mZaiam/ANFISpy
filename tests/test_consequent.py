import pytest
import torch
from ANFISpy.layers import ConsequentsRegression, ConsequentsClassification

n_vars = 2
n_sets = [2, 3]
n_rules = 6
n_samples = 11
n_classes = 5
x = torch.randn(n_samples, n_vars)

def test_consequentreg_initialization():
    cons = ConsequentsRegression(n_sets=n_sets)
    assert isinstance(cons.A, torch.nn.Parameter)
    assert isinstance(cons.b, torch.nn.Parameter)
    assert cons.A.shape[0] == n_rules
    assert cons.A.shape[1] == n_vars
    assert cons.b.shape[0] == n_rules

def test_consequentreg_output():
    cons = ConsequentsRegression(n_sets=n_sets)
    out = cons(x)
    assert out.shape[0] == n_samples
    assert out.shape[1] == n_rules
    
def test_consequentcla_initialization():
    cons = ConsequentsClassification(n_sets=n_sets, n_classes=n_classes)
    assert isinstance(cons.A, torch.nn.Parameter)
    assert isinstance(cons.b, torch.nn.Parameter)
    assert cons.A.shape[0] == n_rules
    assert cons.A.shape[1] == n_classes
    assert cons.A.shape[2] == n_vars
    assert cons.b.shape[0] == n_rules
    assert cons.b.shape[1] == n_classes

def test_consequentcla_output():
    cons = ConsequentsClassification(n_sets=n_sets, n_classes=n_classes)
    out = cons(x)
    assert out.shape[0] == n_rules
    assert out.shape[1] == n_samples
    assert out.shape[2] == n_classes
