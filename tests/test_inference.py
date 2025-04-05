import pytest
import torch
from ANFISpy.layers import InferenceRegression, InferenceClassification

n_rules = 2
n_samples = 11
n_classes = 5

ant = torch.randn(n_samples, n_rules)
cons_reg = torch.randn(n_samples, n_rules)
cons_cla = torch.randn(n_rules, n_samples, n_classes)

def test_inferencereg_initialization():
    inf = InferenceRegression()
    assert isinstance(inf.output_activation, torch.nn.Module)

def test_inferencereg_output():
    inf = InferenceRegression()
    out = inf(ant, cons_reg)
    assert out.shape[0] == n_samples

def test_inferencecla_initialization():
    inf = InferenceClassification()
    assert isinstance(inf, torch.nn.Module)

def test_inferencecla_output():
    inf = InferenceClassification()
    out = inf(ant, cons_cla)
    assert out.shape[0] == n_samples
    assert out.shape[1] == n_classes