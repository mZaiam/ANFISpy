import pytest
import torch
from ANFISpy.layers import RecurrentInferenceRegression, RecurrentInferenceClassification
from ANFISpy.layers import RecurrentLayerRegression, RecurrentLayerClassification
from ANFISpy.layers import LSTMLayerRegression, LSTMLayerClassification

n_rules = 2
n_samples = 11
n_classes = 5

ant = torch.randn(n_samples, n_rules)
cons_reg = torch.randn(n_samples, n_rules)
cons_cla = torch.randn(n_rules, n_samples, n_classes)
h_reg = torch.randn(n_samples, n_rules)
h_cla = torch.randn(n_rules, n_samples, n_classes)
c_reg = h_reg
c_cla = h_cla

def test_inferencereg_initialization():
    inf = RecurrentInferenceRegression()
    assert isinstance(inf.output_activation, torch.nn.Module)

def test_inferencereg_output():
    inf = RecurrentInferenceRegression()
    out = inf(ant, cons_reg, h_reg)
    assert out.shape[0] == n_samples

def test_inferencecla_initialization():
    inf = RecurrentInferenceClassification()
    assert isinstance(inf, torch.nn.Module)

def test_inferencecla_output():
    inf = RecurrentInferenceClassification()
    out = inf(ant, cons_cla, h_cla)
    assert out.shape[0] == n_samples
    assert out.shape[1] == n_classes
    
def test_recurrentreg_initialization():
    rec = RecurrentLayerRegression(n_rules)
    assert isinstance(rec, torch.nn.Module)

def test_recurrentreg_output():
    rec = RecurrentLayerRegression(n_rules)
    out = rec(cons_reg, h_reg)
    assert out.shape[0] == n_samples
    assert out.shape[1] == n_rules
    
def test_recurrentcla_initialization():
    rec = RecurrentLayerClassification(n_rules)
    assert isinstance(rec, torch.nn.Module)

def test_recurrentcla_output():
    rec = RecurrentLayerClassification(n_rules)
    out = rec(cons_cla, h_cla)
    assert out.shape[0] == n_rules
    assert out.shape[1] == n_samples
    assert out.shape[2] == n_classes
    
def test_lstmreg_initialization():
    rec = LSTMLayerClassification(n_rules)
    assert isinstance(rec, torch.nn.Module)

def test_lstmreg_output():
    rec = LSTMLayerRegression(n_rules)
    h, c = rec(cons_reg, h_reg, c_reg)
    assert h.shape[0] == n_samples
    assert h.shape[1] == n_rules
    assert c.shape[0] == n_samples
    assert c.shape[1] == n_rules
    
def test_lstmcla_initialization():
    rec = LSTMLayerClassification(n_rules)
    assert isinstance(rec, torch.nn.Module)

def test_lstmcla_output():
    rec = LSTMLayerClassification(n_rules)
    h, c = rec(cons_cla, h_cla, c_cla)
    assert h.shape[0] == n_rules
    assert h.shape[1] == n_samples
    assert h.shape[2] == n_classes
    assert c.shape[0] == n_rules
    assert c.shape[1] == n_samples
    assert c.shape[2] == n_classes