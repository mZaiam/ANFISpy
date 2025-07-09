import pytest
import torch
from ANFISpy.layers import Inference, RecurrentInference

n_rules = 2
n_samples = 11
n_classes = 5
seq_len = 7

ant = torch.randn(n_samples, n_rules)
ant_rec = torch.randn(n_samples * seq_len, n_rules)
cons_reg = torch.randn(n_samples, n_rules)
cons_cla = torch.randn(n_samples, n_rules * n_classes)
cons_rec_reg = torch.randn(n_samples, seq_len, n_rules)
cons_rec_cla = torch.randn(n_samples, seq_len, n_rules * n_classes)
h_reg = torch.randn(n_samples, seq_len, 1)
h_cla = torch.randn(n_samples, seq_len, n_classes)

def test_inferencereg_output():
    inf = Inference(n_classes=1)
    out = inf(ant, cons_reg)
    assert out.shape[0] == n_samples

def test_inferencecla_output():
    inf = Inference(n_classes=n_classes)
    out = inf(ant, cons_cla)
    assert out.shape[0] == n_samples
    assert out.shape[1] == n_classes
    
def test_inferencereg_rec_output():
    inf = RecurrentInference(n_classes=1, seq_len=seq_len)
    out = inf(ant_rec, cons_rec_reg, h_reg)
    assert out.shape[0] == n_samples
    assert out.shape[1] == seq_len
    assert out.shape[2] == 1

def test_inferencecla_rec_output():
    inf = RecurrentInference(n_classes=n_classes, seq_len=seq_len)
    out = inf(ant_rec, cons_rec_cla, h_cla)
    assert out.shape[0] == n_samples
    assert out.shape[1] == seq_len
    assert out.shape[2] == n_classes