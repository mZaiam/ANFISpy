import pytest
import torch
from ANFISpy.layers import Consequents

n_vars = 2
n_sets = [2, 3]
n_rules = 6
n_samples = 11
n_classes = 5
seq_len = 7
x = torch.randn(n_samples, n_vars)
x_rec = torch.randn(n_samples, seq_len, n_vars)

def test_consequentreg_output():
    cons = Consequents(n_sets=n_sets, n_classes=1)
    out = cons(x)
    assert out.shape[0] == n_samples
    assert out.shape[1] == n_rules * 1

def test_consequentcla_output():
    cons = Consequents(n_sets=n_sets, n_classes=n_classes)
    out = cons(x)
    assert out.shape[0] == n_samples
    assert out.shape[1] == n_rules * n_classes
    
def test_consequentreg_rec_output():
    cons = Consequents(n_sets=n_sets, n_classes=1)
    out = cons(x_rec)
    assert out.shape[0] == n_samples
    assert out.shape[1] == seq_len
    assert out.shape[2] == n_rules * 1

def test_consequentcla_rec_output():
    cons = Consequents(n_sets=n_sets, n_classes=n_classes)
    out = cons(x_rec)
    assert out.shape[0] == n_samples
    assert out.shape[1] == seq_len
    assert out.shape[2] == n_rules * n_classes