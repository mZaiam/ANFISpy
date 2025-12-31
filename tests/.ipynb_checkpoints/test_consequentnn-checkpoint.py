import pytest
import torch
from ANFISpy.layers import ConsequentsNN

n_vars = 2
n_sets = [2, 3]
n_rules = 6
n_samples = 11
n_classes = 5
seq_len = 7
x = torch.randn(n_samples, n_vars)
x_rec = torch.randn(n_samples, seq_len, n_vars)

@pytest.mark.parametrize("nn_layers", [1, 2])
def test_consequentnnreg_output():
    cons = ConsequentsNN(
        in_features=len(n_sets),
        n_rules=n_rules,
        n_classes=1,
        n_layers=nn_layers,
    )
    out = cons(x)
    assert out.shape[0] == n_samples
    assert out.shape[1] == n_rules * 1

@pytest.mark.parametrize("nn_layers", [1, 2])
def test_consequentnncla_output():
    cons = ConsequentsNN(
        in_features=len(n_sets),
        n_rules=n_rules,
        n_classes=n_classes,
        n_layers=nn_layers,
    )
    out = cons(x)
    assert out.shape[0] == n_samples
    assert out.shape[1] == n_rules * n_classes

@pytest.mark.parametrize("nn_layers", [1, 2])
def test_consequentnnreg_rec_output():
    cons = ConsequentsNN(
        in_features=len(n_sets),
        n_rules=n_rules,
        n_classes=1,
        n_layers=nn_layers,
    )
    out = cons(x_rec)
    assert out.shape[0] == n_samples
    assert out.shape[1] == seq_len
    assert out.shape[2] == n_rules * 1

@pytest.mark.parametrize("nn_layers", [1, 2])
def test_consequentnncla_rec_output():
    cons = ConsequentsNN(
        in_features=len(n_sets),
        n_rules=n_rules,
        n_classes=1,
        n_layers=nn_layers,
    )
    out = cons(x_rec)
    assert out.shape[0] == n_samples
    assert out.shape[1] == seq_len
    assert out.shape[2] == n_rules * n_classes