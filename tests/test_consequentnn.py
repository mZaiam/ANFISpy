import pytest
import torch
from ANFISpy.layers import ConsequentsNN

n_vars = 2
n_sets = [2, 3]
n_rules = 6
n_samples = 11
n_classes = 5
x = torch.randn(n_samples, n_vars)

@pytest.mark.parametrize("nn_layers", [1, 2])
def test_consequentnnreg_output(nn_layers):
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
def test_consequentnncla_output(nn_layers):
    cons = ConsequentsNN(
        in_features=len(n_sets),
        n_rules=n_rules,
        n_classes=n_classes,
        n_layers=nn_layers,
    )
    out = cons(x)
    assert out.shape[0] == n_samples
    assert out.shape[1] == n_rules * n_classes
