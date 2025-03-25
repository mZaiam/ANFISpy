import pytest
import torch
from ANFISpy.layers import Antecedents

n_features = 3
n_sets = [2, 5, 7]
n_samples = 11

memberships = [
    torch.randn(n_samples, n_sets[i]) for i in range(n_features)
]

def test_antecedents_initialization():
    ant = Antecedents(n_sets=n_sets, and_operator=torch.prod)
    assert ant.n_sets == n_sets
    assert ant.n_rules == n_sets[0] * n_sets[1] * n_sets[2]
    
def test_antecedents_output():
    ant = Antecedents(n_sets=n_sets, and_operator=torch.prod, mean_rule_activation=True)
    weights = ant(memberships)
    assert weights.shape[0] == n_samples
    assert weights.shape[1] == n_sets[0] * n_sets[1] * n_sets[2]
    assert ant.mean_rule_activation[0].shape[0] == n_sets[0] * n_sets[1] * n_sets[2]
