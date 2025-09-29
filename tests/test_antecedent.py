import pytest
import torch
from ANFISpy.layers import Antecedents
from ANFISpy.tnorms import LukasiewiczAND, MinAND, ProdAND, HamacherAND, FrankAND 

n_features = 3
n_sets = [2, 5, 7]
n_samples = 11

memberships = [
    torch.randn(n_samples, n_sets[i]) for i in range(n_features)
]

@pytest.fixture(
    params=[LukasiewiczAND, MinAND, ProdAND, HamacherAND, FrankAND]
)

def and_class(request):
    return request.param

def test_antecedents_initialization(and_class):
    and_op  = and_class()
    ant = Antecedents(n_sets=n_sets, and_operator=and_op)
    assert ant.n_sets == n_sets
    assert ant.n_rules == n_sets[0] * n_sets[1] * n_sets[2]
    
def test_antecedents_output(and_class):
    and_op  = and_class()
    ant = Antecedents(n_sets=n_sets, and_operator=and_op)
    weights = ant(memberships)
    assert weights.shape[0] == n_samples
    assert weights.shape[1] == n_sets[0] * n_sets[1] * n_sets[2]
