import pytest
import torch
from ANFISpy.mfs import GaussianMF, BellMF, SigmoidMF, TriangularMF

n_sets = 4
uod = [-1, 1]
n_samples = 10
x = torch.randn(n_samples, 1)

@pytest.fixture(
    params=[GaussianMF, BellMF, SigmoidMF, TriangularMF]
)

def mf_class(request):
    return request.param

def test_initialization(mf_class):
    mf = mf_class(n_sets=n_sets, uod=uod)
    assert mf.n_sets == n_sets
    assert mf.uod == uod

def test_output(mf_class):
    mf = mf_class(n_sets=n_sets, uod=uod)
    y = mf(x)
    assert y.shape[0] == n_samples
    assert y.shape[1] == n_sets
