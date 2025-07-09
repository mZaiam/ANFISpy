import pytest
import torch
import numpy as np
from ANFISpy.anfis import ANFIS

n_vars = 3
n_classes = 2
mf_names = [['L', 'M', 'H']]

variables_reg = {
    'inputs': {
        'n_sets': [3, 3, 3],
        'uod': n_vars * [(0, 1)],
        'var_names': ['a', 'b', 'c'],
        'mf_names': n_vars * mf_names,
    },
    'output': {
        'var_names': 'd',
        'n_classes': 1,
    },
}

variables_cla = {
    'inputs': {
        'n_sets': [3, 3, 3],
        'uod': n_vars * [(0, 1)],
        'var_names': ['a', 'b', 'c'],
        'mf_names': n_vars * mf_names,
    },
    'output': {
        'var_names': 'd',
        'n_classes': n_classes,
    },
}

batch_size = 5

x = torch.randn(batch_size, n_vars, dtype=torch.float32)
y_reg = torch.randn(batch_size, dtype=torch.float32)
y_cla = torch.ones(batch_size, dtype=torch.long)

mf_type = ['gaussian', 'bell', 'sigmoid', 'triangular']

@pytest.mark.parametrize("mf_type", mf_type)
def test_initialization_regression(mf_type):
    anfis = ANFIS(variables_reg, mf_type)
    assert len(anfis.input_n_sets) == 3
    assert len(anfis.input_uod) == 3
    assert len(anfis.input_var_names) == 3
    assert anfis.output_var_names == 'd'
    assert anfis.output_n_classes == 1
    
@pytest.mark.parametrize("mf_type", mf_type)
def test_initialization_classification(mf_type):
    anfis = ANFIS(variables_cla, mf_type)
    assert len(anfis.input_n_sets) == 3
    assert len(anfis.input_uod) == 3
    assert len(anfis.input_var_names) == 3
    assert anfis.output_var_names == 'd'
    assert anfis.output_n_classes == n_classes

@pytest.mark.parametrize("mf_type", mf_type)
def test_regression_forward(mf_type):
    anfis = ANFIS(variables_reg, mf_type)
    y_pred = anfis(x)
    assert y_pred.shape[0] == batch_size

@pytest.mark.parametrize("mf_type", mf_type)    
def test_classification_forward(mf_type):
    anfis = ANFIS(variables_cla, mf_type)
    y_pred = anfis(x)
    assert y_pred.shape == (batch_size, n_classes)

@pytest.mark.parametrize("mf_type", mf_type)
def test_training_regression(mf_type):
    anfis = ANFIS(variables_reg, mf_type)
    optimizer = torch.optim.Adam(anfis.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    for _ in range(5):  
        optimizer.zero_grad()
        y_pred = anfis(x)
        loss = criterion(y_pred, y_reg)
        loss.backward()
        optimizer.step()
        
    for name, param in anfis.named_parameters():
        assert param.grad is not None
        assert not torch.all(param.grad == 0)
        
@pytest.mark.parametrize("mf_type", mf_type)
def test_training_classification(mf_type):
    anfis = ANFIS(variables_cla, mf_type)
    optimizer = torch.optim.Adam(anfis.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    for _ in range(5):  
        optimizer.zero_grad()
        y_pred = anfis(x)
        loss = criterion(y_pred, y_cla)
        loss.backward()
        optimizer.step()
        
    for name, param in anfis.named_parameters():
        assert param.grad is not None
        assert not torch.all(param.grad == 0)
