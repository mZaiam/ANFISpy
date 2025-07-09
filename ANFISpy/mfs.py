import torch
import torch.nn as nn

class GaussianMF(nn.Module):
    def __init__(self, n_sets, uod):
        '''Gaussian membership function. Receives the input tensor of a variable and outputs the tensor with the membership            
        values.

        Args:
            n_sets:      int for number of fuzzy sets associated to the variable.
            uod:         list/tuple with the universe of discourse of the variable.

        Returns:
            memberships: tensor (N, n) containing the membership values of the inputs.
        '''

        super(GaussianMF, self).__init__()

        self.n_sets = n_sets
        self.uod = uod
        
        step = ((uod[1] - uod[0]) / (n_sets - 1))
        
        self.mu = nn.Parameter(torch.linspace(*uod, n_sets))
        self.sigma = nn.Parameter((step / 2) * torch.ones(n_sets))

    def forward(self, x):
        x = x.reshape(-1, 1).repeat(1, self.n_sets)
        mu = torch.minimum(torch.maximum(self.mu, torch.tensor(self.uod[0])), torch.tensor(self.uod[1]))
        sigma = nn.functional.relu(self.sigma) + 1e-6
        memberships = torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        return memberships
    
class TriangularMF(nn.Module):
    def __init__(self, n_sets, uod):
        '''Triangular membership function. Receives the input tensor of a variable and outputs the tensor with the                      
        membership values.

        Args:
            n_sets: int for number of fuzzy sets associated to the variable.
            uod:    list/tuple with the universe of discourse of the variable.

        Returns:
            memberships: tensor (N, n) containing the membership values of the inputs.
        '''

        super(TriangularMF, self).__init__()

        self.n_sets = n_sets
        self.uod = uod
        
        step = ((uod[1] - uod[0]) / (n_sets - 1))
        
        self.b = nn.Parameter(torch.linspace(*uod, n_sets))
        self.deltaL = nn.Parameter(step * torch.ones(n_sets))
        self.deltaR = nn.Parameter(step * torch.ones(n_sets))

    def forward(self, x, delta=10e-8):
        x = x.reshape(-1, 1).repeat(1, self.n_sets)
        b = torch.minimum(torch.maximum(self.b, torch.tensor(self.uod[0])), torch.tensor(self.uod[1]))
        deltaL = nn.functional.relu(self.deltaL) + 1e-6
        deltaR = nn.functional.relu(self.deltaR) + 1e-6
        left = (x + deltaL - b) / (deltaL + delta)
        right = (deltaR + b - x) / (deltaR + delta)
        memberships = torch.maximum(torch.minimum(left, right), torch.tensor(0.0))
        return memberships

class BellMF(nn.Module):
    def __init__(self, n_sets, uod):
        '''Generalized bell membership function. Receives the input tensor of a variable and outputs the tensor with the 
        membership values.

        Args:
            n_sets:      int for number of fuzzy sets associated to the variable.
            uod:         list/tuple with the universe of discourse of the variable.

        Returns:
            memberships: tensor (N, n) containing the membership values of the inputs.
        '''

        super(BellMF, self).__init__()

        self.n_sets = n_sets
        self.uod = uod
        
        step = ((uod[1] - uod[0]) / (n_sets - 1))
        
        self.c = nn.Parameter(torch.linspace(*uod, n_sets))
        self.a = nn.Parameter((step / 2) * torch.ones(n_sets))
        self.b = nn.Parameter(step * torch.ones(n_sets))

    def forward(self, x):
        x = x.reshape(-1, 1).repeat(1, self.n_sets)
        c = torch.minimum(torch.maximum(self.c, torch.tensor(self.uod[0])), torch.tensor(self.uod[1]))
        a = nn.functional.relu(self.a) + 1e-6
        b = nn.functional.relu(self.b) + 1e-6
        memberships = 1 / (1 + torch.abs((x - c) / (a)) ** (2 * b))
        return memberships
    
class SigmoidMF(nn.Module):
    def __init__(self, n_sets, uod):
        '''Sigmoid membership function. Receives the input tensor of a variable and outputs the tensor with the 
        membership values.

        Args:
            n_sets:      int for number of fuzzy sets associated to the variable.
            uod:         list/tuple with the universe of discourse of the variable.

        Returns:
            memberships: tensor (N, n) containing the membership values of the inputs.
        '''

        super(SigmoidMF, self).__init__()

        self.n_sets = n_sets
        self.uod = uod
        
        step = ((uod[1] - uod[0]) / (n_sets - 1))
        delta = uod[1] - uod[0]
    
        self.c = nn.Parameter(torch.linspace(uod[0] + 0.05 * delta, uod[1] - 0.05 * delta, n_sets))
        self.a = nn.Parameter(2 * step * torch.ones(n_sets))

    def forward(self, x):
        x = x.reshape(-1, 1).repeat(1, self.n_sets)
        c = torch.minimum(torch.maximum(self.c, torch.tensor(self.uod[0])), torch.tensor(self.uod[1]))
        memberships = 1 / (1 + torch.exp(- self.a * (x - c)))
        return memberships