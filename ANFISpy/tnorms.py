import torch

class MinAND:
    '''Calculates the minimum t-norm along a given dimension.'''
    def __init__(self):
        pass

    def __call__(self, x, dim):
        return torch.min(x, dim=dim).values
    
class ProdAND:
    '''Calculates the product t-norm along a given dimension.'''
    def __init__(self):
        pass

    def __call__(self, x, dim):
        return torch.prod(x, dim=dim)

class FrankAND:
    '''Calculates the Frank t-norm along a given dimension.
    
    Args:
        p: float for parameter p. It should be positive and different than 1.
    '''
    def __init__(self, p=2):
        self.log_p = torch.log(torch.tensor(p))

    def __call__(self, x, dim):
        p = torch.exp(self.log_p).to(x.device)
        arg = 1 + torch.prod(p**x - 1, dim=dim) / (p - 1)
        return torch.log(arg) / self.log_p.to(x.device)

class HamacherAND:
    '''Calculates the Frank t-norm along a given dimension.
    
    Args:
        r: float for parameter r. It should be positive.
    '''
    def __init__(self, r=1):
        self.r = r

    def __call__(self, x, dim):
        prod_x = torch.prod(x, dim=dim)
        sum_x = torch.sum(x, dim=dim)
        return prod_x / (self.r + (1 - self.r) * (sum_x - prod_x))

class LukasiewiczAND:
    '''Calculates the Lukasiewicz t-norm along a given dimension.'''
    def __init__(self):
        pass

    def __call__(self, x, dim):
        n = x.shape[dim]
        sum_x = torch.sum(x, dim=dim)
        out = sum_x - (n - 1)
        return torch.clamp(out, min=0)