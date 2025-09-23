import torch
import torch.nn as nn

import itertools

'''
Notation.

N: batch size;
n: number of features;
m: number of output classes;
nj: number of fuzzy sets for feature j;
R: number of rules;
L: sequence length.
'''

class Antecedents(torch.nn.Module):
    def __init__(self, n_sets, and_operator=torch.prod):
        '''
        Calculates the antecedent values of the rules. Makes all possible combinations from the fuzzy sets 
        defined for each variable, considering rules of the form: var1 is set1 and ... and varn is setn.

        Args:
            n_sets:               list with the number of fuzzy sets associated to each variable.
            and_operator:         torch function for aggregation of the membership values, modeling 
                                  the AND operator.
                                  
        Tensors:
            memberships:          list (n) with tensors (N, nj) containing the membership values of each variable.
            rule_indices:         tensor (R, n) with indices of fuzzy sets for each rule.
            antecedents:          tensor (N, R) with the activation weights for all rules.
        '''
        
        super().__init__()
        
        self.n_sets = n_sets
        self.n_rules = torch.prod(torch.tensor(n_sets)).item()
        self.and_operator = and_operator
        self.mean_rule_activation = []

        grids = torch.meshgrid([torch.arange(s) for s in n_sets], indexing="ij")
        rule_indices_tensor = torch.stack([g.reshape(-1) for g in grids], dim=1)
        self.register_buffer('rule_indices', rule_indices_tensor)

    def forward(self, memberships):
        N = memberships[0].size(0)
        n = len(self.n_sets)
        R = self.n_rules

        max_sets = max(self.n_sets)
        padded = []
        for j, nj in enumerate(self.n_sets):
            pad = (0, max_sets - nj)  
            padded.append(nn.functional.pad(memberships[j], pad))
        memb_tensor = torch.stack(padded, dim=0)  

        idx = self.rule_indices.T.unsqueeze(1).expand(-1, N, -1)  
        gathered = torch.gather(memb_tensor, 2, idx)

        antecedents = self.and_operator(gathered.permute(1, 2, 0), dim=2)
        if isinstance(antecedents, tuple): 
            antecedents = antecedents[0]

        return antecedents

class Consequents(nn.Module):
    def __init__(self, n_sets, n_classes):
        '''Calculates the consequents, considering a linear combination of the input variables.

        Args:
            n_sets:       list with the number of fuzzy sets associated to each variable.
            n_classes:    int with number of classes.

        Tensors:
            x:            tensor (N, n) containing the inputs of a variable.
            consequents:  tensor (N, R) or (N, R * m) containing the consequents of each rule.
        '''

        super(Consequents, self).__init__()

        self.n_vars = len(n_sets)
        self.n_rules = torch.prod(torch.tensor(n_sets)).item()
        self.n_classes = n_classes
        self.mode = 'regression' if n_classes == 1 else 'classification'

        if self.mode == 'regression':
            self.linear = nn.Linear(
                in_features=self.n_vars,
                out_features=self.n_rules,
            )
        
        if self.mode == 'classification':
            self.linear = nn.Linear(
                in_features=self.n_vars,
                out_features=self.n_rules * self.n_classes,
            )
            
    def forward(self, x):
        consequents = self.linear(x)
        return consequents

class Inference(nn.Module):
    def __init__(self, n_classes, output_activation=nn.Identity()):
        '''Performs the Takagi-Sugeno-Kang inference.
        
        Args:
            n_classes:    int with number of classes.
            output_activation: torch function.
        
        Tensors:
            antecedents:       tensor (N, R) with the weights of activation of each rule.
            consequents:       tensor (N, R) with the outputs of each rule.
            Y:                 tensor (N) or (N, m) with the outputs of the system.
            output_activation: torch function.
        '''
        
        super(Inference, self).__init__()
        
        self.n_classes = n_classes
        self.mode = 'regression' if n_classes == 1 else 'classification'
        self.output_activation = output_activation

    def forward(self, antecedents, consequents):
        w = antecedents / torch.sum(antecedents, dim=1, keepdim=True)
        if self.mode == 'classification':
            n_rules = w.shape[1]
            w = w.unsqueeze(-1)
            consequents = consequents.view(-1, n_rules, self.n_classes)
        y_hat = torch.sum(w * consequents, dim=1, keepdim=True).squeeze()
        return self.output_activation(y_hat)
    
class RecurrentInference(nn.Module):
    def __init__(self, n_classes, seq_len, bidirectional=False, output_activation=nn.Identity()):
        '''Performs the Takagi-Sugeno-Kang inference adapted for recurrent models.
        
        Args:
            n_classes:         int with number of classes.
            seq_len:           int with sequence length.
            bidirectional:     bool for directionality of model.
            output_activation: torch function.
        
        Returns:
            y_hat:             tensor (N, L, m) with outputs of the system.
        '''
        
        super(RecurrentInference, self).__init__()
        
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.bidirectional = bidirectional
        self.mode = 'regression' if n_classes == 1 else 'classification'
        self.output_activation = output_activation

    def forward(self, antecedents, consequents, h=None):
        if h is None:
            h = torch.zeros_like(consequents)
        w = antecedents / torch.sum(antecedents, dim=1, keepdim=True)
        n_rules = w.shape[1]
        if self.mode == 'regression':
            if self.bidirectional:
                h = h.view(-1, self.seq_len, 2, n_rules).mean(dim=2)
            consequents = consequents + h
            consequents = consequents.view(antecedents.shape)
        if self.mode == 'classification':
            if self.bidirectional:
                h = h.view(-1, self.seq_len, 2, self.n_classes * n_rules).mean(dim=2)
            w = w.unsqueeze(-1)
            consequents = consequents + h
            consequents = consequents.view(-1, n_rules, self.n_classes)
        y_hat = torch.sum(w * consequents, dim=1, keepdim=True).view(-1, self.seq_len, self.n_classes)
        return self.output_activation(y_hat)
