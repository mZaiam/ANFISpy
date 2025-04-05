import torch
import torch.nn as nn

import itertools

class Antecedents(nn.Module):
    def __init__(self, n_sets, and_operator, mean_rule_activation=False):
        '''Calculates the antecedent values of the rules. Makes all possible combinations from the fuzzy sets defined for              
        each variable, considering rules of the form: var1 is set1 and ... and varn is setn.

        Args:
            n_sets:               list with the number of fuzzy sets associated to each variable.
            and_operator:         torch function for agregation of the membership values, modeling the AND operator.
            mean_rule_activation: bool to keep mean rule activation values.

        Tensors:
            memberships:          tensor (n) with tensors (N, nj) containing the membership values of each variable.
            weight:               tensor (N) representing the activation weights of a certain rule for all inputs.
            antecedents:          tensor (N, R) with the activation weights for all rules.
        '''

        super(Antecedents, self).__init__()

        self.n_sets = n_sets
        self.n_rules = torch.prod(torch.tensor(n_sets))
        self.and_operator = and_operator
        self.combinations = list(itertools.product(*[range(i) for i in n_sets]))
        self.mean_rule_activation = []
        self.bool = mean_rule_activation

    def forward(self, memberships):
        N = memberships[0].size(0)
        antecedents = []

        for combination in self.combinations:
            mfs = [] 
            
            for var_index, set_index in enumerate(combination):
                mfs.append(memberships[var_index][:, set_index])
            
            weight = self.and_operator(torch.stack(mfs, dim=1), dim=1)
            
            if isinstance(weight, tuple):  
                weight = weight[0]  
            
            antecedents.append(weight)

        antecedents = torch.stack(antecedents, dim=1)
        
        if self.bool:
            with torch.no_grad():
                self.mean_rule_activation.append(torch.mean(antecedents, dim=0))    
            
        return antecedents
    
class ConsequentsRegression(nn.Module):
    def __init__(self, n_sets):
        '''Calculates the consequent values of the system for a regression problem, considering a linear combination of the            
        input variables.

        Args:
            n_sets:       list with the number of fuzzy sets associated to each variable.

        Tensors:
            x:            tensor (N, n) containing the inputs of a variable.
            A:            tensor (R, n) with the linear coefficients (opt).
            b:            tensor (R) with the bias coefficients (opt).
            consequents:  tensor (N, R) containing the consequents of each rule.
        '''

        super(ConsequentsRegression, self).__init__()

        n_vars = len(n_sets)
        n_rules = torch.prod(torch.tensor(n_sets))

        self.A = nn.Parameter(torch.randn(n_rules, n_vars))
        self.b = nn.Parameter(torch.randn(n_rules))

    def forward(self, x):
        consequents = x @ self.A.T + self.b 
        return consequents
    
class ConsequentsClassification(nn.Module):
    def __init__(self, n_sets, n_classes):
        '''Calculates the consequent values of the system for a classification problem, considering a linear combination of            
        the input variables.

        Args:
            n_sets:       list with the number of fuzzy sets associated to each variable.
            n_classes:    int with number of n_classes.

        Tensors:
            x:            tensor (N, n) containing the inputs of a variable.
            A:            tensor (R, m, n) with the linear coefficients (opt).
            b:            tensor (R, m) with the bias coefficients (opt).
            consequents:  tensor (R, N, m) containing the consequents of each rule.
        '''

        super(ConsequentsClassification, self).__init__()

        n_vars = len(n_sets)
        n_rules = torch.prod(torch.tensor(n_sets))

        self.A = nn.Parameter(torch.randn(n_rules, n_classes, n_vars))
        self.b = nn.Parameter(torch.randn(n_rules, n_classes))

    def forward(self, x):
        consequents = torch.matmul(self.A, x.T).permute(0, 2, 1) + self.b.unsqueeze(1)
        return consequents

class InferenceRegression(nn.Module):
    def __init__(self, output_activation=nn.Identity()):
        '''Performs the Takagi-Sugeno-Kang inference for a regression problem.
        
        Args:
            output_activation: nn.Module for output activation function.
        
        Tensors:
            antecedents:       tensor (N, R) with the weights of activation of each rule.
            consequents:       tensor (N, R) with the outputs of each rule.
            Y:                 tensor (N) with the outputs of the system.
            output_activation: torch function.
        '''
        
        super(InferenceRegression, self).__init__()
        
        self.output_activation = output_activation

    def forward(self, antecedents, consequents):
        Y = torch.sum(antecedents * consequents, dim=1, keepdim=True) / torch.sum(antecedents, dim=1, keepdim=True)
        Y = self.output_activation(Y)
        return Y
    
class InferenceClassification(nn.Module):
    def __init__(self):
        '''Performs the Takagi-Sugeno-Kang inference for a classification problem.

        Tensors:
            antecedents: tensor (N, R) with the weights of activation of each rule.
            consequents: tensor (R, N, m) with the outputs of each rule.
            Y:           tensor (N, m) with the outputs of the system.
        '''

        super(InferenceClassification, self).__init__()

    def forward(self, antecedents, consequents):
        Y = torch.sum(antecedents.T.unsqueeze(-1) * consequents, dim=0) / torch.sum(antecedents, dim=1, keepdim=True)
        return Y
