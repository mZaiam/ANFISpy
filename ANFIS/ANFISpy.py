import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import itertools

'''
Notation:

N:  batch size.
n:  number of features.
nj: number of fuzzy sets from variable j.
R:  number of rules.
'''

class GaussianMF(nn.Module):
    def __init__(self, n_sets, uod):
        '''Gaussian membership function. Receives the input tensor of a variable and outputs the tensor with the membership            values.

        Args:
            n_sets:      int for number of fuzzy sets associated to the variable.
            uod:         list/tuple with the universe of discourse of the variable.

        Tensors:
            x:           tensor (N) containing the inputs of a variable.
            mu:          tensor (n) representing the mu parameter for the gaussian set (opt).
            sigma:       tensor (n) representing the sigma parameter for the gaussian set (opt).
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
        '''Triangular membership function. Receives the input tensor of a variable and outputs the tensor with the                      membership values.

        Args:
            n_sets: int for number of fuzzy sets associated to the variable.
            uod:    list/tuple with the universe of discourse of the variable.

        Tensors:
            x:      tensor (N) containing the inputs of a variable.
            a:      tensor (N) containing the left point of the triangle (opt).
            b:      tensor (N) containing the center point of the triangle (opt).
            c:      tensor (N) containing the right point of the triangle (opt).
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

        Tensors:
            x:           tensor (N) containing the inputs of a variable.
            a:           tensor (n) representing the a parameter for the bell set (opt).
            b:           tensor (n) representing the b parameter for the bell set (opt).
            c:           tensor (n) representing the c parameter for the bell set (opt).
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

        Tensors:
            x:           tensor (N) containing the inputs of a variable.
            a:           tensor (n) representing the a parameter for the sigmoid set (opt).
            c:           tensor (n) representing the c parameter for the sigmoid set (opt).
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
    
class Antecedents(nn.Module):
    def __init__(self, n_sets, and_operator, mean_rule_activation=False):
        '''Calculates the antecedent values of the rules. Makes all possible combinations from the fuzzy sets defined for              each variable, considering rules of the form: var1 is set1 and ... and varn is setn.

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
        '''Calculates the consequent values of the system for a regression problem, considering a linear combination of the            input variables.

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
        '''Calculates the consequent values of the system for a classification problem, considering a linear combination of            the input variables.

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
    def __init__(self, output_activation=nn.ReLU()):
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

class ANFIS(nn.Module):
    def __init__(
        self, 
        variables, 
        mf_shape, 
        and_operator=torch.prod, 
        output_activation=nn.Identity(), 
        mean_rule_activation=False
    ):
        '''Adaptative Neuro-Fuzzy Inference System with Takagi-Sugeno-Kang architecture. Can perform both regression and
           classification.

        Args:
            variables:            dict with two keys, 'inputs' and 'output'. The 'input' has a dict as its value,
                                  containing four keys: 'n_sets', 'uod', 'var_names' and 'mf_names'. They have lists as                                       their values, containing, respectively: int with number of fuzzy sets associated to the                                     variable, tuple/list with the universe of discourse of the variable, str with the name of                                   the variable and list of str with the name of the fuzzy sets. The lists need to be the                                       same length, and the index of them are all associated, that is, index 0 represents the                                       information of the same variable. Now, 'output' has only the keys 'var_names' and                                           'n_classes', with a str representing the name of the variable and an int with the number                                     of classes (if the model is a regressor, insert 1). 
            mf_shape:             str containing the shape of the fuzzy sets of the system. Supports 'triangular', 'bell'
                                  'sigmoid' and 'gaussian'.
            and_operator:         torch function to model the AND in the antecedents calculation.
            output_activation:    torch function for output activation function.
            mean_rule_activation: bool to keep the mean rule activation values. 
        '''

        super(ANFIS, self).__init__()

        self.input_n_sets = variables['inputs']['n_sets']
        self.input_uod = variables['inputs']['uod']
        self.input_var_names = variables['inputs']['var_names']
        self.input_mf_names = variables['inputs']['mf_names']
        
        self.output_var_names = variables['output']['var_names']
        self.output_n_classes = variables['output']['n_classes']
        
        self.mf_shape = mf_shape
        self.and_operator = and_operator
        
        if mf_shape == 'gaussian':
            self.memberships = nn.ModuleList(
                [GaussianMF(n_sets_, uod_) for n_sets_, uod_ in zip(self.input_n_sets, self.input_uod)]
            )
            
        elif mf_shape == 'triangular':
            self.memberships = nn.ModuleList(
                [TriangularMF(n_sets_, uod_) for n_sets_, uod_ in zip(self.input_n_sets, self.input_uod)]
            )
            
        if mf_shape == 'bell':
            self.memberships = nn.ModuleList(
                [BellMF(n_sets_, uod_) for n_sets_, uod_ in zip(self.input_n_sets, self.input_uod)]
            )
            
        if mf_shape == 'sigmoid':
            self.memberships = nn.ModuleList(
                [SigmoidMF(n_sets_, uod_) for n_sets_, uod_ in zip(self.input_n_sets, self.input_uod)]
            )
            
        self.antecedents = Antecedents(self.input_n_sets, and_operator, mean_rule_activation)
        
        if self.output_n_classes == 1:
            self.consequents = ConsequentsRegression(self.input_n_sets)
            self.inference = InferenceRegression(output_activation)

        else:
            self.consequents = ConsequentsClassification(self.input_n_sets, self.output_n_classes)
            self.inference = InferenceClassification()
        
    def forward(self, x):
        memberships = [mf(x[:, i]) for i, mf in enumerate(self.memberships)]
        antecedents = self.antecedents(memberships)
        consequents = self.consequents(x)
        Y = self.inference(antecedents, consequents)
        
        return Y

    def plot_var(self, var_name, file_name=False):
        '''Plots the membership functions for a certain variable of the model.

        Args:
            var_name:  str with the name of the variable, written in the same way as given in dict variables.
            file_name: str with the name of the file to be saved, if desired.
        '''

        var_index = self.input_var_names.index(var_name)
        n_sets = self.input_n_sets[var_index]
        uod = torch.linspace(*self.input_uod[var_index], 1000)
        mf_names = self.input_mf_names[var_index]
        mf_function = self.memberships[var_index]

        memberships = mf_function(uod)

        for i in range(n_sets):
            plt.plot(uod.numpy(), memberships[:, i].detach().numpy(), label=f'{mf_names[i]}')

        plt.title(f'Membership Functions for Variable {var_name}')
        plt.xlabel('Universe of Discourse')
        plt.ylabel('Membership')
        plt.legend()

        if isinstance(file_name, str):
            plt.savefig(f'{file_name}.png', bbox_inches='tight')

        plt.show()

    def rules(self, mean_rule_activation=False):
        '''Returns a list with the rules of the model in str format.
        
        Args:
            mean_rule_activation: bool to return mean rule activation.
            
        Returns:
            rules:                list of str representing the rules of the system.
            mean:                 numpy array (R) with normalized mean rule activation.
        '''
        
        combinations = list(itertools.product(*[range(i) for i in self.input_n_sets]))
        rules = []

        for i, combination in enumerate(combinations):
            rule = 'IF '
            for var_index, set_index in enumerate(combination):
                if var_index != len(combination) - 1:
                    rule += f'{self.input_var_names[var_index]} IS {self.input_mf_names[var_index][set_index]} AND '
                else:
                    rule += f'{self.input_var_names[var_index]} IS {self.input_mf_names[var_index][set_index]}, '

            rule += f'THEN {self.output_var_names} IS f{i}'
            rules.append(rule)
        
        if mean_rule_activation:
            mean = torch.mean(torch.stack(self.antecedents.mean_rule_activation, dim=0), dim=0)
            mean /= torch.sum(mean)
            mean = mean.numpy()
            
            return rules, mean
            
        return rules
    
    def plot_rules(
        self, 
        var_names, 
        n_points=1000, 
        thr=0.8, 
        levels=10, 
        cmap='viridis', 
        alpha=0.3,
        x_data=None,
        y_data=None,
        file_name=None,
    ):
        '''Plot the projection of the fuzzy rules in a two variable space.
        
        Args:
            var_names: list/tuple with the variables names.
            n_points:  int with 
            thr:       float between 0 and 1 with the threshold value of the fuzzy rules activation.
            levels:    same as matplotlib.pyplot.
            cmap:      same as matplotlib.pyplot.
            alpha:     same as matplotlib.pyplot.
            x_data:    data for scatter plot.
            y_data:    data for scatter plot.
            file_name: str with the name of the file to be saved, if desired.
        '''
        
        var_index = self.input_var_names.index(var_names[0])
        n_sets0 = self.input_n_sets[var_index]
        uod0 = torch.linspace(*self.input_uod[var_index], n_points)
        mf_names0 = self.input_mf_names[var_index]
        mf_function0 = self.memberships[var_index]

        memberships0 = mf_function0(uod0)
        
        var_index = self.input_var_names.index(var_names[1])
        n_sets1 = self.input_n_sets[var_index]
        uod1 = torch.linspace(*self.input_uod[var_index], n_points)
        mf_names1 = self.input_mf_names[var_index]
        mf_function1 = self.memberships[var_index]

        memberships1 = mf_function1(uod1)
        
        x = np.linspace(uod0[0], uod0[-1], n_points)
        y = np.linspace(uod1[0], uod1[-1], n_points)
        X, Y = np.meshgrid(x, y)
        
        fig = plt.figure(figsize=(7, 7))
        gs = fig.add_gridspec(2, 2, width_ratios=[5, 1.5], height_ratios=[1.5, 5], wspace=0.08, hspace=0.08)

        ax_main = fig.add_subplot(gs[1, 0])  
        ax_top = fig.add_subplot(gs[0, 0])  
        ax_right = fig.add_subplot(gs[1, 1])  
        
        for i in range(n_sets0):
            for j in range(n_sets1):
                if self.and_operator == torch.prod:
                    Z = np.outer(memberships1[:, j].detach().numpy(), memberships0[:, i].detach().numpy())
                elif self.and_operator == torch.min:
                    Z = np.minimum.outer(memberships1[:, j].detach().numpy(), memberships0[:, i].detach().numpy())
                
                Z[Z < thr] = np.nan
                ax_main.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=alpha)
                
        ax_main.set_xlabel(var_names[0])
        ax_main.set_ylabel(var_names[1])
        
        cbar_ax = fig.add_axes([ax_main.get_position().x0 - 0.16, 
                        ax_main.get_position().y0,  
                        0.02,  
                        ax_main.get_position().height])  

        cbar = fig.colorbar(ax_main.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=alpha), cax=cbar_ax, orientation="vertical")

        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")
        
        for i in range(n_sets0):
            ax_top.plot(uod0, memberships0[:, i].detach().numpy(), label=f'{mf_names0[i]}', lw=2)
        
        ax_top.legend()
        ax_top.set_xticks([])
        ax_top.set_yticks([])

        for i in range(n_sets1):
            ax_right.plot(memberships1[:, i].detach().numpy(), uod1, label=f'{mf_names1[i]}', lw=2)

        ax_right.legend()
        ax_right.set_xticks([])
        ax_right.set_yticks([])
        
        ax_top.set_xlim(ax_main.get_xlim())
        ax_right.set_ylim(ax_main.get_ylim())

        ax_top.set_frame_on(False)
        ax_right.set_frame_on(False)
        
        if x_data is None:
            pass
        else:
            ax_main.scatter(x_data, y_data, color='red')
        
        if isinstance(file_name, str):
            plt.savefig(f'{file_name}.png', bbox_inches='tight')

        plt.show()
