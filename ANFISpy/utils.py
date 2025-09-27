import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools

def _print_rules(instance, precision=2):
    '''Returns a list with the rules of the model in str format.
        
    Args:
        precision: int for number of decimals to show for the rule parameters. 

    Returns:
        rules:     list of str representing the rules of the system.
    '''
    
    original_weights = instance.consequents.linear.weight
    original_biases = instance.consequents.linear.bias
    
    combinations = list(itertools.product(*[range(i) for i in instance.input_n_sets]))
    n_rules = len(combinations)
    num_classes = instance.output_n_classes
    num_inputs = len(instance.input_var_names)
    class_names = instance.output_var_names
    
    rules = []

    consequent_weights = original_weights.view(num_inputs, n_rules, num_classes)    
    consequent_biases = original_biases.view(n_rules, num_classes)
    
    for i, combination in enumerate(combinations):        
        clauses = []
        for var_index, set_index in enumerate(combination):
            var_name = instance.input_var_names[var_index]
            mf_name = instance.input_mf_names[var_index][set_index]
            clauses.append(f'{var_name} IS {mf_name}')
        antecedent_str = 'IF ' + ' AND '.join(clauses)

        for c in range(num_classes):
            weights = consequent_weights[:, i, c]            
            bias = consequent_biases[i, c].item()
            linear_terms = []
            for var_idx, weight_val in enumerate(weights):
                var_name = instance.input_var_names[var_idx]
                linear_terms.append(f'({weight_val:.{precision}f} * {var_name})')
            
            equation_str = ' + '.join(linear_terms)
            
            if bias >= 0:
                equation_str += f' + {bias:.{precision}f}'
            else:
                equation_str += f' - {-bias:.{precision}f}'

            class_name = class_names[c] if isinstance(class_names, list) else class_names  
            final_rule = f'Rule {i} ({class_name}): {antecedent_str}, THEN f{i}_{c} = {equation_str}'
            rules.append(final_rule)
            
    return rules

def _plot_var(instance, var_name):
        '''Plots the membership functions for a certain variable of the model.

        Args:
            var_name: str with the name of the variable, written in the same way as given in dict variables.
        
        Returns:
            fig, ax:  matplotlib.pyplot fig and axes.
        '''

        fig, ax = plt.subplots()

        var_index = instance.input_var_names.index(var_name)
        n_sets = instance.input_n_sets[var_index]
        uod = torch.linspace(*instance.input_uod[var_index], 1000)
        mf_names = instance.input_mf_names[var_index]
        mf_function = instance.memberships[var_index]

        memberships = mf_function(uod)

        for i in range(n_sets):
            ax.plot(uod.numpy(), memberships[:, i].detach().numpy(), label=f'{mf_names[i]}')

        ax.set_title(f'Membership Functions for Variable {var_name}')
        ax.set_xlabel('Universe of Discourse')
        ax.set_ylabel('Membership')
        ax.legend()

        return fig, ax

def _plot_rules(
        instance, 
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
        
        Returns:
            fig, axes: matplotlib.pyplot fig and axes.
        '''
        
        if len(var_names) != 2:
            raise ValueError('Exactly two variable names should be provided.')
        
        var_index = instance.input_var_names.index(var_names[0])
        n_sets0 = instance.input_n_sets[var_index]
        uod0 = torch.linspace(*instance.input_uod[var_index], n_points)
        mf_names0 = instance.input_mf_names[var_index]
        mf_function0 = instance.memberships[var_index]

        memberships0 = mf_function0(uod0)

        var_index = instance.input_var_names.index(var_names[1])
        n_sets1 = instance.input_n_sets[var_index]
        uod1 = torch.linspace(*instance.input_uod[var_index], n_points)
        mf_names1 = instance.input_mf_names[var_index]
        mf_function1 = instance.memberships[var_index]

        memberships1 = mf_function1(uod1)

        x = np.linspace(uod0[0], uod0[-1], n_points)
        y = np.linspace(uod1[0], uod1[-1], n_points)
        X, Y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(7, 7))
        gs = fig.add_gridspec(2, 2, width_ratios=[5, 1.5], height_ratios=[1.5, 5], wspace=0.08, hspace=0.08)

        ax_main = fig.add_subplot(gs[1, 0])  
        ax_top = fig.add_subplot(gs[0, 0])  
        ax_right = fig.add_subplot(gs[1, 1])  

        Z = None 
        for i in range(n_sets0):
            for j in range(n_sets1):
                mu0 = memberships0[:, i].detach()  
                mu1 = memberships1[:, j].detach()  

                mf0 = mu1.unsqueeze(1).repeat(1, n_points)  
                mf1 = mu0.unsqueeze(0).repeat(n_points, 1)
                rule = torch.stack([mf0, mf1], dim=0)

                Z = instance.and_operator(rule, dim=0)

                if isinstance(Z, tuple):
                    Z = Z[0]

                Z = Z.numpy()
                Z[Z < thr] = np.nan
                ax_main.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=alpha)

        ax_main.set_xlabel(var_names[0])
        ax_main.set_ylabel(var_names[1])

        cbar_ax = fig.add_axes([ax_main.get_position().x0 - 0.16,  
                               ax_main.get_position().y0,  
                               0.02,  
                               ax_main.get_position().height])  

        contour_plot = ax_main.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=alpha)
        cbar = fig.colorbar(contour_plot, cax=cbar_ax, orientation="vertical")

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

        axes = {
            'main': ax_main,
            'top': ax_top,
            'right': ax_right,
            'cbar_ax': cbar.ax  
        }

        return fig, axes
    
def _rule_activations(instance, x):
    '''Returns the normalized rule activation for a given input batch.

    Args:
        x:           tensor (N, n) with input data.

    Returns:
        activations: tensor (N, R) with normalized rule activations.
    '''

    n_inputs = len(instance.input_var_names)
    batch_size = x.shape[0]

    memberships = []
    for i in range(n_inputs):
        m = instance.memberships[i](x[:, i])
        memberships.append(m)

    rule_combinations = list(itertools.product(*[range(n) for n in instance.input_n_sets]))
    n_rules = len(rule_combinations)
    
    antecedent_memberships = torch.zeros(batch_size, n_rules, n_inputs)

    for rule_idx, combination in enumerate(rule_combinations):
        for var_idx, set_idx in enumerate(combination):
            antecedent_memberships[:, rule_idx, var_idx] = memberships[var_idx][:, set_idx]

    activations = instance.and_operator(antecedent_memberships, dim=2)

    return activations
