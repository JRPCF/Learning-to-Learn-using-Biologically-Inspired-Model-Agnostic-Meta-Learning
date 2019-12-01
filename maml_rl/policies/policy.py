import torch
import torch.nn as nn

from collections import OrderedDict

def process(grads, parameters):
    parameters=list(parameters)
    grads=list(grads)
    for i in range(len(grads)):
        if grads[i] is None:
            grads[i]=torch.zeros(parameters[i].size())
    grads=tuple(grads)
    return grads
    
def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()
    elif isinstance(module, nn.RNN):
        nn.init.xavier_uniform_(module.weight_ih_l0)
        module.bias_ih_l0.data.zero_()

class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

    def update_params(self, loss, step_size=0.5, first_order=False):
        """Apply one step of gradient descent on the loss function `loss`, with 
        step-size `step_size`, and returns the updated parameters of the neural 
        network.
        """
        grads = torch.autograd.grad(loss, self.parameters(),create_graph=not first_order, allow_unused=True)
        grads = process(grads,self.parameters())
        updated_params = OrderedDict()
        for (name, param), grad in zip(self.named_parameters(), grads):
            updated_params[name] = param - step_size * grad

        return updated_params
