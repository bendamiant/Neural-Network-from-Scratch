import numpy as np


def cost(fn, a, y):
    function = COST_FUNCTIONS.get(fn.lower())
    if function is None:
        raise ValueError(f"unknown function: {fn}")
    return function(a, y)
    
def cost_derivative(cost_fn, a, y):
    function = COST_DERIVATIVES.get(cost_fn)
    if function is None:
        raise ValueError(f"unknown derivative : {function}")
    return function(a, y)
    
def mse(a, y):
    return (0.5 * np.linalg.norm(a - y)**2)

def cross_entropy(a, y):
    epsilon = 1e-10  # added to prevent log(0)
    return -np.sum(y * np.log(a + epsilon))

def mse_derivative(a, y):
    return (a - y)
    
COST_FUNCTIONS = {
    'mse': mse,
    'cross_entropy': cross_entropy
}

COST_DERIVATIVES = {
    mse: mse_derivative
}

# Assuming a softmax activation output layer
def error_for_cross_entropy_softmax(a, y):
    return (a - y)