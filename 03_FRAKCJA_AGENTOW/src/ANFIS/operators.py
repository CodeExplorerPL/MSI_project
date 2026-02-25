import numpy as np

def productN(args, op):
    return np.prod(args, axis=0)

def zadeh_t(args, op):
    return np.min(args, axis=0)
def zadeh_s(args, op):
    return np.max(args, axis=0)

def algebraic_t(args, op):
    return np.prod(args, axis=0)
def probabilistic_s(args, op):
    return 1 - np.prod(1 - args, axis=0)

def lukasiewicz_t(args, op):
    return np.maximum(np.sum(args, axis=0) - (args.shape[0] - 1), 0)
def lukasiewicz_s(args, op):
    return np.minimum(np.sum(args, axis=0), 1)

def fodor_t(args, op):
    result = args[0]
    for i in range(1, args.shape[0]):
        result = np.where(result + args[i] > 1, np.minimum(result, args[i]), 0)
    return result
def fodor_s(args, op):
    for i in range(1, args.shape[0]):
        result = np.where(result + args[i] < 1, np.maximum(result, args[i]), 1)
    return result

def drastic_t(args, op):
    return np.where(np.all(args == 1.0, axis=0), 1.0, 0.0)
def drastic_s(args, op):
    return np.where(np.all(args == 0.0, axis=0), 0.0, 1.0)

def einstein_t(args, op):
    result = args[0]
    for i in range(1, args.shape[0]):
        result = (result * args[i]) / (2 - (result + args[i] - result * args[i]))
    return result
def einstein_s(args, op):
    result = args[0]
    for i in range(1, args.shape[0]):
        result = (result + args[i]) / (1 + (result * args[i]))
    return result