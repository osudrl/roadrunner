import torch
import numpy
import math

from .colors import FAIL, ENDC, OKGREEN

def is_variable_valid(x):
    """Check if a given number has NaN, Inf, or None.
    Datatype of x in list, tensor, ndarray, floating number
    Returns: True if this number of array has the above values.
    """
    # Check if it's a single float
    try:
        iter(x)
    except:
        if x is None:
            return False
        if not math.isfinite(x):
            return False
        else:
            return True

    # Convert list to ndarray float type
    try:
        if isinstance(x, list):
            x = numpy.array(x, dtype=float)
    except:
        print(f"{FAIL}is_variable_valid gets non-float number in {x}.{ENDC}")
        return False

    # Check by type
    if isinstance(x, torch.Tensor):
        if not torch.isfinite(x).all():
            return False
        else:
            return True
    elif isinstance(x, numpy.ndarray):
        if not numpy.isfinite(x).all():
            return False
        else:
            return True
    else:
        print(f"{FAIL}is_variable_valid gets not qualified input type of {type(x)} for {x}.{ENDC}")
        return False

def unpack_training_error(file_path):
    data = torch.load(file_path)
    for key, value in data.items():
        if not isinstance(value, torch.Tensor):
            print(f"{key} is not tensor, but other type {type(value)}. Need to check further.")
        elif not torch.isfinite(value).all():
            print(f"{key} has non finite values! Check the calculation in optimization!")
    try:
        ratio = (data['log_probs'] - data['old_log_probs']).exp()
        diff = (data['log_probs'] - data['old_log_probs'])
        cpi_loss   = ratio * data['advantages']
        print(torch.max(ratio), torch.min(ratio), torch.max(cpi_loss), torch.min(cpi_loss))
        clip_loss  = ratio.clamp(1.0 - 0.2, 1 + 0.2) * data['advantages']
        actor_loss = -(torch.min(cpi_loss, clip_loss)).sum()
        print(f"Finite values in ratio: {torch.isfinite(ratio).all()}, cpi loss: {torch.isfinite(cpi_loss).all()}, clip loss: {torch.isfinite(clip_loss).all()}, actor loss: {torch.isfinite(actor_loss).all()}, diff: {torch.isfinite(diff).all()}, log prob: {torch.isfinite(data['log_probs']).all()}, old log prob: {torch.isfinite(data['old_log_probs']).all()}")
        # Take mask on the actor loss
        ratio = ((data['log_probs'] - data['old_log_probs']) * data['mask']).exp()
        diff = (data['log_probs'] - data['old_log_probs'])
        cpi_loss   = ratio * data['advantages']
        actor_loss = -(torch.min(cpi_loss, clip_loss)).sum()
        print(f"After masking on the ratio, Finite values in ratio: {torch.isfinite(ratio).all()}, cpi loss: {torch.isfinite(cpi_loss).all()}, clip loss: {torch.isfinite(clip_loss).all()}, actor loss: {torch.isfinite(actor_loss).all()}, diff: {torch.isfinite(diff).all()}, log prob: {torch.isfinite(data['log_probs']).all()}, old log prob: {torch.isfinite(data['old_log_probs']).all()}")
    except:
        pass

if __name__=='__main__':
    # Test cases for this utility function
    a = [None, 1, 2]
    np_arr = numpy.array(a, dtype=float)
    tensor_arr = torch.from_numpy(np_arr)
    assert is_variable_valid(a)==False, "Checking function wrong."
    assert is_variable_valid(np_arr)==False, "Checking function wrong."
    assert is_variable_valid(tensor_arr)==False, "Checking function wrong."

    a = 1.0
    np_arr = numpy.array(a, dtype=float)
    tensor_arr = torch.from_numpy(np_arr)
    assert is_variable_valid(a)==True, "Checking function wrong."
    assert is_variable_valid(np_arr)==True, "Checking function wrong."
    assert is_variable_valid(tensor_arr)==True, "Checking function wrong."

    a = [1, float('Inf')]
    np_arr = numpy.array(a, dtype=float)
    tensor_arr = torch.from_numpy(np_arr)
    assert is_variable_valid(a)==False, "Checking function wrong."
    assert is_variable_valid(np_arr)==False, "Checking function wrong."
    assert is_variable_valid(tensor_arr)==False, "Checking function wrong."

    a = "qwer"
    assert is_variable_valid(a)==False, "Checking function wrong."

    a = ["qwer"]
    assert is_variable_valid(a)==False, "Checking function wrong."

    a = ["qwer", 1]
    assert is_variable_valid(a)==False, "Checking function wrong."

    a = [[1,2,3],[1,2,3]]
    np_arr = numpy.array(a, dtype=float)
    tensor_arr = torch.from_numpy(np_arr)
    assert is_variable_valid(a)==True, "Checking function wrong."
    assert is_variable_valid(np_arr)==True, "Checking function wrong."
    assert is_variable_valid(tensor_arr)==True, "Checking function wrong."

    a = [[1,2,3],[1,2,None]]
    np_arr = numpy.array(a, dtype=float)
    tensor_arr = torch.from_numpy(np_arr)
    assert is_variable_valid(a)==False, "Checking function wrong."
    assert is_variable_valid(np_arr)==False, "Checking function wrong."
    assert is_variable_valid(tensor_arr)==False, "Checking function wrong."

    print(f"{OKGREEN}Passed all number check tests.{ENDC}")