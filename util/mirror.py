import torch

def mirror_tensor(t, indices):
    """
    Mirrors a tensor according to the information provided in 'indices.' The 'indices' list will
    specify which indices of the observation need to be moved and/or negated. For example, if the
    mirror vector is [0.1, 2, -1], the mirror state will be will keep the first (0th) element in
    place, then swap the 2nd and 3rd elements while negating the 2nd element. So the state [1, 2, 3]
    would become [1, 3, -2].

    't' should be a pytorch tensor. The state dimension should be the final dimension, but as long
    as that is satisfied it can have any number of dimensions (batch x state, seq x batch x state,
    1 x state, etc).

    Note that due to how this implemented (need to get sign as well), the zeroth element is index as
    0.1, since 0 and -0 are the same. So this way allows one to specify either 0.1 or -0.1, if you
    need to negate the first element as well.

    Args:
        t (tensor): tensor to be mirrored
        indices (list): List to use a mirror indices
    """
    if type(indices) is list:
        indices = torch.Tensor(indices)
    sign = torch.sign(indices)
    indices = indices.long().abs()
    mirror_t = sign * torch.index_select(t, -1, indices)
    return mirror_t
