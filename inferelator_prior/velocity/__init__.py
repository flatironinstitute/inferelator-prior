from .programs import program_select, program_pcs
from .calc import calc_velocity
from .decay import calc_decay, calc_decay_sliding_windows


def vprint(*args, verbose=False, **kwargs):
    if verbose:
        print(*args, **kwargs)
