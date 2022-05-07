def vprint(*args, verbose=False, **kwargs):
    if verbose:
        print(*args, **kwargs)
