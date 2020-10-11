_YEAST = {
    "use_tss": True,
    "window": (1000, 100),
    "tandem": 25
}

_FLY = {
    "use_tss": True,
    "window": (50000, 2000),
    "tandem": 100
}

_MOUSE = {
    "use_tss": True,
    "window": (50000, 2000),
    "tandem": 100
}

_HUMAN = {
    "use_tss": True,
    "window": (50000, 2000),
    "tandem": 100
}

SPECIES_MAP = {
    "yeast": _YEAST,
    "saccharomyces cerevisiae": _YEAST,
    "fly": _FLY,
    "drosophila melanogaster": _FLY,
    "mouse": _MOUSE,
    "mus musculus": _MOUSE,
    "human": _HUMAN,
    "homo sapiens": _HUMAN
}

DEFAULT_TANDEM = 25
DEFAULT_WINDOW = (5000, 500)
DEFAULT_TSS = True
