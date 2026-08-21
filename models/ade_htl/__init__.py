"""ADE-HTL (Zhang et al., IEEE TMI 43(2) 2024).

    model.py              the network, registered as "ade_htl"
    precompute_ade.py     offline: ROI + 5 anatomical distance field maps
    precompute_targets.py offline: key points + centerline heatmap

The offline scripts are `python -m` entry points and are deliberately not imported
here: registry.py pulls this package in at startup, so importing it must stay as
cheap as importing the model.
"""
from models.ade_htl.model import (  # noqa: F401
    ADEHTL, NEIGHBOUR_OFFSETS, N_CONNECTIVITY, SELF_CHANNEL, connectivity_votes,
)
