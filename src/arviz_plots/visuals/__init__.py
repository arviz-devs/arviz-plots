"""ArviZ intermediate level visuals."""
from importlib import import_module

import arviz as az
import numpy as np


def get_backend(target, kwargs):  # pylint: disable=unused-argument
    """Get the backend to use."""
    # use target here to potentially allow recognizing
    # the backend from the target type
    backend = kwargs.pop("backend", "matplotlib")
    return import_module(f"xrtist.backend.{backend}")
