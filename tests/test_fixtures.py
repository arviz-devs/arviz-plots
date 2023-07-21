"""Test fixtures."""
from importlib import import_module

import pytest


@pytest.mark.usefixtures("no_artist_kwargs")
def test_no_artist_kwargs_fixture():
    none_backend = import_module("arviz_plots.backend.none")
    with pytest.raises(ValueError):
        none_backend.line([1, 2], [0, 1], [], extra_kwarg="yes")
