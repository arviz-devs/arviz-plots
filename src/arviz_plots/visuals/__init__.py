from importlib import import_module

import arviz as az
import numpy as np


def get_backend(target, kwargs):  # pylint: disable=unused-argument
    # use target here to potentially allow recognizing
    # the backend from the target type
    backend = kwargs.pop("backend", "matplotlib")
    return import_module(f"xrtist.backend.{backend}")


def kde(values, target, **kwargs):
    if "preprocessed_data" in kwargs:
        pre_ds = kwargs.pop("preprocessed_data")
        grid = pre_ds["grid"]
        pdf = pre_ds["kde"]
    else:
        grid, pdf = az.kde(np.array(values).flatten())
    bkd = get_backend(target, kwargs)
    y = kwargs.pop("y", 0)
    return bkd.line(grid, pdf + y, target, **kwargs)


def interval(values, target, **kwargs):
    if "preprocessed_data" in kwargs:
        pre_ds = kwargs.pop("preprocessed_data")
        interval_values = pre_ds["interval"]
    else:
        int_func = kwargs.pop("interval_func", az.hdi)
        interval_values = int_func(np.array(values).flatten())
    bkd = get_backend(target, kwargs)
    y = kwargs.pop("y", 0)
    return bkd.line(interval_values, [y, y], target=target, **kwargs)


def point(values, target, **kwargs):
    if "preprocessed_data" in kwargs:
        pre_ds = kwargs.pop("preprocessed_data")
        point_est = pre_ds["point_estimate"].item()
    else:
        point_func = kwargs.pop("point_func", np.mean)
        point_est = point_func(np.array(values).flatten())
    bkd = get_backend(target, kwargs)
    y = kwargs.pop("y", 0)
    return bkd.scatter(point_est, y, target, **kwargs)


def point_label(values, target, **kwargs):
    if "preprocessed_data" in kwargs:
        pre_ds = kwargs.pop("preprocessed_data")
        point_est = pre_ds["point_estimate"]
        point_est_label = kwargs.pop("point_label")
        pdf = pre_ds["kde"]
    else:
        point_func = kwargs.pop("point_func", np.mean)
        point_est_label = kwargs.pop("point_label", point_func.__name__)
        values = np.array(values).flatten()
        point_est = point_func(values)
        _, pdf = az.kde(values)

    top = np.max(pdf)

    bkd = get_backend(target, kwargs)
    return bkd.text(point_est, 0.05 * top, f"{point_est:.2f} {point_est_label}", target, **kwargs)
