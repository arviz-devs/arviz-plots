(kwargs_conventions)=
# Conventions for artist naming and kwargs in the plots module

## Purpose

This document summarizes the recommended conventions for `visuals`, `aes_by_visuals`, and `stats` across all plots in `arviz-plots`. Following these guidelines ensures consistency, intuitive usage, and easier maintenance.

## General rules

- Full names are preferred over abbreviations or acronyms.
- Each visual element has a name which is a valid key in `visuals`. This allows users to deactivate or skip any operation the plot does.
- Keys in `aes_by_visuals` should match keys in `visuals` whenever possible. Some keys may only make sense for `visuals` and can be skipped (e.g., `remove_axis`).
- There are no strict rules for `stats` yet. Also related to [Allow precomputing #40](https://github.com/arviz-devs/arviz-plots/issues/40).

### Handling invalid or ignored keys

- **Invalid keys**: raise an error.
- **Valid keys but ignored**: continue execution, optionally print a warning.
  This allows compound calls like:

```python
plot_dist(..., stats={"kde": {"bw": 4}, "hist": {"bins": 20}})
```

to work and process multiple visualization types without modifying `kind` or other parameters.

### Open questions

- Should we try to match the keys in `stats` to the key in `visuals` where that data is more prominently encoded?

## Syntactic sugar for multiple keys

Multiple keys can be grouped using a delimiter (e.g., `","`) and internally expanded:

```python
visuals = {
    "title,credible_interval": {"color": "red"},
    "point_estimate,remove_axis": False
}
```

## Recommended helper function

We recommend implementing a helper function to standardize kwargs processing across all plots.
A future helper could look like:

```python
def process_kwargs(kwargs_dict: dict, valid_keys: Sequence):
    """
    Processes input kwargs dictionary to ensure consistency.

    Steps:
    - Convert None → empty dict
    - Copy dict input
    - Check for invalid keys and raise errors
    - Expand multiple keys in a single entry
    """
    ...
```

This would ensure consistency across all plots and reduce repetitive validation code.

## Best practices

- Keep keys consistent across all plots.
- Match `aes_by_visuals` keys to `visuals` keys whenever possible.
- Document any exceptions clearly in developer-facing documentation.
- Avoid introducing multiple keys for the same concept unless required.
