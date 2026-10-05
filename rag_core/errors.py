"""Shared exception types used to classify dependency failures."""


class TransientDependencyError(RuntimeError):
    """An external service failure that may clear on a bounded retry."""
