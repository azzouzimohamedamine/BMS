"""Project-specific exceptions."""


class InsufficientDataError(RuntimeError):
    """Raised when a module cannot be computed scientifically with the available data."""


class DatasetMissingError(FileNotFoundError):
    """Raised when a configured raw data file is missing (never silently substituted)."""
