class DatasetError(Exception):
    exit_code = 1


class UsageError(DatasetError):
    """Bad arguments or a refused path (exit 2)."""
    exit_code = 2


class QualityError(DatasetError):
    """SfM result failed a quality gate (exit 3)."""
    exit_code = 3


class Cancelled(DatasetError):
    exit_code = 130
