def faulty_optimizer(name: str, missing_package: str, *args, **kwargs) -> None:
    """
    A placeholder function to indicate that the optimizer is faulty.
    If you try to use this optimizer, it will raise an error.
    This indicates the user that the optimizer is not available and
    additional packages need to be installed.

    Parameters
    ----------
    name : str
        The name of the optimizer that is not available.
    missing_package : str
        The package that needs to be installed to use this optimizer.

    Raises
    ------
    NotImplementedError
        Always raises an error indicating the optimizer is not available.
    """
    raise NotImplementedError(
        f"The optimizer '{name}' is not available. "
        f"Please install the required package '{missing_package}'"
        f"to use this optimizer."
    )
