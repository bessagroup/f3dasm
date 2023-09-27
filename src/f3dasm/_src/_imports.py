"""Classes that manage importing and exceptions of submodules with extension.
This code was adapted from Optuna (https://github.com/optuna/optuna/blob/master/optuna/_imports.py)
and modified to fit the specific case of f3dasm.
"""

#                                                                       Modules
# =============================================================================

# Standard
from types import TracebackType
from typing import Optional, Tuple, Type

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling', 'Optuna']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class _DeferredImportExceptionContextManager:
    """Context manager to defer exceptions from imports.

    Catches :exc:`ImportError` and :exc:`SyntaxError`.
    If any exception is caught, this class raises an :exc:`ImportError` when being checked.

    """

    def __init__(self, extension_name: str) -> None:
        self._deferred: Optional[Tuple[Exception, str]] = None
        self.extension_name = extension_name

    def __enter__(self) -> "_DeferredImportExceptionContextManager":
        """Enter the context manager.

        Returns:
            Itself.

        """
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[Exception]],
        exc_value: Optional[Exception],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        """Exit the context manager.

        Args:
            exc_type:
                Raised exception type. :obj:`None` if nothing is raised.
            exc_value:
                Raised exception object. :obj:`None` if nothing is raised.
            traceback:
                Associated traceback. :obj:`None` if nothing is raised.

        Returns:
            :obj:`None` if nothing is deferred, otherwise :obj:`True`.
            :obj:`True` will suppress any exceptions avoiding them from propagating.

        """
        if isinstance(exc_value, (ImportError, SyntaxError)):
            if isinstance(exc_value, ImportError):
                message = (
                    f"Tried to import '{exc_value.name}' but failed. Please make sure that you have "
                    f"installed the extension '{self.extension_name}' correctly to use this feature. "
                    f"Actual error: {exc_value}."
                )
            elif isinstance(exc_value, SyntaxError):
                message = (
                    f"Tried to import a package but failed due to a syntax error in {exc_value.filename}. Please "
                    f"make sure that the Python version is correct to use this feature. Actual "
                    f"error: {exc_value}."
                )
            else:
                assert False

            self._deferred = (exc_value, message)
            return True
        return None

    def is_successful(self) -> bool:
        """Return whether the context manager has caught any exceptions.

        Returns:
            :obj:`True` if no exceptions are caught, :obj:`False` otherwise.

        """
        return self._deferred is None

    def check(self) -> None:
        """Check whether the context manager has caught any exceptions.

        Raises:
            :exc:`ImportError`:
                If any exception was caught from the caught exception.

        """
        if self._deferred is not None:
            exc_value, message = self._deferred
            raise ImportError(message) from exc_value


def try_import(extension_name: Optional[str] = None) -> _DeferredImportExceptionContextManager:
    """Create a context manager that can wrap imports of optional packages to defer exceptions.

    Args:
        Extension name for this particular context manager

    Returns:
        Deferred import context manager.

    """
    return _DeferredImportExceptionContextManager(extension_name=extension_name)
