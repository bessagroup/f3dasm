"""Parameters for constructing a feasible search space."""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import copy  # noqa: F401
import dataclasses
import pickle
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional, Protocol, Union

import numpy as np

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================

CategoricalType = Union[None, int, float, str]  # noqa

# =============================================================================


class StoreFunction(Protocol):
    """Protocol for a custom function that stores an object to disk."""

    def __call__(object: Any, path: str) -> str:
        """
        Store ``object`` to disk and return the full path it was written to.

        Parameters
        ----------
        object : Any
            The object to store.
        path : str
            The location to store the object to, **without** a file
            extension. The implementation is responsible for choosing the
            extension appropriate to the object type and appending it
            before writing.

        Returns
        -------
        str
            The full path the object was written to, **including** the
            file extension. f3dasm uses this returned path to locate the
            object at load time, so it must reflect the extension actually
            used on disk.

        Examples
        --------
        A minimal store function for a numpy array — note that the
        returned path carries the ``.npy`` suffix:

        >>> import numpy as np
        >>> from pathlib import Path
        >>> def numpy_store(object: np.ndarray, path: str) -> str:
        ...     _path = Path(path).with_suffix(".npy")
        ...     np.save(_path, object)
        ...     return str(_path)  # must include the .npy suffix
        """
        ...


class LoadFunction(Protocol):
    """Protocol for a custom function that loads an object from disk."""

    def __call__(path: str, **kwargs: Any) -> Any:
        """
        Load and return the object previously written by a StoreFunction.

        Parameters
        ----------
        path : str
            The location to load the object from.
        **kwargs : Any
            Auxiliary state required by the load implementation. f3dasm
            forwards any ``load_kwargs`` registered on the matching
            :class:`Parameter` (see issue #285). The built-in loaders
            ignore unknown kwargs, so custom deserialisers can request
            extra state (e.g. an ``equinox`` template via
            ``load_kwargs={"like": template}``) without breaking the
            default path.

        Returns
        -------
        Any
            The loaded object.

        Examples
        --------
        The symmetric load for the numpy ``StoreFunction`` example:

        >>> import numpy as np
        >>> def numpy_load(path: str) -> np.ndarray:
        ...     return np.load(path)
        """
        ...


# =============================================================================


class Parameter:
    """Interface class of a search space parameter."""

    _type: ClassVar[str] = "object"

    def __init__(
        self,
        to_disk: bool = False,
        store_function: Optional[StoreFunction] = None,
        load_function: Optional[LoadFunction] = None,
        load_kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize the Parameter.

        Parameters
        ----------
        to_disk : bool, optional
            Whether the parameter should be saved to disk. Defaults to False.
        store_function : Optional[StoreFunction], optional
            Function to store the parameter to disk. Defaults to None.
        load_function : Optional[LoadFunction], optional
            Function to load the parameter from disk. Defaults to None.
        load_kwargs : dict[str, Any], optional
            Extra keyword arguments forwarded to ``load_function`` each
            time the stored object is loaded. Useful when the
            deserialiser needs auxiliary state (an ``equinox`` template,
            a torch module skeleton for a `state_dict`, etc.) -- see
            issue #285. Defaults to None (no kwargs forwarded).

        Raises
        ------
        ValueError
            If `to_disk` is False but either `store_function`,
            `load_function`, or `load_kwargs` is not None.

        Notes
        -----
        The `store_function` and `load_function` parameters are used to store
        the parameter to disk and load it from disk, respectively. f3dasm has
        built-in support for some common datatypes (numpy array, pandas
        DataFrame, xarray Dataarray and Dataset), for those data types the
        `store_function` and `load_function` are automatically set. If the
        parameter is not one of these types, the user must provide a custom
        `store_function` and `load_function`.

        Examples
        --------
        >>> param = Parameter(to_disk=True)
        >>> print(param)
        Parameter(type=object, to_disk=True)
        """

        if not to_disk and (
            store_function is not None
            or load_function is not None
            or load_kwargs is not None
        ):
            raise ValueError(
                "If 'to_disk' is False, 'store_function', "
                "'load_function', and 'load_kwargs' must be None."
            )

        self.to_disk = to_disk
        self.store_function = store_function
        self.load_function = load_function
        self.load_kwargs = load_kwargs

    def __str__(self):
        """Return a string representation of the Parameter.

        Returns
        -------
        str
            String representation of the Parameter.
        """
        if type(self) is Parameter:
            return f"Parameter(type={self._type}, to_disk={self.to_disk})"
        return repr(self)

    def __repr__(self):
        """Return a representation of the Parameter.

        Returns
        -------
        str
            Representation string of the Parameter.
        """
        return f"{self.__class__.__name__}(to_disk={self.to_disk})"

    def __eq__(self, __o: Parameter):
        """Check equality between two Parameters.

        Parameters
        ----------
        __o : Parameter
            The other Parameter to compare.

        Returns
        -------
        bool
            True if parameters have the same to_disk attribute.
        """
        return self.to_disk == __o.to_disk

    def __add__(self, __o: Parameter) -> Parameter:
        """
        Add two parameters together.

        Parameters
        ----------
        __o : Parameter
            The parameter to add to the current parameter.

        Returns
        -------
        Parameter
            The first parameter

        Notes
        -----
        Adding two parameters together will return the first parameter. This
        is because the parameters are not additive.
        """
        return self

    def _copy(self) -> Parameter:
        """
        Create a copy of the Parameter object.

        Returns
        -------
        Parameter
            A copy of the Parameter object.

        Examples
        --------
        >>> param = Parameter(to_disk=True)
        >>> param_copy = param._copy()
        """
        return Parameter(
            to_disk=self.to_disk,
            store_function=self.store_function,
            load_function=self.load_function,
            load_kwargs=self.load_kwargs,
        )

    def to_dict(self) -> dict:
        """
        Convert the Parameter object to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the Parameter object.

        Notes
        -----
        The dictionary representation of the Parameter object contains
        the type of the parameter, whether it should be saved to disk,
        and the store and load functions. The functions are stored as
        hex strings.

        Examples
        --------
        >>> param = Parameter(to_disk=True)
        >>> param_dict = param.to_dict()
        """
        d: dict = {
            "type": self._type,
            "to_disk": self.to_disk,
            "store_function": (
                pickle.dumps(self.store_function).hex()
                if self.store_function
                else None
            ),
            "load_function": (
                pickle.dumps(self.load_function).hex()
                if self.load_function
                else None
            ),
            "load_kwargs": (
                pickle.dumps(self.load_kwargs).hex()
                if self.load_kwargs is not None
                else None
            ),
        }
        if dataclasses.is_dataclass(self):
            for f in dataclasses.fields(self):
                d[f.name] = getattr(self, f.name)
        return d

    @classmethod
    def from_dict(cls, param_dict: dict) -> Parameter:
        """
        Create a Parameter object from a dictionary.

        Parameters
        ----------
        param_dict : dict
            Dictionary representation of the Parameter object.

        Returns
        -------
        Parameter
            Parameter object created from the dictionary.

        Examples
        --------
        >>> param_dict = {'type': 'object', 'to_disk': False,
        ...               'store_function': None, 'load_function': None}
        >>> param = Parameter.from_dict(param_dict)
        """
        param_type = param_dict["type"]
        store_function = None
        load_function = None
        load_kwargs = None
        if param_dict.get("store_function"):
            store_function = pickle.loads(
                bytes.fromhex(param_dict["store_function"])
            )
        if param_dict.get("load_function"):
            load_function = pickle.loads(
                bytes.fromhex(param_dict["load_function"])
            )
        if param_dict.get("load_kwargs"):
            load_kwargs = pickle.loads(
                bytes.fromhex(param_dict["load_kwargs"])
            )

        if param_type == "object":
            return Parameter(
                to_disk=param_dict["to_disk"],
                store_function=store_function,
                load_function=load_function,
                load_kwargs=load_kwargs,
            )

        param_cls = _PARAM_REGISTRY.get(param_type)
        if param_cls is None:
            raise ValueError(f"Unknown parameter type: {param_type}")
        fields = {f.name for f in dataclasses.fields(param_cls)}
        kwargs = {k: v for k, v in param_dict.items() if k in fields}
        return param_cls(**kwargs)


# =============================================================================


@dataclass
class ConstantParameter(Parameter):
    """
    Create a search space parameter that is constant.

    Parameters
    ----------
    value : Any
        The constant value of the parameter.

    Attributes
    ----------
    _type : str
        The type of the parameter, which is always 'constant'.

    Raises
    ------
    TypeError
        If the value is not hashable.

    Examples
    --------
    >>> param = ConstantParameter(value=5)
    >>> print(param)
    ConstantParameter(value=5)
    """

    _type: ClassVar[str] = "constant"
    value: Any  # required, no default

    def __post_init__(self):
        """Validate that the value is hashable.

        Raises
        ------
        TypeError
            If the value is not hashable.
        """
        super().__init__()
        self._validate_hashable()

    def __add__(self, other: Parameter):
        """Add two ConstantParameters.

        Parameters
        ----------
        other : Parameter
            The parameter to add.

        Returns
        -------
        ConstantParameter or CategoricalParameter
            Returns self if values are equal, otherwise returns a
            CategoricalParameter containing both values.

        Raises
        ------
        ValueError
            If trying to add a ContinuousParameter.
        """
        if isinstance(other, ConstantParameter):
            if self.value == other.value:
                return self
            else:
                return CategoricalParameter(
                    categories=[self.value, other.value]
                )

        if isinstance(other, CategoricalParameter):
            return self.to_categorical() + other

        if isinstance(other, DiscreteParameter):
            return self.to_categorical() + other

        if isinstance(other, ContinuousParameter):
            raise ValueError("Cannot add continuous parameter to constant!")

    def to_categorical(self) -> CategoricalParameter:
        """
        Convert the constant parameter to a categorical parameter.

        Returns
        -------
        CategoricalParameter
            The converted categorical parameter.
        """
        return CategoricalParameter(categories=[self.value])

    def _validate_hashable(self):
        """Check if the value is hashable.

        Raises
        ------
        TypeError
            If the value is not hashable.
        """
        try:
            hash(self.value)
        except TypeError as exc:
            raise TypeError("The value must be hashable.") from exc


# =============================================================================


@dataclass
class ContinuousParameter(Parameter):
    """
    A search space parameter that is continuous.

    Parameters
    ----------
    lower_bound : float, optional
        The lower bound of the parameter. Defaults to -inf.
    upper_bound : float, optional
        The upper bound of the parameter. Defaults to inf.
    log : bool, optional
        Whether the parameter should be on a log scale. Defaults to False.

    Raises
    ------
    ValueError
        If `log` is True and `lower_bound` is less than or equal to 0.
        If `upper_bound` is less than or equal to `lower_bound`.

    Examples
    --------
    >>> param = ContinuousParameter(lower_bound=0.0, upper_bound=1.0)
    >>> print(param)
    ContinuousParameter(lower_bound=0.0, upper_bound=1.0, log=False)
    """

    _type: ClassVar[str] = "float"
    lower_bound: float = field(default_factory=lambda: float("-inf"))
    upper_bound: float = field(default_factory=lambda: float("inf"))
    log: bool = False

    def __post_init__(self):
        """Cast bounds to float and validate the range.

        Raises
        ------
        ValueError
            If ``log`` is True and ``lower_bound <= 0``, or if
            ``upper_bound <= lower_bound``.
        """
        super().__init__()
        self.lower_bound = float(self.lower_bound)
        self.upper_bound = float(self.upper_bound)
        if self.log and self.lower_bound <= 0.0:
            raise ValueError(
                f"The `lower_bound` value must be larger than 0 for a "
                f"log distribution (low={self.lower_bound}, "
                f"high={self.upper_bound})."
            )
        self._validate_range()

    def __add__(self, other: Parameter) -> ContinuousParameter:
        """Add two ContinuousParameters.

        Parameters
        ----------
        other : Parameter
            The parameter to add.

        Returns
        -------
        ContinuousParameter
            Combined continuous parameter with merged bounds.

        Raises
        ------
        ValueError
            If other is not a ContinuousParameter, has different log
            scale, or ranges do not overlap.
        """
        if not isinstance(other, ContinuousParameter):
            raise ValueError(
                "Cannot add non-continuous parameter to continuous!"
            )
        if self.log != other.log:
            raise ValueError(
                "Cannot add continuous parameters with different log scales!"
            )
        if (
            self.lower_bound > other.upper_bound
            or other.lower_bound > self.upper_bound
        ):
            raise ValueError("Ranges do not coincide, cannot add")

        return ContinuousParameter(
            lower_bound=min(self.lower_bound, other.lower_bound),
            upper_bound=max(self.upper_bound, other.upper_bound),
        )

    def _validate_range(self):
        if self.upper_bound <= self.lower_bound:
            raise ValueError(
                f"The `upper_bound` value must be larger than `lower_bound`. "
                f"(lower_bound={self.lower_bound}, "
                f"upper_bound={self.upper_bound})"
            )

    def to_discrete(self, step: int = 1) -> DiscreteParameter:
        """
        Convert the continuous parameter to a discrete parameter.

        Parameters
        ----------
        step : int, optional
            The step size for the discrete parameter. Defaults to 1.

        Returns
        -------
        DiscreteParameter
            The converted discrete parameter.

        Raises
        ------
        ValueError
            If the step size is less than or equal to 0.

        Examples
        --------
        >>> param = ContinuousParameter(lower_bound=0.0, upper_bound=1.0)
        >>> discrete_param = param.to_discrete(step=0.1)
        >>> print(discrete_param)
        DiscreteParameter(lower_bound=0.0, upper_bound=1.0, step=0.1)
        """
        if step <= 0:
            raise ValueError("The step size must be larger than 0.")
        return DiscreteParameter(
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            step=step,
        )


# =============================================================================


@dataclass
class DiscreteParameter(Parameter):
    """
    Create a search space parameter that is discrete.

    Parameters
    ----------
    lower_bound : int, optional
        The lower bound of the parameter. Defaults to 0.
    upper_bound : int, optional
        The upper bound of the parameter. Defaults to 1.
    step : int, optional
        The step size for the parameter. Defaults to 1.

    Raises
    ------
    ValueError
        If `upper_bound` is less than or equal to `lower_bound`.
        If `step` is less than or equal to 0.

    Examples
    --------
    >>> param = DiscreteParameter(lower_bound=0, upper_bound=10, step=1)
    >>> print(param)
    DiscreteParameter(lower_bound=0, upper_bound=10, step=1)
    """

    _type: ClassVar[str] = "int"
    lower_bound: int = 0
    upper_bound: int = 1
    step: int = 1

    def __post_init__(self):
        """Cast bounds to int and validate the range.

        Raises
        ------
        ValueError
            If ``upper_bound <= lower_bound`` or
            ``step <= 0``.
        """
        super().__init__()
        self.lower_bound = int(self.lower_bound)
        self.upper_bound = int(self.upper_bound)
        self._validate_range()

    def __add__(self, other: Parameter) -> DiscreteParameter:
        """Add two Parameters.

        Parameters
        ----------
        other : Parameter
            The parameter to add.

        Returns
        -------
        DiscreteParameter or CategoricalParameter
            Combined parameter based on types.

        Raises
        ------
        ValueError
            If trying to add a ContinuousParameter.
        """
        if isinstance(other, CategoricalParameter):
            return other + self
        if isinstance(other, ConstantParameter):
            return other.to_categorical() + self
        if isinstance(other, ContinuousParameter):
            raise ValueError("Cannot add continuous parameter to discrete!")
        return self  # Assuming the same discrete parameters are being added.

    def _validate_range(self):
        if self.upper_bound <= self.lower_bound:
            raise ValueError("Upper bound must be greater than lower bound.")
        if self.step <= 0:
            raise ValueError("Step size must be positive.")


# =============================================================================


@dataclass(eq=False)
class CategoricalParameter(Parameter):
    """
    Create a search space parameter that is categorical.

    Parameters
    ----------
    categories : Iterable[Any]
        The categories of the parameter.

    Raises
    ------
    ValueError
        If the categories contain duplicates.

    Examples
    --------
    >>> param = CategoricalParameter(categories=['a', 'b', 'c'])
    >>> print(param)
    CategoricalParameter(categories=['a', 'b', 'c'])
    """

    _type: ClassVar[str] = "category"
    categories: list[Any] = field(default_factory=list)

    def __post_init__(self):
        """Convert categories to a list and check for duplicates.

        Raises
        ------
        ValueError
            If the categories contain duplicate values.
        """
        super().__init__()
        self.categories = list(self.categories)
        self._check_duplicates()

    def __add__(self, other: Parameter) -> CategoricalParameter:
        """Add two Parameters to create a combined categorical.

        Parameters
        ----------
        other : Parameter
            The parameter to add.

        Returns
        -------
        CategoricalParameter
            Combined categorical parameter with merged categories.

        Raises
        ------
        ValueError
            If trying to add a ContinuousParameter or unsupported type.
        """
        if isinstance(other, CategoricalParameter):
            joint_categories = list(set(self.categories + other.categories))
        elif isinstance(other, ConstantParameter):
            joint_categories = list(set(self.categories + [other.value]))
        elif isinstance(other, DiscreteParameter):
            joint_categories = list(
                set(
                    self.categories
                    + list(
                        range(other.lower_bound, other.upper_bound, other.step)
                    )
                )
            )
        elif isinstance(other, ContinuousParameter):
            raise ValueError("Cannot add continuous parameter to categorical!")
        else:
            raise ValueError(
                f"Cannot add parameter of type {type(other)} to categorical."
            )
        return CategoricalParameter(joint_categories)

    def __eq__(self, other: CategoricalParameter) -> bool:
        """Check equality with another CategoricalParameter.

        Parameters
        ----------
        other : CategoricalParameter
            The other parameter to compare.

        Returns
        -------
        bool
            True if both have the same categories (order-independent).
        """
        return set(self.categories) == set(other.categories)

    def _check_duplicates(self):
        """Check for duplicate categories.

        Raises
        ------
        ValueError
            If categories contain duplicates.
        """
        if len(self.categories) != len(set(self.categories)):
            raise ValueError("Categories contain duplicates!")


# =============================================================================


@dataclass(eq=False)
class ArrayParameter(Parameter):
    """
    Create a search space parameter that is an array.

    Parameters
    ----------
    shape : int or Iterable[int]
        The dimensions of the array.
    lower_bound : float or np.ndarray, optional
        The lower bound of the parameter, by default -inf.
    upper_bound : float or np.ndarray, optional
        The upper bound of the parameter, by default inf.

    Raises
    ------
    ValueError
        If `shape` is empty or contains non-positive integers.

    Examples
    --------
    >>> param = ArrayParameter(shape=[3, 4], lower_bound=0.0, upper_bound=1.0)
    """

    _type: ClassVar[str] = "array"
    shape: Any = field(default_factory=tuple)
    lower_bound: Any = field(default_factory=lambda: float("-inf"))
    upper_bound: Any = field(default_factory=lambda: float("inf"))

    def __post_init__(self):
        """Normalize shape and bounds, validating dimensions.

        Raises
        ------
        ValueError
            If ``shape`` is empty or contains non-positive
            integers.
        """
        super().__init__()

        if isinstance(self.shape, int):
            self.shape = (self.shape,)
        else:
            self.shape = tuple(int(d) for d in self.shape)

        if not self.shape or any(d <= 0 for d in self.shape):
            raise ValueError(
                "Shape must be a non-empty iterable of positive integers."
            )

        if isinstance(self.lower_bound, float | int):
            self.lower_bound = np.full(self.shape, float(self.lower_bound))
        if isinstance(self.upper_bound, float | int):
            self.upper_bound = np.full(self.shape, float(self.upper_bound))

    def __eq__(self, other: Parameter) -> bool:
        """Check equality with another Parameter.

        Parameters
        ----------
        other : Parameter
            The other Parameter to compare.

        Returns
        -------
        bool
            True if both are ArrayParameters with equal attributes.
        """
        if not isinstance(other, ArrayParameter):
            return False
        return (
            self.shape == other.shape
            and np.array_equal(self.lower_bound, other.lower_bound)
            and np.array_equal(self.upper_bound, other.upper_bound)
        )


PARAMETERS = [
    CategoricalParameter,
    ConstantParameter,
    ContinuousParameter,
    DiscreteParameter,
    ArrayParameter,
]

_PARAM_REGISTRY: dict[str, type[Parameter]] = {
    cls._type: cls for cls in PARAMETERS
}
