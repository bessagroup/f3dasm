# parameter.py Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `parameter.py` from ~1,100 to ~550 lines by applying `@dataclass` to all 5 concrete parameter subclasses, fixing `_type` ClassVar inconsistencies, replacing per-class `to_dict()` overrides with introspection, and swapping the `from_dict()` if/elif chain for a registry.

**Architecture:** `Parameter` (base) stays a regular class. Each concrete subclass becomes a `@dataclass`, letting Python generate `__init__`, `__repr__`, and `__eq__`. Validation moves to `__post_init__`. A module-level `_PARAM_REGISTRY` dict replaces the manual factory. `Domain._copy()` is updated to use `copy.copy()` instead of the removed `_copy()` methods.

**Tech Stack:** Python 3.10+, `dataclasses` stdlib, `copy` stdlib, `numpy`

**Spec:** `docs/superpowers/specs/2026-03-20-parameter-simplification-design.md`

---

## File Map

| File | Change |
|---|---|
| `src/f3dasm/_src/design/parameter.py` | Main refactor |
| `src/f3dasm/_src/design/domain.py` | Replace `v._copy()` with `copy.copy(v)` |

---

## Task 1: Fix `_type` ClassVar inconsistencies and add imports

**Files:**
- Modify: `src/f3dasm/_src/design/parameter.py`

- [ ] **Step 1: Add `dataclasses` and `copy` imports**

At the top of `parameter.py`, after the existing `import pickle` line, add:
```python
import copy
import dataclasses
from dataclasses import dataclass, field
```

- [ ] **Step 2: Add `_type: ClassVar[str] = "constant"` to `ConstantParameter`**

Inside `ConstantParameter`, directly after the class docstring (before `def __init__`), add:
```python
_type: ClassVar[str] = "constant"
```

- [ ] **Step 3: Fix `CategoricalParameter._type`**

`CategoricalParameter` currently has `_type: ClassVar[str] = "object"`. Change it to:
```python
_type: ClassVar[str] = "category"
```

- [ ] **Step 4: Fix `DiscreteParameter._type`**

`DiscreteParameter` currently sets `self._type = "int"` inside `__init__` (line 717). Replace with a class-level declaration directly after the class docstring:
```python
_type: ClassVar[str] = "int"
```
Then **delete** the `self._type = "int"` line from `__init__`.

- [ ] **Step 5: Add `_type` to `ArrayParameter`**

`ArrayParameter` inherits `_type = "object"` from `Parameter` but its `to_dict()` overrides it to `"array"`. Add at class level (after the docstring):
```python
_type: ClassVar[str] = "array"
```

- [ ] **Step 6: Run tests to confirm nothing broke**

```bash
cd /Users/martin/Documents/GitHub/f3dasm/.claude/worktrees/dazzling-hellman
pytest tests/design/test_parameters.py -v
```
Expected: all tests pass (green).

- [ ] **Step 7: Commit**

```bash
git add src/f3dasm/_src/design/parameter.py
git commit -m "fix: correct _type ClassVars and add dataclasses import in parameter.py"
```

---

## Task 2: Refactor `ConstantParameter` to `@dataclass`

**Files:**
- Modify: `src/f3dasm/_src/design/parameter.py`

- [ ] **Step 1: Replace the class body**

Replace the current `ConstantParameter` class (everything from `class ConstantParameter(Parameter):` through its last method) with:

```python
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
    value: Any  # required, no default — preserves existing TypeError on missing value

    def __post_init__(self):
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
                return CategoricalParameter(categories=[self.value, other.value])

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
```

Note: `__init__`, `__str__`, `__repr__`, `__eq__`, `_copy()`, and `to_dict()` are all removed — the dataclass generates `__init__`/`__repr__`/`__eq__`, `__str__` defaults to `__repr__`, and `to_dict()` will be handled by the base class in Task 5.

- [ ] **Step 2: Run tests**

```bash
pytest tests/design/test_parameters.py -v -k "Constant"
```
Expected: all `Constant`-related tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/f3dasm/_src/design/parameter.py
git commit -m "refactor: convert ConstantParameter to dataclass"
```

---

## Task 3: Refactor `ContinuousParameter` and `DiscreteParameter` to `@dataclass`

**Files:**
- Modify: `src/f3dasm/_src/design/parameter.py`

- [ ] **Step 1: Replace `ContinuousParameter` class body**

Replace with:

```python
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
```

- [ ] **Step 2: Replace `DiscreteParameter` class body**

Replace with:

```python
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
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/design/test_parameters.py -v -k "Continuous or Discrete"
```
Expected: all matching tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/f3dasm/_src/design/parameter.py
git commit -m "refactor: convert ContinuousParameter and DiscreteParameter to dataclasses"
```

---

## Task 4: Refactor `CategoricalParameter` and `ArrayParameter` to `@dataclass`

**Files:**
- Modify: `src/f3dasm/_src/design/parameter.py`

- [ ] **Step 1: Replace `CategoricalParameter` class body**

```python
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
```

- [ ] **Step 2: Replace `ArrayParameter` class body**

```python
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
        If `upper_bound` is less than or equal to `lower_bound`.

    Examples
    --------
    >>> param = ArrayParameter(shape=[3, 4], lower_bound=0.0, upper_bound=1.0)
    >>> print(param)
    ArrayParameter(shape=(3, 4), lower_bound=[[0. 0. 0. 0.] ...], upper_bound=[[1. 1. 1. 1.] ...])
    """

    _type: ClassVar[str] = "array"
    shape: tuple = field(default_factory=tuple)
    lower_bound: Any = field(default_factory=lambda: float("-inf"))
    upper_bound: Any = field(default_factory=lambda: float("inf"))

    def __post_init__(self):
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
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/design/test_parameters.py -v -k "Categorical or Array"
```
Expected: all matching tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/f3dasm/_src/design/parameter.py
git commit -m "refactor: convert CategoricalParameter and ArrayParameter to dataclasses"
```

---

## Task 5: Simplify `to_dict()` and replace `from_dict()` with registry

**Files:**
- Modify: `src/f3dasm/_src/design/parameter.py`

- [ ] **Step 1: Update `Parameter.to_dict()` to use dataclass introspection**

Replace the current `to_dict()` method on `Parameter` with:

```python
def to_dict(self) -> dict:
    """
    Convert the Parameter object to a dictionary.

    Returns
    -------
    dict
        Dictionary representation of the Parameter object.
    """
    d = {
        "type": self._type,
        "to_disk": self.to_disk,
        "store_function": pickle.dumps(self.store_function).hex() if self.store_function else None,
        "load_function": pickle.dumps(self.load_function).hex() if self.load_function else None,
    }
    if dataclasses.is_dataclass(self):
        for f in dataclasses.fields(self):
            d[f.name] = getattr(self, f.name)
    return d
```

- [ ] **Step 2: Add `_PARAM_REGISTRY` at the end of the module**

At the very end of `parameter.py`, after the `PARAMETERS` list, add:

```python
_PARAM_REGISTRY: dict[str, type[Parameter]] = {
    cls._type: cls for cls in PARAMETERS
}
```

> **Important:** Add this BEFORE updating `from_dict()`. `from_dict` references `_PARAM_REGISTRY` at call time (not import time), but the intermediate file state needs this name to be present for `ruff` checks to pass.

- [ ] **Step 3: Update `Parameter.from_dict()` to use a registry**

Replace the current `from_dict()` classmethod body with:

```python
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
    if param_dict.get("store_function"):
        store_function = pickle.loads(
            bytes.fromhex(param_dict["store_function"])
        )
    if param_dict.get("load_function"):
        load_function = pickle.loads(
            bytes.fromhex(param_dict["load_function"])
        )

    if param_type == "object":
        return Parameter(
            to_disk=param_dict["to_disk"],
            store_function=store_function,
            load_function=load_function,
        )

    param_cls = _PARAM_REGISTRY.get(param_type)
    if param_cls is None:
        raise ValueError(f"Unknown parameter type: {param_type}")
    fields = {f.name for f in dataclasses.fields(param_cls)}
    kwargs = {k: v for k, v in param_dict.items() if k in fields}
    return param_cls(**kwargs)
```

> **Note on `ArrayParameter` shape:** When a dict is deserialised from JSON, `shape` comes back as a `list`. `ArrayParameter.__post_init__` converts it to `tuple` via `tuple(int(d) for d in self.shape)`, so passing a list here is safe.

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/design/test_parameters.py tests/design/test_domain.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/f3dasm/_src/design/parameter.py
git commit -m "refactor: simplify to_dict() with introspection and replace from_dict() with registry"
```

---

## Task 6: Update `domain.py` and run full verification

**Files:**
- Modify: `src/f3dasm/_src/design/domain.py`

- [ ] **Step 1: Add `import copy` to `domain.py`**

At the top of `domain.py`, add `import copy` alongside the existing stdlib imports.

- [ ] **Step 2: Replace `v._copy()` calls in `Domain._copy()`**

Change lines 175–176 from:
```python
return Domain(
    input_space={k: v._copy() for k, v in self.input_space.items()},
    output_space={k: v._copy() for k, v in self.output_space.items()},
)
```
to:
```python
return Domain(
    input_space={k: copy.copy(v) for k, v in self.input_space.items()},
    output_space={k: copy.copy(v) for k, v in self.output_space.items()},
)
```

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v --tb=short 2>&1 | tail -30
```
Expected: all tests pass.

- [ ] **Step 4: Run ruff**

```bash
cd /Users/martin/Documents/GitHub/f3dasm/.claude/worktrees/dazzling-hellman
ruff check src/f3dasm/_src/design/parameter.py src/f3dasm/_src/design/domain.py
ruff format src/f3dasm/_src/design/parameter.py src/f3dasm/_src/design/domain.py
```
Fix any issues found.

- [ ] **Step 5: Confirm line count reduction**

```bash
wc -l src/f3dasm/_src/design/parameter.py
```
Expected: ~550 lines (down from ~1,100).

- [ ] **Step 6: Commit**

```bash
git add src/f3dasm/_src/design/domain.py src/f3dasm/_src/design/parameter.py
git commit -m "refactor: replace Parameter._copy() with copy.copy() in Domain._copy()"
```
