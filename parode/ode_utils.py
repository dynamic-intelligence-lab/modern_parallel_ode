import inspect
from typing import Callable


def prepare_ode_function(ode: Callable) -> Callable:
    """
    Wrap the supplied ODE so downstream code can consistently call it with
    the (t, x, u) signature; an ODE of (t, x) is called without the input, and
    its wrapper raises ValueError when it is given one.
    """

    signature = inspect.signature(ode)
    positional = [
        param
        for param in signature.parameters.values()
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
    ]
    nb_args = len(positional)

    if nb_args == 2:
        def ode_without_input(t, x, u=None):
            if u is not None:
                raise ValueError("the ODE takes (t, x) but an input was given (input_fn= or inputs=); "
                                 "write the ODE as f(t, x, u)")
            return ode(t, x)

        return ode_without_input

    return lambda t, x, u=None: ode(t, x, u)


def evaluate_input(input_fn, t):
    if input_fn is None:
        return None
    return input_fn(t)
