"""
wfpt.py: aesara implementation of the Wiener First Passage Time Distribution
This code is based on Sam Mathias's Aesara/Theano implementation
of the WFPT distribution here:
https://gist.github.com/sammosummo/c1be633a74937efaca5215da776f194b
"""

from __future__ import annotations

from typing import Callable

import aesara
import aesara.tensor as at
import numpy as np
from pymc.distributions.dist_math import check_parameters

aesara.config.floatX = "float32"


def k_small(rt: np.ndarray, err: float) -> np.ndarray:
    """Determines number of terms needed for small-t expansion.
    Args:
        rt: A 1D numpy array of flipped R.... T.....s. (0, inf).
        err: Error bound
    Returns: a 1D at array of k_small.
    """
    ks = 2 + at.sqrt(-2 * rt * at.log(2 * np.sqrt(2 * np.pi * rt) * err))
    ks = at.max(at.stack([ks, at.sqrt(rt) + 1]), axis=0)
    ks = at.switch(2 * at.sqrt(2 * np.pi * rt) * err < 1, ks, 2)

    return ks


def k_large(rt: np.ndarray, err: float) -> np.ndarray:
    """Determine number of terms needed for large-t expansion.
    Args:
        rt: An 1D numpy array of flipped RTs. (0, inf).
        err: Error bound
    Returns: a 1D at array of k_large.
    """
    kl = at.sqrt(-2 * at.log(np.pi * rt * err) / (np.pi**2 * rt))
    kl = at.max(at.stack([kl, 1.0 / (np.pi * at.sqrt(rt))]), axis=0)
    kl = at.switch(np.pi * rt * err < 1, kl, 1.0 / (np.pi * at.sqrt(rt)))

    return kl


def compare_k(rt: np.ndarray, err: float) -> np.ndarray:
    """Computes and compares k_small with k_large.
    Args:
        rt: An 1D numpy of flipped RTs. (0, inf).
        err: Error bound
    Returns: a 1D boolean at array of which implementation should be used.
    """
    ks = k_small(rt, err)
    kl = k_large(rt, err)

    return ks < kl


def decision_func() -> Callable[[np.ndarray, float], np.ndarray]:
    """Produces a decision function that determines whether the pdf should be calculated
    with large-time or small-time expansion.
    Returns: A decision function with saved state to avoid repeated computation.
    """

    internal_rt: np.ndarray | None = None
    internal_err: float | None = None
    internal_result: np.ndarray | None = None

    def inner_func(rt: np.ndarray, err: float = 1e-7) -> np.ndarray:
        """For each element in `rt`, return `True` if the large-time expansion is
        more efficient than the small-time expansion and `False` otherwise.
        This function uses a closure to save the result of past computation.
        If `rt` and `err` passed to it does not change, then it will directly
        return the results of the previous computation.
        Args:
            rt: An 1D numpy array of flipped RTs. (0, inf).
            err: Error bound
        Returns: a 1D boolean at array of which implementation should be used.
        """

        nonlocal internal_rt
        nonlocal internal_err
        nonlocal internal_result

        if (
            internal_result is not None
            and err == internal_err
            and np.all(rt == internal_rt)
        ):
            # This order is to promote short circuiting to avoid
            # unnecessary computation.
            return internal_result

        internal_rt = rt
        internal_err = err

        lambda_rt = compare_k(rt, err)

        internal_result = lambda_rt

        return lambda_rt

    return inner_func


# This decision function keeps an internal state of `tt`
# and does not repeat computation if a new `tt` passed to
# it is the same
decision = decision_func()


def get_ks(k_terms: int, fast: bool) -> np.ndarray:
    """Returns an array of ks given the number of terms needed to
    approximate the sum of the infinite series.
    Args:
        k_terms: number of terms needed
        fast: whether the function is used in the fast of slow expansion.
    Returns: An array of ks.
    """
    if fast:
        return at.arange(-at.floor((k_terms - 1) / 2), at.ceil((k_terms - 1) / 2) + 1)
    return at.arange(1, k_terms + 1).reshape((-1, 1))


def ftt01w_fast(tt: np.ndarray, w: float, k_terms: int) -> np.ndarray:
    """Density function for lower-bound first-passage times with drift rate set to 0 and
    upper bound set to 1, calculated using the fast-RT expansion.
    Args:
        tt: Flipped, normalized RTs. (0, inf).
        w: Normalized decision starting point. (0, 1).
        k_terms: number of terms to use to approximate the PDF.
    Returns:
        The approximated function f(tt|0, 1, w).
    """

    # Slightly changed the original code to mimic the paper and
    # ensure correctness
    k = get_ks(k_terms, fast=True)

    # A log-sum-exp trick is used here
    y = w + 2 * k.reshape((-1, 1))
    r = -at.power(y, 2) / (2 * tt)
    c = at.max(r, axis=0)
    p = at.exp(c + at.log(at.sum(y * at.exp(r - c), axis=0)))
    # Normalize p
    p = p / at.sqrt(2 * np.pi * at.power(tt, 3))

    return p


def ftt01w_slow(tt: np.ndarray, w: float, k_terms: int) -> np.ndarray:
    """Density function for lower-bound first-passage times with drift rate set to 0 and
    upper bound set to 1, calculated using the slow-RT expansion.
    Args:
        tt: Flipped, normalized RTs. (0, inf).
        w: Normalized decision starting point. (0, 1).
        k_terms: number of terms to use to approximate the PDF.
    Returns:
        The approximated function f(tt|0, 1, w).
    """
    k = get_ks(k_terms, fast=False)
    y = k * at.sin(k * np.pi * w)
    r = -at.power(k, 2) * at.power(np.pi, 2) * tt / 2
    p = at.sum(y * at.exp(r), axis=0) * np.pi

    return p


def ftt01w(
    rt: np.ndarray,
    a: float,
    w: float,
    err: float = 1e-7,
    k_terms: int = 10,
) -> np.ndarray:
    """Compute the appproximated density of f(tt|0,1,w) using the method
    and implementation of Navarro & Fuss, 2009.
    Args:
        rt: Flipped Response Rates. (0, inf).
        a: Value of decision upper bound. (0, inf).
        w: Normalized decision starting point. (0, 1).
        err: Error bound.
        k_terms: number of terms to use to approximate the PDF.
    """
    lambda_rt = decision(rt, err)
    tt = rt / a**2

    p_fast = ftt01w_fast(tt, w, k_terms)
    p_slow = ftt01w_slow(tt, w, k_terms)

    p = at.switch(lambda_rt, p_fast, p_slow)

    return p * (p > 0)  # Making sure that p > 0


def log_pdf_sv(
    data: np.ndarray,
    v: float,
    sv: float,
    a: float,
    z: float,
    t: float,
    err: float = 1e-7,
    k_terms: int = 10,
) -> np.ndarray:
    """Computes the log-likelihood of the drift diffusion model f(t|v,a,z) using
    the method and implementation of Navarro & Fuss, 2009.
    Args:
        data: RTs. (-inf, inf) except 0. Negative values correspond to the lower bound.
        v: Mean drift rate. (-inf, inf).
        sv: Standard deviation of the drift rate [0, inf).
        a: Value of decision upper bound. (0, inf).
        z: Normalized decision starting point. (0, 1).
        t: Non-decision time [0, inf).
        err: Error bound.
        k_terms: number of terms to use to approximate the PDF.
    """

    # First, flip data to positive
    data = at.reshape(data, (-1, 2))
    rt = data[:, 0]
    choice = data[:, 1]
    v_flipped = at.switch(choice, -v, v)  # transform v if x is upper-bound response
    z_flipped = at.switch(choice, 1 - z, z)  # transform z if x is upper-bound response
    rt = rt - t  # remove nondecision time

    p = ftt01w(rt, a, z_flipped, err, k_terms)

    # This step does 3 things at the same time:
    # 1. Computes f(t|v, a, z) from the pdf when setting a = 0 and z = 1.
    # 2. Computes the log of above value
    # 3. Computes the integration given the sd of v
    logp = (
        at.log(p)
        + (
            (a * z_flipped * sv) ** 2
            - 2 * a * v_flipped * z_flipped
            - (v_flipped**2) * rt
        )
        / (2 * (sv**2) * rt + 2)
        - at.log(sv**2 * rt + 1) / 2
        - 2 * at.log(a)
    )

    checked_logp = check_parameters(
        logp,
        sv >= 0,
        msg="sv >= 0",
    )
    checked_logp = check_parameters(checked_logp, a >= 0, msg="a >= 0")
    # checked_logp = check_parameters(checked_logp, 0 < z < 1, msg="0 < z < 1")
    # checked_logp = check_parameters(checked_logp, np.all(rt > 0), msg="t <= min(rt)")

    return checked_logp
