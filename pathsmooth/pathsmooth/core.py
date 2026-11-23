"""Laplacian path smoothing.

The implementation is intentionally stateless: a :class:`PathSmoother`
instance holds only immutable configuration, so one instance may be shared
by any number of threads without locking.
"""

import math

__all__ = ["PathSmoother", "DEFAULT_MAX_DISPLACEMENT"]

# Default upper bound for the displacement of any single point, measured
# from its position in the input path.
DEFAULT_MAX_DISPLACEMENT = 1.0

# Stop iterating once no point moved more than this during a whole pass.
_CONVERGENCE_EPS = 1e-12


def _as_finite_number(value, name):
    """Return *value* as a finite float, or raise ``ValueError``."""
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be a real number, got {type(value).__name__}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a real number, got {type(value).__name__}"
        ) from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _validate_iterations(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"iterations must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"iterations must be >= 1, got {value}")
    return value


def _validate_alpha(value):
    alpha = _as_finite_number(value, "alpha")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {value}")
    return alpha


def _validate_bound(value):
    bound = _as_finite_number(value, "max_displacement")
    if bound < 0.0:
        raise ValueError(f"max_displacement must be >= 0, got {value}")
    return bound


def _as_coordinate(value, index):
    """Convert one coordinate to a finite float or raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"point {index} has a non-numeric coordinate "
            f"of type {type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"point {index} has a non-finite coordinate: {value!r}")
    return result


def _coerce_points(points):
    """Validate *points* and return a fresh list of ``(float, float)`` tuples.

    The input object is never mutated.  Every malformed input (too few
    points, wrong arity, non-numeric or non-finite coordinates) is reported
    as ``ValueError`` so callers can distinguish bad data from programming
    errors.
    """
    if isinstance(points, (str, bytes)):
        raise ValueError("points must be an iterable of 2D points")
    try:
        raw = list(points)
    except TypeError as exc:
        raise ValueError("points must be an iterable of 2D points") from exc
    if len(raw) < 3:
        raise ValueError(f"at least 3 points are required, got {len(raw)}")

    result = []
    for index, point in enumerate(raw):
        if isinstance(point, (str, bytes)):
            raise ValueError(f"point {index} is not a 2D point")
        try:
            coords = tuple(point)
        except TypeError as exc:
            raise ValueError(f"point {index} is not a 2D point") from exc
        if len(coords) != 2:
            raise ValueError(
                f"point {index} must have exactly 2 coordinates, got {len(coords)}"
            )
        result.append((_as_coordinate(coords[0], index),
                       _as_coordinate(coords[1], index)))
    return result


class PathSmoother:
    """Smooth polylines with the Laplacian (neighbour-average) filter.

    Parameters
    ----------
    iterations:
        Number of smoothing passes; must be a positive integer.  Passes stop
        early once the path has converged.
    alpha:
        Blend factor in ``(0, 1]``; ``1.0`` moves a point all the way to the
        midpoint of its neighbours.
    max_displacement:
        Upper bound on how far any point may end up from its original
        position.  Defaults to :data:`DEFAULT_MAX_DISPLACEMENT`.

    Instances are immutable after construction and safe to share between
    threads.
    """

    __slots__ = ("_iterations", "_alpha", "_max_displacement")

    def __init__(self, iterations=1, alpha=0.5,
                 max_displacement=DEFAULT_MAX_DISPLACEMENT):
        self._iterations = _validate_iterations(iterations)
        self._alpha = _validate_alpha(alpha)
        self._max_displacement = _validate_bound(max_displacement)

    @property
    def iterations(self):
        return self._iterations

    @property
    def alpha(self):
        return self._alpha

    @property
    def max_displacement(self):
        return self._max_displacement

    def smooth(self, points, max_displacement=None):
        """Return a smoothed copy of *points* as a list of float tuples.

        The first and last points are never moved.  The input sequence is
        not modified.  ``max_displacement`` may override the bound configured
        on the instance for this call only.
        """
        bound = (self._max_displacement if max_displacement is None
                 else _validate_bound(max_displacement))
        original = _coerce_points(points)
        count = len(original)
        alpha = self._alpha

        current = original
        for _ in range(self._iterations):
            previous = current
            updated = previous[:]  # endpoints stay locked
            largest_move = 0.0
            for index in range(1, count - 1):
                ax, ay = previous[index - 1]
                bx, by = previous[index]
                cx, cy = previous[index + 1]
                step_x = (ax + cx) * 0.5 - bx
                step_y = (ay + cy) * 0.5 - by
                new_x = bx + alpha * step_x
                new_y = by + alpha * step_y
                updated[index] = (new_x, new_y)
                if abs(new_x - bx) > largest_move:
                    largest_move = abs(new_x - bx)
                if abs(new_y - by) > largest_move:
                    largest_move = abs(new_y - by)
            current = updated
            if largest_move <= _CONVERGENCE_EPS:
                break

        # Enforce the cumulative displacement bound relative to the input.
        smoothed = [original[0]]
        append = smoothed.append
        for index in range(1, count - 1):
            ox, oy = original[index]
            x, y = current[index]
            dx = x - ox
            dy = y - oy
            distance = math.sqrt(dx * dx + dy * dy)
            if distance > bound:
                scale = bound / distance
                x = ox + dx * scale
                y = oy + dy * scale
            append((x, y))
        append(original[-1])
        return smoothed
