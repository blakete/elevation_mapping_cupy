"""Planning-only occupancy grid for MIGHTY's HGP/A*.

Pure numpy/scipy (no ROS, no cupy) so it is unit-testable on any box. Imported by
elevation_mapping_node.py, which publishes the result on planning_occ_2d_topic.
"""
import numpy as np
from scipy import ndimage

FREE = 0
UNKNOWN = -1
OCCUPIED = 100


def make_planning_occupancy_grid(
    data: np.ndarray, max_small_size: int = 2, exempt_border_components: bool = True
) -> np.ndarray:
    """Build a planning-only occupancy grid from a raw OccupancyGrid `data` array.

    The raw tri-state convention is FREE=0, UNKNOWN=-1, OCCUPIED=100. Small isolated
    pockets of UNKNOWN are traversable holes a planner can route through, but a large
    contiguous UNKNOWN region *inside* the observed map is a genuine gap in perception
    that MIGHTY's HGP/A* must treat as blocked. This returns a COPY where every
    8-connected UNKNOWN component of size > max_small_size is set to OCCUPIED (100);
    components of size <= max_small_size are left as UNKNOWN (-1). FREE/OCCUPIED cells
    are untouched.

    exempt_border_components (default True, added 2026-09): an UNKNOWN component that
    touches the grid's outer edge is left UNKNOWN regardless of size. That ring is the
    boundary of what the sensor has observed so far -- the map's edge, not a perception
    gap -- and flipping it to OCCUPIED turned the edge of the local map into a wall with
    gaps wherever the sensor happened to see free ground. With MIGHTY planning to goals
    beyond the local map, A* then left the map through whichever gap was cheapest that
    cycle and the path flipped between edges every update. MIGHTY now prices UNKNOWN
    per step (w_unknown) instead of treating it as free, so leaving the ring UNKNOWN
    is safe: it is costly, not open. Interior components are still flipped.

    Classification is recomputed from scratch on every call -- there is no history.
    Implemented as one vectorized connected-components labeling (scipy.ndimage.label,
    8-connectivity) plus a per-label size count (np.bincount); a 400x400 grid takes
    ~1 ms. `data` itself is never mutated (only the returned copy is written).
    """
    planning = data.copy()
    # Label every 8-connected UNKNOWN component. Background (non-UNKNOWN) -> label 0.
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labels, num = ndimage.label(data == UNKNOWN, structure=structure)
    if num == 0:
        return planning  # no UNKNOWN cells at all
    # sizes[k] = number of cells in component k (sizes[0] = background count).
    sizes = np.bincount(labels.ravel())
    is_large = sizes > max_small_size  # same threshold as before: > max_small_size
    is_large[0] = False                # never block background / FREE / OCCUPIED cells
    if exempt_border_components and labels.ndim == 2:
        border = np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
        is_large[np.unique(border)] = False  # includes label 0, already False
    # is_large[labels] broadcasts the per-label decision back to the grid; only cells
    # belonging to a large interior UNKNOWN component are flipped to OCCUPIED.
    planning[is_large[labels]] = OCCUPIED
    return planning
