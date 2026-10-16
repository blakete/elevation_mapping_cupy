"""make_planning_occupancy_grid: interior large UNKNOWN -> OCCUPIED, border-touching
UNKNOWN stays UNKNOWN (2026-09), small pockets stay UNKNOWN, FREE/OCCUPIED untouched,
input never mutated. Pure numpy/scipy; runs without ROS or a GPU."""
import numpy as np
import pytest

from elevation_mapping_cupy.planning_occupancy import (
    FREE, OCCUPIED, UNKNOWN, make_planning_occupancy_grid)


def grid(n=12):
    return np.full((n, n), FREE, dtype=np.int8)


def test_interior_large_component_becomes_occupied():
    g = grid()
    g[4:8, 4:8] = UNKNOWN  # 16 cells, interior
    out = make_planning_occupancy_grid(g, 4)
    assert (out[4:8, 4:8] == OCCUPIED).all()
    assert (out[g == FREE] == FREE).all()


def test_border_touching_component_stays_unknown():
    g = grid()
    g[0, :] = UNKNOWN          # whole top row
    g[:, -1] = UNKNOWN         # whole right column (joined to the row)
    g[4:8, 4:8] = UNKNOWN      # plus an interior block
    out = make_planning_occupancy_grid(g, 4)
    assert (out[0, :] == UNKNOWN).all()
    assert (out[:, -1] == UNKNOWN).all()
    assert (out[4:8, 4:8] == OCCUPIED).all(), "interior gap must still be blocked"


def test_border_exemption_can_be_disabled():
    g = grid()
    g[0, :] = UNKNOWN
    out = make_planning_occupancy_grid(g, 4, exempt_border_components=False)
    assert (out[0, :] == OCCUPIED).all()


def test_small_pockets_stay_unknown_and_diagonal_connectivity():
    g = grid()
    g[2, 2] = UNKNOWN                    # 1 cell
    g[6, 6] = g[7, 7] = g[8, 8] = UNKNOWN  # 3 cells, 8-connected diagonal
    out = make_planning_occupancy_grid(g, 4)
    assert out[2, 2] == UNKNOWN
    assert out[6, 6] == out[7, 7] == out[8, 8] == UNKNOWN
    out5 = make_planning_occupancy_grid(g, 2)
    assert out5[2, 2] == UNKNOWN
    assert out5[6, 6] == out5[7, 7] == out5[8, 8] == OCCUPIED, "3 > 2 and 8-connected"


def test_free_and_occupied_untouched_and_input_not_mutated():
    g = grid()
    g[3, 3] = OCCUPIED
    g[5:9, 5:9] = UNKNOWN
    before = g.copy()
    out = make_planning_occupancy_grid(g, 4)
    assert out[3, 3] == OCCUPIED
    assert (g == before).all()
    assert out is not g


def test_no_unknown_is_identity_copy():
    g = grid()
    g[1, 1] = OCCUPIED
    out = make_planning_occupancy_grid(g, 4)
    assert (out == g).all() and out is not g
