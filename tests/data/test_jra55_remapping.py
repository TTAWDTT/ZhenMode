"""Closed-form spherical-area controls, periodic seam and missing-cell refusal."""
import numpy as np
import pytest

from zhenmode.model.inputs.forcing.jra55 import (
    conservative_rectilinear_weights,
    remap_rectilinear_means,
)


def test_unequal_latitude_bands_area_integral_and_constant():
    # Two bands occupy 1/4 and 3/4 of the sphere; their average is 4.
    weights = conservative_rectilinear_weights([[0, 180], [180, 360]],
                                               [[-90, -30], [-30, 90]],
                                               [[-180, 0], [0, 180]], [[-90, 90]])
    np.testing.assert_allclose(remap_rectilinear_means([[1, 1], [5, 5]], weights), [[4, 4]])
    np.testing.assert_allclose(remap_rectilinear_means(np.ones((2, 2)), weights), 1)


def test_periodic_seam_and_independent_longitude_means():
    weights = conservative_rectilinear_weights([[0, 180], [180, 360]], [[-90, 90]],
                                               [[-90, 90], [90, 270]], [[-90, 90]])
    np.testing.assert_allclose(remap_rectilinear_means([[1, 3]], weights), [[2, 2]])


def test_refinement_and_coarsening_against_analytic_linear_cell_averages():
    # f(sin(lat))=2+sin(lat), integrates analytically across every band.
    latitude = np.linspace(-90, 90, 9)
    bands = np.column_stack((latitude[:-1], latitude[1:]))
    values = 2 + (np.sin(np.deg2rad(bands[:, 0])) + np.sin(np.deg2rad(bands[:, 1]))) / 2
    weights = conservative_rectilinear_weights([[0, 360]], bands, [[0, 360]], [[-90, 0], [0, 90]])
    np.testing.assert_allclose(remap_rectilinear_means(values[:, None], weights), [[1.5], [2.5]], atol=1e-14)


@pytest.mark.parametrize('bad', [[[0, 170], [180, 360]], [[0, 180], [170, 360]], [[0, np.nan]], [[0, 180]]])
def test_gaps_overlaps_nonfinite_or_partial_longitudes_rejected(bad):
    with pytest.raises(ValueError):
        conservative_rectilinear_weights(bad, [[-90, 90]], [[0, 360]], [[-90, 90]])


def test_missing_cells_and_mutated_weights_rejected():
    weights = conservative_rectilinear_weights([[0, 360]], [[-90, 90]], [[0, 360]], [[-90, 90]])
    for data in (np.array([[np.nan]]), np.ma.array([[1]], mask=[[True]]), np.ones((2, 1))):
        with pytest.raises(ValueError):
            remap_rectilinear_means(data, weights)
    weights[0][0, 0] = .9
    with pytest.raises(ValueError):
        remap_rectilinear_means([[1]], weights)
