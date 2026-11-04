# SPDX-License-Identifier: Apache-2.0
"""Live plate scale: gating and median of SENPAI solves, and the stream consumer."""

import asyncio
from datetime import UTC, datetime

import pytest

pytest.importorskip("burr")

from burr.core.config import AppConfig, PlottingConfig  # noqa: E402
from burr.models.hardware import HardwareConfig, TelescopeConfig  # noqa: E402
from burr.models.site import SiteMetadata  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from sensorkit.astro.common import AltAzPointing  # noqa: E402
from sensorkit.burr.models import MeasuredPlateScaleConfig  # noqa: E402
from sensorkit.burr.plate_scale import (  # noqa: E402
    PlateScaleEstimator,
    consume_senpai_results,
    with_telescope,
)
from sensorkit.burr.program import BurrConfig  # noqa: E402
from sensorkit.senpai.models import SenpaiResult  # noqa: E402

CONTROLLER = "taurus400"
FALLBACK = TelescopeConfig(ifov_no_binning_arcsec=10.0, field_of_view_degree=10.0)
PLOTTING = PlottingConfig(debug=False, review=False)

_frame_counter = 0


def make_result(scale=11.5, *, solved=True, controller=CONTROLLER, x_binning=1, file_path=None):
    global _frame_counter
    _frame_counter += 1
    return SenpaiResult(
        file_path=file_path or f"/data/raw/frame_{_frame_counter:05d}.fits",
        timestamp=datetime(2026, 11, 3, 4, 0, tzinfo=UTC),
        track_mode="SIDEREAL",
        controller_name=controller,
        x_binning=x_binning,
        y_binning=x_binning,
        n_sources=100,
        median_fwhm_pixels=None,
        std_fwhm_pixels=None,
        pixel_scale_arcsec=scale,
        median_fwhm_arcsec=None,
        std_fwhm_arcsec=None,
        solved=solved,
        detections=[],
    )


def estimator(**config) -> PlateScaleEstimator:
    return PlateScaleEstimator(MeasuredPlateScaleConfig(**config), FALLBACK, CONTROLLER)


class TestGating:
    def test_counts_solved_result_from_controller(self):
        assert estimator().unbinned_scale(make_result(11.5)) == 11.5

    @pytest.mark.parametrize(
        "result",
        [
            make_result(solved=False),
            make_result(scale=None),
            make_result(scale=0.0),
            make_result(controller="other"),
            make_result(controller=None),
        ],
        ids=["unsolved", "no-scale", "zero-scale", "other-controller", "no-controller"],
    )
    def test_ignores(self, result):
        assert estimator().unbinned_scale(result) is None

    def test_normalizes_binned_scale(self):
        assert estimator().unbinned_scale(make_result(23.0, x_binning=2)) == 11.5

    def test_missing_binning_is_unbinned(self):
        assert estimator().unbinned_scale(make_result(11.5, x_binning=None)) == 11.5

    @pytest.mark.parametrize("scale", [4.9, 20.1, 230.0])
    def test_rejects_scale_outside_sanity_bounds(self, scale):
        assert estimator().unbinned_scale(make_result(scale)) is None

    @pytest.mark.parametrize("scale", [5.0, 20.0])
    def test_bounds_are_inclusive(self, scale):
        assert estimator().unbinned_scale(make_result(scale)) == scale


class TestMedian:
    def test_waits_for_min_solves(self):
        est = estimator(min_solves=3)
        assert est.add(make_result(11.5)) is None
        assert est.add(make_result(11.6)) is None

        telescope = est.add(make_result(11.4))
        assert telescope is not None
        assert telescope.ifov_no_binning_arcsec == pytest.approx(11.5)
        assert est.applied == pytest.approx(11.5)

    def test_rescales_field_of_view(self):
        est = estimator(min_solves=1)
        telescope = est.add(make_result(12.0))
        assert telescope.field_of_view_degree == pytest.approx(12.0)

    def test_within_tolerance_keeps_current_scale(self):
        est = estimator(min_solves=1, tolerance=0.05)
        assert est.add(make_result(10.4)) is None
        assert est.applied == 10.0

    def test_reapplies_only_on_further_drift(self):
        est = estimator(min_solves=1, window=1, tolerance=0.05)
        assert est.add(make_result(11.5)) is not None
        assert est.add(make_result(11.8)) is None  # within 5% of 11.5
        assert est.add(make_result(12.5)).ifov_no_binning_arcsec == 12.5

    def test_median_resists_outlier_solve(self):
        est = estimator(min_solves=3)
        for scale in (11.5, 11.5, 19.0):  # a bad solve inside the sanity bounds
            telescope = est.add(make_result(scale))
        assert telescope.ifov_no_binning_arcsec == 11.5

    def test_rejected_solves_do_not_count(self):
        est = estimator(min_solves=2)
        assert est.add(make_result(11.5)) is None
        assert est.add(make_result(100.0)) is None
        assert est.add(make_result(solved=False)) is None
        assert est.n_samples == 1

    def test_redelivered_frame_counted_once(self):
        est = estimator(min_solves=2)
        est.add(make_result(11.5, file_path="a.fits"))
        assert est.add(make_result(11.5, file_path="a.fits")) is None
        assert est.n_samples == 1

    def test_window_drops_old_solves(self):
        est = estimator(min_solves=1, window=2)
        for scale in (11.0, 11.0, 15.0, 15.0):
            est.add(make_result(scale))
        assert est.median == 15.0

    def test_starts_from_applied_scale(self):
        est = PlateScaleEstimator(
            MeasuredPlateScaleConfig(min_solves=1), FALLBACK, CONTROLLER, applied=11.5
        )
        assert est.add(make_result(11.6)) is None


class TestConfig:
    def test_with_telescope_replaces_frozen_config(self):
        config = AppConfig(
            stage="test",
            plotting=PLOTTING,
            site=SiteMetadata(name="site", latitude=32.0, longitude=-110.0, altitude_km=1.0),
            hardware=HardwareConfig(telescope=FALLBACK),
        )
        telescope = TelescopeConfig(ifov_no_binning_arcsec=11.5, field_of_view_degree=11.5)

        updated = with_telescope(config, telescope)

        assert updated.hardware.telescope == telescope
        assert config.hardware.telescope == FALLBACK
        assert updated.schedule is config.schedule

    def test_measured_requires_fallback_telescope(self):
        with pytest.raises(ValidationError, match="measured_plate_scale"):
            BurrConfig(
                controller="c", plotting=PLOTTING, measured_plate_scale=MeasuredPlateScaleConfig()
            )

    def test_disabled_measured_needs_no_telescope(self):
        BurrConfig(
            controller="c",
            plotting=PLOTTING,
            measured_plate_scale=MeasuredPlateScaleConfig(enabled=False),
        )


@pytest.mark.asyncio
async def test_consumer_hands_senpai_results_to_estimator(program_impl):
    """SenpaiResults published to the stream reach the handler; other keywords don't."""
    est = estimator(min_solves=2)
    applied: list[TelescopeConfig] = []
    received: list[SenpaiResult] = []

    async def handle(result):
        received.append(result)
        if telescope := est.add(result):
            applied.append(telescope)

    ready = asyncio.Event()
    consumer = asyncio.create_task(
        consume_senpai_results(program_impl.sensorkit(), handle, ready=ready)
    )
    try:
        async with asyncio.timeout(5.0):
            await ready.wait()

        await program_impl.publish(AltAzPointing(altitude_degrees=45.0, azimuth_degrees=90.0))
        await program_impl.publish(make_result(11.5))
        await program_impl.publish(make_result(11.5, solved=False))
        await program_impl.publish(make_result(11.7))

        async with asyncio.timeout(2.0):
            while len(received) < 3:
                await asyncio.sleep(0.01)
    finally:
        consumer.cancel()

    assert [t.ifov_no_binning_arcsec for t in applied] == [pytest.approx(11.6)]
