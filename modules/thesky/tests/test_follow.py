import pytest

from sensorkit.thesky.follow import FollowPath, angular_offset, lead_slew, slew_time

SECONDS_PER_DAY = 86400.0
JD0 = 2461000.5


def test_slew_time_is_set_by_the_larger_axis_move():
    assert slew_time(10.0, 0.0, 30.0, 5.0, overhead=3.0, rate=4.0) == pytest.approx(8.0)
    assert slew_time(10.0, 0.0, 12.0, 40.0, overhead=3.0, rate=4.0) == pytest.approx(13.0)
    # RA moves the short way across 0/360.
    assert slew_time(359.0, 0.0, 1.0, 0.0, overhead=0.0, rate=1.0) == pytest.approx(2.0)


def test_angular_offset_scales_ra_by_cos_dec():
    assert angular_offset(10.0, 60.0, 12.0, 60.0) == pytest.approx(1.0, rel=1e-3)
    assert angular_offset(359.5, 0.0, 0.5, 0.0) == pytest.approx(1.0)


def test_path_interpolates_across_ra_wrap():
    jds = [JD0 + k * 2 / SECONDS_PER_DAY for k in range(3)]
    path = FollowPath(jds, [359.0, 1.0, 3.0], [0.0, 1.0, 2.0])

    ra, dec = path.position(JD0 + 1 / SECONDS_PER_DAY)
    assert ra == pytest.approx(0.0, abs=1e-6) or ra == pytest.approx(360.0, abs=1e-6)
    assert dec == pytest.approx(0.5)
    assert path.rate(JD0, 2.0) == pytest.approx((1.0, 0.5))
    # Past the last sample it extrapolates the final segment.
    assert path.position(JD0 + 6 / SECONDS_PER_DAY) == pytest.approx((5.0, 3.0), abs=1e-3)


def test_lead_slew_solves_for_arrival():
    # 1 deg/s in RA from RA 100; mount at RA 80: lead t satisfies t = 2 + (20 + t) / 5.
    jds = [JD0 + k / SECONDS_PER_DAY for k in range(0, 60)]
    path = FollowPath(jds, [100.0 + k for k in range(60)], [0.0] * 60)

    arrival_jd, ra, dec = lead_slew(path, 80.0, 0.0, now_jd=JD0, overhead=2.0, rate=5.0)

    t = (2.0 + 20.0 / 5.0) / (1 - 1 / 5.0)
    assert (arrival_jd - JD0) * SECONDS_PER_DAY == pytest.approx(t, rel=1e-3)
    assert ra == pytest.approx(100.0 + t, rel=1e-4)
    assert dec == 0.0
