import asyncio

import astropy.units as u
import pytest
import pytest_asyncio
from astropy.coordinates import EarthLocation
from astropy.time import Time

from sensorkit.astro.common import ReferenceFrame
from sensorkit.astro.coords import Equatorial, Geodetic
from sensorkit.astro.target import EphemerisTarget, ICRSTarget
from sensorkit.common.aio import AsyncLoop
from sensorkit.std import Connect, FollowTarget, MoveToPark, Stop
from sensorkit.thesky.telescope import TheSkyTelescopeConfig, TheSkyTelescopeState

SECONDS_PER_DAY = 86400.0

# Slews land at once in the simulator; no lead means the slew aims at the target as it is.
NO_LEAD = {"follow_slew_overhead": 0.0, "follow_slew_rate": 1e9}


@pytest_asyncio.fixture
async def make_telescope(simulator):
    host, port = simulator
    made = []

    async def make(**overrides):
        config = TheSkyTelescopeConfig(
            device_type="telescope",
            host=host,
            port=port,
            needs_homed=False,
            timeout=5.0,
            status_frequency=0.1,
            **overrides,
        )
        m = config.create_device()
        m.state = TheSkyTelescopeState()
        # Normally set during entity_init (on_attach) from TheSky's site info, which
        # these connect-only tests bypass. _geodetic is unused for pass-through adapts;
        # _location is what status_publish needs to reach the alt/az rate conversion.
        m._geodetic = Geodetic(lon=149.0, lat=-31.0, elev=1100.0)
        m._location = EarthLocation(lat=-31.0 * u.deg, lon=149.0 * u.deg, height=1100.0 * u.m)
        m.status_loop = AsyncLoop(m.status_publish, interval=config.status_frequency_slow)
        m.fast_loop = AsyncLoop(m._publish_telescope_status, interval=config.status_frequency_fast)
        made.append(m)

        await m.telescope_connect(Connect())
        await m.telescope_unpark()
        return m

    yield make

    for m in made:
        await m._stop_follow()
        await m.status_loop.stop()
        await m.fast_loop.stop()


@pytest_asyncio.fixture
async def telescope(make_telescope):
    return await make_telescope()


def make_path(ra_rate, dec_rate, ra_accel=0.0, span=(-10, 180)):
    """Return an ICRF EphemerisTarget sampled every 2 s, and its exact position function.

    The path starts at RA 100, Dec 20 now, moving at the given rates (deg/s) with an
    optional RA acceleration (deg/s^2).
    """
    now_jd = Time.now().jd

    def at(jd):
        t = (jd - now_jd) * SECONDS_PER_DAY
        return 100.0 + ra_rate * t + 0.5 * ra_accel * t * t, 20.0 + dec_rate * t

    jds = [now_jd + t / SECONDS_PER_DAY for t in range(span[0], span[1], 2)]
    points = [Equatorial(ra=ra, dec=dec) for ra, dec in map(at, jds)]
    return EphemerisTarget(frame=ReferenceFrame.ICRF, jds=jds, points=points), at


async def mount_radec(telescope):
    """Read the mount's (RA deg, Dec deg), advancing the simulated tracking."""
    ra, dec = await telescope._get_radec()
    return ra * 15, dec


async def reads(telescope, field):
    return float(await telescope.execute(f"sky6RASCOMTele.{field};"))


@pytest.mark.asyncio
async def test_telescope_follow_ephemeris(make_telescope):
    """An ICRF EphemerisTarget slews onto the path and tracks at its offset rate.

    A linear path moving 0.02 deg in RA and 0.01 deg in Dec per 2 s sample should apply
    offset rates of 36 arcsec/s (RA) and 18 arcsec/s (Dec) via SetTracking -- TheSky
    takes plain angular arcsec/s, with no sidereal scaling.
    """
    telescope = await make_telescope(**NO_LEAD, follow_update_period=60.0)
    target, at = make_path(0.01, 0.005)

    before = Time.now().jd
    await telescope.telescope_follow_target(FollowTarget(target=target))
    after = Time.now().jd

    assert (await telescope.execute("sky6RASCOMTele.IsTracking;")).strip() == "1"

    # Slewed to the target as it was during the call. SlewToRaDec takes RA in hours,
    # so the 100 deg path goes out as 100/15 hr; Dec is degrees on both sides.
    ra = await reads(telescope, "dRa") * 15
    assert at(before)[0] - 1e-3 <= ra <= at(after)[0] + 1e-3

    # rel=1e-3 absorbs float64 precision loss from differencing ~2.46e6 JDs.
    assert await reads(telescope, "dRaTrackingRate") == pytest.approx(36.0, rel=1e-3)
    assert await reads(telescope, "dDecTrackingRate") == pytest.approx(18.0, rel=1e-3)


@pytest.mark.asyncio
async def test_follow_slew_leads_the_target(make_telescope):
    """The slew aims where the target will be once the estimated slew time has passed.

    From the simulator's park pointing (RA 180, Dec 30) to RA ~100 is an 80 deg RA move,
    the larger axis: at 8 deg/s plus a 5 s overhead the slew should lead by 15 s.
    """
    telescope = await make_telescope(
        follow_slew_overhead=5.0, follow_slew_rate=8.0, follow_update_period=60.0
    )
    target, at = make_path(0.01, 0.005)

    before = Time.now().jd
    await telescope.telescope_follow_target(FollowTarget(target=target))
    after = Time.now().jd

    lead = 15.0 / SECONDS_PER_DAY
    ra = await reads(telescope, "dRa") * 15
    dec = await reads(telescope, "dDec")
    # The 0.15 deg lead is well clear of the ~0.02 deg the call's own duration spans.
    assert at(before + lead)[0] - 2e-3 <= ra <= at(after + lead)[0] + 2e-3
    assert at(before + lead)[1] - 2e-3 <= dec <= at(after + lead)[1] + 2e-3
    assert ra > at(after)[0] + 0.1


@pytest.mark.asyncio
async def test_follow_rates_track_a_changing_path(make_telescope):
    """The update loop keeps re-rating the mount as the target's motion changes.

    The path accelerates in RA by 0.001 deg/s^2, so a single starting rate would fall
    behind by ~0.01 deg within 5 s; the loop should keep both the rate and the pointing
    on the path.
    """
    telescope = await make_telescope(**NO_LEAD, follow_update_period=0.5)
    target, at = make_path(0.01, 0.004, ra_accel=0.001)

    await telescope.telescope_follow_target(FollowTarget(target=target))
    first_rate = await reads(telescope, "dRaTrackingRate")
    await asyncio.sleep(3.0)

    now = Time.now().jd
    rate = await reads(telescope, "dRaTrackingRate")
    path_rate = (at(now + 0.5 / SECONDS_PER_DAY)[0] - at(now)[0]) / 0.5 * 3600

    # The rate grows 3.6 arcsec/s every second, so it has moved well past where it began.
    assert rate - first_rate > 5.0
    assert rate == pytest.approx(path_rate, abs=4.0)
    assert await reads(telescope, "dDecTrackingRate") == pytest.approx(0.004 * 3600, abs=2.0)

    ra, dec = await mount_radec(telescope)
    path_ra, path_dec = at(Time.now().jd)
    assert ra == pytest.approx(path_ra, abs=0.005)
    assert dec == pytest.approx(path_dec, abs=0.005)


@pytest.mark.asyncio
async def test_follow_reslews_past_the_threshold(make_telescope):
    """A pointing error past the threshold slews the mount back onto the path."""
    telescope = await make_telescope(
        **NO_LEAD,
        follow_update_period=0.5,
        follow_reslew_threshold=0.05,
        follow_reslew_interval=0.0,
    )
    target, at = make_path(0.01, 0.005)

    await telescope.telescope_follow_target(FollowTarget(target=target))

    # Knock the mount 5 deg off the path, as a missed slew or a bump would.
    await telescope.execute("sky6RASCOMTele.dRa = 7.0;")
    # One update period to see it, then the re-slew's ~2 s onset wait in the simulator.
    await asyncio.sleep(4.0)

    ra, dec = await mount_radec(telescope)
    path_ra, path_dec = at(Time.now().jd)
    assert ra == pytest.approx(path_ra, abs=0.01)
    assert dec == pytest.approx(path_dec, abs=0.01)
    assert telescope._follow_loop is not None and telescope._follow_loop.active


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["stop", "park", "follow"])
async def test_follow_loop_stops_on_new_command(make_telescope, command):
    """A Stop, a park or another FollowTarget ends the update loop for good."""
    telescope = await make_telescope(**NO_LEAD, follow_update_period=0.2)
    target, _ = make_path(0.01, 0.005)

    await telescope.telescope_follow_target(FollowTarget(target=target))
    loop = telescope._follow_loop
    assert loop is not None and loop.active

    match command:
        case "stop":
            await telescope.telescope_stop(Stop())
        case "park":
            await telescope.telescope_park(MoveToPark())
        case "follow":
            await telescope.telescope_follow_target(
                FollowTarget(target=ICRSTarget(coords=Equatorial(ra=50.0, dec=10.0)))
            )

    assert telescope._follow_loop is None and not loop.active

    # A loop still running would turn offset-rate tracking back on within a period.
    await asyncio.sleep(0.6)
    if command == "follow":
        assert (await telescope.execute("sky6RASCOMTele._offsetRates;")).strip() == "false"
    else:
        assert (await telescope.execute("sky6RASCOMTele.IsTracking;")).strip() == "0"


@pytest.mark.asyncio
async def test_telescope_follow_tle_without_raven3(telescope):
    """Without Raven3, a TLE target follows its propagated track and never touches Raven3."""
    from sensorkit.astro.common import TLE
    from sensorkit.astro.target import TLETarget

    telescope._has_raven3 = False
    # Poison Raven3 so any reference fails the test, as it would stall a real TheSky.
    await telescope.execute("Raven3 = null;")

    target = TLETarget(
        tle=TLE(
            line0="0 ISS (ZARYA)",
            line1="1 25544U 98067A   24100.50000000  .00016717  00000-0  10270-3 0  9002",
            line2="2 25544  51.6400 200.0000 0001234  90.0000 270.0000 15.49000000400000",
        )
    )

    await telescope.telescope_follow_target(FollowTarget(target=target))

    assert (await telescope.execute("sky6RASCOMTele.IsTracking;")).strip() == "1"
    ra_rate = float(await telescope.execute("sky6RASCOMTele.dRaTrackingRate;"))
    dec_rate = float(await telescope.execute("sky6RASCOMTele.dDecTrackingRate;"))
    assert abs(ra_rate) + abs(dec_rate) > 0
    # Followed as an unbounded track, re-rated in the background.
    assert telescope._follow_loop is not None and telescope._follow_loop.active
