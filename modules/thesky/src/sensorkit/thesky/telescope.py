# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import os
import tempfile
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, override

import astropy.units as u
from astropy.coordinates import ICRS, AltAz, EarthLocation, SkyCoord
from astropy.time import Time

# Prevent IERS-A (Earth orientation parameters) download, which can take long enough to cause a lease expiry
from astropy.utils import iers
from loguru import logger

import sensorkit.api as sk
from sensorkit.astro.common import TLE, AltAzPointing, RADecPointing, ReferenceFrame
from sensorkit.astro.coords import Geodetic
from sensorkit.astro.target import (
    AltAzTarget,
    EphemerisTarget,
    FrameTarget,
    ICRSTarget,
    RateTarget,
    TargetTrack,
    TLETarget,
)
from sensorkit.common.aio import AsyncLoop
from sensorkit.std import (
    AxisRate,
    AxisRates,
    Connect,
    Connected,
    Deinit,
    Disconnect,
    FollowTarget,
    Home,
    Init,
    MountAxis,
    MoveToPark,
    SetParkPosition,
    Slewing,
    StandardMount,
    Stop,
    Tracking,
)
from sensorkit.thesky.device import (
    CommandNotSupportedError,
    DomeCommandInProgressError,
    MountCommandInProgressError,
    MountNotHomedError,
    ProcessAbortedError,
    ScriptBusyError,
    TheSkyDevice,
    TheSkyDeviceConfig,
    TheSkyDeviceState,
    TheSkyError,
)
from sensorkit.thesky.follow import (
    SECONDS_PER_DAY,
    FollowPath,
    angular_offset,
    lead_slew,
    wrap_degrees,
)

iers.conf.auto_download = False
# Suppress the warning that results from the above decision
from astropy.utils.iers import conf  # noqa: E402

conf.auto_max_age = None
LEO_WAITING, LEO_SLEWING, LEO_SETTLING, LEO_TRACKING = 3, 4, 5, 6
# Max time to wait for a commanded slew to *start* before treating it as a
# positional no-op.
_MOUNT_ONSET_TIMEOUT = 2.0


@dataclass
class _Follow:
    """A path follow in progress: the path and what the mount was last told and seen."""

    path: FollowPath
    # Commanded offset rates, deg/s of RA and Dec.
    rate: tuple[float, float] = (0.0, 0.0)
    # Last mount readback as (ra deg, dec deg, jd); None until one follows a slew.
    mount: tuple[float, float, float] | None = None
    # time.monotonic() of the last slew onto the path.
    slewed_at: float = 0.0


def radec_rates_to_altaz_rates(
    ra_hr: float,
    dec_deg: float,
    ra_rate_deg_per_sec: float,
    dec_rate_deg_per_sec: float,
    location: EarthLocation,
    time: Time,
) -> tuple[float, float]:
    """Convert equatorial angular rates to altitude and azimuth rates.

    Args:
        ra_hr: Right ascension in hours.
        dec_deg: Declination in degrees.
        ra_rate_deg_per_sec: Right ascension rate in degrees per second.
        dec_rate_deg_per_sec: Declination rate in degrees per second.
        location: Observer location.
        time: Observation time.

    Returns:
        Altitude and azimuth rates, in degrees per second.
    """
    # Convert RA from hours to degrees
    ra_deg = ra_hr * 15.0

    # Create the AltAz frame
    altaz_frame = AltAz(obstime=time, location=location)

    # Create coordinate at current RA/Dec position
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame=ICRS())

    # Small time step for numerical differentiation
    dt = 0.01  # seconds

    # Calculate new RA/Dec position after dt
    new_ra = ra_deg + ra_rate_deg_per_sec * dt
    new_dec = dec_deg + dec_rate_deg_per_sec * dt

    # Create coordinate at new position
    new_coord = SkyCoord(ra=new_ra * u.deg, dec=new_dec * u.deg, frame=ICRS())

    # Transform both to AltAz (new position at an obstime advanced by dt)
    altaz = coord.transform_to(altaz_frame)
    new_altaz = new_coord.transform_to(AltAz(obstime=time + dt * u.s, location=location))

    # Calculate alt/az rates in deg/sec
    alt_rate_deg_per_sec = (new_altaz.alt.deg - altaz.alt.deg) / dt
    az_rate_deg_per_sec = (new_altaz.az.deg - altaz.az.deg) / dt

    return alt_rate_deg_per_sec, az_rate_deg_per_sec


@sk.declare_device(type=StandardMount)
class TheSkyTelescope(TheSkyDevice):
    """TheSky Telescope implementation."""

    config: TheSkyTelescopeConfig
    device_name = "Telescope"

    # Site info, populated from TheSky in entity_init. Defaulted here so a status
    # publish that lands before the attach completes skips the alt/az rate
    # conversion, which is what the `is not None` guard in status_publish assumes.
    _geodetic: Geodetic | None = None
    _location: EarthLocation | None = None

    # Whether TheSky has the Raven3 satellite-tracking plugin, probed in entity_init.
    # Without it, any reference to Raven3 throws, which parks TheSky's script server
    # in its debugger, so TLE targets follow their propagated track instead (_start_follow).
    _has_raven3: bool = True

    # Re-rates the mount along a moving target's path; see _follow_tick.
    _follow_loop: AsyncLoop | None = None
    _follow: _Follow | None = None

    # NOTE: For a TheSky telescope, you have to home it before any commands at all, and you have to "Unpark" it
    # (if in the "Park" position) before any motion. If you "Park" the telescope and "Disconnect" it, note that
    # it will not be in the software "Park" position upon a reconnection, even if it is actually in the hardware
    # "Park" position.

    @sk.on_attach
    async def entity_init(self):
        device = sk.device()

        # Restore state
        try:
            self.state = await device.kv_get_model(TheSkyTelescopeState)
            logger.debug(f"restored state for {device.entity}")
        except Exception:
            logger.warning(f"No saved state for {device.entity}")
            self.state = TheSkyTelescopeState()

        self.status_loop = AsyncLoop(
            self.status_publish, interval=self.config.status_frequency_slow, log=True
        )
        self.fast_loop = AsyncLoop(
            self._publish_telescope_status, interval=self.config.status_frequency_fast, log=True
        )

        # Get TheSky location
        resp = await self.execute(
            """
            var Out;
            var sk6DocProp_Latitude = 0;
            var sk6DocProp_Longitude = 1;
            var sk6DocProp_Elevation = 3;
            sky6StarChart.DocumentProperty(sk6DocProp_Latitude);
            dLat = sky6StarChart.DocPropOut
            sky6StarChart.DocumentProperty(sk6DocProp_Longitude);
            dLon = sky6StarChart.DocPropOut
            sky6StarChart.DocumentProperty(sk6DocProp_Elevation);
            dEle = sky6StarChart.DocPropOut
            Out = [
                dLat,
                dLon,
                dEle
            ];
            """
        )
        self._site_lat, self._site_lon, self._site_elev = [float(x) for x in resp.split(",")]

        # TheSky reports longitude positive-WEST; Geodetic and EarthLocation are
        # positive-EAST, so the sign always flips. Verified empirically against
        # TheSky's own ComputeLocalSiderealTime.
        self._site_lon = -self._site_lon
        self._geodetic = Geodetic(
            lon=self._site_lon,
            lat=self._site_lat,
            elev=self._site_elev,
        )
        self._location = EarthLocation(
            lat=self._site_lat * u.deg,
            lon=self._site_lon * u.deg,
            height=(self._site_elev) * u.m,
        )
        logger.debug(
            f"site info: lat={self._site_lat}, lon={self._site_lon}, "
            f"height={self._site_elev} m"
        )

        self._has_raven3 = (await self.execute("""typeof Raven3;""")).strip() != "undefined"
        if not self._has_raven3:
            logger.warning("TheSky has no Raven3 plugin; TLE targets will follow their propagated track")

    @sk.on_detach
    async def entity_deinit(self):
        await asyncio.sleep(self.config.status_frequency_slow)
        await self.status_loop.stop()
        await self.fast_loop.stop()
        await self.telescope_disconnect(Disconnect())
        await sk.device().kv_put_model(self.state)

    @sk.command_handler
    async def telescope_init(self, cmd: Init):
        # Connect to the hardware
        self._reconnect = lambda: self.telescope_connect(Connect())

        await self.telescope_connect(Connect())
        self.status_loop.start()

        # Home, as needed
        if not self.state.has_been_homed:
            await self.telescope_home(Home())

    @sk.command_handler
    async def telescope_deinit(self, cmd: Deinit):
        if not self.device_connected:
            # Init may not have run, but telescope may still be tracking from a failed run.
            # Connect so we can ensure it's parked.
            try:
                await self.telescope_connect(Connect())
            except Exception:
                logger.warning("Unable to connect telescope for Deinit park; skipping")
                return

        await self.fast_loop.stop()
        await self._stop_follow()
        await self.telescope_stop(Stop())
        await self.telescope_park(MoveToPark())

    @sk.command_handler
    async def telescope_connect(self, cmd: Connect):
        logger.debug("connecting to telescope")

        await self.execute(
            """
            sky6RASCOMTele.Asynchronous = 1;
            sky6RASCOMTele.ConnectAndDoNotUnpark();
            """
        )

        async with asyncio.timeout(self.config.timeout):
            await self.poll("""sky6RASCOMTele.IsConnected;""", "1")

        self.device_connected = True
        await sk.device().publish(Connected(is_connected=True))

        logger.debug("connected to telescope")

    @sk.command_handler
    async def telescope_disconnect(self, cmd: Disconnect):
        await self._stop_follow()
        logger.debug("disconnecting from telescope")

        await self.execute(
            """
            sky6RASCOMTheSky.DisconnectTelescope();
            """
        )

        async with asyncio.timeout(self.config.timeout):
            await self.poll("""sky6RASCOMTele.IsConnected;""", "0")

        self.device_connected = False
        await sk.device().publish(Connected(is_connected=False))

        logger.debug("disconnected from telescope")

    @sk.command_handler
    async def telescope_stop(self, cmd: Stop):
        await self._stop_follow()
        await self.fast_loop.stop()
        await self.telescope_unpark()
        logger.debug("stopping telescope")

        try:
            await self.execute(
                """
                sky6RASCOMTele.Abort();
                sky6RASCOMTele.SetTracking(0,1,0,0);
                """
            )

            async with asyncio.timeout(self.config.timeout):
                await self.poll("""sky6RASCOMTele.IsTracking;""", "0", interval=0.2)
        except CommandNotSupportedError:
            logger.warning("Mount does not support SetTracking; skipping")

        await self.fast_loop.stop()
        logger.debug("stopped telescope")

    @sk.command_handler
    async def telescope_park(self, cmd: MoveToPark):
        await self._stop_follow()
        await self.fast_loop.stop()
        await self.require_connected()
        logger.debug("parking telescope")
        await self._homed_retry(self._start_park)

        async with asyncio.timeout(self.config.timeout):
            await self.poll("""sky6RASCOMTele.IsParked();""", "true")

        logger.debug("parked telescope")

    async def _start_park(self):
        async with asyncio.timeout(self.config.timeout):
            while True:
                try:
                    await self.execute("""sky6RASCOMTele.ParkAndDoNotDisconnect();""")
                    return
                except (MountCommandInProgressError, ScriptBusyError):
                    await asyncio.sleep(0.5)

    @sk.command_handler
    async def telescope_set_park_position(self, cmd: SetParkPosition):
        await self.require_connected()
        logger.debug("setting park position")

        await self.execute(
            """
            sky6RASCOMTele.SetParkPosition();
            """
        )

        logger.debug("set park position")

    async def telescope_unpark(self):
        # This is unique to TheSky. It requires you to unpark the telescope before issuing any
        # other motion command.
        await self.require_connected()
        logger.debug("unparking telescope")

        await self.execute(
            """
            var Out;
            Out = sky6RASCOMTele.IsParked();
            if (Out) {
                sky6RASCOMTele.Unpark();
            }
            """
        )

        async with asyncio.timeout(self.config.timeout):
            await self.poll("""sky6RASCOMTele.IsParked();""", "false", interval=0.2)

        logger.debug("unparked telescope")

    @sk.command_handler
    async def telescope_home(self, cmd: Home):
        await self._stop_follow()
        await self.fast_loop.stop()
        await self.telescope_unpark()
        logger.debug("homing telescope")

        try:
            async with asyncio.timeout(self.config.timeout):
                while True:
                    try:
                        await self.execute(
                            """
                            sky6RASCOMTele.FindHome();
                            """
                        )
                        break
                    except (MountCommandInProgressError, ScriptBusyError):
                        await asyncio.sleep(0.5)

            await self.poll("""sky6RASCOMTele.LastSlewError;""", "0")
        except CommandNotSupportedError:
            logger.warning("Mount does not support FindHome; skipping")

        # Turn off sidereal tracking
        try:
            await self.execute(
                """
                sky6RASCOMTele.SetTracking(0,1,0,0);
                """
            )

            async with asyncio.timeout(self.config.timeout):
                await self.poll("""sky6RASCOMTele.IsTracking;""", "0")
        except CommandNotSupportedError:
            logger.warning("Mount does not support SetTracking; skipping")

        self.state.has_been_homed = True
        await sk.device().kv_put_model(self.state)

        logger.debug("homed telescope")

    async def _homed_retry(self, action: Callable[[], Coroutine]):
        """Run *action*, homing the mount and retrying once if TheSky says it isn't homed.

        TheSky exposes no scripting property for the homed state, and `has_been_homed` only
        records that this driver homed the mount once: after the mount loses power or its
        home reference, TheSky refuses slews and parks until it is homed again.
        """
        try:
            return await action()
        except MountNotHomedError:
            logger.warning("mount is not homed; homing before retrying")
            self.state.has_been_homed = False
            await self.telescope_home(Home())
            return await action()

    @sk.command_handler
    async def telescope_follow_target(self, cmd: FollowTarget):
        await self._homed_retry(lambda: self._follow_target(cmd))

    async def _follow_target(self, cmd: FollowTarget):
        await self._stop_follow()
        await self.fast_loop.stop()
        await self.telescope_unpark()

        # Must first abort any in-progress Raven3.trackLEOCommand.
        aborted_satellite_track = await self._abort_satellite_track()

        target = await cmd.target.adapt(
            ICRSTarget,
            AltAzTarget,
            (FrameTarget, ReferenceFrame.ICRF),
            (FrameTarget, ReferenceFrame.ALTAZ),
            *([TLETarget] if self._has_raven3 else []),
            (RateTarget, ReferenceFrame.ICRF),
            (EphemerisTarget, ReferenceFrame.ICRF),
            # Satellites Raven3 can't take (or any trajectory) follow an unbounded track.
            (TargetTrack, ReferenceFrame.ICRF),
            observer=self._geodetic,
        )

        if self.config.require_coupled_dome:
            await self._require_dome_coupling()

        match target:
            case ICRSTarget():
                logger.debug("executing RADec follow")

                await self.execute(
                    f"""
                    sky6RASCOMTele.SlewToRaDec(
                        {target.coords.ra / 15:0.6f},
                        {target.coords.dec:0.6f},
                        "object"
                    );
                    """
                )

                tracked = await self._set_tracking(1, 1, 0, 0)
                await self._wait_for_mount(tracking=tracked)
                self.fast_loop.start()

                logger.debug("following RADec target")

            case AltAzTarget():
                logger.debug("executing AltAz follow")

                await self.execute(
                    f"""
                    sky6RASCOMTele.SlewToAzAlt(
                        {target.coords.az:0.6f},
                        {target.coords.alt:0.6f},
                        "object"
                    );
                    """
                )

                await self._wait_for_mount(tracking=False)
                self.fast_loop.start()

                logger.debug("following AltAz target")

            case TLETarget():
                logger.debug("executing TLE follow")

                # Update line0
                # NOTE: TheSky requires this syntax, so we force it here.
                tle = TLE(
                    line0=f"0 {target.tle.line2.split()[1]}",
                    line1=target.tle.line1,
                    line2=target.tle.line2,
                )

                # Update the TLE file
                tle_path = self.write_tle(tle)
                designator = tle.line1.split()[1]  # e.g. "12345U"
                await self.execute(
                    f"""
                    Raven3.trackLEODoCommand(
                        100,
                        '.'
                    );
                    Raven3.trackLEODoCommand(
                        100,
                        '{tle_path}'
                    );
                    """
                )
                await asyncio.sleep(0.5)

                # Find the just-loaded satellite, verify the chart selection
                # actually points at it, and only then call trackLEOBegin.
                # If Find returns no matches, TheSky raises ObjectNotFoundError
                # (250) at the script level. If it matches but the first hit
                # isn't our satellite, trackLEOBegin would error 7503 ("Target
                # not a satellite") -- detect that here and surface a diagnostic
                # message instead.
                result = await self.execute(
                    f"""
                    var Out;
                    sky6StarChart.Find('{designator}');
                    var n = sky6ObjectInformation.Count;
                    var parts = ['count=' + n];
                    var ok = false;
                    for (var i = 0; i < n; i++) {{
                        sky6ObjectInformation.Index = i;
                        sky6ObjectInformation.Property(0);
                        var name = sky6ObjectInformation.ObjInfoPropOut;
                        parts.push(i + ':' + name);
                        if (i === 0 && name.indexOf('#{designator}') >= 0) {{
                            ok = true;
                        }}
                    }}
                    if (ok) {{
                        Raven3.trackLEOBegin();
                        Out = 'OK|' + parts.join('|');
                    }} else {{
                        Out = 'BAD|' + parts.join('|');
                    }}
                    """
                )

                if result.startswith("BAD|"):
                    logger.warning(
                        f"TLE follow chart-selection mismatch for {designator};"
                        f" skipped trackLEOBegin. Find diagnostic: {result}"
                    )
                    raise TheSkyError(
                        message=(
                            f"chart selection isn't the loaded satellite for"
                            f" {designator}: {result}"
                        ),
                        code=-1,
                    )

                logger.debug(f"TLE follow Find diagnostic for {designator}: {result}")

                async with asyncio.timeout(self.config.timeout):
                    await self.poll("""Raven3.trackLEOStatus;""", str(LEO_TRACKING), interval=0.2)
                self.fast_loop.start()

                logger.debug("tracking TLE follow")

            case RateTarget():
                logger.debug("executing Rate follow")

                await self.execute(
                    f"""
                    sky6RASCOMTele.SlewToRaDec(
                        {target.initial_coords.ra / 15:0.6f},
                        {target.initial_coords.dec:0.6f},
                        "object"
                    );
                    """
                )

                # Apply custom offset rates (degrees/sec -> arcsec/sec)
                ra_rate_arcsec = target.rates.ra * 3600
                dec_rate_arcsec = target.rates.dec * 3600
                tracked = await self._set_tracking(1, 0, ra_rate_arcsec, dec_rate_arcsec)
                await self._wait_for_mount(tracking=tracked)
                self.fast_loop.start()

                logger.debug("following rate target")

            case EphemerisTarget():
                logger.debug("executing Ephemeris follow")
                await self._start_follow(FollowPath.from_ephemeris(target))
                logger.debug("following Ephemeris target")

            case TargetTrack():
                logger.debug("executing track follow")
                await self._start_follow(await FollowPath.from_track(target))
                logger.debug("following track")

            case FrameTarget():
                match target.frame:
                    case ReferenceFrame.ALTAZ:
                        await self.fast_loop.stop()
                        logger.debug("disabling tracking")
                        await self._set_tracking(0, 1, 0, 0)
                        await self._wait_for_mount(tracking=False, await_onset=False)
                        logger.debug("disabled tracking")
                    case ReferenceFrame.ICRF:
                        logger.debug("enabling sidereal tracking")
                        # If we just aborted a satellite track, the abort latch
                        # leaves IsSlewComplete stuck; clear it with a real slew
                        # before SetTracking, or _wait_for_mount would poll forever.
                        if aborted_satellite_track:
                            await self._slew_to_current_radec()
                        tracked = await self._set_tracking(1, 1, 0, 0)
                        await self._wait_for_mount(tracking=tracked, await_onset=False)
                        self.fast_loop.start()
                        logger.debug("enabled sidereal tracking")

                    case _:
                        raise RuntimeError(f"Need specific target to track {target.frame}")

            case _:
                track_type = type(target).__name__
                raise NotImplementedError(f"{track_type} tracking via TheSky is not supported")

        async with asyncio.timeout(self.config.timeout):
            await self._wait_for_coupled_dome()

        try:
            await self._publish_telescope_status()
        except Exception as e:
            logger.warning(f"Immediate telescope status publish failed: {e}")

    async def _dome_follow_status(self, *, check_motion: bool = True) -> str:
        script = """
            var Out;
            if (sky6Dome.IsConnected == 0) {
                Out = "disconnected";
            } else if (sky6Dome.IsCoupled == 0) {
                Out = "uncoupled";
            } else {
                Out = "coupled";
            }
        """
        if check_motion:
            script += """
                if (Out == "coupled") {
                    Out = sky6Dome.IsGotoComplete == 0 ? "moving" : "complete";
                }
            """
        return (await self.execute(script)).strip()

    async def _require_dome_coupling(self):
        status = await self._dome_follow_status(check_motion=False)
        if status != "coupled":
            raise RuntimeError(f"Required slew-coupled dome is {status}")

    async def _wait_for_coupled_dome(self):
        """Wait until the slew-coupled dome finishes its current goto."""
        while True:
            status = await self._dome_follow_status()

            match status:
                case "disconnected" | "uncoupled":
                    if self.config.require_coupled_dome:
                        raise RuntimeError(f"Required slew-coupled dome is {status}")
                    return
                case "complete":
                    return
                case "moving":
                    await asyncio.sleep(0.1)
                case _:
                    raise RuntimeError(f"Unexpected dome status {status!r}")

    async def _start_follow(self, path: FollowPath):
        """Follow a moving target along *path*.

        TheSky has no path-follow primitive, only a slew and a constant offset rate. So
        slew to where the target will be when the slew ends, track at its rate from
        there, and keep re-rating the mount along the path in the background.
        """
        follow = _Follow(path)
        tracked = await self._follow_slew(follow)

        if tracked:
            self._follow = follow
            self._follow_loop = AsyncLoop(
                self._follow_tick, interval=self.config.follow_update_period, log=True
            )
            self._follow_loop.start()

        self.fast_loop.start()

    async def _stop_follow(self):
        """Stop re-rating the mount, if a path follow is running."""
        if self._follow_loop is not None:
            await self._follow_loop.stop()
            self._follow_loop = None
        self._follow = None

    async def _follow_slew(self, follow: _Follow) -> bool:
        """Slew onto the path ahead of the target and track at its rate on arrival.

        Returns whether the offset rates were applied (see _set_tracking).
        """
        config = self.config
        follow.slewed_at = time.monotonic()
        follow.mount = None

        ra_hr, dec = await self._get_radec()
        await follow.path.ensure(Time.now().jd)
        arrival_jd, ra, dec = lead_slew(
            follow.path,
            ra_hr * 15,
            dec,
            overhead=config.follow_slew_overhead,
            rate=config.follow_slew_rate,
        )
        follow.rate = follow.path.rate(arrival_jd, config.follow_update_period)
        lead = (arrival_jd - Time.now().jd) * SECONDS_PER_DAY

        await self.execute(
            f"""
            sky6RASCOMTele.SlewToRaDec(
                {ra / 15:0.6f},
                {dec:0.6f},
                "object"
            );
            """
        )

        # Offset rates (degrees/sec -> arcsec/sec), matching RateTarget
        tracked = await self._set_tracking(1, 0, follow.rate[0] * 3600, follow.rate[1] * 3600)
        await self._wait_for_mount(tracking=tracked)

        # Compare to tune follow_slew_overhead/follow_slew_rate: the slew lands on the
        # target only as well as the estimate matches.
        took = time.monotonic() - follow.slewed_at
        logger.debug(f"follow slew led the target by {lead:.1f} s; the slew took {took:.1f} s")

        return tracked

    async def _follow_tick(self):
        """Re-rate the mount along the followed path.

        Each tick is one script: it reads the mount's position and sets the new rate,
        the path's mean rate over the next update period plus a correction that
        removes `follow_correction_gain` of the pointing error by then. The error
        comes from the previous tick's readback, carried forward at the rate commanded
        since. Past `follow_reslew_threshold`, slew back onto the path instead.
        """
        follow = self._follow
        if follow is None:
            return

        config = self.config
        period = config.follow_update_period
        now = Time.now().jd

        await follow.path.ensure(now + period / SECONDS_PER_DAY)
        ra, dec = follow.path.position(now)
        ra_rate, dec_rate = follow.path.rate(now, period)

        if follow.mount is not None:
            mount_ra, mount_dec, mount_jd = follow.mount
            elapsed = (now - mount_jd) * SECONDS_PER_DAY
            mount_ra += follow.rate[0] * elapsed
            mount_dec += follow.rate[1] * elapsed

            error = angular_offset(ra, dec, mount_ra, mount_dec)
            if error > config.follow_reslew_threshold:
                if time.monotonic() - follow.slewed_at >= config.follow_reslew_interval:
                    logger.info(f"follow pointing error {error:.3f} deg; slewing back onto the path")
                    await self._follow_slew(follow)
                    return
                # Too soon to slew again; correct at most a threshold's worth meanwhile.
                scale = config.follow_reslew_threshold / error
            else:
                scale = 1.0

            correction = config.follow_correction_gain * scale / period
            ra_rate += wrap_degrees(ra - mount_ra) * correction
            dec_rate += (dec - mount_dec) * correction

        sent = Time.now().jd
        resp = await self.execute(
            f"""
            var Out;
            if (sky6RASCOMTele.IsConnected) {{
                sky6RASCOMTele.GetRaDec();
                Out = [sky6RASCOMTele.dRa, sky6RASCOMTele.dDec];
                sky6RASCOMTele.SetTracking(1, 0, {ra_rate * 3600:0.6f}, {dec_rate * 3600:0.6f});
            }} else {{
                Out = "disconnected";
            }}
            """
        )
        received = Time.now().jd

        if resp.strip() == "disconnected":
            follow.mount = None
            return

        mount_ra_hr, mount_dec = (float(x) for x in resp.split(","))
        follow.mount = (mount_ra_hr * 15, mount_dec, (sent + received) / 2)
        follow.rate = (ra_rate, dec_rate)

    def write_tle(self, target: TLE) -> str:
        """Write TLE to a temp file and return the path for TheSky use."""

        lines = []
        if target.line0 is not None:
            lines.append(target.line0)
        lines.extend([target.line1, target.line2])

        # Use satellite designator (e.g. "NNNNNU") for a unique filename
        designator = target.line1.split()[1]
        filename = f"tle_{designator}.txt"

        host_data_path = os.environ.get("SENSORKIT_DATA_PATH")

        if host_data_path:
            write_path = Path("/data") / filename
            thesky_path = f"{host_data_path.replace('\\', '/')}/{filename}"
        else:
            write_path = Path(tempfile.gettempdir()) / filename
            thesky_path = write_path.as_posix()

        write_path.write_text("\n".join(lines) + "\n")
        return thesky_path

    async def _wait_for_mount(
        self,
        *,
        slewing: bool = False,
        tracking: bool = False,
        await_onset: bool = True,
    ):
        """Poll until IsSlewComplete/IsTracking match the target flags.

        When `await_onset` (the default, for slewing commands)
        first wait briefly for the slew to *start*; if none is seen within
        _MOUNT_ONSET_TIMEOUT the command was a positional no-op and we fall
        through to the settle check. Non-slewing commands (enable/disable
        tracking) pass `await_onset=False`. The settle wait is bounded by
        config.timeout; both phases poll every 0.1 s.
        """
        flags = """var Out; Out=[sky6RASCOMTele.IsSlewComplete, sky6RASCOMTele.IsTracking];"""

        if await_onset:
            try:
                async with asyncio.timeout(_MOUNT_ONSET_TIMEOUT):
                    while True:
                        slew_complete, _ = (await self.execute(flags)).split(",")
                        if slew_complete.strip() != "1":
                            break
                        await asyncio.sleep(0.1)
            except TimeoutError:
                pass  # no slew observed -> positional no-op; fall through to settle

        async with asyncio.timeout(self.config.timeout):
            while True:
                slew_complete, is_tracking = (await self.execute(flags)).split(",")
                is_slewing = slew_complete.strip() != "1"
                if is_slewing == slewing and (is_tracking.strip() == "1") == tracking:
                    break
                await asyncio.sleep(0.1)

    async def _set_tracking(
        self, on: int, ignore_rates: int, ra_rate: float = 0.0, dec_rate: float = 0.0
    ) -> bool:
        """Set tracking mode/rates. Returns True if applied, False only if the
        mount doesn't implement SetTracking (error 228) -- in which case the
        tracking degrades to a bare slew instead of crashing.
        Any other error propagates.
        """
        try:
            await self.execute(
                f"""sky6RASCOMTele.SetTracking({on}, {ignore_rates}, {ra_rate}, {dec_rate});"""
            )
        except CommandNotSupportedError:
            logger.warning("Mount does not support SetTracking; tracking no-op")
            return False
        return True

    async def _abort_satellite_track(self) -> bool:
        """Abort a Raven3 satellite track and wait for it to become idle.

        Returns True if a track was active and aborted, False if none was
        running (or the mount doesn't implement Raven3).
        """
        if not self._has_raven3:
            return False

        try:
            status = (await self.execute("""Raven3.trackLEOStatus;""")).strip()
        except CommandNotSupportedError:
            return False

        if status == "0":
            return False

        logger.debug("aborting satellite track")
        await self.execute("""Raven3.trackLEOAbort();""")

        async with asyncio.timeout(self.config.timeout):
            while True:
                try:
                    if (await self.execute("""Raven3.trackLEOStatus;""")).strip() == "0":
                        break
                except ProcessAbortedError:
                    pass  # transient 212 after abort; keep polling
                await asyncio.sleep(0.2)

        logger.debug("aborted satellite track")
        return True

    async def _slew_to_current_radec(self):
        """Slew to the mount's current RA/Dec.

        After trackLEOAbort(), IsSlewComplete stays latched on error 212 until a
        real slew resets it -- but the mount is NOT command-ready the instant
        trackLEOStatus reads 0. A slew fired too soon is rejected with error 121
        ("A Mount command is already in progress"), which leaves the latch stuck
        forever and _wait_for_mount hanging. Retry the slew until it is *accepted*
        (bounded by config.timeout), then wait for it to settle.

        With the dome slaved, the settle wait can instead latch on error 123 ("A
        Dome command is already in progress"): IsSlewComplete/IsTracking keep
        returning it even after the dome stops, and TheSky stops slaving the dome
        meanwhile. Sending the same slew again clears it (#55), so do that.
        """
        ra, dec = await self._get_radec()

        async with asyncio.timeout(self.config.timeout):
            while True:
                try:
                    await self.execute(
                        f"""
                        sky6RASCOMTele.SlewToRaDec({ra:0.6f}, {dec:0.6f}, "object");
                        """
                    )
                    await self._wait_for_mount(tracking=True)
                    break
                except MountCommandInProgressError:
                    await asyncio.sleep(0.1)
                except DomeCommandInProgressError:
                    # 123 latch (#55): the next pass re-sends the slew to clear it
                    logger.debug("settle latched on TheSky error 123; re-sending slew")
                    await asyncio.sleep(0.1)

    async def _get_radec(self) -> tuple[float, float]:
        """Return the mount's (RA hours, Dec degrees)."""
        resp = await self.execute(
            """
            var Out;
            sky6RASCOMTele.GetRaDec();
            Out = [sky6RASCOMTele.dRa, sky6RASCOMTele.dDec];
            """
        )
        ra, dec = (float(x) for x in resp.split(","))
        return ra, dec

    async def _publish_telescope_status(self):
        try:
            resp = await self.execute(
                """
                var LeoStatus = -1;
                try { LeoStatus = Raven3.trackLEOStatus; } catch (e) {}
                var SlewComplete = -1;
                try { SlewComplete = sky6RASCOMTele.IsSlewComplete; } catch (e) {}
                var Out;
                // GetRaDec throws on a disconnected mount, which would park TheSky in its debugger.
                if (sky6RASCOMTele.IsConnected) {
                    sky6RASCOMTele.GetRaDec();
                    sky6RASCOMTele.GetAzAlt();
                    Out = [
                        sky6RASCOMTele.IsConnected,
                        sky6RASCOMTele.dRa,
                        sky6RASCOMTele.dRaTrackingRate,
                        sky6RASCOMTele.dDec,
                        sky6RASCOMTele.dDecTrackingRate,
                        sky6RASCOMTele.dAlt,
                        sky6RASCOMTele.dAz,
                        SlewComplete,
                        sky6RASCOMTele.IsTracking,
                        LeoStatus
                    ];
                } else {
                    Out = [0, 0, 0, 0, 0, 0, 0, -1, 0, -1];
                }
                """
            )
        except ProcessAbortedError:
            # TheSky returns 212 transiently after an `abort`; telescope is fine
            return

        (
            connected,
            ra,
            ra_rate,
            dec,
            dec_rate,
            alt,
            az,
            slew_complete,
            tracking,
            leo_status
        ) = [float(x) for x in resp.split(",")]

        connected = bool(connected)
        self.device_connected = connected

        if not connected:
            await sk.device().publish(Connected(is_connected=False))
            return

        leo_status = int(leo_status)
        if leo_status in (LEO_SLEWING, LEO_SETTLING):
            is_slewing, is_tracking = True, False
        elif leo_status == LEO_TRACKING:
            is_slewing, is_tracking = False, True
        else:
            if slew_complete < 0:
                is_slewing = False
            else:
                is_slewing = not bool(slew_complete)
            is_tracking = bool(tracking)

        device = sk.device()
        await device.publish(Connected(is_connected=connected))
        await device.publish(Slewing(is_slewing=is_slewing))
        await device.publish(Tracking(is_tracking=is_tracking))
        await device.publish(RADecPointing(right_ascension_hours=ra, declination_degrees=dec))
        await device.publish(AltAzPointing(altitude_degrees=alt, azimuth_degrees=az))

        # Convert RA/Dec rates from arcsec/sec to deg/sec
        ra_rate /= 3600
        dec_rate /= 3600

        icrf_ra = ra_rate if is_tracking else 0.0
        icrf_dec = dec_rate if is_tracking else 0.0

        if self._location is not None:
            if is_tracking:
                alt_rate, az_rate = radec_rates_to_altaz_rates(
                    ra_hr=ra,
                    dec_deg=dec,
                    ra_rate_deg_per_sec=icrf_ra,
                    dec_rate_deg_per_sec=icrf_dec,
                    location=self._location,
                    time=Time.now(),
                )
            else:
                alt_rate = az_rate = 0.0

            await device.publish(
                AxisRates(
                    azimuth=AxisRate(velocity=az_rate, axis=MountAxis.AZIMUTH),
                    altitude=AxisRate(velocity=alt_rate, axis=MountAxis.ALTITUDE),
                    right_ascension=AxisRate(
                        velocity=icrf_ra,
                        axis=MountAxis.RIGHT_ASCENSION,
                    ),
                    declination=AxisRate(
                        velocity=icrf_dec,
                        axis=MountAxis.DECLINATION,
                    ),
                )
            )

    async def status_publish(self):
        # The fast loop publishes the full status while it runs, so defer to it and
        # report connectivity only.
        if self.fast_loop.active:
            await sk.device().publish(Connected(is_connected=self.device_connected))
        else:
            await self._publish_telescope_status()


class TheSkyTelescopeConfig(TheSkyDeviceConfig[TheSkyTelescope]):
    """TheSky Telescope configuration."""

    device_type: Literal["telescope"] = "telescope"
    status_frequency_slow: float = 1.0
    status_frequency_fast: float = 0.1
    timeout: float = 300.0
    require_coupled_dome: bool = False

    # Following a moving target (an ephemeris, or a satellite without Raven3): the slew
    # leads the target by its estimated duration, overhead + the larger axis move at
    # follow_slew_rate, then the rate is updated every follow_update_period seconds.
    follow_slew_overhead: float = 3.0
    follow_slew_rate: float = 4.0
    # One script per update (a readback and a SetTracking), on top of the status polls.
    follow_update_period: float = 2.0
    # Fraction of the pointing error each update removes by the next one.
    follow_correction_gain: float = 0.5
    # Pointing error (degrees) past which to slew back onto the path, at most this often (s).
    follow_reslew_threshold: float = 0.2
    follow_reslew_interval: float = 30.0

    @override
    def create_device(self):
        return TheSkyTelescope(self)


class TheSkyTelescopeState(TheSkyDeviceState):
    """TheSky Telescope state."""

    device_type: Literal["telescope"] = "telescope"
    has_been_homed: bool = False
