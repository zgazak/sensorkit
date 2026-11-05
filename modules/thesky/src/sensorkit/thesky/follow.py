# SPDX-License-Identifier: Apache-2.0
"""Path following for moving targets on mounts with no path-follow primitive.

TheSky can only slew and track at a constant offset rate, so a moving target (an
ephemeris, or a satellite track without the Raven3 plugin) is followed by slewing
ahead of it and then re-rating the mount along its path. This module holds the
geometry; the telescope drives the mount.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from astropy.time import Time
from loguru import logger

from sensorkit.astro.target import EphemerisTarget, TargetTrack

SECONDS_PER_DAY = 86400.0

# A track is sampled in windows this long, at this step, and re-sampled once less than
# the margin remains, so a follow can run for as long as the target is up.
_TRACK_WINDOW = timedelta(minutes=10)
_TRACK_STEP = timedelta(seconds=2)
_TRACK_MARGIN = timedelta(minutes=2)

# Solving for the slew lead: each pass shrinks the error by the ratio of target to slew
# speed (~1/4 for a fast LEO), so a few passes reach the tolerance (s).
_LEAD_MAX_ITERATIONS = 10
_LEAD_TOLERANCE = 0.01


def wrap_degrees(angle: float) -> float:
    """Wrap an angle difference into [-180, 180) degrees."""
    return (angle + 180.0) % 360.0 - 180.0


def angular_offset(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Return the small-angle on-sky separation in degrees between two RA/Dec points."""
    dra = wrap_degrees(ra1 - ra2) * math.cos(math.radians((dec1 + dec2) / 2))
    return math.hypot(dra, dec1 - dec2)


def slew_time(
    ra_from: float, dec_from: float, ra_to: float, dec_to: float, *, overhead: float, rate: float
) -> float:
    """Estimate a slew's duration in seconds between two RA/Dec points (degrees).

    The two axes of an equatorial mount move at once, so the slower axis sets the time:
    a fixed overhead (acceleration, settling) plus the larger axis move at *rate* deg/s.
    """
    axis_move = max(abs(wrap_degrees(ra_to - ra_from)), abs(dec_to - dec_from))
    return overhead + axis_move / rate


class FollowPath:
    """The RA/Dec path of a moving target, evaluated at any time.

    Samples are interpolated linearly. An ephemeris is fixed and extrapolates past its
    ends from the nearest two samples; a track re-samples itself as time advances.
    """

    def __init__(self, jds: Sequence[float], ras: Sequence[float], decs: Sequence[float]):
        if len(jds) < 2:
            raise ValueError("a path needs at least two samples")

        self._set(jds, ras, decs)
        self._track: TargetTrack | None = None
        self._warned_end = False

    @classmethod
    def from_ephemeris(cls, target: EphemerisTarget) -> FollowPath:
        return cls(target.jds, [p.ra for p in target.points], [p.dec for p in target.points])

    @classmethod
    async def from_track(cls, track: TargetTrack) -> FollowPath:
        path = cls(*await cls._sample_track(track))
        path._track = track
        return path

    @staticmethod
    async def _sample_track(track: TargetTrack):
        # Start one step back so the first sample sits at now.
        jds, points = await track.sample(datetime.now(UTC) - _TRACK_STEP, _TRACK_WINDOW, _TRACK_STEP)
        return jds, [p.ra for p in points], [p.dec for p in points]

    def _set(self, jds: Sequence[float], ras: Sequence[float], decs: Sequence[float]):
        # Times as seconds from the first sample, which keeps finite differences precise
        # where raw JDs (~2.46e6) would lose most of their digits.
        self._jd0 = jds[0]
        self._t = [(jd - self._jd0) * SECONDS_PER_DAY for jd in jds]
        # Unwrap RA so interpolation never runs the long way round through 0/360.
        self._ra = [ras[0]]
        for ra in ras[1:]:
            self._ra.append(self._ra[-1] + wrap_degrees(ra - self._ra[-1]))
        self._dec = list(decs)

    @property
    def end_jd(self) -> float:
        return self._jd0 + self._t[-1] / SECONDS_PER_DAY

    async def ensure(self, jd: float):
        """Make sure a track is sampled through *jd*; an ephemeris is left as is."""
        if self._track is None:
            if jd > self.end_jd and not self._warned_end:
                logger.warning("following past the end of the ephemeris; extrapolating")
                self._warned_end = True
            return

        if jd + _TRACK_MARGIN.total_seconds() / SECONDS_PER_DAY > self.end_jd:
            self._set(*await self._sample_track(self._track))

    def position(self, jd: float) -> tuple[float, float]:
        """Return (ra, dec) in degrees at *jd*, RA in [0, 360)."""
        t = (jd - self._jd0) * SECONDS_PER_DAY
        i = min(max(bisect.bisect_right(self._t, t) - 1, 0), len(self._t) - 2)
        f = (t - self._t[i]) / (self._t[i + 1] - self._t[i])
        ra = self._ra[i] + f * (self._ra[i + 1] - self._ra[i])
        dec = self._dec[i] + f * (self._dec[i + 1] - self._dec[i])
        return ra % 360.0, dec

    def rate(self, jd: float, span: float) -> tuple[float, float]:
        """Return the mean (ra, dec) rate in deg/s over the *span* seconds from *jd*."""
        ra0, dec0 = self.position(jd)
        ra1, dec1 = self.position(jd + span / SECONDS_PER_DAY)
        return wrap_degrees(ra1 - ra0) / span, (dec1 - dec0) / span


def lead_slew(
    path: FollowPath,
    ra_from: float,
    dec_from: float,
    *,
    now_jd: float | None = None,
    overhead: float,
    rate: float,
) -> tuple[float, float, float]:
    """Return (arrival_jd, ra, dec): where the target will be when a slew there ends.

    Solved by fixed-point iteration on the slew time, since the destination depends on
    the duration and the duration on the destination.
    """
    now_jd = Time.now().jd if now_jd is None else now_jd
    duration = 0.0

    for _ in range(_LEAD_MAX_ITERATIONS):
        ra, dec = path.position(now_jd + duration / SECONDS_PER_DAY)
        previous, duration = duration, slew_time(
            ra_from, dec_from, ra, dec, overhead=overhead, rate=rate
        )
        if abs(duration - previous) < _LEAD_TOLERANCE:
            break

    arrival_jd = now_jd + duration / SECONDS_PER_DAY
    ra, dec = path.position(arrival_jd)
    return arrival_jd, ra, dec
