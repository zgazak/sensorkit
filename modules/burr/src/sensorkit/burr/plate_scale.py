# SPDX-License-Identifier: Apache-2.0
"""Live plate scale from SENPAI's per-frame WCS solves.

Burr plans streak rates from `hardware.telescope.ifov_no_binning_arcsec`. Rather than relying
only on a configured value, the program can follow the pixel scale SENPAI measures on each
solved frame: `PlateScaleEstimator` gates and accumulates those solves, and
`consume_senpai_results` feeds it from the SensorKit stream.
"""

import asyncio
import statistics
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import datetime

from burr.core.config import AppConfig
from burr.models.hardware import TelescopeConfig
from loguru import logger

import sensorkit.api as sk
from sensorkit.backend.base import SpecialProperty, Subject
from sensorkit.burr.models import MeasuredPlateScaleConfig
from sensorkit.common.keyword import get_keyword_info, validate_keyword_json
from sensorkit.senpai.models import SenpaiResult

_SEEN_FILES_LIMIT = 4096
_RECONNECT_DELAY_S = 10.0


class PlateScaleEstimator:
    """Running median of accepted unbinned plate scales, with change gating.

    `fallback` is the configured telescope: the reference for the sanity bounds, and the
    field of view that is rescaled along with the pixel scale. `applied` is the pixel scale
    currently in use (defaults to the fallback's).
    """

    def __init__(
        self,
        config: MeasuredPlateScaleConfig,
        fallback: TelescopeConfig,
        controller: str,
        applied: float | None = None,
    ):
        self.config = config
        self.fallback = fallback
        self.controller = controller
        self.applied = applied if applied is not None else fallback.ifov_no_binning_arcsec
        self._samples: deque[float] = deque(maxlen=config.window)
        # Results can be redelivered (e.g. on a stream reconnect); count each frame once.
        self._seen: dict[str, None] = {}

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    @property
    def median(self) -> float | None:
        return statistics.median(self._samples) if self._samples else None

    def unbinned_scale(self, result: SenpaiResult) -> float | None:
        """The result's unbinned pixel scale, or None when it should not be counted."""
        if result.controller_name != self.controller:
            return None

        scale = result.pixel_scale_arcsec
        if not result.solved or scale is None or scale <= 0:
            return None

        # Missing binning means a frame written without XBINNING; treat it as unbinned.
        scale /= result.x_binning or 1

        ratio = scale / self.fallback.ifov_no_binning_arcsec
        if not self.config.min_ratio <= ratio <= self.config.max_ratio:
            logger.warning(
                f'plate scale {scale:.3f}"/px from {result.file_path} is {ratio:.2f}x the '
                f'configured {self.fallback.ifov_no_binning_arcsec:.3f}"/px (outside '
                f"{self.config.min_ratio}x-{self.config.max_ratio}x); ignoring it"
            )
            return None

        return scale

    def add(self, result: SenpaiResult) -> TelescopeConfig | None:
        """Count one SENPAI result; return the telescope to apply when the scale has moved."""
        if result.file_path in self._seen:
            return None

        scale = self.unbinned_scale(result)
        if scale is None:
            return None

        self._seen[result.file_path] = None
        if len(self._seen) > _SEEN_FILES_LIMIT:
            del self._seen[next(iter(self._seen))]

        self._samples.append(scale)
        if len(self._samples) < self.config.min_solves:
            return None

        median = statistics.median(self._samples)
        if abs(median / self.applied - 1) <= self.config.tolerance:
            return None

        self.applied = median
        return self.telescope()

    def telescope(self) -> TelescopeConfig:
        """The fallback telescope with the applied pixel scale (field of view scaled to match)."""
        ratio = self.applied / self.fallback.ifov_no_binning_arcsec
        return TelescopeConfig(
            ifov_no_binning_arcsec=self.applied,
            field_of_view_degree=self.fallback.field_of_view_degree * ratio,
        )


def with_telescope(config: AppConfig, telescope: TelescopeConfig) -> AppConfig:
    """A copy of the (frozen) burr config with `hardware.telescope` replaced."""
    hardware = config.hardware.model_copy(update={"telescope": telescope})
    return config.model_copy(update={"hardware": hardware})


async def consume_senpai_results(
    kit: sk.SensorKit,
    handle: Callable[[SenpaiResult], Awaitable[None]],
    *,
    start_at: datetime | None = None,
    ready: asyncio.Event | None = None,
) -> None:
    """Hand every SenpaiResult published anywhere in SensorKit to `handle`, until cancelled.

    `start_at` replays results published since then (e.g. the start of the night), so a
    restarted program recovers the solves it already missed. Stream errors reconnect.
    """
    info = get_keyword_info(SenpaiResult)
    keyword_key = info.key if info else SenpaiResult.__name__
    subject = Subject(path=(), prop=SpecialProperty.ALL_DESCENDANTS)

    while True:
        try:
            consumer = await kit.backend.impl.stream_consume(subject, start_at=start_at)
            logger.debug("burr consuming SenpaiResults for the plate scale")
            if ready is not None:
                ready.set()

            async for msg in consumer:
                if msg.subject.prop != keyword_key:
                    continue

                try:
                    result = validate_keyword_json(keyword_key, msg.data)
                except Exception as e:
                    logger.debug(f"ignoring unparseable SenpaiResult: {e}")
                    continue

                if not isinstance(result, SenpaiResult):
                    continue

                try:
                    await handle(result)
                except Exception:
                    logger.exception(f"error handling SenpaiResult for {result.file_path}")
        except Exception as e:
            # CancelledError is a BaseException, so cancellation still stops the loop.
            logger.warning(
                f"SenpaiResult stream error: {e}; reconnecting in {_RECONNECT_DELAY_S:.0f}s"
            )
            await asyncio.sleep(_RECONNECT_DELAY_S)
        finally:
            if ready is not None:
                ready.clear()
