# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import array
import asyncio
import uuid
from datetime import UTC, datetime
from typing import Literal, override

import numpy as np
from astropy.io import fits
from astropy.time import Time
from loguru import logger

import sensorkit.api as sk
from sensorkit.common.aio import AsyncLoop
from sensorkit.data.filesys import FileNameTemplate
from sensorkit.data.fits import ArrayInfo, BayerPattern, ImageInfo
from sensorkit.std import (
    Binning,
    CameraCapture,
    CameraSensorSize,
    CameraSensorTemperature,
    ConfigureCameraCooler,
    ConfigureCameraSensor,
    Connect,
    Connected,
    Disconnect,
    ExposureInfo,
    FrameType,
    StandardCamera,
    Stop,
    TemperatureUnit,
)
from sensorkit.thesky.device import (
    ProcessAbortedError,
    TheSkyDevice,
    TheSkyDeviceConfig,
    TheSkyDeviceState,
)


@sk.declare_device(type=StandardCamera)
class TheSkyCamera(TheSkyDevice):
    """TheSky Camera implementation."""

    config: TheSkyCameraConfig
    device_name = "Camera"

    # Set while a capture runs. Status polls skip it: querying the camera mid-capture has hung
    # TheSky's script server (observed with a Canon DSLR), stalling every other device.
    _capturing: bool = False

    @sk.on_attach
    async def entity_init(self):
        device = sk.device()

        # Restore state
        try:
            self.state = await device.kv_get_model(TheSkyCameraState)
            logger.debug(f"restored state for {device.entity}")
        except Exception:
            logger.warning(f"No saved state for {device.entity}")
            self.state = TheSkyCameraState()

        # Initialize the camera
        await self._initialize()
        self.status_loop = AsyncLoop(
            self.status_publish, interval=self.config.status_frequency, log=True
        ).start()

    @sk.on_detach
    async def entity_deinit(self):
        await asyncio.sleep(self.config.status_frequency)
        await self.status_loop.stop()
        await self.camera_disconnect(Disconnect())
        await sk.device().kv_put_model(self.state)

    async def _initialize(self):
        # Connect to the hardware
        self._reconnect = lambda: self.camera_connect(Connect())
        await self.camera_connect(Connect())

        # Set the camera temperature, unless the camera has no cooler (e.g. a DSLR, whose
        # TheSky plugin raises on TemperatureSetPoint)
        if self.config.temperature is None:
            return
        await self.camera_set_temperature(
            ConfigureCameraCooler(
                enable=True,
                setpoint=CameraSensorTemperature(
                    temperature=self.config.temperature, units=TemperatureUnit.CELSIUS
                ),
            )
        )

    @sk.command_handler
    async def camera_connect(self, cmd: Connect):
        logger.debug("connecting to camera")

        await self.execute(
            """
            ccdsoftCamera.Asynchronous = 1;
            ccdsoftCamera.ShutDownTemperatureRegulationOnDisconnect = 1;
            ccdsoftCamera.Connect();
            """
        )

        async with asyncio.timeout(self.config.timeout):
            await self.poll("""ccdsoftCamera.Status;""", "Ready")

        self.device_connected = True
        await sk.device().publish(Connected(is_connected=True))

        logger.debug("connected to camera")

    @sk.command_handler
    async def camera_disconnect(self, cmd: Disconnect):
        logger.debug("disconnecting from camera")

        await self.execute(
            """
            ccdsoftCamera.Disconnect();
            """
        )

        async with asyncio.timeout(self.config.timeout):
            await self.poll("""ccdsoftCamera.Status;""", "Not Connected")

        self.device_connected = False
        await sk.device().publish(Connected(is_connected=False))

        logger.debug("disconnected from camera")

    @sk.command_handler
    async def camera_stop(self, cmd: Stop):
        await self.camera_abort(sk.Abort())

    @sk.command_handler
    async def camera_abort(self, cmd: sk.Abort):
        await self.require_connected()
        logger.debug("aborting camera capture")

        await self.execute(
            """
            ccdsoftCamera.Abort();
            """
        )

        async with asyncio.timeout(self.config.timeout):
            await self.poll("""ccdsoftCamera.Status;""", "Ready")

        logger.debug("aborted camera capture")

    @sk.command_handler
    async def camera_set_temperature(self, cmd: ConfigureCameraCooler):
        await self.require_connected()
        logger.debug(f"setting camera temperature to {cmd.setpoint.temperature}")

        target = cmd.setpoint.temperature
        await self.execute(
            f"""
            ccdsoftCamera.TemperatureSetPoint = {target};
            ccdsoftCamera.RegulateTemperature = {1 if cmd.enable else 0};
            """
        )

        resp = await self.execute(
            """
            ccdsoftCamera.TemperatureSetPoint;
            """
        )
        temperature_set_point = float(resp)
        if temperature_set_point != target:
            logger.warning(f"Requested camera temperature of {target} C, got {temperature_set_point} C")
        else:
            logger.debug(f"set camera temperature to {target} C")

    async def camera_set_readout_mode(self, cmd: ConfigureCameraSensor):
        if cmd.readout_mode is None:
            return
        logger.warning("TheSky camera does not support setting readout mode; ignoring")

    async def camera_set_gain(self, cmd: ConfigureCameraSensor):
        if cmd.gain is None:
            return
        logger.warning("TheSky camera does not support setting gain; ignoring")

    async def camera_set_binning(self, cmd: ConfigureCameraSensor):
        if cmd.binning is None:
            return
        await self.require_connected()
        bin_x, bin_y = int(cmd.binning.x), int(cmd.binning.y)
        logger.debug(f"setting camera binning to ({bin_x}, {bin_y})")
        await self.execute(
            f"""
            ccdsoftCamera.BinX = {bin_x};
            ccdsoftCamera.BinY = {bin_y};
            """
        )

        resp = await self.execute(
            """
            var Out;
            Out = [
                ccdsoftCamera.BinX,
                ccdsoftCamera.BinY
            ];
            """
        )
        actual_x, actual_y = [int(x) for x in resp.split(",")]
        if actual_x != bin_x or actual_y != bin_y:
            raise RuntimeError(
                f"Requested camera binning of ({bin_x}, {bin_y}), "
                f"got ({actual_x}, {actual_y})"
            )
        else:
            logger.debug(f"set camera binning to ({actual_x}, {actual_y})")

    @sk.command_handler
    async def camera_configure_sensor(self, cmd: ConfigureCameraSensor):
        await self.camera_set_readout_mode(cmd)
        await self.camera_set_gain(cmd)
        await self.camera_set_binning(cmd)

    @sk.command_handler
    async def camera_capture(self, cmd: CameraCapture):
        self._capturing = True
        try:
            await self._capture(cmd)
        finally:
            self._capturing = False

    async def _capture(self, cmd: CameraCapture):
        await self.require_connected()
        logger.info(f"Requesting {cmd.integration_time:.1f} sec capture from camera")
        logger.debug("starting camera capture")

        # Start the exposure
        frame_code = {
            FrameType.LIGHT: 1,
            FrameType.BIAS: 2,
            FrameType.DARK: 3,
            FrameType.FLAT: 4,
        }[cmd.frame_type]

        # In file readout, TheSky saves the frame itself; remember the previous file so a
        # stale name is never mistaken for this exposure's.
        previous_file = None
        if self.config.readout == "file":
            previous_file = _image_file_name(await self.execute("""ccdsoftCamera.LastImageFileName;"""))
            await self.execute("""ccdsoftCamera.AutoSaveOn = 1;""")

        await self.execute(
            f"""
            ccdsoftCamera.Frame = {frame_code};
            ccdsoftCamera.ExposureTime = {cmd.integration_time};
            ccdsoftCamera.TakeImage();
            """
        )

        async with asyncio.timeout(cmd.integration_time + self.config.timeout):
            await self.poll("""ccdsoftCamera.IsExposureComplete;""", "1", interval=0.25)

        if self.config.readout == "file":
            data, date_obs, temp, binx, biny = await self._read_saved_frame(previous_file)
        else:
            data, date_obs, temp, binx, biny = await self._read_scanlines()

        if data is None or not len(data) or not len(data[0]):
            logger.error("No data returned from camera!")
            return

        if isinstance(data, np.ndarray):
            array_info = ArrayInfo.from_array(data)
        else:
            array_info = ArrayInfo.from_array(data[0], shape=(len(data), len(data[0])))
        logger.debug(f"got image array with {len(data)} rows, {len(data[0])} cols, dtype {array_info.dtype}")

        # Write it to the DataGraph
        if graph := await sk.device().data_graph():
            source = graph.app_source()

            detector = await self.execute("""SelectedHardware.cameraModel;""")
            cmd.context.set(
                ImageInfo(array=array_info, bayer=self.config.bayer, binning=(int(binx), int(biny))),
                ExposureInfo(
                    date_obs=date_obs,
                    exposure_time=cmd.integration_time,
                    detector=detector,
                    image_type=cmd.frame_type,
                    ccd_temperature=temp,
                ),
            )

            if not cmd.context.get(FileNameTemplate):
                cmd.context.set(FileNameTemplate(template=f"{str(uuid.uuid1())}.fits"))

            writer = source.produce(cmd.context)
            n = 0

            for row in data:
                row_bytes = row.tobytes()
                writer.write(row_bytes)
                n += len(row_bytes)

            logger.debug(f"wrote {n} bytes to DataGraph")
            await writer.drain()

            writer.close()
            await writer.wait_closed()

        else:
            logger.warning("discarding data since no DataGraph is defined!")

        logger.debug("completed camera capture")

    async def _read_scanlines(self):
        """Pull the active image through TheSky's script server as CSV scan lines.

        Rows come back `scanline_chunk_rows` per script. Building a whole large frame as one
        string (~90 MB for 18 MP) inside TheSky has left it unresponsive for tens of seconds.
        """
        height = int(
            await self.execute(
                """
                ccdsoftCameraImage.AttachToActiveImager();
                ccdsoftCameraImage.HeightInPixels;
                """
            )
        )

        data: list[array.array] = []
        for start in range(0, height, self.config.scanline_chunk_rows):
            end = min(start + self.config.scanline_chunk_rows, height)
            resp = await self.execute(
                f"""
                ccdsoftCameraImage.AttachToActiveImager();
                var Out = "";
                for (var y = {start}; y < {end}; y++) {{
                    if (y > {start}) Out += ";";
                    Out += ccdsoftCameraImage.scanLine(y);
                }}
                """
            )

            # Convert it to list[array.array] - this is CPU-bound, so background it
            data += await asyncio.to_thread(self._pixels_from_csv_block, resp, "uint16")

        resp = await self.execute(
            """
            var Out;
            Out = [
                ccdsoftCameraImage.JulianDay,
                ccdsoftCameraImage.Temperature,
                ccdsoftCamera.BinX,
                ccdsoftCamera.BinY
            ];
            """
        )
        jd, temp, binx, biny = resp.split(",")
        date_obs = Time(float(jd), format="jd", scale="utc").to_datetime(timezone=UTC)
        return data, date_obs, _float_or_none(temp), float(binx), float(biny)

    async def _read_saved_frame(self, previous_file: str | None):
        """Read the frame TheSky auto-saved, once it reports a file newer than *previous_file*.

        Large frames (e.g. an 18 MP DSLR) never pass through TheSky's script engine this way;
        building them as one CSV string there has crashed TheSky.
        """
        async with asyncio.timeout(self.config.timeout):
            while True:
                path = _image_file_name(await self.execute("""ccdsoftCamera.LastImageFileName;"""))
                if path and path != previous_file:
                    break
                await asyncio.sleep(0.2)

        def read():
            with fits.open(path, memmap=False) as hdul:
                return np.asarray(hdul[0].data, dtype=np.uint16), hdul[0].header

        data, header = await asyncio.to_thread(read)
        logger.debug(f"read TheSky frame {path}")

        date_obs = datetime.fromisoformat(header["DATE-OBS"]).replace(tzinfo=UTC)
        temp = _float_or_none(header.get("CCD-TEMP"))
        return data, date_obs, temp, float(header.get("XBINNING", 1)), float(header.get("YBINNING", 1))

    def _pixels_from_csv_block(self, csv_block: str, dtype: str = "uint16") -> list[array.array]:
        """
        Convert a single CSV block string like:
            "1,2,3;10,20,30;100,200,300"
        into a list of array.array rows.
        """

        # `array.array` typecode for the target dtype (e.g. uint16 -> "H").
        typecode = np.dtype(dtype).char

        img = []
        for row_str in csv_block.strip().split(";"):
            if not row_str:
                continue
            nums = list(map(int, row_str.split(",")))
            img.append(array.array(typecode, nums))
        return img

    async def status_publish(self):
        if self._capturing:
            return

        try:
            resp = await self.execute(
                """
                var Out;
                // Reading a disconnected camera throws, which would park TheSky in its debugger.
                if (ccdsoftCamera.Status == "Not Connected") {
                    Out = ["Not Connected"];
                } else {
                    Out = [
                        ccdsoftCamera.Status,
                        ccdsoftCamera.Temperature,
                        ccdsoftCamera.SubframeLeft,
                        ccdsoftCamera.SubframeTop,
                        ccdsoftCamera.SubframeRight,
                        ccdsoftCamera.SubframeBottom,
                        ccdsoftCamera.BinX,
                        ccdsoftCamera.BinY
                    ];
                }
                """
            )
        except ProcessAbortedError:
            # TheSky returns 212 transiently after an `abort`; camera is fine
            return

        parts = resp.split(",")
        connected = parts[0] != "Not Connected"
        self.device_connected = connected

        device = sk.device()
        if not connected:
            await device.publish(Connected(is_connected=False))
            return

        temp = _float_or_none(parts[1])
        left, top, right, bottom, binx, biny = map(float, parts[2:])

        width = int(right - left)
        height = int(bottom - top)

        await device.publish(Connected(is_connected=connected))
        if temp is not None:
            await device.publish(
                CameraSensorTemperature(temperature=temp, units=TemperatureUnit.CELSIUS)
            )
        await device.publish(CameraSensorSize(x=width, y=height))
        await device.publish(Binning(x=int(binx), y=int(biny)))


class TheSkyCameraConfig(TheSkyDeviceConfig[TheSkyCamera]):
    """TheSky Camera configuration."""

    device_type: Literal["camera"] = "camera"
    temperature: float | None = -10.0  # C; None leaves the cooler alone
    # How frames get back from TheSky: "scanline" streams them through the script server;
    # "file" has TheSky auto-save each frame and reads that FITS (needs a shared filesystem).
    readout: Literal["scanline", "file"] = "scanline"
    scanline_chunk_rows: int = 128  # rows per script in scanline readout
    bayer: BayerPattern | None = None  # raw color sensor's CFA pattern in scanLine row order
    status_frequency: float = 1.0
    timeout: float

    @override
    def create_device(self):
        return TheSkyCamera(self)


def _image_file_name(value) -> str | None:
    """TheSky's LastImageFileName, or None when it reports no file."""
    name = str(value).strip() if value is not None else ""
    return None if name in ("", "undefined", "null") else name


def _float_or_none(value) -> float | None:
    """A numeric TheSky reading, or None where it has none (e.g. a DSLR's sensor temperature)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class TheSkyCameraState(TheSkyDeviceState):
    """TheSky Camera state."""

    device_type: Literal["camera"] = "camera"
