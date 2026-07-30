"""Visualization helpers for signed event voxel sequences."""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError as exc:  # pragma: no cover
    raise ImportError("Visualization requires Pillow. Install it with: pip install pillow") from exc


ALL_CHANNELS = -1
LIGHT_POSITIVE = np.array([214, 39, 40], dtype=np.float32)
LIGHT_NEGATIVE = np.array([31, 119, 180], dtype=np.float32)
DARK_POSITIVE = np.array([255, 60, 60], dtype=np.float32)
DARK_NEGATIVE = np.array([80, 150, 255], dtype=np.float32)
WHITE = np.array([255, 255, 255], dtype=np.float32)
BLACK = np.array([0, 0, 0], dtype=np.float32)


def signed_log_scale(values: np.ndarray, clip_value: float) -> np.ndarray:
    if clip_value <= 0:
        raise ValueError("clip_value must be positive")
    values = np.nan_to_num(values, nan=0.0, posinf=clip_value, neginf=-clip_value)
    values = np.clip(values, -clip_value, clip_value)
    return np.sign(values) * np.log1p(np.abs(values)) / np.log1p(clip_value)


def auto_clip_value(
    *arrays: np.ndarray,
    percentile: float = 99.0,
    min_clip: float = 0.05,
) -> float:
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    if min_clip <= 0:
        raise ValueError("min_clip must be positive")
    values = []
    for array in arrays:
        finite_abs = np.abs(array[np.isfinite(array)])
        finite_abs = finite_abs[finite_abs > 0]
        if finite_abs.size:
            values.append(finite_abs)
    if not values:
        return min_clip
    merged = np.concatenate(values)
    return max(float(np.percentile(merged, percentile)), min_clip)


def apply_gamma(values: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    return np.power(np.clip(values, 0.0, 1.0), gamma)


def collapse_channels(frame: np.ndarray, reduce: str = "sum", channel: int | None = None) -> np.ndarray:
    if channel is not None:
        if channel < 0 or channel >= frame.shape[0]:
            raise ValueError(f"channel {channel} is out of range for {frame.shape[0]} channels")
        return frame[channel]
    if frame.shape[0] == 1:
        return frame[0]
    if reduce == "sum":
        return frame.sum(axis=0)
    if reduce == "mean":
        return frame.mean(axis=0)
    if reduce == "maxabs":
        indices = np.abs(frame).argmax(axis=0)
        return np.take_along_axis(frame, indices[None, ...], axis=0)[0]
    raise ValueError(f"Unknown reduce mode: {reduce}")


def signed_to_rgb(scaled: np.ndarray, theme: str = "light", gamma: float = 0.55) -> np.ndarray:
    scaled = np.clip(scaled, -1.0, 1.0)
    positive = apply_gamma(np.clip(scaled, 0.0, 1.0), gamma)
    negative = apply_gamma(np.clip(-scaled, 0.0, 1.0), gamma)
    amount = np.clip(positive + negative, 0.0, 1.0)[..., None]
    if theme == "light":
        event_color = positive[..., None] * LIGHT_POSITIVE + negative[..., None] * LIGHT_NEGATIVE
        rgb = WHITE * (1.0 - amount) + event_color
    elif theme == "dark":
        event_color = positive[..., None] * DARK_POSITIVE + negative[..., None] * DARK_NEGATIVE
        rgb = BLACK * (1.0 - amount) + event_color
    else:
        raise ValueError("theme must be 'light' or 'dark'")
    return np.clip(rgb, 0, 255).round().astype(np.uint8)


def gray_to_rgb(scaled: np.ndarray, theme: str = "light", gamma: float = 0.55) -> np.ndarray:
    intensity = apply_gamma(np.abs(np.clip(scaled, -1.0, 1.0)), gamma)
    if theme == "light":
        gray = (255.0 * (1.0 - intensity)).round().astype(np.uint8)
    elif theme == "dark":
        gray = (255.0 * intensity).round().astype(np.uint8)
    else:
        raise ValueError("theme must be 'light' or 'dark'")
    return np.repeat(gray[..., None], 3, axis=2)


def rgb_mode(frame: np.ndarray, clip_value: float) -> np.ndarray:
    if frame.shape[0] < 3:
        padded = np.zeros((3, *frame.shape[1:]), dtype=frame.dtype)
        padded[: frame.shape[0]] = frame
        frame = padded
    channels = signed_log_scale(frame[:3], clip_value)
    rgb = ((np.moveaxis(channels, 0, -1) + 1.0) * 127.5).round()
    return np.clip(rgb, 0, 255).astype(np.uint8)


def grid_mode(frame: np.ndarray, clip_value: float, theme: str, gamma: float) -> np.ndarray:
    panels = [signed_to_rgb(signed_log_scale(channel, clip_value), theme, gamma) for channel in frame]
    if len(panels) == 1:
        return panels[0]
    height = panels[0].shape[0]
    gap_value = 230 if theme == "light" else 32
    gap = np.full((height, 4, 3), gap_value, dtype=np.uint8)
    pieces: list[np.ndarray] = []
    for index, panel in enumerate(panels):
        if index:
            pieces.append(gap)
        pieces.append(panel)
    return np.concatenate(pieces, axis=1)


def render_frame(
    frame: np.ndarray,
    mode: str = "signed",
    reduce: str = "sum",
    channel: int | None = None,
    clip_value: float = 100.0,
    theme: str = "light",
    gamma: float = 0.55,
) -> np.ndarray:
    if mode == "grid":
        if channel is not None:
            frame = frame[channel : channel + 1]
        return grid_mode(frame, clip_value, theme, gamma)
    if mode == "rgb":
        return rgb_mode(frame, clip_value)
    collapsed = collapse_channels(frame, reduce=reduce, channel=channel)
    scaled = signed_log_scale(collapsed, clip_value)
    if mode == "signed":
        return signed_to_rgb(scaled, theme, gamma)
    if mode == "gray":
        return gray_to_rgb(scaled, theme, gamma)
    raise ValueError(f"Unknown mode: {mode}")


def select_indices(length: int, count: int) -> list[int]:
    if count <= 0:
        return []
    if length <= count:
        return list(range(length))
    return [int(round(x)) for x in np.linspace(0, length - 1, count)]


def make_sequence_comparison(
    condition: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    max_frames: int = 8,
    mode: str = "signed",
    reduce: str = "sum",
    channel: int | None = None,
    clip_value: float | None = None,
    clip_percentile: float = 99.0,
    min_clip: float = 0.05,
    theme: str = "light",
    gamma: float = 0.55,
) -> Image.Image:
    """Create a labeled condition/target/prediction grid."""

    target_indices = select_indices(target.shape[0], max_frames)
    condition_indices = select_indices(condition.shape[0], min(condition.shape[0], max_frames))
    effective_clip = (
        clip_value
        if clip_value is not None
        else auto_clip_value(condition, target, prediction, percentile=clip_percentile, min_clip=min_clip)
    )
    rows: list[tuple[str, list[np.ndarray]]] = [
        (
            "condition",
            [
                render_frame(condition[index], mode, reduce, channel, effective_clip, theme, gamma)
                for index in condition_indices
            ],
        ),
        (
            "target",
            [
                render_frame(target[index], mode, reduce, channel, effective_clip, theme, gamma)
                for index in target_indices
            ],
        ),
        (
            "prediction",
            [
                render_frame(prediction[index], mode, reduce, channel, effective_clip, theme, gamma)
                for index in target_indices
            ],
        ),
    ]

    tile_h, tile_w = rows[1][1][0].shape[:2]
    gap = 6
    label_w = 92
    header_h = 22
    max_cols = max(len(frames) for _, frames in rows)
    width = label_w + max_cols * tile_w + (max_cols - 1) * gap
    height = len(rows) * (tile_h + header_h) + (len(rows) - 1) * gap
    background = (246, 247, 249) if theme == "light" else (0, 0, 0)
    text = (20, 20, 20) if theme == "light" else (255, 255, 255)
    subtext = (90, 90, 90) if theme == "light" else (220, 220, 220)
    canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas)
    y = 0
    for label, frames in rows:
        draw.text((8, y + 5), label, fill=text)
        for col, frame in enumerate(frames):
            x = label_w + col * (tile_w + gap)
            if label != "condition":
                draw.text((x + 4, y + 5), f"t+{target_indices[col] + 1}", fill=subtext)
            else:
                draw.text((x + 4, y + 5), f"c{condition_indices[col]}", fill=subtext)
            canvas.paste(Image.fromarray(frame, mode="RGB"), (x, y + header_h))
        y += tile_h + header_h + gap
    return canvas


def make_video_frame(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    mode: str = "signed",
    reduce: str = "sum",
    channel: int | None = None,
    clip_value: float = 100.0,
    theme: str = "light",
    gamma: float = 0.55,
) -> np.ndarray:
    left = render_frame(target, mode, reduce, channel, clip_value, theme, gamma)
    right = render_frame(prediction, mode, reduce, channel, clip_value, theme, gamma)
    header_h = 24
    gap = 6
    height = max(left.shape[0], right.shape[0]) + header_h
    width = left.shape[1] + right.shape[1] + gap
    background = (246, 247, 249) if theme == "light" else (0, 0, 0)
    text = (20, 20, 20) if theme == "light" else (255, 255, 255)
    canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), "target", fill=text)
    draw.text((left.shape[1] + gap + 8, 6), "prediction", fill=text)
    canvas.paste(Image.fromarray(left, mode="RGB"), (0, header_h))
    canvas.paste(Image.fromarray(right, mode="RGB"), (left.shape[1] + gap, header_h))
    return np.asarray(canvas)


def riff_chunk(tag: bytes, data: bytes) -> bytes:
    padding = b"\x00" if len(data) % 2 else b""
    return tag + struct.pack("<I", len(data)) + data + padding


def riff_list(list_type: bytes, data: bytes) -> bytes:
    size = 4 + len(data)
    padding = b"\x00" if size % 2 else b""
    return b"LIST" + struct.pack("<I", size) + list_type + data + padding


class MjpegAviWriter:
    """Small MJPEG AVI writer backed by Pillow and the standard library."""

    def __init__(self, path: Path, width: int, height: int, fps: float, total_frames: int) -> None:
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.total_frames = int(total_frames)
        self.index: list[tuple[int, int]] = []
        self.file = path.open("wb")
        self.movi_size_pos = 0
        self.movi_data_start = 0
        self._write_header()

    def _write_header(self) -> None:
        fps_int = max(1, int(round(self.fps)))
        microseconds_per_frame = int(round(1_000_000 / self.fps))
        suggested_buffer = self.width * self.height * 3
        mjpg = struct.unpack("<I", b"MJPG")[0]
        avih = struct.pack(
            "<IIIIIIIIII4I",
            microseconds_per_frame,
            suggested_buffer * fps_int,
            0,
            0x10,
            self.total_frames,
            0,
            1,
            suggested_buffer,
            self.width,
            self.height,
            0,
            0,
            0,
            0,
        )
        strh = struct.pack(
            "<4s4sIHHIIIIIIIIiiii",
            b"vids",
            b"MJPG",
            0,
            0,
            0,
            0,
            1,
            fps_int,
            0,
            self.total_frames,
            suggested_buffer,
            0xFFFFFFFF,
            0,
            0,
            0,
            self.width,
            self.height,
        )
        strf = struct.pack(
            "<IiiHHIIiiII",
            40,
            self.width,
            self.height,
            1,
            24,
            mjpg,
            suggested_buffer,
            0,
            0,
            0,
            0,
        )
        hdrl = riff_chunk(b"avih", avih) + riff_list(
            b"strl",
            riff_chunk(b"strh", strh) + riff_chunk(b"strf", strf),
        )
        self.file.write(b"RIFF")
        self.file.write(struct.pack("<I", 0))
        self.file.write(b"AVI ")
        self.file.write(riff_list(b"hdrl", hdrl))
        self.file.write(b"LIST")
        self.movi_size_pos = self.file.tell()
        self.file.write(struct.pack("<I", 0))
        self.file.write(b"movi")
        self.movi_data_start = self.file.tell()

    def add_frame(self, rgb: np.ndarray) -> None:
        image = Image.fromarray(rgb, mode="RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, subsampling=0)
        payload = buffer.getvalue()
        chunk_start = self.file.tell()
        self.file.write(b"00dc")
        self.file.write(struct.pack("<I", len(payload)))
        self.file.write(payload)
        if len(payload) % 2:
            self.file.write(b"\x00")
        self.index.append((chunk_start - self.movi_data_start, len(payload)))

    def close(self) -> None:
        idx_start = self.file.tell()
        idx_payload = b"".join(
            struct.pack("<4sIII", b"00dc", 0x10, offset, size) for offset, size in self.index
        )
        self.file.write(riff_chunk(b"idx1", idx_payload))
        file_end = self.file.tell()
        self.file.seek(self.movi_size_pos)
        self.file.write(struct.pack("<I", idx_start - self.movi_size_pos - 4))
        self.file.seek(4)
        self.file.write(struct.pack("<I", file_end - 8))
        self.file.close()


def save_comparison_video(
    target: np.ndarray,
    prediction: np.ndarray,
    path: Path,
    *,
    fps: float = 20.0,
    mode: str = "signed",
    reduce: str = "sum",
    channel: int | None = None,
    clip_value: float | None = None,
    clip_percentile: float = 99.0,
    min_clip: float = 0.05,
    theme: str = "light",
    gamma: float = 0.55,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    effective_clip = (
        clip_value
        if clip_value is not None
        else auto_clip_value(target, prediction, percentile=clip_percentile, min_clip=min_clip)
    )
    first = make_video_frame(
        target[0],
        prediction[0],
        mode=mode,
        reduce=reduce,
        channel=channel,
        clip_value=effective_clip,
        theme=theme,
        gamma=gamma,
    )
    writer = MjpegAviWriter(path, first.shape[1], first.shape[0], fps, target.shape[0])
    try:
        writer.add_frame(first)
        for index in range(1, target.shape[0]):
            writer.add_frame(
                make_video_frame(
                    target[index],
                    prediction[index],
                    mode=mode,
                    reduce=reduce,
                    channel=channel,
                    clip_value=effective_clip,
                    theme=theme,
                    gamma=gamma,
                )
            )
    finally:
        writer.close()
