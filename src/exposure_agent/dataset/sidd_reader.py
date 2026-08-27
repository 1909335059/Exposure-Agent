from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

from exposure_agent.camera.exposure import compute_relative_ev

KEEP_METADATA_KEYS = {
    "activearea",
    "aperture",
    "asshotneutral",
    "bitdepth",
    "blacklevel",
    "blacklevelrepeatdim",
    "camera",
    "cameracalibration1",
    "cameracalibration2",
    "cfalayout",
    "cfaplanecolor",
    "colormatrix1",
    "colormatrix2",
    "colortemperature",
    "defaultcroporigin",
    "defaultcropsize",
    "exposuretime",
    "filename",
    "fnumber",
    "focallength",
    "format",
    "height",
    "isospeedratings",
    "iso",
    "make",
    "model",
    "orientation",
    "photometricinterpretation",
    "uniquecameramodel",
    "whitebalance",
    "whitelevel",
    "width",
}

XYZ_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float32,
)

COLOR_CODE_TO_CHANNEL = {
    0: "R",
    1: "G",
    2: "B",
}

DEFAULT_OUTPUT_WHITE_LEVEL = 1.0


@dataclass
class ExposureSample:
    image: np.ndarray
    raw_noisy: np.ndarray
    raw_gt: np.ndarray | None
    iso: int
    shutter: float
    ev: float | None
    camera: str
    scene_id: str
    brightness_level: str
    metadata: dict[str, Any] = field(default_factory=dict)
    image_path: str | None = None
    physical_scene_id: str | None = None
    gt_image: np.ndarray | None = None
    gt_image_path: str | None = None


@dataclass(frozen=True)
class SIDDSceneInfo:
    scene_id: str
    physical_scene_id: str
    camera: str
    iso: int
    shutter: float
    color_temperature: int | None
    brightness_level: str


class SIDDReader:
    def __init__(
        self,
        data_root: str | Path,
        *,
        preview_dir: str | Path | None = None,
        linear_rgb_dir: str | Path | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.data_dir = self.data_root / "Data"
        self.preview_dir = Path(preview_dir) if preview_dir is not None else None
        self.linear_rgb_dir = Path(linear_rgb_dir) if linear_rgb_dir is not None else None

    def scene_directories(self) -> list[Path]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"SIDD Data directory not found: {self.data_dir}")
        return sorted(path for path in self.data_dir.iterdir() if path.is_dir())

    def scene_ids(self) -> list[str]:
        return [parse_sidd_scene_name(path.name).scene_id for path in self.scene_directories()]

    def physical_scene_ids(self) -> list[str]:
        return [
            parse_sidd_scene_name(path.name).physical_scene_id
            for path in self.scene_directories()
        ]

    def iter_samples(self, max_samples: int | None = None) -> Iterator[ExposureSample]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"SIDD Data directory not found: {self.data_dir}")

        yielded = 0
        for scene_dir in self.scene_directories():
            if max_samples is not None and yielded >= max_samples:
                break
            try:
                sample = self.read_scene(scene_dir)
            except Exception as exc:
                warnings.warn(f"Skipping damaged SIDD sample {scene_dir.name}: {exc}")
                continue
            yielded += 1
            yield sample

    def read_scene(self, scene_dir: str | Path) -> ExposureSample:
        scene_path = Path(scene_dir)
        info = parse_sidd_scene_name(scene_path.name)
        noisy_raw = _first_array(_load_mat(scene_path / "NOISY_RAW_010.MAT"))
        gt_raw = self._try_load_gt(scene_path / "GT_RAW_010.MAT")
        metadata = self._try_load_metadata(scene_path / "METADATA_RAW_010.MAT")

        iso = int(_find_metadata_number(metadata, ["iso"]) or info.iso)
        shutter = float(
            _find_metadata_number(
                metadata,
                ["exposuretime", "exposure_time", "shutter", "shutterspeed"],
            )
            or info.shutter
        )
        ev = compute_relative_ev(iso=iso, shutter=shutter)
        isp_metadata = {
            **metadata,
            "color_temperature": info.color_temperature,
        }
        preview_raw = gt_raw if gt_raw is not None else noisy_raw
        linear_rgb = raw_to_linear_rgb(preview_raw, metadata=isp_metadata)
        if self.linear_rgb_dir is not None:
            save_rgb_png(linear_rgb, self.linear_rgb_dir / f"{info.scene_id}.png")
        rgb = rgb_to_srgb(linear_rgb)
        image_path = None
        if self.preview_dir is not None:
            image_path = str(save_rgb_png(rgb, self.preview_dir / f"{info.scene_id}.png"))

        compact_metadata = compact_sidd_metadata(
            metadata,
            folder_name=scene_path.name,
            color_temperature=info.color_temperature,
        )

        return ExposureSample(
            image=rgb,
            raw_noisy=noisy_raw,
            raw_gt=gt_raw,
            iso=iso,
            shutter=shutter,
            ev=ev,
            camera=info.camera,
            scene_id=info.scene_id,
            brightness_level=info.brightness_level,
            metadata=compact_metadata,
            image_path=image_path,
            physical_scene_id=info.physical_scene_id,
        )

    @staticmethod
    def _try_load_gt(path: Path) -> np.ndarray | None:
        if not path.exists():
            return None
        return _first_array(_load_mat(path))

    @staticmethod
    def _try_load_metadata(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        data = _load_mat(path)
        return _to_plain_metadata(data)


def parse_sidd_scene_name(folder_name: str) -> SIDDSceneInfo:
    parts = folder_name.split("_")
    if len(parts) < 7:
        raise ValueError(f"Unexpected SIDD scene folder name: {folder_name}")
    scene_id = f"{parts[0]}_{parts[1]}"
    physical_scene_id = parts[1]
    camera = parts[2]
    iso = int(parts[3])
    shutter = _parse_shutter_token(parts[4])
    color_temperature = int(parts[5]) if parts[5].isdigit() else None
    brightness_level = parts[6]
    return SIDDSceneInfo(
        scene_id=scene_id,
        physical_scene_id=physical_scene_id,
        camera=camera,
        iso=iso,
        shutter=shutter,
        color_temperature=color_temperature,
        brightness_level=brightness_level,
    )


class SIDDSRGBReader:
    """Reads official SIDD sRGB pairs while using NOISY sRGB as model input."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        preview_dir: str | Path | None = None,
        preview_max_dimension: int = 1024,
        load_gt_image: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.data_dir = (
            self.data_root / "Data"
            if (self.data_root / "Data").is_dir()
            else self.data_root
        )
        self.preview_dir = Path(preview_dir) if preview_dir is not None else None
        self.preview_max_dimension = preview_max_dimension
        self.load_gt_image = load_gt_image

    def scene_directories(self) -> list[Path]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"SIDD sRGB Data directory not found: {self.data_dir}")
        return sorted(path for path in self.data_dir.iterdir() if path.is_dir())

    def scene_ids(self) -> list[str]:
        return [parse_sidd_scene_name(path.name).scene_id for path in self.scene_directories()]

    def physical_scene_ids(self) -> list[str]:
        return [
            parse_sidd_scene_name(path.name).physical_scene_id
            for path in self.scene_directories()
        ]

    def iter_samples(self, max_samples: int | None = None) -> Iterator[ExposureSample]:
        yielded = 0
        for scene_dir in self.scene_directories():
            if max_samples is not None and yielded >= max_samples:
                break
            try:
                sample = self.read_scene(scene_dir)
            except Exception as exc:
                warnings.warn(f"Skipping damaged SIDD sRGB sample {scene_dir.name}: {exc}")
                continue
            yielded += 1
            yield sample

    def read_scene(self, scene_dir: str | Path) -> ExposureSample:
        scene_path = Path(scene_dir)
        info = parse_sidd_scene_name(scene_path.name)
        noisy_path = _find_srgb_file(scene_path, "NOISY_SRGB_")
        gt_path = _find_srgb_file(scene_path, "GT_SRGB_", required=False)

        noisy_image = _load_rgb_image(noisy_path)
        image_path = noisy_path
        if self.preview_dir is not None:
            image_path = self.preview_dir / f"{info.scene_id}.png"
            _save_resized_preview(
                noisy_path,
                image_path,
                max_dimension=self.preview_max_dimension,
            )
            noisy_image = _load_rgb_image(image_path)

        gt_image = (
            _load_rgb_image(gt_path)
            if self.load_gt_image and gt_path is not None
            else None
        )
        metadata = {
            "source": "SIDD official NOISY sRGB",
            "folder_name": scene_path.name,
            "noisy_srgb_path": str(noisy_path),
            "gt_srgb_path": str(gt_path) if gt_path is not None else None,
            "color_temperature": info.color_temperature,
        }
        return ExposureSample(
            image=noisy_image,
            raw_noisy=np.empty((0, 0), dtype=np.float32),
            raw_gt=None,
            iso=info.iso,
            shutter=info.shutter,
            ev=compute_relative_ev(info.iso, info.shutter),
            camera=info.camera,
            scene_id=info.scene_id,
            brightness_level=info.brightness_level,
            metadata=metadata,
            image_path=str(image_path),
            physical_scene_id=info.physical_scene_id,
            gt_image=gt_image,
            gt_image_path=str(gt_path) if gt_path is not None else None,
        )


def _find_srgb_file(
    scene_path: Path,
    prefix: str,
    *,
    required: bool = True,
) -> Path | None:
    matches = sorted(
        path
        for path in scene_path.iterdir()
        if path.is_file()
        and path.name.upper().startswith(prefix)
        and path.suffix.upper() == ".PNG"
    )
    if matches:
        return matches[0]
    if required:
        raise FileNotFoundError(f"Missing {prefix}*.PNG in {scene_path}")
    return None


def _load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _save_resized_preview(
    source_path: Path,
    output_path: Path,
    *,
    max_dimension: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as source:
        image = source.convert("RGB")
        if max(image.size) > max_dimension:
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        image.save(output_path)
    return output_path


def compact_sidd_metadata(
    metadata: dict[str, Any],
    *,
    folder_name: str,
    color_temperature: int | None,
) -> dict[str, Any]:
    compact = {
        "folder_name": folder_name,
        "color_temperature": color_temperature,
    }
    flattened = _flatten_metadata(metadata)
    for key, value in flattened.items():
        if (
            _normalize_key(key) in KEEP_METADATA_KEYS
            or _normalize_key(key.rsplit(".", 1)[-1]) in KEEP_METADATA_KEYS
        ):
            compact[key] = _limit_metadata_value(value)
    return compact


def raw_to_rgb(raw: np.ndarray, metadata: dict[str, Any] | None = None) -> np.ndarray:
    return rgb_to_srgb(raw_to_linear_rgb(raw, metadata=metadata))


def raw_to_linear_rgb(
    raw: np.ndarray,
    metadata: dict[str, Any] | None = None,
    *,
    apply_crop: bool = False,
    apply_orientation: bool = False,
    output_white_level: float = DEFAULT_OUTPUT_WHITE_LEVEL,
) -> np.ndarray:
    metadata = metadata or {}
    arr = np.asarray(raw)
    normalized = _normalize_raw(arr, metadata)
    if apply_crop:
        normalized = _apply_default_crop(normalized, metadata)

    if normalized.ndim == 3 and normalized.shape[-1] == 3:
        rgb = normalized
    elif normalized.ndim == 3 and normalized.shape[-1] >= 4:
        rgb = np.stack(
            [
                normalized[..., 0],
                (normalized[..., 1] + normalized[..., 2]) / 2.0,
                normalized[..., 3],
            ],
            axis=-1,
        )
    elif normalized.ndim == 2:
        rgb = _demosaic_bayer(normalized, _find_cfa_pattern(metadata))
    else:
        raise ValueError(f"Unsupported RAW shape: {normalized.shape}")

    rgb = _white_balance(rgb, metadata)
    rgb = _camera_rgb_to_srgb(rgb, metadata)
    rgb = _scale_linear_rgb(rgb, output_white_level=output_white_level)
    if apply_orientation:
        rgb = _apply_orientation(rgb, metadata)
    return rgb.astype(np.float32)


def rgb_to_srgb(
    rgb: np.ndarray,
    *,
    normalize: bool = False,
    output_white_level: float = DEFAULT_OUTPUT_WHITE_LEVEL,
) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    if normalize:
        arr = _scale_linear_rgb(arr, output_white_level=output_white_level)
    else:
        arr = np.clip(arr, 0.0, 1.0)
    return _srgb_transfer(arr).astype(np.float32)


def save_rgb_png(image: np.ndarray, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image, dtype=np.float32)
    if arr.max(initial=0) <= 1.0:
        arr = arr * 255.0
    Image.fromarray(np.uint8(np.clip(np.round(arr), 0, 255))).save(output)
    return output


def _parse_shutter_token(token: str) -> float:
    value = float(token)
    if value <= 0:
        raise ValueError(f"Invalid shutter token: {token}")
    if value >= 10:
        return 1.0 / value
    return value


def _normalize_raw(raw: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    original = np.asarray(raw)
    if np.issubdtype(original.dtype, np.floating) and 0.0 <= float(arr.min(initial=0.0)):
        max_value = float(arr.max(initial=0.0))
        if max_value <= 1.0:
            return np.clip(arr, 0.0, 1.0)

    black = _find_metadata_number(metadata, ["blacklevel", "black_level"])
    white = _find_metadata_number(metadata, ["whitelevel", "white_level"])

    if black is None:
        black = float(np.percentile(arr, 0.1))
    if white is None:
        if np.issubdtype(np.asarray(raw).dtype, np.integer):
            white = float(np.iinfo(np.asarray(raw).dtype).max)
        else:
            white = float(np.percentile(arr, 99.9))
    if white <= black:
        white = float(arr.max(initial=black + 1.0))
    return np.clip((arr - black) / (white - black), 0.0, 1.0)


def _apply_default_crop(raw: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    origin = _find_metadata_value(metadata, "DefaultCropOrigin")
    size = _find_metadata_value(metadata, "DefaultCropSize")
    if origin is None or size is None or raw.ndim < 2:
        return raw
    origin_arr = np.asarray(origin, dtype=int).reshape(-1)
    size_arr = np.asarray(size, dtype=int).reshape(-1)
    if origin_arr.size < 2 or size_arr.size < 2:
        return raw

    row = int(origin_arr[1])
    col = int(origin_arr[0])
    height = int(size_arr[1])
    width = int(size_arr[0])
    if row < 0 or col < 0 or height <= 0 or width <= 0:
        return raw
    return raw[row : row + height, col : col + width, ...]


def _demosaic_bayer(raw: np.ndarray, pattern: str) -> np.ndarray:
    even_h = raw.shape[0] - raw.shape[0] % 2
    even_w = raw.shape[1] - raw.shape[1] % 2
    cropped = raw[:even_h, :even_w]
    masks = _bayer_masks(cropped.shape, pattern)
    channels = []
    for channel_name in ("R", "G", "B"):
        mask = masks[channel_name]
        channels.append(_interpolate_channel(cropped * mask, mask))
    return np.stack(channels, axis=-1)


def _bayer_masks(shape: tuple[int, int], pattern: str) -> dict[str, np.ndarray]:
    if len(pattern) != 4 or any(channel not in "RGB" for channel in pattern):
        pattern = "RGGB"
    h, w = shape
    masks = {
        "R": np.zeros((h, w), dtype=np.float32),
        "G": np.zeros((h, w), dtype=np.float32),
        "B": np.zeros((h, w), dtype=np.float32),
    }
    positions = [
        (pattern[0], (0, 0)),
        (pattern[1], (0, 1)),
        (pattern[2], (1, 0)),
        (pattern[3], (1, 1)),
    ]
    for channel, position in positions:
        row_offset, col_offset = position
        masks[channel][row_offset::2, col_offset::2] = 1.0
    return masks


def _interpolate_channel(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import convolve  # type: ignore

        kernel = np.array(
            [
                [1.0, 2.0, 1.0],
                [2.0, 4.0, 2.0],
                [1.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        )
        weighted = convolve(values, kernel, mode="mirror")
        weights = convolve(mask, kernel, mode="mirror")
        interpolated = weighted / np.maximum(weights, 1e-8)
        return np.where(mask > 0, values, interpolated)
    except ImportError:
        return _downsample_demosaic_fallback(values, mask)


def _downsample_demosaic_fallback(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    filled = values.copy()
    known = mask > 0
    mean_value = float(values[known].mean()) if np.any(known) else 0.0
    filled[~known] = mean_value
    return filled


def _demosaic_rggb(raw: np.ndarray) -> np.ndarray:
    even_h = raw.shape[0] - raw.shape[0] % 2
    even_w = raw.shape[1] - raw.shape[1] % 2
    cropped = raw[:even_h, :even_w]
    red = cropped[0::2, 0::2]
    green = (cropped[0::2, 1::2] + cropped[1::2, 0::2]) / 2.0
    blue = cropped[1::2, 1::2]
    return np.stack([red, green, blue], axis=-1)


def _white_balance(rgb: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    gains = _find_wb_gains(metadata)
    if gains is None:
        means = np.maximum(rgb.mean(axis=(0, 1)), 1e-6)
        gains = means[1] / means
    return rgb * np.asarray(gains, dtype=np.float32).reshape(1, 1, 3)


def _find_wb_gains(metadata: dict[str, Any]) -> np.ndarray | None:
    for key in ["whitebalance", "white_balance", "asneutral", "asshotneutral", "wb"]:
        value = _find_metadata_value(metadata, key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size >= 3 and np.all(arr[:3] > 0):
            gains = arr[:3]
            if "neutral" in key:
                gains = 1.0 / gains
            return gains / gains[1]
    return None


def _find_cfa_pattern(metadata: dict[str, Any]) -> str:
    pattern = _find_metadata_value(metadata, "CFAPattern")
    if pattern is None:
        pattern = _find_unknown_tag_value(metadata, 33422)
    if pattern is None:
        return "RGGB"
    values = np.asarray(pattern).reshape(-1)
    if values.size < 4:
        return "RGGB"
    channels = [COLOR_CODE_TO_CHANNEL.get(int(value)) for value in values[:4]]
    if any(channel is None for channel in channels):
        return "RGGB"
    return "".join(channel for channel in channels if channel is not None)


def _find_unknown_tag_value(metadata: dict[str, Any], tag_id: int) -> Any | None:
    tags = _find_metadata_value(metadata, "UnknownTags")
    if tags is None:
        return None
    tag_items = tags if isinstance(tags, list) else [tags]
    for tag in tag_items:
        if not isinstance(tag, dict):
            continue
        if int(tag.get("ID", -1)) == tag_id:
            return tag.get("Value")
    return None


def _camera_rgb_to_srgb(rgb: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    color_matrix = _select_color_matrix(metadata)
    if color_matrix is None:
        return rgb
    try:
        camera_to_xyz = np.linalg.inv(color_matrix)
    except np.linalg.LinAlgError:
        return rgb
    transform = XYZ_TO_SRGB @ camera_to_xyz
    corrected = np.tensordot(rgb, transform.T, axes=1)
    return np.asarray(corrected, dtype=np.float32)


def _select_color_matrix(metadata: dict[str, Any]) -> np.ndarray | None:
    color_temperature = _find_metadata_number(
        metadata,
        ["color_temperature", "ColorTemperature"],
    )
    key = "ColorMatrix1" if color_temperature is not None and color_temperature < 5000 else "ColorMatrix2"
    matrix = _find_metadata_value(metadata, key) or _find_metadata_value(metadata, "ColorMatrix1")
    if matrix is None:
        return None
    arr = np.asarray(matrix, dtype=np.float32).reshape(-1)
    if arr.size != 9:
        return None
    xyz_to_camera = arr.reshape(3, 3)
    row_sums = xyz_to_camera.sum(axis=1, keepdims=True)
    row_sums = np.where(np.abs(row_sums) < 1e-8, 1.0, row_sums)
    return xyz_to_camera / row_sums


def _scale_linear_rgb(
    rgb: np.ndarray,
    *,
    output_white_level: float = DEFAULT_OUTPUT_WHITE_LEVEL,
) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, None)
    white = max(float(output_white_level), 1e-8)
    if white > 1e-6:
        rgb = rgb / white
    return np.clip(rgb, 0.0, 1.0)


def _srgb_transfer(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * np.power(rgb, 1 / 2.4) - 0.055,
    )


def _apply_orientation(rgb: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    orientation = _find_metadata_number(metadata, ["Orientation"])
    if orientation is None:
        return rgb
    orientation_value = int(orientation)
    if orientation_value == 3:
        return np.rot90(rgb, 2)
    if orientation_value == 6:
        return np.rot90(rgb, 3)
    if orientation_value == 8:
        return np.rot90(rgb, 1)
    return rgb


def _load_mat(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        from scipy.io import loadmat  # type: ignore

        return loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        return _load_hdf5_mat(path)
    except ImportError as exc:
        try:
            return _load_hdf5_mat(path)
        except ImportError as hdf_exc:
            raise RuntimeError(
                "Reading SIDD .MAT files requires scipy or h5py. "
                "Install project dependencies before running SIDD inference."
            ) from hdf_exc
    except ValueError:
        return _load_hdf5_mat(path)


def _load_hdf5_mat(path: Path) -> dict[str, Any]:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ImportError("h5py is required for MATLAB v7.3 files") from exc

    def convert(obj: Any) -> Any:
        if hasattr(obj, "keys"):
            return {key: convert(obj[key]) for key in obj.keys()}
        data = np.asarray(obj)
        return data.T if data.ndim >= 2 else data

    with h5py.File(path, "r") as handle:
        return {key: convert(handle[key]) for key in handle.keys()}


def _first_array(data: dict[str, Any]) -> np.ndarray:
    candidates: list[np.ndarray] = []
    for key, value in data.items():
        if key.startswith("__"):
            continue
        for arr in _iter_arrays(value):
            if arr.size > 1 and np.issubdtype(arr.dtype, np.number):
                candidates.append(np.asarray(arr))
    if not candidates:
        raise ValueError("No numeric RAW array found in MAT file")
    return max(candidates, key=lambda arr: arr.size)


def _iter_arrays(value: Any) -> Iterator[np.ndarray]:
    if isinstance(value, np.ndarray):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_arrays(item)
    elif hasattr(value, "__dict__"):
        for item in vars(value).values():
            yield from _iter_arrays(item)


def _to_plain_metadata(data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _plain_value(value)
        for key, value in data.items()
        if not key.startswith("__")
    }


def _plain_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _plain_value(value.item())
        if value.dtype == object:
            return [_plain_value(item) for item in value.reshape(-1)]
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            key: _plain_value(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def _flatten_metadata(
    metadata: dict[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in metadata.items():
        current_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_metadata(value, prefix=current_key))
        else:
            flattened[current_key] = value
    return flattened


def _limit_metadata_value(value: Any, max_items: int = 32) -> Any:
    if isinstance(value, list):
        if len(value) > max_items:
            return {
                "truncated": True,
                "length": len(value),
                "head": value[:max_items],
            }
        return [_limit_metadata_value(item, max_items=max_items) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _limit_metadata_value(item, max_items=max_items)
            for key, item in value.items()
        }
    return value


def _find_metadata_number(
    metadata: dict[str, Any],
    names: list[str],
) -> float | None:
    for name in names:
        value = _find_metadata_value(metadata, name)
        if value is None:
            continue
        arr = np.asarray(value).reshape(-1)
        if arr.size:
            try:
                return float(arr[0])
            except (TypeError, ValueError):
                continue
    return None


def _find_metadata_value(metadata: dict[str, Any], name: str) -> Any | None:
    normalized_name = _normalize_key(name)
    for key, value in metadata.items():
        if _normalize_key(key) == normalized_name:
            return value
        if isinstance(value, dict):
            found = _find_metadata_value(value, name)
            if found is not None:
                return found
    return None


def _normalize_key(key: str) -> str:
    return "".join(ch for ch in key.lower() if ch.isalnum())
