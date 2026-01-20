import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile


@dataclass(frozen=True)
class TifSummary:
    path: str
    shape: Tuple[int, ...]
    dtype: str
    min: float
    max: float
    mean: float
    std: float


def _list_tifs(data_dir: str) -> List[str]:
    if not os.path.isdir(data_dir):
        return []
    files = []
    for name in os.listdir(data_dir):
        lower = name.lower()
        if lower.endswith(".tif") or lower.endswith(".tiff"):
            files.append(os.path.join(data_dir, name))
    files.sort()
    return files


def _safe_read_frame(path: str, frame_index: int) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if len(series.shape) == 2:
            return series.asarray().copy()
        if len(series.pages) > frame_index:
            return series.pages[frame_index].asarray().copy()
        stack = series.asarray().copy()
        if stack.ndim >= 3 and frame_index < stack.shape[0]:
            return stack[frame_index]
        return stack


def _summarize_one_tif(path: str, sample_frames: int) -> TifSummary:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        shape = tuple(int(x) for x in series.shape)
        dtype = str(series.dtype)

    sample: np.ndarray
    if len(shape) <= 2:
        sample = _safe_read_frame(path, 0).astype(np.float64)
    else:
        t = int(shape[0])
        indices = [0, t // 2, max(0, t - 1)]
        if sample_frames > 3 and t > 3:
            rng = np.random.default_rng(0)
            extra = rng.integers(0, t, size=sample_frames - 3).tolist()
            indices.extend(extra)
        indices = sorted(set(int(i) for i in indices if 0 <= i < t))

        frames: List[np.ndarray] = []
        try:
            block = tifffile.imread(path, key=indices)
            if block.ndim == 2:
                frames = [block]
            else:
                frames = [block[i] for i in range(block.shape[0])]
        except Exception:
            frames = [_safe_read_frame(path, i) for i in indices]

        sample = np.stack([f.astype(np.float64) for f in frames], axis=0)

    return TifSummary(
        path=path,
        shape=shape,
        dtype=dtype,
        min=float(np.min(sample)),
        max=float(np.max(sample)),
        mean=float(np.mean(sample)),
        std=float(np.std(sample)),
    )


def _infer_tyx(shape: Sequence[int]) -> Optional[Tuple[int, int, int]]:
    if len(shape) == 3:
        t, y, x = (int(shape[0]), int(shape[1]), int(shape[2]))
        return t, y, x
    if len(shape) == 4:
        t, z, y, x = (int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]))
        t2 = t * z
        return t2, y, x
    return None


def _round_down(value: int, base: int) -> int:
    if base <= 0:
        return value
    return (value // base) * base


def _suggest_patch_xy(y: int, x: int) -> int:
    min_dim = max(1, min(y, x))
    patch_xy = min(200, min_dim)
    patch_xy = _round_down(patch_xy, 10)
    patch_xy = max(64, patch_xy)
    patch_xy = min(patch_xy, min_dim)
    return int(patch_xy)


def _suggest_patch_t(t: int) -> int:
    if t <= 0:
        return 20
    patch_t = min(150, max(40, t // 20))
    patch_t = min(patch_t, max(10, t // 2))
    return int(patch_t)


def _suggest_test_overlap(patch_xy: int) -> float:
    if patch_xy <= 0:
        return 0.6
    required = 90.0 / float(patch_xy)
    overlap = max(0.4, min(0.8, required))
    return float(overlap)


def _estimate_w_h_nums(x: int, y: int, patch_xy: int, overlap_factor: float) -> Tuple[int, int]:
    gap = max(1, int(round(patch_xy * (1 - overlap_factor))))
    w_num = max(1, math.floor((x - patch_xy) / gap) + 1) if x >= patch_xy else 1
    h_num = max(1, math.floor((y - patch_xy) / gap) + 1) if y >= patch_xy else 1
    return int(w_num), int(h_num)


def suggest_params(
    summaries: Sequence[TifSummary],
    gpus: str,
    train_overlap: float,
) -> Dict[str, Any]:
    shapes = [s.shape for s in summaries]
    inferred = [_infer_tyx(shape) for shape in shapes]
    inferred = [v for v in inferred if v is not None]
    if not inferred:
        return {"error": "unsupported_tif_shape", "shapes": shapes}

    t = int(min(v[0] for v in inferred))
    y = int(min(v[1] for v in inferred))
    x = int(min(v[2] for v in inferred))
    dtype = summaries[0].dtype
    vmax = float(max(s.max for s in summaries))

    patch_xy = _suggest_patch_xy(y, x)
    patch_t = _suggest_patch_t(t)

    train_datasets_size = 6000
    test_datasize_quick = min(400, t) if t > 0 else 400
    test_datasize_full = t if t > 0 else 100000

    overlap_test = _suggest_test_overlap(patch_xy)

    w_num, h_num = _estimate_w_h_nums(x, y, patch_xy, train_overlap)
    stack_num = max(1, len(summaries))
    approx_xy_tiles = w_num * h_num * stack_num

    scale_factor = 1
    if vmax > 1e6:
        scale_factor = 1000
    elif vmax > 1e5:
        scale_factor = 100

    return {
        "data": {
            "files": [s.path for s in summaries],
            "inferred_min_TYX": {"T": t, "Y": y, "X": x},
            "dtype": dtype,
            "max_value_estimate": vmax,
        },
        "suggested_train": {
            "datasets_path": "datasets/my_data",
            "GPU": gpus,
            "patch_xy": patch_xy,
            "patch_t": patch_t,
            "overlap_factor": float(train_overlap),
            "train_datasets_size": int(train_datasets_size),
            "scale_factor": int(scale_factor),
            "num_workers": 4,
        },
        "suggested_test": {
            "datasets_path": "datasets/my_data",
            "GPU": gpus,
            "patch_xy": patch_xy,
            "patch_t": patch_t,
            "overlap_factor": float(overlap_test),
            "test_datasize_quick": int(test_datasize_quick),
            "test_datasize_full": int(test_datasize_full),
            "scale_factor": int(scale_factor),
            "num_workers": 4,
        },
        "notes": {
            "approx_xy_tiles_per_stack": int(approx_xy_tiles),
            "train_requires_T_ge_2_patch_t": bool(t >= 2 * patch_t),
            "test_overlap_rule": "overlap_factor ~= max(0.4, min(0.8, 90/patch_xy))",
        },
    }


def _print_human_readable(summaries: Sequence[TifSummary], suggested: Dict[str, Any]) -> None:
    print("Found tif files:")
    for s in summaries:
        print(f"- {s.path}")
        print(f"  shape={s.shape} dtype={s.dtype} min={s.min:.3f} max={s.max:.3f} mean={s.mean:.3f} std={s.std:.3f}")

    print("\nSuggested parameters (copy into demo_train_pipeline.py / demo_test_pipeline.py):")
    if "error" in suggested:
        print(json.dumps(suggested, indent=2, ensure_ascii=False))
        return

    tr = suggested["suggested_train"]
    te = suggested["suggested_test"]

    print("\n[Train]")
    print(f"download_demo_file = False")
    print(f"datasets_path = '{tr['datasets_path']}'")
    print(f"GPU = '{tr['GPU']}'")
    print(f"patch_xy = {tr['patch_xy']}")
    print(f"patch_t = {tr['patch_t']}")
    print(f"overlap_factor = {tr['overlap_factor']}")
    print(f"train_datasets_size = {tr['train_datasets_size']}")
    print(f"scale_factor = {tr['scale_factor']}")
    print(f"num_workers = {tr['num_workers']}")

    print("\n[Test]")
    print(f"download_demo_file = False")
    print(f"datasets_path = '{te['datasets_path']}'")
    print(f"GPU = '{te['GPU']}'")
    print(f"patch_xy = {te['patch_xy']}")
    print(f"patch_t = {te['patch_t']}")
    print(f"overlap_factor = {te['overlap_factor']}")
    print(f"test_datasize = {te['test_datasize_full']}")
    print(f"scale_factor = {te['scale_factor']}")
    print(f"num_workers = {te['num_workers']}")

    print("\n[Quick sanity test (optional)]")
    print(f"test_datasize = {te['test_datasize_quick']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=os.path.join(os.path.dirname(__file__), "my_data"),
    )
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--train_overlap", type=float, default=0.25)
    parser.add_argument("--sample_frames", type=int, default=8)
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    tifs = _list_tifs(data_dir)
    if not tifs:
        print(f"No tif files found in: {data_dir}")
        return 2

    summaries = [_summarize_one_tif(p, sample_frames=args.sample_frames) for p in tifs]
    suggested = suggest_params(summaries, gpus=args.gpus, train_overlap=float(args.train_overlap))
    _print_human_readable(summaries, suggested)

    if args.json_out:
        out_path = os.path.abspath(args.json_out)
        payload = {
            "summaries": [asdict(s) for s in summaries],
            "suggested": suggested,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved JSON report to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
