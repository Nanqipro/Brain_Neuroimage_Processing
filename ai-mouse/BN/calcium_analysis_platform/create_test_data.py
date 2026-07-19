"""Generate a deterministic synthetic workbook for the Web demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_test_data(time_points: int = 1000, neurons: int = 5) -> pd.DataFrame:
    """Build synthetic calcium traces without reading any research data."""

    rng = np.random.default_rng(42)
    data: dict[str, np.ndarray] = {"Time": np.arange(time_points)}

    for index in range(1, neurons + 1):
        calcium_signal = rng.normal(100, 5, time_points)
        for _ in range(rng.integers(3, 8)):
            start = int(rng.integers(0, time_points - 100))
            duration = int(rng.integers(20, 80))
            amplitude = float(rng.uniform(20, 50))
            x = np.arange(duration)
            transient = amplitude * np.exp(-x / 20) * (1 - np.exp(-x / 5))
            end = min(start + duration, time_points)
            calcium_signal[start:end] += transient[: end - start]
        data[f"Neuron_{index}"] = calcium_signal

    return pd.DataFrame(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "test_data.xlsx",
        help="Output workbook path (default: next to this script)",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output) as writer:
        build_test_data().to_excel(writer, sheet_name="dF", index=False)
    print(f"Synthetic test workbook created: {args.output}")


if __name__ == "__main__":
    main()
