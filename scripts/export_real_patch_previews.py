from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / "data" / "images_real"
OUTPUT_DIR = BASE_DIR / "data" / "images_real_preview"


def to_rgb(patch: np.ndarray) -> np.ndarray:
    # Input band order: B2, B3, B4, B8. RGB uses B4, B3, B2.
    rgb = np.stack([patch[2], patch[1], patch[0]], axis=-1).astype(np.float32)
    p2, p98 = np.percentile(rgb, [2, 98])
    if p98 > p2:
        rgb = (rgb - p2) / (p98 - p2)
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def main(limit: int = 40) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(INPUT_DIR.glob("*.npy"))[:limit]

    for file_path in files:
        patch = np.load(file_path)
        if patch.ndim != 3:
            continue
        if patch.shape[0] != 4 and patch.shape[-1] == 4:
            patch = np.transpose(patch, (2, 0, 1))
        if patch.shape[0] < 4:
            continue

        rgb = to_rgb(patch[:4])
        out_path = OUTPUT_DIR / f"{file_path.stem}.png"
        plt.imsave(out_path, rgb)

    print(f"Saved {len(files)} previews to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main(limit=40)
