"""
Predspracovanie datasetov CASIA-WebFace a LFW pomocou MTCNN.
MTCNN detekuje tvár, zarovná ju podľa očí a oreže na 112x112 pixelov.
Výsledky sa ukladajú do data/processed/ v rovnakej adresárovej štruktúre.
"""

import argparse
import os
from pathlib import Path

from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm


def create_mtcnn(device: str = "cpu") -> MTCNN:
    return MTCNN(
        image_size=112,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        post_process=False,
        device=device,
    )


def process_dataset(
    src_root: str,
    dst_root: str,
    mtcnn: MTCNN,
) -> tuple[int, int]:
    """Spracuje dataset - detekuje a oreže tváre.

    Returns:
        (processed, skipped) počet úspešne spracovaných a preskočených obrázkov.
    """
    src_path = Path(src_root)
    dst_path = Path(dst_root)

    image_files: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_files.extend(src_path.rglob(ext))

    processed = 0
    skipped = 0

    for img_file in tqdm(image_files, desc=f"Processing {src_path.name}"):
        rel = img_file.relative_to(src_path)
        out_file = dst_path / rel

        if out_file.exists():
            processed += 1
            continue

        try:
            img = Image.open(img_file).convert("RGB")
        except Exception:
            skipped += 1
            continue

        face = mtcnn(img)

        if face is None:
            skipped += 1
            continue

        # MTCNN vracia tensor (C, H, W) s hodnotami 0-255 (post_process=False)
        face_img = Image.fromarray(face.permute(1, 2, 0).byte().numpy())

        out_file.parent.mkdir(parents=True, exist_ok=True)
        face_img.save(str(out_file))
        processed += 1

    return processed, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="MTCNN face crop pre CASIA a LFW")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
        help="Zariadenie pre MTCNN (cuda/cpu)",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    mtcnn = create_mtcnn(args.device)
    print(f"MTCNN beží na: {args.device}")

    # --- CASIA-WebFace ---
    casia_src = base / "data" / "raw" / "cassia" / "casia-webface"
    casia_dst = base / "data" / "processed" / "casia"

    if casia_src.exists():
        print(f"\n=== CASIA-WebFace ===")
        print(f"Zdroj:  {casia_src}")
        print(f"Cieľ:   {casia_dst}")
        proc, skip = process_dataset(str(casia_src), str(casia_dst), mtcnn)
        print(f"Spracované: {proc}  |  Preskočené (tvár nenájdená): {skip}")
    else:
        print(f"CASIA dataset nenájdený: {casia_src}")

    # --- LFW ---
    lfw_src = base / "data" / "raw" / "lfw" / "lfw-deepfunneled"
    lfw_dst = base / "data" / "processed" / "lfw"

    if lfw_src.exists():
        print(f"\n=== LFW ===")
        print(f"Zdroj:  {lfw_src}")
        print(f"Cieľ:   {lfw_dst}")
        proc, skip = process_dataset(str(lfw_src), str(lfw_dst), mtcnn)
        print(f"Spracované: {proc}  |  Preskočené (tvár nenájdená): {skip}")
    else:
        print(f"LFW dataset nenájdený: {lfw_src}")


if __name__ == "__main__":
    main()
