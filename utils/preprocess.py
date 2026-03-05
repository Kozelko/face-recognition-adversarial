"""
Predspracovanie datasetov CASIA-WebFace a LFW pomocou MTCNN.
MTCNN detekuje tvár, zarovná ju podľa očí a oreže na 112x112 pixelov.
Výsledky sa ukladajú do data/processed/ v rovnakej adresárovej štruktúre.

Používa multiprocessing — každý worker má vlastnú MTCNN inštanciu na CPU,
čo umožňuje paralelné spracovanie na viacerých jadrách.
"""

import argparse
import multiprocessing as mp
import os
from pathlib import Path

from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm


def _worker_init(device: str) -> None:
    """Inicializuje MTCNN v každom worker procese."""
    import torch

    torch.set_num_threads(1)  # každý worker 1 vlákno — zamedzí contention
    global _mtcnn
    _mtcnn = MTCNN(
        image_size=112,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        post_process=False,
        device=device,
    )


def _worker_process(args: tuple[str, str]) -> bool:
    """Spracuje jeden obrázok. Vracia True ak OK, False ak skip."""
    src_path, dst_path = args
    try:
        img = Image.open(src_path).convert("RGB")
    except Exception:
        return False

    face = _mtcnn(img)
    if face is None:
        return False

    face_img = Image.fromarray(face.permute(1, 2, 0).byte().numpy())
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    face_img.save(dst_path)
    return True


def process_dataset(
    src_root: str,
    dst_root: str,
    device: str,
    num_workers: int,
) -> tuple[int, int]:
    """Spracuje dataset - detekuje a oreže tváre.

    Returns:
        (processed, skipped) počet úspešne spracovaných a preskočených obrázkov.
    """
    src_path = Path(src_root)
    dst_path = Path(dst_root)

    # Zozbieraj všetky obrázky
    image_files: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_files.extend(src_path.rglob(ext))

    # Prefiltruj — spracuj len tie, čo ešte neexistujú
    todo: list[tuple[str, str]] = []
    already_done = 0
    for img_file in image_files:
        rel = img_file.relative_to(src_path)
        out_file = dst_path / rel
        if out_file.exists():
            already_done += 1
        else:
            todo.append((str(img_file), str(out_file)))

    if already_done:
        print(f"  Preskočených (už existujú): {already_done}")

    if not todo:
        return already_done, 0

    processed = already_done
    skipped = 0

    with mp.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(device,),
    ) as pool:
        results = pool.imap_unordered(_worker_process, todo, chunksize=32)
        for ok in tqdm(
            results, total=len(todo), desc=f"Processing {src_path.name}", unit="img"
        ):
            if ok:
                processed += 1
            else:
                skipped += 1

    return processed, skipped


def main() -> None:
    cpu_count = os.cpu_count() or 4

    parser = argparse.ArgumentParser(description="MTCNN face crop pre CASIA a LFW")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Zariadenie pre MTCNN (cuda/cpu). Pre multiprocessing sa odporúča cpu.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cpu_count - 2),
        help=f"Počet workerov (default: {max(1, cpu_count - 2)}, máš {cpu_count} jadier)",
    )
    args = parser.parse_args()

    print(f"MTCNN beží na: {args.device}  |  Workerov: {args.workers}")

    base = Path(__file__).resolve().parent.parent

    # --- CASIA-WebFace ---
    casia_src = base / "data" / "raw" / "cassia" / "casia-webface"
    casia_dst = base / "data" / "processed" / "casia"

    if casia_src.exists():
        print(f"\n=== CASIA-WebFace ===")
        print(f"Zdroj:  {casia_src}")
        print(f"Cieľ:   {casia_dst}")
        proc, skip = process_dataset(
            str(casia_src), str(casia_dst), args.device, args.workers
        )
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
        proc, skip = process_dataset(
            str(lfw_src), str(lfw_dst), args.device, args.workers
        )
        print(f"Spracované: {proc}  |  Preskočené (tvár nenájdená): {skip}")
    else:
        print(f"LFW dataset nenájdený: {lfw_src}")


if __name__ == "__main__":
    main()
