import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.benchmark_cnn import BenchmarkCNN

# ──────────────────────────────────────────────
# Konfigurácia
# ──────────────────────────────────────────────
BATCH_SIZE = 128
EPOCHS = 30
LR = 0.1
DATA_DIR = "data/processed/casia"
CHECKPOINT_PATH = "models/checkpoints/benchmark_cnn.pth"
LOG_PATH = "results/training_log.csv"


def main():
    # ──────────────────────────────────────────────
    # Dataset a augmentácie
    # ──────────────────────────────────────────────
    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # ImageFolder automaticky priradí triedu podľa podpriečinka
    train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    num_classes = len(train_dataset.classes)

    print(f"Počet tried: {num_classes}")
    print(f"Počet obrázkov: {len(train_dataset)}")

    # ──────────────────────────────────────────────
    # Model, stratová funkcia, optimalizátor
    # ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BenchmarkCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Mixed precision - zrýchli tréning na GPU použitím float16 kde je to bezpečné
    scaler = torch.cuda.amp.GradScaler()

    # Načítanie checkpointu ak existuje (resume tréning)
    start_epoch = 1
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Pokračujem od epochy {start_epoch}")

    # Príprava log súboru
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log_exists = os.path.exists(LOG_PATH)
    log_file = open(LOG_PATH, "a", newline="")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(["epoch", "loss", "accuracy", "time_s"])

    # ──────────────────────────────────────────────
    # Trénovacia slučka
    # ──────────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(train_loader)
        epoch_start = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 100 == 0:
                elapsed = time.time() - epoch_start
                batch_acc = 100.0 * correct / total
                print(
                    f"  Batch {batch_idx}/{num_batches}  "
                    f"Loss: {loss.item():.4f}  Acc: {batch_acc:.2f}%  "
                    f"[{elapsed:.0f}s]"
                )

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch}/{EPOCHS}  Loss: {epoch_loss:.4f}  "
            f"Acc: {epoch_acc:.2f}%  Čas: {epoch_time:.0f}s"
        )

        scheduler.step()

        # Zapis metrík do CSV
        log_writer.writerow(
            [epoch, f"{epoch_loss:.4f}", f"{epoch_acc:.2f}", f"{epoch_time:.0f}"]
        )
        log_file.flush()

        # Uloženie checkpointu po každej epoche
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "num_classes": num_classes,
                "embedding_size": 512,
            },
            CHECKPOINT_PATH,
        )
        print(f"Checkpoint uložený (epocha {epoch})")

    log_file.close()
    print(f"Tréning dokončený. Model uložený do {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
