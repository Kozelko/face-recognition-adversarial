import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.sgd import SGD
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
CHECKPOINT_PATH = "models/checkpoints/benchmark_cnn_best_run1.pth"
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
    
    val_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    full_train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    full_val_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transforms)

    # 80% train, 20% validation split
    total_size = len(full_train_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = torch.utils.data.random_split(
        range(total_size), [train_size, val_size], generator=generator
    )

    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices.indices)
    val_dataset = torch.utils.data.Subset(full_val_dataset, val_indices.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    num_classes = len(full_train_dataset.classes)

    print(f"Počet tried: {num_classes}")
    print(f"Počet trénovacích obrázkov: {len(train_dataset)}")
    print(f"Počet validačných obrázkov: {len(val_dataset)}")

    # ──────────────────────────────────────────────
    # Model, stratová funkcia, optimalizátor
    # ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BenchmarkCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Mixed precision - zrýchli tréning na GPU použitím float16 kde je to bezpečné
    scaler = torch.cuda.amp.GradScaler()

    # Načítanie checkpointu ak existuje (resume tréning)
    start_epoch = 1
    best_val_acc = 0.0
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        print(f"Pokračujem od epochy {start_epoch}, najlepšia val_acc: {best_val_acc:.2f}%")

    # Príprava log súboru
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log_exists = os.path.exists(LOG_PATH)
    log_file = open(LOG_PATH, "a", newline="")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "time_s"])

    # ──────────────────────────────────────────────
    # Trénovacia a validačná slučka
    # ──────────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()

        # --- Trénovanie ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        num_batches = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            if batch_idx % 100 == 0:
                elapsed = time.time() - epoch_start
                batch_acc = 100.0 * train_correct / train_total
                print(
                    f"  Batch {batch_idx}/{num_batches}  "
                    f"Train Loss: {loss.item():.4f}  Train Acc: {batch_acc:.2f}%  "
                    f"[{elapsed:.0f}s]"
                )

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total

        # --- Validácia ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}% | "
            f"Čas: {epoch_time:.0f}s"
        )

        scheduler.step()

        # Zapis metrík do CSV
        log_writer.writerow(
            [epoch, f"{epoch_train_loss:.4f}", f"{epoch_train_acc:.2f}", f"{epoch_val_loss:.4f}", f"{epoch_val_acc:.2f}", f"{epoch_time:.0f}"]
        )
        log_file.flush()

        # Uloženie najlepšieho checkpointu
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "num_classes": num_classes,
            "embedding_size": 512,
            "best_val_acc": best_val_acc,
        }
        
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        # Ulož latest vždy
        torch.save(state, CHECKPOINT_PATH)
        
        # Ak je to zatiaľ najlepší model, ulož ho špeciálne
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            state["best_val_acc"] = best_val_acc
            best_model_path = CHECKPOINT_PATH.replace(".pth", "_best.pth")
            if "_best_best" in best_model_path:
                 best_model_path = best_model_path.replace("_best_best", "_best")
            torch.save(state, best_model_path)
            print(f"Nový najlepší model uložený! (val_acc: {best_val_acc:.2f}%)")

    log_file.close()
    print(f"Tréning dokončený. Najlepší model má val_acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
