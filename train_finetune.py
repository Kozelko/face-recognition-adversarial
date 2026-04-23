import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.benchmark_cnn import BenchmarkCNN

def run_finetuning(dataset_dir="data/custom_dataset", epochs=15, lr=0.001, progress=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Skontroluj, či dataset vôbec existuje a či má aspoň 1 triedu (osobu)
    if not os.path.exists(dataset_dir):
        return False, f"Zložka s datasetom {dataset_dir} neexistuje."
        
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    num_new_classes = len(classes)
    
    if num_new_classes < 2:
        # Pre klasifikáciu potrebujeme aspoň 2 triedy (alebo jednu novú + nejaké "neznáme" fotky, ale dajme pre jednoduchosť že musí mať aspoň 2)
        # Pre demo účely môžeme pridať "dummy" triedu, ale lepšie je nechať používateľa nafotiť aspoň 2 ľudí (napr. "Ja" a "Ostatní")
        pass # PyTorch zvládne aj 1 triedu, ale CrossEntropyLoss s 1 triedou nedáva zmysel.
        
    if num_new_classes == 0:
        return False, "V datasete nie sú žiadne osoby. Najprv nazbieraj nejaké fotky."

    # 2. Príprava DataLoaderu
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    try:
        dataset = datasets.ImageFolder(dataset_dir, transform=transform)
    except Exception as e:
        return False, f"Chyba pri načítaní datasetu: {e}"
        
    num_classes = len(dataset.classes)
    if num_classes < 2:
        return False, "Na klasifikáciu sú potrebné aspoň 2 rôzne osoby (zložky) v datasete."
        
    # Batch size menší, lebo máme málo fotiek
    batch_size = min(8, len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Načítanie pôvodného modelu
    checkpoint_path = "models/checkpoints/benchmark_cnn_best_run1.pth"
    if not os.path.exists(checkpoint_path):
        return False, f"Chýba pôvodný model {checkpoint_path}."
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    orig_num_classes = checkpoint.get("num_classes", 10575)
    
    model = BenchmarkCNN(num_classes=orig_num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if progress:
        progress(0.1, desc="Model načítaný. Zmrazujem vrstvy...")

    # 4. Zmrazenie vrstiev (Feature Extractor)
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = False
    for param in model.bn.parameters():
        param.requires_grad = False
        
    # 5. Nahradenie klasifikátora
    model.classifier = nn.Linear(512, num_classes).to(device)
    
    # 6. Optimalizátor len pre nový klasifikátor
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    
    if progress:
        progress(0.2, desc=f"Spúšťam trénovanie na {num_classes} triedach...")

    # 7. Trénovacia slučka
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = total_loss / total
        epoch_acc = 100. * correct / total
        
        if progress:
            progress(0.2 + 0.7 * (epoch / epochs), desc=f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.1f}%")
            
        time.sleep(0.1) # Aby sa to v UI aktualizovalo plynule
        
    # 8. Uloženie finetuned modelu
    finetuned_path = "models/checkpoints/benchmark_cnn_finetuned.pth"
    state = {
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "embedding_size": 512,
        "classes": dataset.classes # Uložíme si aj zoznam mien
    }
    torch.save(state, finetuned_path)
    
    if progress:
        progress(1.0, desc="Dokončené!")
        
    return True, f"Dotrénovanie úspešne dokončené! (Acc: {epoch_acc:.1f}%). Nový model bol uložený a je pripravený v záložke útokov."
