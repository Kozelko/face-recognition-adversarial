import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceModelWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError("Podtrieda musí implementovať metódu forward.")

class BenchmarkCNNWrapper(FaceModelWrapper):
    def __init__(self, num_classes=None, checkpoint_path=None, device="cpu"):
        super().__init__()
        from models.benchmark_cnn import BenchmarkCNN
        
        checkpoint = None
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if num_classes is None:
                state = checkpoint.get("model_state_dict", checkpoint)
                if isinstance(state, dict) and "classifier.weight" in state:
                    num_classes = state["classifier.weight"].shape[0]
                elif isinstance(checkpoint, dict) and "num_classes" in checkpoint:
                    num_classes = checkpoint["num_classes"]

        if num_classes is None:
            num_classes = 10572
        
        self.model = BenchmarkCNN(num_classes=num_classes)
        if checkpoint is not None:
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        
        self.model.to(device)
        self.model.eval()
        self.classes = checkpoint.get("classes", None) if (checkpoint and isinstance(checkpoint, dict)) else None

    def forward(self, x):
        emb = self.model(x, return_embedding=True)
        return F.normalize(emb, p=2, dim=1)

    def predict_identity(self, x):
        """Vráti predikciu triedy a pravdepodobnosti, ak má model finetuned klasifikačnú hlavu."""
        if not self.classes:
            return None
        x = x.to(next(self.model.parameters()).device)
        with torch.no_grad():
            logits = self.model(x, return_embedding=False)
            probs = F.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)
            label = self.classes[idx.item()]
            readable_label = "Peter Brandajský" if "brandajsky" in label or "peter" in label else label.replace("_", " ").title()
            return {
                "raw_label": label,
                "label": readable_label,
                "confidence": conf.item(),
                "probabilities": {self.classes[i]: probs[0, i].item() for i in range(len(self.classes))}
            }

class FaceNetWrapper(FaceModelWrapper):
        # Wrapper pre model FaceNet (InceptionResnetV1) z balíka facenet-pytorch.

    def __init__(self, pretrained="vggface2", device="cpu"):
        super().__init__()
        try:
            from facenet_pytorch import InceptionResnetV1
        except ImportError:
            raise ImportError("Nainštalujte balík: pip install facenet-pytorch")
        
        # InceptionResnetV1 defaultne očakáva na vstupe tensor v rozsahu <0, 1>
        # alebo normalizovaný podobne ako u nás: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        self.model = InceptionResnetV1(pretrained=pretrained).to(device)
        self.model.eval()

    def forward(self, x):
        # facenet-pytorch vracia embeddingy, ale pre porovnávanie ich ešte
        # L2 znormalizujeme, čo vylepšuje stabilitu kosínusovej podobnosti
        emb = self.model(x)
        return F.normalize(emb, p=2, dim=1)

class ArcFaceWrapper(FaceModelWrapper):
    """
    Wrapper pre ArcFace model (napr. z InsightFace).
    Používa štandardnú IResNet50 architektúru.
    """
    def __init__(self, model_path="models/checkpoints/arcface_iresnet50.pth", device="cpu"):
        super().__init__()
        import os
        from models.iresnet import iresnet50
        
        self.model = iresnet50(num_features=512)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            # Podpora pre rôzne formáty uloženia PyTorch checkpointov
            state_dict = checkpoint.get('state_dict', checkpoint)
            # Vyčistenie kľúčov v prípade DataParallel
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
            print(f"✅ ArcFace: Váhy úspešne načítané z {model_path}")
        else:
            print(f"⚠️ Upozornenie: ArcFace váhy nenájdené na '{model_path}'.")
            
        self.model = self.model.to(device)
        self.model.eval()

    def forward(self, x):
        # Zabezpečíme, že vstup je na rovnakom zariadení ako model
        x = x.to(next(self.model.parameters()).device)
        emb = self.model(x)
        return F.normalize(emb, p=2, dim=1)

class AdaFaceWrapper(FaceModelWrapper):
    """
    Wrapper pre model AdaFace.
    Využíva vlastnú implementáciu IResNet architektúry priamo z repozitára AdaFace.
    """
    def __init__(self, model_path="models/checkpoints/adaface_iresnet50.pth", device="cpu"):
        super().__init__()
        import os
        
        try:
            from models.adaface_net import build_model
            self.model = build_model('ir_50')
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=device)
                state_dict = checkpoint.get('state_dict', checkpoint)
                
                # AdaFace váhy začínajú prefixom 'model.'
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if 'model.' in k}
                self.model.load_state_dict(state_dict, strict=True)
                print(f"✅ AdaFace: Váhy úspešne načítané z {model_path}")
            else:
                print(f"⚠️ Upozornenie: AdaFace váhy nenájdené na '{model_path}'.")
                
            self.model = self.model.to(device)
            self.model.eval()
        except ImportError:
            print("⚠️ Upozornenie: Modul models.adaface_net nenájdený.")

    def forward(self, x):
        x = x.to(next(self.model.parameters()).device).contiguous()
        emb = self.model(x)
        if isinstance(emb, tuple):
            emb = emb[0] 
        return F.normalize(emb, p=2, dim=1)

# Príklad použitia:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, 3, 112, 112).to(device)
    
    # Pre test wrapperov
    try:
        facenet = FaceNetWrapper(device=device)
        emb = facenet(dummy_input)
        print("FaceNet embedding shape:", emb.shape)
    except Exception as e:
        print("FaceNet error:", e)
