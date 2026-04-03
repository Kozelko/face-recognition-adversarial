import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceModelWrapper(nn.Module):
    """
    Spoločné rozhranie pre modely tvárovej biometrie.
    Každý wrapper zabezpečuje, že dopredný prechod (forward) 
    vráti L2-normalizovaný embedding (vektor príznakov).
    To nám zaručí, že adversariálne útoky budú môcť byť implementované
    univerzálne nad týmto rozhraním.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Vstup: Tenzor obrázkov tvaru (B, C, H, W).
               Predpokladáme, že vstupné obrázky sú normalizované pre daný model
               alebo použijeme transformáciu priamo tu.
        Výstup: L2-normalizovaný tenzor embeddingov tvaru (B, embedding_size).
        """
        raise NotImplementedError("Podtrieda musí implementovať metódu forward.")

class BenchmarkCNNWrapper(FaceModelWrapper):
    """
    Wrapper pre náš vlastný natrénovaný model (BenchmarkCNN).
    """
    def __init__(self, num_classes, checkpoint_path=None, device="cpu"):
        super().__init__()
        from models.benchmark_cnn import BenchmarkCNN
        
        self.model = BenchmarkCNN(num_classes=num_classes)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        
        self.model.to(device)
        self.model.eval() # Vždy používam modely v eval() móde pre útoky

    def forward(self, x):
        # Získame vnútorný embedding (pred klasifikačnou vrstvou)
        emb = self.model(x, return_embedding=True)
        # Normalizácia embeddingu (štandard v tvárovej biometrii)
        return F.normalize(emb, p=2, dim=1)

class FaceNetWrapper(FaceModelWrapper):
    """
    Wrapper pre model FaceNet (InceptionResnetV1) z balíka facenet-pytorch.
    """
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
    Wrapper pre ArcFace model z InsightFace (alebo inej implementácie).
    Momentálne pripravené ako štruktúra, bude sa dopĺňať po stiahnutí konkrétneho modelu.
    """
    def __init__(self, model_path=None, device="cpu"):
        super().__init__()
        # Zatiaľ placeholder, implementácia sa doplní na základe toho,
        # aké konkrétne váhy (ONNX / PyTorch pt) pre ArcFace použijem.
        pass

    def forward(self, x):
        raise NotImplementedError("ArcFaceWrapper ešte nie je úplne implementovaný.")

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
