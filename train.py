import torch
import torch.nn as nn


class SimpleFaceCNN(nn.Module):
    """
    Jednoduchý 'benchmark' CNN model pre extrakciu príznakov a rozpoznávanie tvárí.
    Vstup: RGB obrázok, ideálne s predvykrojenou (cropped) tvárou, veľkosť 3x112x112.
    Výstup: Vektor príznakov / embeddings (zhluky), resp. klasifikačné skóre.
    """

    def __init__(self, num_classes: int = 100, embedding_dim: int = 128) -> None:
        super().__init__()

        # Konvolučné vrstvy
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv3: nn.Conv2d = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )

        # Pooling a aktivácia
        self.pool: nn.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu: nn.ReLU = nn.ReLU()

        # Ak je na vstupe tensor (batch, 3, 112, 112):
        # pool 1 (112 -> 56)
        # pool 2 (56 -> 28)
        # pool 3 (28 -> 14)
        # Spojené dimenzie (128 kanálov, 14 x 14 grid) -> 128*14*14
        self.fc_features: nn.Linear = nn.Linear(128 * 14 * 14, embedding_dim)

        # Finálna vrstva pre priamu klasifikáciu osôb počas trénovania
        self.fc_classifier: nn.Linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Extrakcia príznakov
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        # Sploštenie všetkých priestorových dimenzií do jedného vektora pre plne prepojenú vrstvu
        x = x.view(-1, 128 * 14 * 14)

        # Embedding vo vrstve fc_features reprezentuje abstraktné informácie o tvári
        embeddings: torch.Tensor = self.fc_features(x)
        features_activated: torch.Tensor = self.relu(embeddings)

        # Skóre pre zhodnotenie tváre s danými triedami (z osobnostnej databázy napr. 100 osôb)
        logits: torch.Tensor = self.fc_classifier(features_activated)

        # Ak chceme získať len črty na porovnanie dvoch identít (napr. Cosine Similarity), používame 'embeddings'
        return embeddings, logits


# Overenie priamym spustením
if __name__ == "__main__":
    model = SimpleFaceCNN(num_classes=50)  # Predpokladáme dataset s 50 ľuďmi

    # Dummy prechod obrázka: (1 obrázok, 3 kanály-RGB, šírka 112, výška 112)
    dummy_input = torch.randn(1, 3, 112, 112)
    embeddings, logits = model(dummy_input)

    print("Vytvorený model: SimpleFaceCNN")
    print(
        f"Výstupný rozmer pre embedding: {tuple(embeddings.shape)} (Očakávané: [1, 128])"
    )
    print(f"Výstupný rozmer pre triedy: {tuple(logits.shape)} (Očakávané: [1, 50])")
