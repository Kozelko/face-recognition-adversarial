import torch.nn as nn


class ConvBlock(nn.Module):
    """Konvolučný blok: 2x (Conv -> BN -> PReLU) -> MaxPool.

    Dva konvolučné prechody extrahujú príznaky, MaxPool zmenší
    priestorový rozmer na polovicu.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # --- Prvý konvolučný prechod ---
        # 3x3 konvolúcia, padding=1 zachováva priestorový rozmer, bias=False lebo BN má vlastný bias
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        # Batch normalizácia - normalizuje výstupy pre stabilnejší a rýchlejší tréning
        self.bn1 = nn.BatchNorm2d(out_channels)
        # PReLU - aktivačná funkcia s učiteľným sklonom pre záporné hodnoty
        self.act1 = nn.PReLU(out_channels)

        # --- Druhý konvolučný prechod ---
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.PReLU(out_channels)

        # MaxPool 2x2 - vyberie maximálnu hodnotu z každého 2x2 okna, zmenší rozmer na polovicu
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))  # Conv1 -> BN1 -> PReLU1
        x = self.act2(self.bn2(self.conv2(x)))  # Conv2 -> BN2 -> PReLU2
        return self.pool(x)  # MaxPool 2x2


class BenchmarkCNN(nn.Module):
    """CNN na rozpoznávanie tvárí.

    Vstup: RGB obrázok 112x112.
    Feature extractor: 4 ConvBloky postupne zvyšujú počet kanálov
    (3→64→128→256→512) a zmenšujú priestorový rozmer na polovicu
    (112→56→28→14→7).
    """

    def __init__(self, num_classes, embedding_size=512):
        super().__init__()

        # --- Feature extractor: 4 konvolučné bloky ---
        self.features = nn.Sequential(
            ConvBlock(3, 64),  # 112×112×3   → 56×56×64
            ConvBlock(64, 128),  # 56×56×64    → 28×28×128
            ConvBlock(128, 256),  # 28×28×128   → 14×14×256
            ConvBlock(256, 512),  # 14×14×256   → 7×7×512
        )

        # --- Embedding hlava ---
        # Flatten roztiahne 512×7×7 feature mapu do vektora s 25 088 hodnotami
        self.flatten = nn.Flatten()
        # Lineárna vrstva premietne 25 088-rozmerný vektor na embedding požadovanej veľkosti
        self.fc = nn.Linear(512 * 7 * 7, embedding_size)
        # Batch normalizácia nad embeddingom - stabilizuje rozloženie embeddingov
        self.bn = nn.BatchNorm1d(embedding_size)

        # --- Klasifikačná hlava ---
        # Lineárna vrstva z embeddingu na počet tried (používa sa len pri trénovaní)
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x, return_embedding=False):
        x = self.features(x)  # Feature extractor: 112×112×3 → 7×7×512
        x = self.flatten(x)  # 7×7×512 → 25088
        x = self.fc(x)  # 25088 → embedding_size
        x = self.bn(x)  # Batch normalizácia embeddingu

        if return_embedding:
            return x

        return self.classifier(x)
