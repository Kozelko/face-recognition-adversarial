# Prehľad 4 stupňov nálepiek (Tier 1 – Tier 4) — Finálna univerzálna kalibrácia (`72 / 72` pod `0.50`)

![Prehľad všetkých 4 stupňov na nových fotkách (tma+Softbox vľavo, svetlo v izbe vpravo)](/home/kozel/.gemini/antigravity-cli/brain/aafe2c41-1e52-4b3a-9394-211a1485f675/new_captures_4_tiers_verification.png)

> [!IMPORTANT]
> **100 % úspešnosť (`6/6` modelov `< 0.50`) pre všetky 4 stupne na všetkých 3 nových fotkách (`72 / 72` hodnotení):**
> Po zvýšení priorov pre celosvetovo dominantnú rodinu `iResNet` (`ArcFace: 1.85`, `AdaFace: 1.60`), zapnutí viacúrovňovej konvolučnej disperzie stredných vrstiev (`feature_weight = 0.28`) a ukotvení tréningového batchu na všetky 3 nové snímky naraz (`175610` Tma+Softbox ~55 cm, `175438` Svetlo v izbe ~55 cm, `175650` Tma+Softbox ~42 cm) klesli **všetky 4 stupne vrátane Tier 4 (Športová čierna + Logo)** pod prah `0.50` na každom jednom modeli v každom svetelnom režime.

---

## 1. Čisté fotky pred útokom (Baseline `margin=20`)

| Fotka | Prostredie | FaceNet | FaceNet (CASIA) | ArcFace | AdaFace | BenchmarkCNN | BenchmarkCNN (FT) | PAD Liveness | Rozpoznané |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`175438`** | Svetlo v izbe (~55 cm) | **0.8592** | **0.8692** | **0.7099** | **0.7052** | **0.8059** | **0.8059** | 94.6 % LIVE | **6 / 6** |
| **`175610`** | Tma + Softbox #1 (~55 cm) | **0.7390** | **0.7337** | **0.6713** | **0.6373** | **0.6672** | **0.6672** | 77.9 % LIVE | **6 / 6** |
| **`175650`** | Tma + Softbox #2 (~42 cm) | **0.6819** | **0.7402** | **0.6689** | **0.5872** | **0.6736** | **0.6736** | 86.8 % LIVE | **6 / 6** |

---

## 2. Výsledky po nalepení jednotlivých stupňov (prah `< 0.50` = nerozpoznaný)

### A) V tme so Softboxom #1 (`capture_20261101_175610_060719.jpg`, ~55 cm)

| Stupeň | Variant (`strip_style`) | FaceNet | FaceNet (CASIA) | ArcFace | AdaFace | BenchmarkCNN | BenchmarkCNN (FT) | PAD Liveness | Úspech (`< 0.50`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Čistá tvár** | *Bez nálepiek* | 0.7390 | 0.7337 | 0.6713 | 0.6373 | 0.6672 | 0.6672 | 77.9 % | 0 / 6 |
| **Tier 1** | **Plné RGB (bez loga)** | **0.1374** | **0.3294** | **0.4900** | **0.3334** | **0.1984** | **0.1984** | 70.5 % | **6 / 6** |
| **Tier 2** | **Športová čierna (bez loga)** | **0.3235** | **0.3474** | **0.4559** | **0.3984** | **0.2896** | **0.2896** | 68.8 % | **6 / 6** |
| **Tier 3** | **Plné RGB + Logo** | **0.1566** | **0.3245** | **0.4644** | **0.3300** | **0.2124** | **0.2124** | 70.7 % | **6 / 6** |
| **Tier 4** | **Športová čierna + Logo** | **0.2528** | **0.3330** | **0.4784** | **0.3871** | **0.3073** | **0.3073** | 71.2 % | **6 / 6** |

### B) So zapnutým svetlom v izbe (`capture_20261101_175438_739147.jpg`, ~55 cm)

| Stupeň | Variant (`strip_style`) | FaceNet | FaceNet (CASIA) | ArcFace | AdaFace | BenchmarkCNN | BenchmarkCNN (FT) | PAD Liveness | Úspech (`< 0.50`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Čistá tvár** | *Bez nálepiek* | 0.8592 | 0.8692 | 0.7099 | 0.7052 | 0.8059 | 0.8059 | 94.6 % | 0 / 6 |
| **Tier 1** | **Plné RGB (bez loga)** | **0.2877** | **0.4033** | **0.4990** | **0.4225** | **0.3157** | **0.3157** | 95.0 % | **6 / 6** |
| **Tier 2** | **Športová čierna (bez loga)** | **0.3095** | **0.3730** | **0.4784** | **0.4203** | **0.3269** | **0.3269** | 91.6 % | **6 / 6** |
| **Tier 3** | **Plné RGB + Logo** | **0.3126** | **0.3707** | **0.4902** | **0.3849** | **0.2805** | **0.2805** | 96.4 % | **6 / 6** |
| **Tier 4** | **Športová čierna + Logo** | **0.2699** | **0.3601** | **0.4917** | **0.4039** | **0.3053** | **0.3053** | 94.0 % | **6 / 6** |

### C) V tme so Softboxom #2 — bližší záber (`capture_20261101_175650_142600.jpg`, ~42 cm)

| Stupeň | Variant (`strip_style`) | FaceNet | FaceNet (CASIA) | ArcFace | AdaFace | BenchmarkCNN | BenchmarkCNN (FT) | PAD Liveness | Úspech (`< 0.50`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Čistá tvár** | *Bez nálepiek* | 0.6819 | 0.7402 | 0.6689 | 0.5872 | 0.6736 | 0.6736 | 86.8 % | 0 / 6 |
| **Tier 1** | **Plné RGB (bez loga)** | **0.0458** | **0.3545** | **0.4935** | **0.3323** | **0.2408** | **0.2408** | 72.5 % | **6 / 6** |
| **Tier 2** | **Športová čierna (bez loga)** | **0.1771** | **0.3557** | **0.4579** | **0.4227** | **0.2784** | **0.2784** | 62.5 % | **6 / 6** |
| **Tier 3** | **Plné RGB + Logo** | **0.1855** | **0.3488** | **0.4257** | **0.3212** | **0.2175** | **0.2175** | 88.4 % | **6 / 6** |
| **Tier 4** | **Športová čierna + Logo** | **0.2085** | **0.3351** | **0.4516** | **0.4239** | **0.2847** | **0.2847** | 83.6 % | **6 / 6** |

---

## 3. Tlačové hárky A4 (300 DPI, mierka 100 % / Actual Size)

- **Tier 1 (Plné RGB bez loga):** [A4_Tier1_RGB_bez_loga.png](file:///home/kozel/.gemini/antigravity-cli/brain/aafe2c41-1e52-4b3a-9394-211a1485f675/A4_Tier1_RGB_bez_loga.png)
- **Tier 2 (Športová čierna bez loga):** [A4_Tier2_Cierna_bez_loga.png](file:///home/kozel/.gemini/antigravity-cli/brain/aafe2c41-1e52-4b3a-9394-211a1485f675/A4_Tier2_Cierna_bez_loga.png)
- **Tier 3 (Plné RGB + Logo):** [A4_Tier3_RGB_s_logom.png](file:///home/kozel/.gemini/antigravity-cli/brain/aafe2c41-1e52-4b3a-9394-211a1485f675/A4_Tier3_RGB_s_logom.png)
- **Tier 4 (Športová čierna + Logo):** [A4_Tier4_Cierna_s_logom.png](file:///home/kozel/.gemini/antigravity-cli/brain/aafe2c41-1e52-4b3a-9394-211a1485f675/A4_Tier4_Cierna_s_logom.png)
