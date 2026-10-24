# Stav plnenia úloh diplomovej práce (DP II / DP III)

### 1. Stav kľúčových úloh

1. ⏳ **Fyzický test pred kamerou (Print-to-Camera validácia):**
   - Softvér a hárok pripravené: [`results/biomesh_specpatch_sheet_A4.png`](file:///home/kozel/fiit/ING/diplomka/face-recognition-adversarial/results/biomesh_specpatch_sheet_A4.png) (300 DPI, 1:1 mierka).
   - Čaká na doplnenie farebnej náplne do tlačiarne alebo čiernobielu tlač na laserovej tlačiarni.
   - Po vytlačení: vystrihnúť, nalepiť na líca + bradu, otestovať live cez `app.py`.

2. ✅ **Presentation Attack Detection (PAD / FAS modul):**
   - Úspešne integrovaný **MiniFASNetV2** (váhy `weights/2.7_80x80_MiniFASNetV2.pth`).
   - Podporovaný v `models/pad_minifasnet.py`, v reálnom čase v `app.py` a ako penalizačný regularizér v `attacks/ensemble_hybrid_patch.py`.
   - Nameraný liveness bypass: **60.0 %** na celom datasete, až **97.7 %** na autorových fotografiách.

3. ✅ **Cross-Model Black-Box Transferabilita & Held-Out Benchmark:**
   - 4×4 transfer matica a Leave-One-Out (3 $\rightarrow$ 1): [`results/transfer_matrix.csv`](file:///home/kozel/fiit/ING/diplomka/face-recognition-adversarial/results/transfer_matrix.csv) (priemerný black-box transfer **75.0 %**).
   - Veľký Held-Out benchmark na nevidenom **AdaFace** ($N=25$ fotiek, 125 behov): [`results/heldout_large_transfer.csv`](file:///home/kozel/fiit/ING/diplomka/face-recognition-adversarial/results/heldout_large_transfer.csv).
   - Nárast ASR z **4.0 %** (Single ArcFace) na **44.0 %** (**BioMesh-SpecPatch**).
   - Detailná dokumentácia výsledkov zapísaná v [`RESULTS.md`](file:///home/kozel/fiit/ING/diplomka/face-recognition-adversarial/RESULTS.md).

4. ⏳ **Zápis výsledkov a metodiky do Typst práce (`vintage-fiit-thesis/main.typ`):**
   - Doplniť Kapitolu 3: formalizácia hybridného útoku BioMesh-SpecPatch (3D MediaPipe FaceMesh, EoT, Hardness-Aware GA, Input Diversity).
   - Doplniť Kapitolu 4: tabuľky ablačných experimentov, transferovej matice a PAD vyhodnotenia z `RESULTS.md`.

---

### 2. Najbližšie odporúčané kroky

1. **Pripraviť grafy / vizualizácie:**
   - Heatmapa transferovej matice 4×4.
   - Stĺpcový graf konfigurácií A až E na modeli AdaFace.
2. **Fyzická tlač a webkamera:**
   - Vytlačiť hárok a spraviť live screenshoty v `app.py` do prílohy práce.
3. **Zapracovanie do Typst textu práce.**