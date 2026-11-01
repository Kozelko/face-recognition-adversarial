# Face Recognition & Adversarial Attacks Platform

Platforma na testovanie a vyhodnocovanie modelov tvárovej biometrie voči digitálnym a fyzickým adverzným útokom (Breathe Right páska na nos + hydrokoloidné nálepky na líca).

---

## 🚀 Rýchly štart

### 1. Aktivácia prostredia
```bash
conda activate cnn-benchmark
```
*(V prípade nového prostredia: `conda env create -f environment.yml` a `pip install gradio pandas opencv-python facenet-pytorch`)*

### 2. Spustenie interaktívnej aplikácie
```bash
python app.py
```
Aplikácia beží lokálne na adrese: **`http://localhost:7860`** (alebo `http://127.0.0.1:7860`).

---

## 🖥️ Ako testovať v aplikácii (GUI)

Aplikácia má 2 hlavné záložky:

### Záložka 1: Testovanie útokov (Živá kamera / Fotka)
1. **Zdroj fotky:**
   - **Webkamera:** Stlač *Zapnúť kameru* -> *Odfotiť*. (Pri horšom svetle je k dispozícii tlačidlo *Prisvietenie tváre (Softbox)*).
   - **Nahrať súbor:** Môžeš nahrať ľubovoľnú uloženú fotografiu.
2. **Režimy testovania:**
   - **A) Digitálna simulácia útoku:**
     - Vyber typ útoku (napr. `BioMesh-SpecPatch (Ensemble)`), cieľový model a štýl nálepiek (`adversarial_rgb`, `athlete_black`, verzie s logom).
     - Klikni na **Spustiť útok**.
     - Vpravo sa zobrazí tvár po útoku, kosínusová podobnosť naprieč modelmi (prah `< 0.50` = úspešné oklamanie), liveness skóre (MiniFASNetV2) a v spodnej časti náhľad vygenerovaného tlačového hárku.
   - **B) Fyzické overenie vytlačených nálepiek:**
     - Nalep si vytlačenú pásku na nos a nálepky na líca.
     - V appke stačí spraviť fotku z webkamery.
     - Sekcia *Prehľad všetkých modelov* okamžite vyhodnotí podobnosť tvojej fyzickej tváre voči databáze.

### Záložka 2: Zber dát a Finetuning
- Slúži na registráciu vlastnej tváre do databázy (`data/custom_dataset/meno_priezvisko/`).
- Integrovaný MTCNN detektor automaticky vyrezáva a ukladá tváre (`112×112 px`, `margin=20`).
- Tlačidlom *Spustiť finetuning* sa spustí dotrénovanie klasifikačnej hlavy modelu `BenchmarkCNN (Finetuned)`.

---

## 🖨️ Fyzická tlač nálepiek na papier (A4, 300 DPI)

Tlačové hárky sa generujú priamo v aplikácii (Záložka 1) a ukladajú do lokálneho priečinka `printable_patches/` (ignorovaný gitom):

### Pokyny k tlači:
1. V dialógu tlače zvoľ **Mierka: 100 % / Skutočná veľkosť (Actual Size)** — nevypĺňať na celú stranu (Fit to page)!
2. Po vytlačení pravítkom over **kalibračné meradlo v hlavičke (musí mať presne 50,0 mm)**.
3. Tlač na matný samolepiaci fotopapier alebo bežný papier podlepený obojstrannou páskou / tyčinkovým lepidlom (neprelepovať lesklou páskou zvrchu kvôli odleskom).

---

## ⚙️ Dávkové spúšťanie cez terminál (CLI)

```bash
# Vyhodnotenie útokov na vybranej sade snímok
python evaluate_attacks.py

# Dávkové experimenty fyzických útokov (Séria 1 až 4 + transfer matica)
python run_patch_experiments.py --series 3,4,T

# Trénovanie bázového modelu BenchmarkCNN od nuly
python train.py

# Dotrénovanie vlastných identít (Transfer Learning)
python train_finetune.py
```

---

## 📁 Štruktúra repozitára

```text
face-recognition-adversarial/
├── app.py                     # Hlavná Gradio webová aplikácia (kamera, útoky, tlač)
├── printable_patches/         # Lokálne generované tlačové hárky A4 (v .gitignore)
├── attacks/                   # Implementácie adverzných útokov
│   ├── hybrid_patch.py        # 3D geometria, textúry, logo emblémy a generovanie hárku
│   ├── ensemble_hybrid_patch.py # Multi-model útok, feature dispersion, PAD hook
│   ├── nasal_strip_patch.py   # Breathe Right nosová páska
│   └── constellation_patch.py # Lícne hydrokoloidné nálepky
├── models/                    # Architektúry a wrappery modelov
│   ├── wrappers.py            # Zjednotené rozhranie (BenchmarkCNN, FaceNet, ArcFace, AdaFace)
│   ├── checkpoints/           # Váhy neurónových sietí (.pth)
│   └── pad_minifasnet.py      # MiniFASNetV2 detektor živosti tváre (PAD)
├── data/                      # Dátové priečinky (databáza tvárí, uložené snímky)
│   ├── custom_dataset/        # Registrované tváre
│   └── eval_captures/         # Uložené snímky z webkamery
└── results/                   # Výstupné CSV a grafy z experimentov
```
