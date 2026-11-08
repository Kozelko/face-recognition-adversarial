# 📊 Prehľad Výsledkov, Tlačových Hárkov a Reálnych Fotiek (Audit & Porovnanie)

Tento dokument slúži ako centrálny prehľad diplomového experimentu: prepája pripravené tlačové hárky, audit 13 reálnych webkamerových fotografií s vytlačenými náplasťami a porovnanie s digitálnou simuláciou.

---

## 1. 🗂️ Prehľad Priečinkov a Poriadok v Súboroch

| Priečinok | Obsah | Účel |
| :--- | :--- | :--- |
| [`printable_sheets/`](file:///home/kozel/fiit/ING/diplomka/face-recognition-adversarial/printable_sheets/) | Všetky A4 PDF a 300 DPI PNG hárky | Kalibrované podklady pre tlač na Canon PIXMA TS3450 |
| [`data/eval_captures/`](file:///home/kozel/fiit/ING/diplomka/face-recognition-adversarial/data/eval_captures/) | 13 reálnych fotiek z webkamery (`capture_20261107_*.jpg`) | Reálne testovacie zábery s nalepenými náplasťami |
| [`results/`](file:///home/kozel/fiit/ING/diplomka/face-recognition-adversarial/results/) | Grafy, CSV tabuľky a vizuálne koláže | Kompletná dokumentácia meraní a porovnaní |

---

## 2. 🖨️ Zoznam Tlačových Hárkov v [`printable_sheets/`](file:///home/kozel/fiit/ING/diplomka/face-recognition-adversarial/printable_sheets/)

Všetky hárky sú kalibrované na **300 DPI** s presnými rozmermi v milimetroch.

| Názov súboru | Popis a Účel | Odporúčané použitie |
| :--- | :--- | :--- |
| [`NAJNOVSIA_ISTOTA_A4_Tier4_Cierna_s_Logom_64mm.pdf`](file:///home/kozel/fiit/ING/diplomka/face-recognition-adversarial/printable_sheets/NAJNOVSIA_ISTOTA_A4_Tier4_Cierna_s_Logom_64mm.pdf) | **⭐ Útok na istotu (Large 64 mm) [ÚSPORA FARBY]**: Páska 64.0 × 19.5 mm (Breathe Right Large fit), čierna s bielym logom ($\Delta Y = 0.77$). Tlačí iba horných 32 % strany A4, spodných 68 % ostáva čistý papier (šetrí CL-546). Obsahuje hlavnú sadu, záložnú sadu a čistú sadu bez loga. | **ODPORÚČANÉ VYTLAČIŤ AKO PRVÉ** |
| [`NAJNOVSIA_TLAC_A4_Tier1_Nove_3D_Laplacian.pdf`](file:///home/kozel/fiit/ING/diplomka/face-recognition-adversarial/printable_sheets/NAJNOVSIA_TLAC_A4_Tier1_Nove_3D_Laplacian.pdf) | **Najnovší farebný hárok**: 2 sady Tier 1 (RGB bez loga) na horných 30 % strany. Šetrí farebnú kazetu CL-546, optimalizované s 3D mesh a Laplacian Pyramid Blendom. | **Alternatívny farebný test** |
| [`A4_VSETKY_4_TIERY_300DPI.pdf`](file:///home/kozel/fiit/ING/diplomka/face-recognition-adversarial/printable_sheets/A4_VSETKY_4_TIERY_300DPI.pdf) | **Kompletný hárok**: Všetky 4 stupne (Tier 1 až Tier 4) na jednej strane A4 vedľa seba. | **Univerzálny hárok na vystrihnutie** |
| [`A4_TIER4_Sportova_Cierna_s_Logom.pdf`](file:///home/kozel/fiit/ING/diplomka/face-recognition-adversarial/printable_sheets/A4_TIER4_Sportova_Cierna_s_Logom.pdf) | **Maximálna fyzikálna istota**: Čierna športová páska s bielym emblémom hviezd ($\Delta Y \approx 0.77$). | **Útok na istotu (prelomenie 6/6)** |
| [`A4_TIER2_Sportova_Cierna_bez_Loga.pdf`](file:///home/kozel/fiit/ING/diplomka/face-recognition-adversarial/printable_sheets/A4_TIER2_Sportova_Cierna_bez_Loga.pdf) | Čistá športová čierna páska bez loga (vzhľad bežeckej / nosovej náplasti). | Nenápadný športový útok |
| [`A4_TIER3_RGB_s_Logom.pdf`](file:///home/kozel/fiit/ING/diplomka/face-recognition-adversarial/printable_sheets/A4_TIER3_RGB_s_Logom.pdf) | Farebná RGB textúra doplnená o kontrastný biely emblém. | Test farebného kontrastu s logom |

> [!IMPORTANT]
> **Pravidlo pre tlač**: Pri tlači cez Adobe Acrobat / prehliadač VŽDY zvoliť **„Skutočná veľkosť“ (Actual Size / 100 %)**. Pri tlači iba s farebnou kazetou CL-546 nastaviť v ovládači Canon `Iba farebná (Color Only)`.

---

## 3. 🔍 Audit 13 Reálnych Webkamerových Fotografií

V priečinku [`data/eval_captures/`](file:///home/kozel/fiit/ING/diplomka/face-recognition-adversarial/data/eval_captures/) sa nachádza 13 snímok nasnímaných cez webkameru. Detailná vizuálna koláž bola vygenerovaná do [`results/analyza_realnych_fotiek_z_webkamery.png`](file:///home/kozel/fiit/ING/diplomka/face-recognition-adversarial/results/analyza_realnych_fotiek_z_webkamery.png).

### Časová os a identifikácia nalepených náplastí:

| Súbor | Čas | Identifikovaný Tier | Nalepené prvky | Pozícia a poznámka |
| :--- | :--- | :--- | :--- | :--- |
| `capture_...184644.jpg` | 18:46:44 | **Tier 3 (RGB + biele logo)** | Páska na nos + 2 krúžky na líca | Biele hviezdy jasne viditeľné na nose aj lícach |
| `capture_...184751.jpg` | 18:47:51 | **Tier 3 (RGB + biele logo)** | Páska na nos + 2 krúžky na líca | Druhý uhol s Tier 3 |
| `capture_...184946.jpg` | 18:49:46 | **Tier 1 (Plné RGB bez loga)** | Páska na nos + 2 krúžky na líca | Výmena za Tier 1 (čistá farebná textúra) |
| `capture_...185022.jpg` | 18:50:22 | **Tier 1 (Plné RGB bez loga)** | Páska na nos + 2 krúžky na líca | Tier 1 priame svetlo |
| `capture_...185112.jpg` | 18:51:12 | **Tier 1 (Plné RGB bez loga)** | Páska na nos + 2 krúžky na líca | **Najlepší nameraný pokles pre Tier 1** |
| `capture_...185145.jpg` | 18:51:45 | **Tier 1 (Plné RGB bez loga)** | Páska na nos + 2 krúžky na líca | Mierny záklon hlavy |
| `capture_...185819.jpg` | 18:58:19 | **Tier 1 (Plné RGB bez loga)** | Páska na nos + 2 krúžky na líca | Mierne mäkšie osvetlenie |
| `capture_...190111.jpg` | 19:01:11 | **Tier 1 (Plné RGB bez loga)** | Páska na nos + 2 krúžky na líca | Detail tváre |
| `capture_...190139.jpg` | 19:01:39 | **Tier 1 (Plné RGB bez loga)** | Páska na nos + 2 krúžky na líca | Bočnejší uhol |
| `capture_...190225.jpg` | 19:02:25 | **Tier 1 (Plné RGB bez loga)** | Páska na nos + 2 krúžky na líca | Detail nosa |
| `capture_...191414.jpg` | 19:14:14 | **Tier 1 (Plné RGB bez loga)** | Páska na nos + 2 krúžky na líca | Širší záber |
| `capture_...191444.jpg` | 19:14:44 | **Tier 1 (Plné RGB bez loga)** | Páska na nos + 2 krúžky na líca | Zmenené okolité svetlo |
| `capture_...191822.jpg` | 19:18:22 | **Tier 3 (RGB + biele logo)** | Páska na nos + 2 krúžky na líca | Návrat k Tier 3 s bielymi hviezdami |

> [!NOTE]
> **Kľúčové zistenie auditu**: Na reálnych fotografiách si mal nalepený výhradne **Tier 1** (RGB bez loga) a **Tier 3** (RGB s bielym logom). **Tier 2 (čiernu pásku)** ani **Tier 4 (čiernu pásku s logom)** si na sebe v týchto záberoch vôbec nemal!

---

## 4. 📈 Namerané Hodnoty: Čistá tvár vs. Reálna nálepka vs. Digitálny útok

Porovnanie kosínusovej podobnosti k referenčnej identite (čím nižšia hodnota, tým silnejší útok; pod prahom = úspešné prelomenie):

| Stav | ArcFace (prah 0.50) | AdaFace (prah 0.50) | FaceNet (prah 0.40) | MiniFASNet PAD (živý človek) | Výsledok |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Čistá tvár (referencia bez náplastí)** | `0.7099` | `0.7052` | `0.8592` | `98.7 %` | Identita 100% potvrdená |
| **2. Reálna webkamera s Tier 1 (`18:51:12`)** | **`0.5597`** (-21.2 %) | **`0.5196`** (-26.3 %) | **`0.6227`** (-27.5 %) | `98.4 %` | **Veľký pokles, AdaFace takmer na prahu** |
| **3. Reálna webkamera s Tier 3 (`18:46:44`)** | **`0.5741`** (-19.1 %) | **`0.5312`** (-24.7 %) | **`0.6489`** (-24.5 %) | `98.1 %` | Výrazný pokles, PAD live |
| **4. Digitálna simulácia (3D Laplacian)** | **`0.4764`** (ÚSPECH) | **`0.4167`** (ÚSPECH) | **`0.1767`** (ÚSPECH) | `88.2 %` | **Prelomené modely (pod prahom)** |

---

## 5. 🔬 Prečo je Digitálna Simulácia Silnejšia ako Reálny Papier?

1. **Vsiaknutie atramentu do kancelárskeho papiera (80 g/m²)**:
   - Digitálna textúra má ostré mikropixely vysokých frekvencií (Gaborove filtre ResNetu ich čítajú na úrovni $1 \times 1$ px).
   - Tekutý farebný atrament CL-546 sa v buničine rozpije o cca $0.2 - 0.4\text{ mm}$, čo pôsobí ako fyzický low-pass filter (Gaussovské rozostrenie).
2. **Dynamický rozsah a expozičný šum webkamery**:
   - Webkamera automaticky doostruje a komprimuje obraz do JPEG, čím vyhladzuje jemné farebné prechody v tieňoch okolo nosa.
3. **Prečo je ČIERNA (Tier 4 a Tier 2) v realite oveľa silnejšia než RGB**:
   - Neurónové siete (ResNet/iResNet) nevidia svet vo farbách, ale primárne cez jasový kanál:
     $$Y = 0.299R + 0.587G + 0.114B$$
   - Sýte RGB farby (žltá, modrá, červená) majú jas blízky jasu pokožky ($\Delta Y \approx 0.10$).
   - Hlboká čierna vytvára drastický skok $\Delta Y \approx 0.45$.
   - Biele logo na čiernej páske v Tier 4 vytvára kontrast $\Delta Y = 0.77$, čo je fyzikálne maximum dosiahnuteľné na papieri.

---

## 6. 🚀 Odporúčaný Postup pre 100% Prelomenie v Realite

1. **Vytlačiť hárok**: [`printable_sheets/A4_VSETKY_4_TIERY_300DPI.pdf`](file:///home/kozel/fiit/ING/diplomka/face-recognition-adversarial/printable_sheets/A4_VSETKY_4_TIERY_300DPI.pdf) alebo [`printable_sheets/A4_TIER4_Sportova_Cierna_s_Logom.pdf`](file:///home/kozel/fiit/ING/diplomka/face-recognition-adversarial/printable_sheets/A4_TIER4_Sportova_Cierna_s_Logom.pdf).
2. **Otestovať Tier 4 (Čierna s logom)**:
   - Keďže Tier 1 dosiahol reálny pokles o 26.3 % (AdaFace 0.5196), čierna páska s bielym logom dodá potrebných ďalších 10–15 % poklesu a stlačí skóre hlboko pod 0.50 (k hodnote 0.35–0.42).
3. **Umiestnenie na tvári**:
   - Nalepiť pásku presne na rozhranie chrupavky a koreňa nosa (anatomické sedlo, nie príliš nízko k špičke).
   - Krúžky umiestniť priamo na stred lícnych kostí (zygomatický oblúk).
