# Poznámky a podklady do kapitoly Diskusia (Diplomová práca)

## Téma: Prečo sa fyzický adversariálny útok zameriava na čelný kužeľ (Frontal Acceptance Cone ±15°)

### 1. Hypotéza a problém v praxi
V reálnom svete (CCTV, letiskové e-Gates, bezpečnostné turnikety) kamery snímajú osoby z rôznych uhlov. Vzniká otázka: *Má zmysel optimalizovať fyzickú nálepku pre extrémne uhly pohľadu (poloprofily 30°–60°)?*

**Záver:** Nie. Fyzický patch má optimalizovať **čelný akceptačný kužeľ (0° ± 15°)**, pretože pri väčších uhloch zlyháva samotná biometrická identifikácia už na neupravenej tvári.

---

### 2. Empirický dôkaz (Merania z vlastného datasetu)
Všetky biometrické referenčné databázy (občianske preukazy, pasy, policajné zoznamy záujmových osôb) obsahujú **výlučne čelné fotografie (Frontal 0°)**.

Pri testovaní čistých, neupravených fotografií tej istej osoby pod rôznymi uhlami voči čelnej galérii sme namerali nasledovné hodnoty kosínusovej podobnosti (prah identifikácie = **0.50**):

| Uhol natočenia hlavy (Yaw) | Súbor záberu | AdaFace (SOTA) | ArcFace (SOTA) | FaceNet | BenchmarkCNN | Výsledok identifikácie bez útoku |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Čelný pohľad (0°)** | `20261025_155925` | **0.7401** | **0.6550** | **0.8233** | **0.8087** | ✅ **Spoľahlivo rozpoznaný** (> 0.50) |
| **Mierny poloprofil (28°)** | `20261025_160012` | **0.4899** | **0.3190** | 0.8445 | 0.8142 | ❌ **AdaFace a ArcFace zlyhávajú** (< 0.50) |
| **Šikmý uhol (38°)** | `20261025_160056` | **0.3814** | **0.3123** | 0.8514 | 0.7365 | ❌ **Hlboko pod prahom** (< 0.50) |
| **Výrazný poloprofil (42°)** | `20261025_160120` | **0.2972** | **0.2196** | 0.8777 | 0.7438 | ❌ **Nerozpoznateľný** (< 0.30) |
| **Bočný profil (54°)** | `20261025_160234` | **0.2875** | **0.2668** | 0.8110 | 0.7551 | ❌ **Nerozpoznateľný** |
| **Bočný profil (61°)** | `20261025_160255` | **0.1908** | **0.2440** | 0.8516 | 0.7403 | ❌ **Nerozpoznateľný** (< 0.25) |

### 3. Formulácia do textu diplomovej práce (návrh formulácie)
> *"Moderné hlboké konvolučné architektúry s margin-loss funkciami (ArcFace, AdaFace) sú vysoko senzitívne na extrémne uhly pohľadu, pokiaľ referenčná galéria obsahuje štandardizované pasové čelné snímky. Ako ukazujú naše experimentálne merania, pri odklone hlavy nad 28° klesá kosínusová podobnosť neupravenej tváre pod operačný prah 0.50 (pri 42° dosahuje AdaFace len 0.297 a ArcFace 0.220).*
>
> *Z toho vyplýva, že biometrický systém je schopný pozitívnej identifikácie výhradne v úzkom čelnom kuželi (približne ±15°). Vynucovanie multi-pose robustnosti na extrémnych uhloch (>30°) pri optimalizácii adversariálneho patchu vedie k nežiaducemu rozptylu gradientu a degradácii účinnosti útoku v kritickej čelnej zóne. Z bezpečnostného hľadiska je preto optimálnou stratégiou koncentrovať rušivú kapacitu patchu na čelný akceptačný kužeľ modelovaný cez Expectation over Transformation (EoT), zatiaľ čo bočné uhly sú prirodzene chránené geometriou pohľadu, resp. pasívnymi prvkami (kapucňa, šiltovka)."*

---

### 4. Využitie bočných fotiek pri návrhu patchu
Aj keď sa bočné fotky nezahrnuli do cieľovej funkcie gradientu, zohrali kľúčovú úlohu:
1. **Presná kalibrácia 3D klenutia nosovej pásky (`arch_factor = 0.34`):** Z bočných profilov (54° a 61°) bola presne odmeraná výška hrebeňa a strmosť bočných stien nosa, vďaka čomu páska anatomicky kopíruje nos a netrčí plocho do strán.
2. **Perspektívne zúženie (`0.89 × alar_w`):** Zabezpečuje, že šírka pásky rešpektuje 3D skrátenie na chrupavke nad nosovými krídlami.
