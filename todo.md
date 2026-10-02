 ### Čo ti ešte CHÝBA

  1. ❌ Fyzický test pred kamerou (Print-to-Camera validácia):
      • Fyzicky vytlačiť hárok, nalepiť na tvár a nasnímať reálny pád podobnosti cez webkameru.
  2. ❌ Presentation Attack Detection (PAD / FAS modul):
      • V práci máš hypotézu (Téza C a D), že hybridný útok oklame aj PAD systémy (liveness). V kóde zatiaľ PAD model
      chýba (treba integrovať napr. MiniFASNet / Silent-Face-Anti-Spoofing).
  3. ❌ Cross-Model Black-Box Transferabilita:
      • Spustiť dávkový test: útok vygenerovaný na BenchmarkCNN otestovať na ArcFace a AdaFace bez gradientov (3 × 3
      matica úspešnosti).
  4. ❌ Dopísanie textu v Typst (main.typ):
      • Text je momentálne v stave DP1 (iba digitálne útoky). Treba doplniť Kapitolu 3 a 4 o fyzické a hybridné útoky.

  ──────
  ### Čo by si mal spraviť NAJSKÔR (Odporúčané poradie)

  1. NAJSKÔR (Dnes / Zajtra): Fyzický Print-to-Camera test
      • Vytlač printable_patch_sheet_last.png na samolepiaci matný papier (alebo bežný papier + lepidlo/obojstranná
      páska).
      • Vystrihni, nalep na nos a líca, sadni pred webkameru v app.py a klikni na 🔍 Rozpoznať tvár.
      • Sprav screenshot – máš hotový reálny dôkaz funkčnosti pre školiteľa.
  2. POTOM (Krok 2): Dávkový skript na Transferabilitu
      • Spustíme automatický skript, ktorý vygeneruje tabuľku prenosnosti útokov medzi modelmi (máme na to všetky
      modely aj funkcie pripravené).
  3. POTOM (Krok 3): Zápis výsledkov do textu práce
      • Do Kapitoly 3 dopísať vzorce a architektúru hybridného útoku.
      • Do Kapitoly 4 vložiť tabuľku nameraných výsledkov a porovnanie digitálnych vs. fyzických útokov.
  4. NA ZÁVER (Krok 4): Integrácia PAD (Anti-Spoofing)
      • Stiahnuť váhy pre ľahký model MiniFASNet a doplniť ho do app.py ako prepínač "Overiť živosť (PAD)".