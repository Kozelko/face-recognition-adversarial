#import "@preview/vintage-fiit-thesis:1.1.0": *

#show: fiit-thesis.with(
  title: "Odolnosť modelov AI pre tvárovú biometriu voči adversariálnym útokom",
  thesis: "dp1",
  author: "Bc. Peter Brandajský",
  supervisor: "Mgr. Ing. Emma Macháčová",
  abstract: (
    sk: [Táto priebežná správa o riešení diplomovej práce (DP1) sa zaoberá hodnotením odolnosti modelov umelej inteligencie pre tvárovú biometriu voči adversariálnym útokom. V teoretickej časti systematicky analyzujeme princípy rozpoznávania tvárí, architektúry hlbokých konvolučných sietí a matematické formulácie strát. Zameriavame sa na porovnanie vzdialenostných a margin-based stratových funkcií. V praktickej časti navrhujeme a implementujeme komplexnú testovaciu platformu s jednotným rozhraním pre integráciu modelov FaceNet, ArcFace, AdaFace a vlastného BenchmarkCNN. Implementujeme a vyhodnocujeme päť white-box digitálnych útokov (FGSM, PGD, BIM, MI-FGSM a C&W) na vzorke 2000 snímok z datasetu CASIA-WebFace. Výsledky ukazujú takmer 100 % úspešnosť iteratívnych útokov, pričom C&W útok preukazuje výrazne vyššiu vizuálnu neviditeľnosť za cenu vyššej výpočtovej náročnosti. Priebežné výsledky vytvárajú robustný základ pre návrh nového hybridného a adaptívneho útoku v ďalších semestroch.],
    en: [This interim report on the master's thesis (DP1) addresses the robustness evaluation of artificial intelligence models for facial biometrics against adversarial attacks. In the theoretical part, we systematically analyze the principles of face recognition, deep convolutional neural network architectures, and mathematical loss formulations. We focus on comparing distance-based and margin-based loss functions. In the practical part, we design and implement a comprehensive evaluation platform with a unified interface for integrating FaceNet, ArcFace, AdaFace, and our custom BenchmarkCNN. We implement and evaluate five white-box digital attacks (FGSM, PGD, BIM, MI-FGSM, and C&W) on a sample of 2000 images from the CASIA-WebFace dataset. The results indicate nearly 100% success rate for iterative attacks, with the C&W attack demonstrating significantly higher visual imperceptibility at the cost of increased computational complexity. These preliminary findings establish a solid foundation for designing a new hybrid and adaptive attack in subsequent semesters.],
  ), // abstract
  id: "FIIT-12345-123456",
  lang: "sk", // this controls how the layout is presented, be careful!
  // remove the argument or made the value none to hide
  acknowledgment: [I would like to thank my supervisor for all the help and
    guidance I have received. I would also like to thank my friends and family
    for supporting during this work.],
  // remove the argument or leave the array empty to hide the list of
  // abbreviations
  abbreviations-outline: (
    ("AI", [Umelá inteligencia (_angl._ Artificial Intelligence)]),
    ("CNN", [Konvolučná neurónová sieť (_angl._ Convolutional Neural Network)]),
    ("SOTA", [Najnovšie metódy (_angl._ State-of-the-Art)]),
    ("FGSM", [Metóda rýchleho znaku gradientu (_angl._ Fast Gradient Sign Method)]),
    ("PGD", [Projected Gradient Descent]),
    ("BIM", [Basic Iterative Method]),
    ("MI-FGSM", [Momentum Iterative FGSM]),
    ("C&W", [Carlini & Wagner L2 útok]),
    ("PAD", [Presentation Attack Detection]),
    ("FAS", [Face Anti-Spoofing]),
    ("MTCNN", [Multi-task Cascaded Convolutional Networks]),
    ("LFW", [Labeled Faces in the Wild dataset]),
    ("SGD", [Stochastic Gradient Descent]),
    ("EoT", [Expectation over Transformation]),
    ("GPU", [Grafická procesorová jednotka (_angl._ Graphics Processing Unit)]),
    ("VRAM", [Video Random Access Memory]),
    ("CSV", [Comma-Separated Values]),
    ("ASR", [Miera úspešnosti útoku (_angl._ Attack Success Rate)]),
  ),
  figures-outline: true,
  tables-outline: true,
  style: "regular",
)

#set par(first-line-indent: (amount: 1.5em, all: true))
#show heading: set par(first-line-indent: 0pt)
= Úvod <introduction>

V súčasnosti zažívajú systémy automatického rozpoznávania tvárí (založené na hlbokom učení) masívny rozmach v komerčnej sfére, bankovníctve i v štátnych bezpečnostných zložkách. Hoci moderné State-of-the-Art (SOTA) modely dosahujú presnosť presahujúcu 99 %, ich integrácia do kritických aplikácií so sebou prináša zásadné bezpečnostné riziká. Jedným z najzávažnejších zraniteľných miest deep-learningových modelov je ich náchylnosť na *adversariálne útoky* – zámerne navrhnuté, pre človeka často nepatrné perturbácie vstupných dát, ktoré dokážu úplne skresliť výstupy neurónovej siete.

Hlavným cieľom tejto diplomovej práce je vyhodnotenie a komparatívna analýza robustnosti modelov umelej inteligencie pre tvárovú biometriu voči spektru digitálnych adversariálnych hrozieb, a následný návrh nového obranného a útočného mechanizmu. Na usmernenie nášho výskumu pracujeme so štyrmi hlavnými vedeckými tézami a pridruženými výskumnými otázkami:

*Téza A: Open-source modely rozpoznávania tváre*
- *Téza:* Open-source modely rozpoznávania tváre (FaceNet, ArcFace) vykazujú výraznú zraniteľnosť voči digitálnym adversariálnym útokom, čo potvrdzujú súčasné hodnotenia v realistických podmienkach.
- *Výskumné otázky:*
  1. Ako ovplyvňujú štandardné digitálne útoky (FGSM, PGD) presnosť týchto modelov?
  2. Aké sú ich limity a prenosnosť v black-box scenároch?

*Téza B: Vlastné modely s jednoduchšou architektúrou*
- *Téza:* Jednoduchšie vlastné modely sú náchylnejšie na adversariálne útoky než komplexné open-source riešenia, no odhaľujú špecifické slabiny v praktickom nasadení.
- *Výskumná otázka:* Aký je rozdiel v odolnosti vlastného BenchmarkCNN modelu oproti open-source benchmarkom pri rovnakej sadbe útokov?

*Téza C: Fyzické a presentation attacks (DP II / DP III)*
- *Téza:* Fyzické útoky (okuliare, masky, print attacks) dosahujú vysokú úspešnosť na všetkých typoch modelov, najmä v reálnych podmienkach s variabilným osvetlením.
- *Výskumné otázky:*
  1. Ako sa líši efektivita fyzických útokov medzi open-source a vlastnými modelmi?
  2. Dokážu presentation attack detection (PAD) mechanizmy tieto útoky spoľahlivo detegovať?

*Téza D: Nový hybridný adversariálny útok (DP II / DP III)*
- *Téza:* Kombinácia digitálnych perturbácií s fyzickými prvkami umožní vytvoriť efektívny útok, ktorý prekoná existujúce obrany v jednotnom experimentálnom prostredí.
- *Výskumné otázky:* Dokáže navrhnutý hybridný útok prelomiť pokročilé obranné mechanizmy ako adversariálne trénovanie a PAD?

*Jasné vymedzenie rozsahu práce a semestrálnych etáp:*
Pre predchádzanie nedorozumeniam v celkovom rozsahu riešenia striktne rozdeľujeme ciele, ktoré boli splnené v rámci tohto semestra (DP I), a ciele plánované do nasledujúceho výskumu:
1. *Súčasný stav (Tento semester - DP I):* Práca sa sústreďuje primárne na teoretický prehľad problematiky, analýzu chybových funkcií a vybudovanie robustného testovacieho prostredia (bázy pre experimenty). Praktickým výstupom je Python/PyTorch platforma s implementáciou 5 white-box digitálnych útokov (FGSM, PGD, BIM, MI-FGSM, C&W L2), integrácia troch SOTA modelov s vlastným BenchmarkCNN baseliningom a realizácia systematických digitálnych testov na podmnožine 2000 vzoriek z CASIA-WebFace.
2. *Budúci vývoj (Nasledujúce semestre - DP II a DP III):* Kým v tejto fáze sa zameriavame na čisto digitálne útoky, v nasledujúcich etapách rozšírime experimentálne prostredie o *fyzické útoky* (napr. potlačené rámy okuliarov, EOT transformácie) a *Presentation Attack Detection (PAD)* systémy. Hlavným prínosom celej práce v neskorších fázach bude *návrh, implementácia a experimentálna validácia nového robustného hybridného a adaptívného útoku*, ktorý efektívne prepojí digitálnu presnosť s fyzickou prenosnosťou na oklamanie multimodálnych obrán.



= Prehľad súčasného stavu problematiky <sota>
== Hlboké modely pre tvárovú biometriu
Moderné systémy tvárovej biometrie sú dnes prevažne formulované ako úloha overovania identity (_angl._ face verification), pri ktorej model neurčuje iba triedu zo známej množiny osôb, ale vytvára vektorovú reprezentáciu tváre v príznakovom priestore (tzv. embedding) a následne porovnáva podobnosť dvoch vzoriek. Typická pipeline takéhoto systému pozostáva z detekcie tváre, geometrického zarovnania, extrakcie embeddingu a rozhodnutia na základe podobnostnej metriky alebo prahovej hodnoty. Z pohľadu bezpečnosti je rozhodujúca najmä kvalita embedding modelu, pretože práve jeho reprezentácia určuje, ako citlivo bude systém reagovať na prirodzené zmeny aj na zámerné manipulácie vstupu @sohairkilany_2025_a.

Z architektonického hľadiska sú pre súčasnú tvárovú biometriu dôležité najmä hlboké konvolučné chrbticové siete (_angl._ backbone networks), ktoré zabezpečujú extrakciu diskriminačných čŕt z obrazu tváre. Architektúra ResNet patrí medzi najčastejšie používané základy moderných systémov, pretože reziduálne spojenia umožňujú efektívne trénovanie hlbších modelov a podporujú tvorbu kvalitných príznakov pre následné overovanie identity. Naopak, MobileNet predstavuje ľahšiu alternatívu optimalizovanú pre výpočtovo obmedzené prostredia, kde je prioritou nižšia latencia a menšie hardvérové zaťaženie, hoci typicky za cenu kompromisu medzi efektivitou a presnosťou. V kontexte výskumu bezpečnosti AI je toto porovnanie kľúčové, pretože robustnosť voči adversariálnym útokom je priamo závislá od kapacity a architektonického návrhu chrbticovej siete @sohairkilany_2025_a @fadel_2025_facial @wang_2026_a.

Popri chrbticovej architektúre zohráva kľúčovú úlohu aj spôsob učenia embedding priestoru, teda použitá stratová funkcia. Model FaceNet je reprezentantom prístupu založeného na trojitej strate (_angl._ triplet loss), kde sa optimalizuje vzdialenosť medzi ukotvenou vzorkou, pozitívnou vzorkou tej istej identity a negatívnou vzorkou inej identity. Modernejšie modely ako ArcFace využívajú optimalizáciu založenú na marži (_angl._ margin-based formulation), konkrétne aditívny uhlový margin v normalizovanom črtovom priestore, čím zvyšujú separovateľnosť identít a zlepšujú diskriminačnú schopnosť modelu. Táto formulácia funguje tak, že penalizuje uhol medzi embeddingom tváre a váhovým vektorom prislúchajúcej triedy v Softmax vrstve, čím núti model stláčať vnútrotriedny rozptyl a maximalizovať medzitriedne rozdiely. Model AdaFace tento princíp ďalej rozširuje o adaptívny margin závislý od kvality vzorky, čo je dôležité najmä pri nekvalitných alebo degradovaných tvárových snímkach. Pri analýze zraniteľností je preto nevyhnutné skúmať nielen topológiu konvolučných sietí, ale predovšetkým vzťah medzi stratovou funkciou a výslednou odolnosťou formovaného embedding priestoru @sohairkilany_2025_a.


== Klasifikácia a formalizácia adversariálnych útokov
Adversariálne útoky predstavujú rastúcu hrozbu pre spoľahlivosť systémov počítačového videnia, pričom ich cieľom je zaviesť model k nesprávnej predikcii pomocou cielenej úpravy vstupných dát. V kontexte tvárovej biometrie, z hľadiska zámeru útočníka, sa tieto hrozby delia na dva základné scenáre: dodging a impersonation. Kým dodging útoky sa zameriavajú na zlyhanie správnej identifikácie (útočník sa snaží skryť vlastnú identitu, čo znamená zväčšenie vzdialenosti vo feature priestore medzi jeho tvárou a referenčnou vzorkou), impersonation útoky sú výrazne náročnejšie, pretože vyžadujú manipuláciu vstupu tak, aby sa zhodoval s konkrétnou identitou obete v databáze @wang_2026_a @zhou_2024_ppr. Z hľadiska aplikačného priestoru sa následne tieto hrozby klasifikujú na čisto digitálne útoky, fyzické útoky v reálnom svete a ich hybridné formy @sohairkilany_2025_a.

=== Princíp a matematika digitálnych útokov
Digitálne adversariálne útoky modifikujú obrazové dáta priamo v digitálnej vrstve predtým, ako sú spracované neurónovou sieťou. Matematicky možno generovanie takéhoto útoku na biometrický systém formalizovať ako hľadanie optimálnej perturbácie $delta$. Cieľom je nájsť takú perturbáciu, ktorá po pripočítaní k originálnemu vstupu $x$ vedie k zlyhaniu verifikačného modelu, pričom zmena je ohraničená konkrétnou $L_p$ normou (najčastejšie využívané sú $L_(infinity)$ pre obmedzenie maximálnej zmeny jedného pixelu alebo $L_2$ pre celkovú energetickú vzdialenosť), aby zostala vizuálne nepozorovateľná pre človeka @wang_2026_a @zhou_2024_improving @carlini_2017_towards @mao_2023_boosting @wang_2022_boosting @zhang_2023_boosting.

Štandardom pre analýzu robustnosti modelov rozpoznávania tváre sú metódy založené na gradientoch, ktoré využívajú spätnú propagáciu (_angl._ backpropagation) na maximalizáciu chybovosti. Medzi najznámejšie patrí Fast Gradient Sign Method (FGSM), ktorú prvýkrát predstavil Goodfellow a kol., čo je jednokroková technika generujúca útok v smere gradientu stratovej funkcie @goodfellow_2014_explaining. Tento útok sa riadi nasledujúcou rovnicou @dong_2018_boosting:

$ x_"adv" = x + epsilon dot text("sign")(nabla_x L(theta, x, y)) $

Hoci je výpočtovo veľmi rýchla, v súčasnom biometrickom výskume často zlyháva voči komplexným nelineárnym architektúram, čo viedlo k posunu k iteratívnym prístupom @musa_2021_attack. Na systematické hodnotenie hlbokých modelov sa dnes ako zlatý štandard používa Projected Gradient Descent (PGD). Ide o iteratívny variant, ktorý aplikuje princíp FGSM vo viacerých krokoch, pričom v každom kroku projektuje vygenerovaný šum späť do dovoleného $epsilon$-okolia pôvodnej vzorky. Matematicky sa krok tohto iteratívneho procesu zapisuje ako @wang_2022_boosting @chen_2025_boosting:

$ x_"adv"^(t+1) = Pi_(x+S) (x_"adv"^t + alpha dot text("sign")(nabla_x L(theta, x_"adv"^t, y))) $

Týmto postupom dokáže PGD efektívnejšie hľadať lokálne minimá vo funkčnom priestore vysoko-dimenzionálnych biometrických modelov @sohairkilany_2025_a @zhou_2024_improving. Okrem týchto prístupov sa často využíva aj Basic Iterative Method (BIM) @kurakin_2016_adversarial, čo je základná iteračná metóda veľmi podobná PGD. Pre predchádzanie problémom s uviaznutím útoku v lokálnom minime bola vyvinutá metóda Momentum Iterative FGSM (MI-FGSM) @dong_2018_boosting. Tá na stabilizáciu gradientu využíva „momentum“, čím zabezpečuje hladší a spoľahlivejší postup optimalizácie pri generovaní útoku @dong_2018_boosting @kurakin_2016_adversarial @carlini_2017_adversarial @wei_2026_physical. Moderný posun v digitálnych útokoch predstavujú metódy založené na generatívnych sieťach (GAN), ako napríklad framework AdvFaces @deb2020advfaces. Tento prístup sa nespolieha na priamy výpočet gradientu pre každý pixel, ale učí sa generovať minimálne, vizuálne nepozorovateľné perturbácie špecificky v oblastiach tváre, ktoré sú pre biometrický model najvýznamnejšie (oči, nos, ústa).


=== Fyzické adversariálne útoky
Na rozdiel od digitálnych manipulácií si fyzické adversariálne útoky nevyžadujú prístup do infraštruktúry systému (tzv. white-box útok priamo na dáta). Útočník manipuluje svoj reálny vzhľad pred kamerovým senzorom pomocou špeciálne navrhnutých artefaktov @wang_2026_a. V tejto oblasti je kritické rozlišovať medzi klasickým podvrhom a skutočnými adversariálnymi hrozbami. Zatiaľ čo Presentation Attacks (Spoofing) spočívajú napríklad v ukázaní vytlačenej fotografie (print attack) alebo v prehraní videa na mobilnom zariadení pred kamerou (replay attack), fyzické adversariálne artefakty fungujú odlišne.

Medzi priekopnícke metódy v tejto oblasti patrí útok AdvHat @komkov2021advhat, ktorý využíva špecifickú farebnú nálepku umiestnenú na šilte baseballovej čiapky. Pri návrhu takejto perturbácie autori modelujú nelineárne zakrivenie nálepky v priestore, aby zabezpečili jej účinnosť aj po vytlačení a fyzickej deformácii. Významným koncepčným pilierom kamufláže pred biometrickými systémami je projekt CV Dazzle @harvey_2010_cvdazzle, ktorý preukázal, že cielené narušenie vizuálnej symetrie tváre, lícnych čŕt a koreňa nosa pomocou asymetrického make-upu a účesu dokáže vyradiť detekciu kľúčových tvárových bodov ešte pred extrakciou identity. Iným typom sú reálne nositeľné objekty, ako napríklad špeciálne vzorované rámy okuliarov optimalizované tak, aby vytvorili presne ten šum, ktorý dokáže oklamať embedding model @cortellazzi_2019_intriguing @zhang_2024_adversarial @wang_2024_sustainable @mao_2023_boosting, či hardvérové anti-surveillance riešenia typu Reflectacles @reflectacles_2015, využívajúce retroreflexné materiály a filtre pohlcujúce infračervené žiarenie na oslepenie 3D hĺbkových a nočných biometrických senzorov.

Aktuálne štúdie v rokoch 2025 a 2026 kategorizujú tieto hrozby do niekoľkých skupín, medzi ktoré patria špeciálne potlačené rámy okuliarov, lokalizované nálepky a adversariálne leukoplasty (patches/bandages), masky (2D a 3D), make-up či dokonca optické útoky prostredníctvom neviditeľných svetelných lúčov. Príkladom sofistikovaného optického útoku je metóda Agile (Invisible Polyjuice Potion) @wang_2024_the, ktorá využíva miniatúrne infračervené lasery zabudované v okuliaroch, či novšia metóda UVHat @yuan_2025_uvhat (ICML 2025), ktorá využíva pre človeka neviditeľné ultrafialové žiariče na šilte čiapky s optimalizáciou cez posilňovacie učenie na prekonanie biometrie z ľubovoľného uhla. Alternatívu k nositeľným doplnkom predstavuje projektorový útok ProjAttacker @liu_2025_projattacker (CVPR 2025), ktorý pomocou premietania svetelných vzorov a modelovania odrazivosti kože modifikuje tvár bez nutnosti nosenia fyzických masiek. V doméne multimodálnej bezpečnosti zase práca VIPatch @vipatch_2026 demonštruje zraniteľnosť fúzovaných RGB a infračervených biometrických systémov prostredníctvom fyzickej nálepky pripomínajúcej bežný leukoplast (band-aid sticker), ktorá dosahuje vyše 90 % úspešnosť v reálnom svete.

Najnovším smerom vo výskume je tiež testovanie robustnosti voči neočakávaným fyzickým zmenám oblečenia, ako napríklad nosenie tričiek s potlačou ľudskej tváre, ktoré môžu zmiasť detektory a verifikačné systémy zamerané na priestorovú konzistenciu @ibsen2026detection. Hlavnou výzvou pri návrhu takýchto útokov je ich citlivosť na podmienky fyzického prostredia - napríklad zmenu uhla snímania, ohniskovej vzdialenosti či osvetlenia scény. Na prekonanie týchto rozdielov medzi ideálnym digitálnym priestorom a reálnym svetom výskumníci v posledných rokoch vo veľkom integrujú techniku Expectation over Transformation (EoT). Táto metóda počas trénovania adversariálneho vzoru simuluje rôzne fyzikálne podmienky a geometrické transformácie, čím zabezpečuje, že vygenerovaný útok bude po zachytení reálnym senzorom stále účinný a povedie k úspešnému obídeniu biometrickej verifikácie @zheng_2022_robust @wang_2026_a @wang_2025_boosting @wang_2021_boosting @mao_2023_boosting.

=== Koncept hybridných útokov a ich význam
V aktuálnom výskume bezpečnosti tvárovej biometrie sa do popredia dostávajú hybridné útoky. Tento prístup je dôležitý pre realistické posúdenie zraniteľností, keďže prepája presnosť digitálnych útokov s reálnou aplikovateľnosťou fyzických hrozieb @wang_2026_a. Čisto digitálne PGD útoky vykazujú v laboratórnych podmienkach vysokú mieru úspešnosti, no zlyhávajú na nutnosti priameho prístupu k dátam. Fyzické útoky sú zase prakticky nasediteľné, ale vyznačujú sa nízkou schopnosťou prenosu medzi rôznymi modelmi (tzv. transferability), čo znamená, že útok optimalizovaný na jeden model nedokáže oklamať iný model @zhou_2024_improving @wang_2026_a.

Hybridné útoky riešia tento rozpor kombinovaním fyzických nosičov (napríklad špecifického artefaktu na tvári) s digitálne vypočítanými perturbáciami zameranými výlučne na tento priestor. Návrh a systematické overenie takýchto hybridných modelov hrozieb v experimentálnych podmienkach je preto nevyhnutným krokom k odhaleniu skrytých nedostatkov moderných architektúr. Len pochopením správania týchto komplexných vektorov útokov je možné navrhovať adekvátne obranné stratégie do reálnych komerčných aplikácií @sohairkilany_2025_a.

== Obranné mechanizmy a ich limity

S narastajúcou sofistikovanosťou adversariálnych útokov sa výskum v oblasti počítačového videnia intenzívne zameriava na vývoj robustných obranných mechanizmov pre systémy tvárovej biometrie. Tieto mechanizmy možno vo všeobecnosti rozdeliť do dvoch hlavných kategórií: prístupy zamerané na vnútornú úpravu samotného verifikačného modelu (architektonické a trénovacie zmeny) a externé moduly zamerané na filtrovanie vstupov či detekciu lživých prezentácií @sohairkilany_2025_a.

=== Metódy zvyšovania robustnosti modelov

Najvyužívanejšou a doteraz najefektívnejšou stratégiou obrany na úrovni samotnej neurónovej siete je adversariálne trénovanie (_angl._ Adversarial Training). Tento proces spočíva v zámernom obohacovaní trénovacej množiny o vopred vygenerované adversariálne príklady (typicky pomocou metódy PGD), čím je model nútený učiť sa robustnejšie reprezentácie a ignorovať umelo pridaný šum v embedding priestore @brindha_2025_face. Ďalšou bežne nasadzovanou metódou je predspracovanie vstupného obrazu (_angl._ input preprocessing), ktoré zahŕňa techniky ako Gaussovská filtrácia, kompresia obrazu alebo priestorová normalizácia s cieľom zničiť vysokofrekvenčný adversariálny šum ešte pred tým, než vstúpi do konvolučnej siete @brindha_2025_face.

Tieto proaktívne obrany však majú preukázateľné teoretické aj praktické limity. Adversariálne trénovanie je výpočtovo extrémne náročné a spravidla vedie k fenoménu zvanému robustness-accuracy trade-off, pri ktorom sa so zvyšovaním odolnosti voči útokom znižuje celková presnosť modelu na čistých, nemanipulovaných dátach. Navyše, modely chránené touto metódou vykazujú nízku schopnosť generalizácie, čo znamená, že robustnosť voči jednému typu útoku (napr. PGD) neposkytuje ochranu voči novým, nepredvídaným typom útokov alebo fyzickým zmenám @jootremoo_2025_adversarial @sohairkilany_2025_a.

=== Presentation Attack Detection (PAD) a jeho limity

Na obranu voči fyzickým hrozbám (ako sú vytlačené masky, fotografie či prehrávané videá na displejoch) sa do biometrickej pipeline štandardne nasadzujú systémy Presentation Attack Detection (PAD), často označované aj ako Face Anti-Spoofing (FAS) modely. PAD moduly fungujú ako bezpečnostná brána pred samotnou extrakciou identity a využívajú techniky analýzy textúry, detekcie živosti (_angl._ liveness detection) alebo dáta z multispektrálnych senzorov na odlíšenie reálnej ľudskej tváre od fyzického falzifikátu @matinehpooshideh_2024_presentation @riaz_2025_improving.

Z hľadiska adversariálnej bezpečnosti sa však práve na úrovni PAD systémov ukazuje kritická zraniteľnosť. Najnovšie výskumy demonštrujú, že hoci sú aktuálne State-of-the-Art PAD algoritmy efektívne proti bežným fyzickým podvrhom, sú vysoko náchylné na zlyhanie pri strete s multimodálnymi (hybridnými) útokmi. Ak útočník aplikuje optimalizovaný adversariálny šum priamo na fyzický artefakt (napríklad na rám okuliarov), dokáže takouto fúziou oklamať nielen samotný model rozpoznávania identity, ale súčasne úplne vyradiť z činnosti aj predsadený PAD systém, ktorý takúto anomáliu nedokáže správne klasifikovať @agarwal_2025_on @zhou_2025_adversarial.

=== Adaptívne útoky a potreba systematického testovania


Vzhľadom na tieto zistenia sa v aktuálnom výskume bezpečnosti upúšťa od izolovaného testovania obrán. Výskumná komunita zdôrazňuje potrebu systematického testovania odolnosti prostredníctvom štandardizovaných benchmarkov a komplexných experimentálnych prostredí. Integrácia otvorených produkčných modelov (ako sú ArcFace či AdaFace) s vlastnými referenčnými architektúrami, a ich následné vystavenie hybridným hrozbám, je dnes považovaná za jediný relevantný spôsob, ako identifikovať zraniteľnosti (medzery v stave poznania) a navrhnúť skutočne odolné biometrické systémy @jootremoo_2025_adversarial @brindha_2025_face.

== Metodika hodnotenia robustnosti a metriky
Aby bolo možné objektívne a systematicky porovnávať úspešnosť rôznych typov útokov voči rozdielnym biometrickým architektúram, moderný výskum integruje štandardizované testovacie sady a robustnostné benchmarky. Tieto benchmarky presne definujú jednotný hodnotiaci protokol (datasety, prahy podobnosti a parametre útokov) na zaistenie reprodukovateľnosti výsledkov, pričom sa spoliehajú na overené kvantitatívne metriky @jootremoo_2025_adversarial. Základným a najčastejšie uvádzaným indikátorom je Attack Success Rate (ASR), čo je percentuálny podiel útokov, ktoré úspešne oklamali model. Výpočet ASR sa priamo viaže na typ útoku: pri dodging scenároch sa za úspech považuje, ak vzdialenosť medzi embeddingami klesne (resp. stúpne) pod/nad stanovenú prahovú hodnotu (threshold) podobnosti, zatiaľ čo pri impersonation musí útočníkov embedding prekonať prahovú hodnotu voči zhluku cudzej identity @jootremoo_2025_adversarial @zhou_2024_ppr.

Druhým kritickým parametrom je miera prenositeľnosti (Transferability Rate). Zatiaľ čo ASR primárne hodnotí úspešnosť útoku na modeli, pre ktorý bol priamo optimalizovaný (white-box scenár), transferabilita meria, do akej miery je vygenerovaný adversariálny vzor účinný proti úplne inému modelu (black-box scenár). Táto metrika je obzvlášť kľúčová pri fyzických a hybridných útokoch, pretože v reálnom prostredí útočník spravidla nemá prístup k architektúre nasadeného obranného systému @zhou_2024_improving.

== Súvisiaca práca a identifikácia medzier vo výskume
Analýza súčasného stavu (State-of-the-Art) poukazuje na intenzívny, no často fragmentovaný výskum v oblasti biometrickej bezpečnosti. Komplexné štúdie, ako napríklad systematický prehľad Kilany & Mahfouz (2025), potvrdzujú, že hoci deep-learningové modely (ArcFace, AdaFace) dosahujú v ideálnych podmienkach presnosť nad 99%, ich robustnosť voči adversariálnym perturbáciám zostáva kritickým problémom @sohairkilany_2025_a.

Z hľadiska útokov sa ukazuje posun od čisto digitálnych metód, akými sú FGSM či PGD, k štúdiu fyzických útokov, pri ktorých sa pozornosť sústreďuje na prekonávanie reálnych fyzikálnych transformácií (EoT) @wang_2026_a @zheng_2022_robust. Napriek tomu viacerí autori upozorňujú na limitovanú prenosnosť týchto fyzických artefaktov medzi modelmi @zhou_2024_improving. V oblasti obranných mechanizmov zas výskum Boutrosa a kol. (2025) či štúdie zamerané na PAD systémy @agarwal_2025_on @zhou_2025_adversarial demonštrujú, že súčasné obrany nedokážu efektívne čeliť hybridným, multimodálnym hrozbám.


=== Medzery v stave poznania a východiská pre návrh riešenia
Na základe vyššie uvedenej analýzy možno konštatovať, že v súčasnom výskume chýba systematické porovnanie vplyvu hybridných útokov naprieč rôznymi trénovacími paradigmami (Triplet loss vs. Margin-based loss), najmä v kontraste s kontrolnými "baseline" modelmi. Súčasné práce sa spravidla zameriavajú na vylepšenie útoku pre jeden špecifický produkčný model, pričom izolujú vplyv predspracovania a samotnej topológie siete.

Táto diplomová práca preto nadväzuje na identifikovanú medzeru. Na základe poznatkov z teoretickej časti bol navrhnutý systematický postup, ktorého cieľom je vytvorenie kontrolovaného experimentálneho prostredia (s integráciou ArcFace, FaceNet, AdaFace a vlastnej baseline CNN) pre porovnanie digitálnych a fyzických hrozieb, čo následne vytvorí fundament pre návrh a testovanie nového hybridného adversariálneho útoku. Detailný návrh tohto prostredia a vybraných architektúr je predmetom nasledujúcej kapitoly.


= Implementácia <implementation>
V rámci praktickej časti diplomovej práce bolo navrhnuté a implementované komplexné experimentálne prostredie určené na trénovanie biometrických modelov, generovanie adversariálnych útokov a vyhodnocovanie robustnosti. Celé riešenie je napísané v jazyku Python s využitím knižnice PyTorch, ktorá poskytuje potrebnú flexibilitu pre prácu s tenzormi, výpočet gradientov a definíciu architektúr neurónových sietí.

Pre jasné metodologické vymedzenie práce rozdeľujeme softvérové komponenty platformy na prevzaté knižnice a vlastný prínos:

#figure(
  block(
    width: 100%,
    stroke: 0.5pt + gray.lighten(50%),
    fill: rgb(250, 252, 255),
    inset: 12pt,
    radius: 6pt,
  )[
    #set text(size: 8.5pt)
    #grid(
      columns: (1fr, 1.2fr),
      gutter: 15pt,
      align(center + top)[
        #block(
          width: 100%,
          fill: rgb(235, 243, 250),
          inset: 8pt,
          radius: 4pt,
          stroke: 0.5pt + rgb(180, 210, 240),
        )[
          #set text(weight: "bold", fill: rgb(25, 75, 125))
          A. Prevzaté knižnice a modely \ (Tretie strany)
        ]
        #v(8pt)
        #rect(width: 100%, fill: white, stroke: 0.5pt + gray.lighten(30%), radius: 4pt, inset: 6pt)[
          *MTCNN Detektor tvárí* \
          _Knižnica:_ `facenet-pytorch` \
          #set text(size: 7.5pt, fill: gray.darken(30%))
          Použité na automatickú lokalizáciu, orezanie a geometrické zarovnanie tvárí na $112 times 112$ px.
        ]
        #v(4pt)
        #rect(width: 100%, fill: white, stroke: 0.5pt + gray.lighten(30%), radius: 4pt, inset: 6pt)[
          *Predtrénované SOTA modely* \
          _Architektúry:_ \
          - *FaceNet* (`InceptionResnetV1`) \
          - *ArcFace* (`IResNet50`) \
          - *AdaFace* (`IResNet50`) \
          #set text(size: 7.5pt, fill: gray.darken(30%))
          Prebraté predtrénované produkčné váhy pre extrakciu diskriminačných embeddingov tváre.
        ]
      ],
      align(center + top)[
        #block(
          width: 100%,
          fill: rgb(234, 250, 241),
          inset: 8pt,
          radius: 4pt,
          stroke: 0.5pt + rgb(163, 228, 187),
        )[
          #set text(weight: "bold", fill: rgb(20, 90, 50))
          B. Vlastný vývoj autora \ (Implementované od nuly)
        ]
        #v(8pt)
        #rect(width: 100%, fill: white, stroke: 0.5pt + rgb(163, 228, 187), radius: 4pt, inset: 6pt)[
          *Predspracovanie dát* (`utils/preprocess.py`) \
          #set text(size: 7.5pt, fill: gray.darken(30%))
          Dátová pipeline s integrovaným MTCNN a paralelným spracovaním (`multiprocessing`) pre rýchlu prípravu CASIA-WebFace datasetu.
        ]
        #v(4pt)
        #rect(width: 100%, fill: white, stroke: 0.5pt + rgb(163, 228, 187), radius: 4pt, inset: 6pt)[
          *FaceModelWrapper* (`models/wrappers.py`) \
          #set text(size: 7.5pt, fill: gray.darken(30%))
          Jednotné Python/PyTorch rozhranie, ktoré unifikuje dopredný prechod pre všetky modely a vynucuje striktnú $L_2$-normalizáciu embeddingov.
        ]
        #v(4pt)
        #rect(width: 100%, fill: white, stroke: 0.5pt + rgb(163, 228, 187), radius: 4pt, inset: 6pt)[
          *BenchmarkCNN* (`models/benchmark_cnn.py`) \
          #set text(size: 7.5pt, fill: gray.darken(30%))
          Vlastná konštrukcia baseline neurónovej siete, trénovacie skripty s Mixed Precision (`torch.cuda.amp`) a Cosine Annealing rozvrhom (`train.py`, `train_finetune.py`).
        ]
        #v(4pt)
        #rect(width: 100%, fill: white, stroke: 0.5pt + rgb(163, 228, 187), radius: 4pt, inset: 6pt)[
          *Modul útokov* (`attacks/`) \
          #set text(size: 7.5pt, fill: gray.darken(30%))
          Natívna implementácia piatich gradientových útokov (FGSM, PGD, BIM, MI-FGSM, C&W L2) prispôsobených na minimalizáciu kosínusovej podobnosti.
        ]
        #v(4pt)
        #rect(width: 100%, fill: white, stroke: 0.5pt + rgb(163, 228, 187), radius: 4pt, inset: 6pt)[
          *Evaluátor & GUI* (`evaluate_batched.py`, `app.py`) \
          #set text(size: 7.5pt, fill: gray.darken(30%))
          Riadiaci skript pre hromadné vyhodnocovanie robustnosti na GPU a interaktívna webová platforma (Gradio) s podporou streaming fine-tuningu.
        ]
      ],
    )
  ],
  caption: [Schéma softvérovej architektúry navrhnutej testovacej platformy s rozdelením medzi prevzaté knižnice a vlastný prínos autora.],
) <fig-arch-schema>


== Architektúra systému a predspracovanie dát
Základným predpokladom pre úspešné trénovanie a vyhodnocovanie modelov tvárovej biometrie je konzistentný vstup. Pre tento účel bol implementovaný modul na predspracovanie dát (`utils/preprocess.py`), ktorý využíva detektor MTCNN (Multi-task Cascaded Convolutional Networks). Tento modul automaticky deteguje tváre na vstupných snímkach, vykonáva ich geometrické zarovnanie (alignment) na základe pozície očí a iných kľúčových bodov a následne ich orezáva na štandardizované rozlíšenie $112 times 112$ pixelov. Pre urýchlenie tohto procesu nad rozsiahlymi datasetmi (ako napr. CASIA-WebFace alebo LFW) bolo implementované paralelné spracovanie s využitím modulu `multiprocessing`.

== Návrh a trénovanie referenčného modelu (Benchmark CNN)
Pre porovnanie robustnosti produkčných modelov (State-of-the-Art) s jednoduchšou architektúrou bol navrhnutý vlastný model, tzv. `BenchmarkCNN`. Tento model slúži ako baseline pre ďalšie experimenty.

Architektúra modelu pozostáva zo štyroch konvolučných blokov (`ConvBlock`), ktoré slúžia ako extraktor príznakov. Každý blok obsahuje dve konvolučné vrstvy (jadro $3 times 3$) spojené s dávkovou normalizáciou (`BatchNorm2d`) a aktivačnou funkciou PReLU (Parametric ReLU). Na znižovanie priestorovej dimenzionality je na konci každého bloku použitá vrstva `MaxPool2d`. Počiatočný vstupný rozmer $112 times 112 times 3$ je postupne redukovaný až na $7 times 7 times 512$. Následne sa aplikuje globálne priemerné zhlukovanie (`AdaptiveAvgPool2d`), vrstva Dropout (s pravdepodobnosťou 0.5) a plne prepojená vrstva, ktorá transformuje príznaky do výsledného embedding priestoru o veľkosti 512. Výsledný vektor je opäť normalizovaný pomocou `BatchNorm1d`.

Model bol trénovaný pomocou štandardnej optimalizácie stochastickým gradientovým zostupom (SGD) s využitím Momentum a metódy Cosine Annealing pre plánovanie rýchlosti učenia (`lr_scheduler`). Pre maximalizáciu efektivity trénovania na GPU bola implementovaná podpora pre zmiešanú presnosť (Mixed Precision) pomocou `torch.cuda.amp.GradScaler`.

Pre účely adaptácie modelu na nové identity bol implementovaný aj mechanizmus dotrénovania (fine-tuning). Tento proces "zmrazí" váhy extraktora príznakov a trénuje iba novú klasifikačnú hlavu nad poskytnutými dátami, čo umožňuje rýchle pridávanie nových identít.

== Integrácia produkčných SOTA modelov
Aby bolo možné objektívne hodnotiť úspešnosť útokov naprieč rôznymi paradigmami učenia, experimentálne prostredie integruje aj trojicu známych otvorených (open-source) modelov: FaceNet, ArcFace a AdaFace.

Pre zabezpečenie jednotného rozhrania pri evaluácii bola navrhnutá trieda `FaceModelWrapper`. Každý integrovaný model má vlastný wrapper (`FaceNetWrapper`, `ArcFaceWrapper`, `AdaFaceWrapper`), ktorý prekrýva špecifiká načítavania váh a spracovania výstupov. Kľúčovou vlastnosťou tohto rozhrania je, že dopredný prechod (forward pass) vždy vracia $L_2$-normalizovaný embedding, čo je nutným predpokladom pre korektný výpočet kosínusovej podobnosti pri generovaní adversariálnych útokov.

== Implementácia adversariálnych útokov
Jadrom praktickej časti je implementácia piatich typov bielych (white-box) digitálnych útokov: FGSM, PGD, BIM, MI-FGSM a C&W. Všetky tieto útoky sú implementované v netargetovanom (untargeted) režime s cieľom maximalizovať vzdialenosť (minimalizovať kosínusovú podobnosť) medzi pôvodným a modifikovaným embeddingom tej istej osoby.

- *Fast Gradient Sign Method (FGSM):* Jednokrokový útok, ktorý vypočíta gradient stratovej funkcie vzhľadom na vstupný obrázok a posunie pixely v smere tohto gradientu o konštantu $epsilon$. Kľúčovým implementačným detailom je pridanie minimálneho náhodného šumu do originálneho obrázka pred výpočtom gradientu. Bez tohto kroku by bol gradient kosínusovej podobnosti identických vektorov nulový.
- *Projected Gradient Descent (PGD) a Basic Iterative Method (BIM):* Iteratívne varianty FGSM, ktoré aplikujú zmeny s menším krokom $alpha$ viackrát po sebe, pričom po každom kroku projektujú perturbáciu späť do stanoveného $L_infinity$ okolia (obmedzeného hodnotou $epsilon$) originálneho obrázka.
- *Momentum Iterative FGSM (MI-FGSM):* Rozšírenie iteratívnych útokov o momentový člen. Implementácia zhromažďuje gradienty z predchádzajúcich krokov a L1 normalizuje ich pre stabilnejší posun smerom k optimu, čím sa predchádza uviaznutiu v lokálnych minimách.
- *Carlini & Wagner (C&W) L2 Attack:* Výpočtovo najnáročnejší, optimalizačne orientovaný útok, ktorý bol špeciálne prispôsobený pre doménu tvárovej biometrie. Namiesto klasického C&W úmyslu pre klasifikáciu (ktorý minimalizuje logit správnej triedy), naša prispôsobená verzia priamo minimalizuje kosínusovú podobnosť voči pôvodnému embeddingu. Útok rieši optimalizačný problém:

  $ min_w |x_"adv" - x|_2^2 + c dot max(text("sim")(f(x), f(x_"adv")) - kappa, 0) $

  kde $x_"adv"$ je vyjadrené pomocou pomocnej premennej $w$ ako $x_"adv" = text("tanh")(w)$ pre elegantné splnenie hraničných obmedzení pixelov v rozsahu $[-1, 1]$ bez nutnosti orezávania (clippingu) v každom kroku gradientového zostupu, $text("sim")$ je kosínusová podobnosť, $f(dot)$ reprezentuje príznakový extraktor (wrapper), $kappa$ je cieľová marža (nastavená na $0.0$) a $c$ je hyperparameter vyvažujúci neviditeľnosť šumu a úspešnosť útoku (nastavený na $10.0$ pre optimálnu konvergenciu) @carlini_2017_towards.


== Vyhodnocovacie a používateľské rozhranie
Na masové testovanie bola implementovaná skriptovacia logika (`evaluate_batched.py`), ktorá spúšťa útoky nad veľkými sadami (batche) z datasetu. Systém automaticky agreguje a ukladá do CSV štruktúry rozšírenú sadu metrík pre detailnú analýzu: Attack Success Rate (percentuálny podiel úspešných útokov pri thresholde 0.5), priemernú kosínusovú podobnosť, energetickú náročnosť šumu (normy $L_2$ a $L_infinity$) a časovú náročnosť prepočítanú na jeden obrázok v milisekundách.

Na demonštráciu a manuálne testovanie zraniteľností bola vytvorená aj interaktívna webová aplikácia pomocou knižnice Gradio (`app.py`). Rozhranie je rozdelené na dve hlavné časti. Prvá slúži na vizualizáciu útokov, kde môže používateľ nahrať vlastnú fotografiu, vybrať si cieľový model, typ útoku a jeho parametre (napr. veľkosť šumu $epsilon$). Systém následne v reálnom čase vygeneruje adversariálny obrázok, zobrazí vizualizáciu zosilneného šumu a reportuje, či sa model podarilo oklamať. Druhá časť rozhrania umožňuje zber dát cez webkameru v režime nepretržitého snímania (streaming), spracovanie tváre detektorom MTCNN a spustenie fine-tuning procesu priamo z prehliadača. Tento streaming mód výrazne urýchľuje proces tvorby custom datasetov, keďže umožňuje snímať viaceré zábery v rýchlom slede bez nutnosti manuálneho reštartovania kamery. Prepojenie týchto modulov dovoľuje rýchle prispôsobenie `BenchmarkCNN` na nové identity priamo počas demonštrácie.

#figure(
  image("gradio_app_screenshot.png", width: 90%),
  caption: [Vizualizácia prvej záložky interaktívneho používateľského rozhrania Gradio platformy, slúžiacej na nahrávanie snímok, výber cieľových modelov a útokov, a real-time vizualizáciu vygenerovaného šumu a úspešnosti oklamania identity.],
) <fig-gradio-app1>

#figure(
  image("gradio_app_screenshot2.png", width: 90%),
  caption: [Vizualizácia druhej záložky interaktívneho používateľského rozhrania Gradio platformy, slúžiacej na streaming dát cez webkameru, automatickú detekciu a zarovnanie tváre detektorom MTCNN a okamžité spustenie dotrénovania BenchmarkCNN modelu.],
) <fig-gradio-app2>


// has the right format, goes before appendices
= Experimentálne výsledky a diskusia <results>

V tejto časti sú prezentované a analyzované výsledky systematického testovania robustnosti vybraných modelov tvárovej biometrie voči digitálnym adversariálnym útokom.

== Metodika experimentov

Pre zabezpečenie štatistickej významnosti a reprodukovateľnosti výsledkov sme zadefinovali presný postup meraní a vyhodnocovania:

1. *Dataset a výber snímok:* Experimenty boli spustené nad testovacou podmnožinou *2000 náhodne vybraných obrázkov* z databázy CASIA-WebFace. Tieto obrázky boli vopred orezané detektorom MTCNN na rozlíšenie $112 times 112$ pixelov, čím sme zaručili jednotný a konzistentný vstup pre všetky extraktory príznakov @sohairkilany_2025_a.
2. *Hardvérová konfigurácia:* Všetky evaluácie prebiehali na lokálnej pracovnej stanici vybavenej dedikovanou grafickou kartou *NVIDIA GeForce RTX 3060 s 6GB VRAM*, procesorom AMD Ryzen 5 5600h a 32GB systémovej pamäte RAM, pracujúcej na systéme Linux. Veľkosť dávky (batch size) bola stanovená na $32$, čo predstavuje optimálny kompromis pre maximálne vyťaženie pamäte VRAM hlbokých modelov bez rizika pretečenia pamäte (Out of Memory error).
3. *Definícia úspešnosti a prah podobnosti:* Pracujeme v scenári dodgingu (skrytia identity). Úspešnosť útoku (ASR - _angl._ Attack Success Rate) meriame ako podiel snímok, pri ktorých kosínusová podobnosť medzi pôvodným embeddingom $e_"orig"$ a modifikovaným embeddingom $e_"adv"$ klesne pod stanovenú prahovú hodnotu podobnosti $T$ @sohairkilany_2025_a @jootremoo_2025_adversarial:

  $ text("ASR") = N_"success" / N_"total" times 100 \% $

  kde prahová hodnota bola konzervatívne určená ako $T = 0.5$ a je zhodná pre všetky modely. Keďže jednotný programový interfejs `FaceModelWrapper` vynucuje $L_2$-normalizáciu výstupných vektorov ($|e|_2 = 1.0$), kosínusová podobnosť sa matematicky zjednodušuje na čistý skalárny súčin normalizovaných embeddingov:

  $ text("sim")(e_"orig", e_"adv") = e_"orig" dot e_"adv" $

  Ak táto hodnota klesne pod $0.5$ (čo zodpovedá kosínusovej vzdialenosti väčšej ako $0.5$), verifikačný systém vyhodnotí identitu ako nezhodnú, a útok je klasifikovaný ako úspešný.
4. *Parametre útokov:* Pre digitálne útoky obmedzené normou $L_infinity$ (FGSM, PGD, BIM, MI-FGSM) bol nastavený limit šumu $epsilon = 8/255$ (približne $0.0314$), čo je štandardná hodnota v biometrickej literatúre zaručujúca nízku viditeľnosť perturbácie @sohairkilany_2025_a. Pre iteratívne metódy bol krok stanovený na $alpha = 2/255$ pri maximálnom počte $20$ iterácií @dong_2018_boosting. Pre optimalizačný útok Carlini & Wagner (C&W) L2 bola marža $kappa$ nastavená na $0.0$ a vyvažovací koeficient $c$ na $10.0$ pri maximálne $100$ iteráciách optimalizátora Adam.

== Analýza úspešnosti útokov

Získané kvantitatívne dáta jasne potvrdzujú zásadné rozdiely medzi jednokrokovými a iteratívnymi útokmi a ilustrujú vplyv stratovej funkcie modelov na ich odolnosť. Kompletné výsledky úspešnosti útokov (ASR) a priemerných hodnôt kosínusovej podobnosti po útoku uvádzame v Tabuľke @tab-success.

#figure(
  table(
    columns: (1.5fr, 1.2fr, 1.2fr, 1.2fr, 1.2fr, 1.2fr),
    align: (left, center, center, center, center, center),
    stroke: 0.5pt + gray,
    fill: (x, y) => if y == 0 { rgb(240, 240, 240) } else { none },
    table.header([*Model*], [*FGSM*], [*PGD*], [*BIM*], [*MI-FGSM*], [*C&W*]),
    [FaceNet],
    [5.95% \ (0.6613)],
    [99.75% \ (-0.2988)],
    [99.80% \ (-0.2967)],
    [99.20% \ (-0.1055)],
    [91.40% \ (0.1217)],

    [ArcFace],
    [23.95% \ (0.5565)],
    [100.00% \ (-0.5940)],
    [100.00% \ (-0.6005)],
    [100.00% \ (-0.6658)],
    [100.00% \ (-0.0297)],

    [AdaFace],
    [12.45% \ (0.6036)],
    [100.00% \ (-0.3009)],
    [100.00% \ (-0.3048)],
    [100.00% \ (-0.2835)],
    [99.35% \ (0.0401)],

    [BenchmarkCNN],
    [5.80% \ (0.6274)],
    [100.00% \ (-0.8261)],
    [100.00% \ (-0.8214)],
    [100.00% \ (-0.7747)],
    [99.95% \ (-0.0374)],
  ),
  caption: [Úspešnosť útokov (ASR %) a priemerná kosínusová podobnosť po útoku.],
  kind: table,
) <tab-success>

Pri analýze dát môžeme vysloviť nasledovné kľúčové závery:

- *Zlyhanie jednokrokového útoku (FGSM):* Útok FGSM vykazoval najnižšiu úspešnosť na všetkých modeloch (od $5.80\%$ pri BenchmarkCNN po $23.95\%$ pri ArcFace). Dôvodom je, že jednokrokový lineárny posun v smere gradientu nedokáže prekonať vysoko komplexné nelineárne rozhodovacie hranice hlbokých sietí a často uviazne v lokálnych sedlových bodoch. Prekvapivo najzraniteľnejším modelom voči FGSM bol ArcFace, čo naznačuje, že hoci jeho margin-based loss vytvára extrémne husté zhluky identít, smer gradientu k najbližšej hranici je veľmi priamy a ľahko zneužiteľný jednokrokovým posunom.
- *Absolútna účinnosť iteratívnych útokov:* Metódy PGD, BIM a MI-FGSM dosiahli na všetkých produkčných modeloch takmer $100\%$ úspešnosť. Postupné, iteratívne vyhľadávanie lokálnych miním v smere gradientu s drobným krokom $alpha$ a následnou projekciou späť do $epsilon$-okolia umožňuje útokom systematicky "odtlačiť" embeddingy ďaleko za hranicu podobnosti. Priemerná kosínusová podobnosť po PGD útoku klesá až do záporných hodnôt (napr. $-0.5940$ pre ArcFace a až $-0.8261$ pre BenchmarkCNN), čo znamená, že embeddingy boli otočené takmer do protismeru v 512-dimenzionálnom priestore.
- *Vysoká odolnosť SOTA strát:* Pri porovnaní priemerných podobností po útoku vidíme, že modely ArcFace a AdaFace si udržiavajú vyššie (menej záporné) hodnoty podobnosti v porovnaní s naším BenchmarkCNN (ktorý bol trénovaný čistým Softmaxom). To dokazuje, že optimalizácia na báze marže (margin-based) vytvára robustnejší embedding priestor, kde sú hranice identít podstatne širšie a stabilnejšie, čo sťažuje útokom maximalizovať vzdialenosť, hoci ASR napriek tomu dosahuje $100\%$.

== Analýza pridávaného šumu a neviditeľnosti

Kľúčovým parametrom pre praktické nasadenie útokov je ich vizuálna neviditeľnosť, ktorá priamo súvisí s energetickou náročnosťou šumu vyjadrenou prostredníctvom noriem $L_2$ (celková energia perturbácie) a $L_infinity$ (maximálna lokálna zmena pixelu). V Tabuľke @tab-noise uvádzame priemerné veľkosti pridávaného šumu pre vybrané reprezentatívne útoky.

#figure(
  table(
    columns: (1.5fr, 1.2fr, 1.2fr, 1.2fr, 1.2fr, 1.2fr),
    align: (left, center, center, center, center, center),
    stroke: 0.5pt + gray,
    fill: (x, y) => if y == 0 { rgb(240, 240, 240) } else { none },
    table.header([*Model*], [*Norma*], [*FGSM*], [*PGD*], [*MI-FGSM*], [*C&W*]),
    [FaceNet], [$L_2$ \ $L_infinity$], [5.9073 \ 0.0324], [5.2290 \ 0.0314], [5.6707 \ 0.0314], [1.9250 \ 0.1427],
    [ArcFace], [$L_2$ \ $L_infinity$], [6.0727 \ 0.0324], [4.6434 \ 0.0314], [5.7153 \ 0.0314], [1.7812 \ 0.0916],
    [AdaFace], [$L_2$ \ $L_infinity$], [6.0723 \ 0.0324], [4.8554 \ 0.0314], [5.7295 \ 0.0314], [1.8536 \ 0.1083],
    [BenchmarkCNN], [$L_2$ \ $L_infinity$], [6.0720 \ 0.0324], [4.7516 \ 0.0314], [5.7254 \ 0.0314], [1.8342 \ 0.1088],
  ),
  caption: [Priemerná veľkosť pridávaného šumu ($L_2$ a $L_infinity$ normy perturbácií).],
  kind: table,
) <tab-noise>

- *PGD a plošný šum:* PGD útok striktne dodržiava ohraničenie normy $L_infinity$ na úrovni $epsilon = 0.0314$. Avšak, jeho priemerná $L_2$ norma je pomerne vysoká (pohybuje sa od $4.64$ do $5.22$), čo znamená, že perturbácia je aplikovaná plošne na úplne všetky pixely obrázka rovnomerne. Pre ľudské oko to pri zosilnení šumu vytvára viditeľný "zrnitý" závoj na celom povrchu tváre.
- *C&W a lokalizovaný šum:* Útok Carlini & Wagner dosahuje mimoriadnu precíznosť. Hoci jeho $L_infinity$ norma je v niektorých prípadoch vyššia (od $0.09$ do $0.14$), jeho celková energetická vzdialenosť $L_2$ je dramaticky nižšia - len *1.78 až 1.92*. To znamená, že C&W útok nešumí plošne, ale selektívne koncentruje modifikácie len do kritických oblastí tváre (napr. okolie očí a nosa), ktoré sú pre biometrický model najvýznamnejšie. Vďaka tomu zostáva zvyšok tváre úplne čistý, čo robí útok C&W pre človeka takmer neviditeľným a vizuálne najsofistikovanejším.

== Časová náročnosť a vplyv architektúry

Dôležitým aspektom pre nasadenie útokov v reálnom čase (napríklad v streamovacích systémoch) je výpočtový čas potrebný na vygenerovanie perturbácie pre jeden obrázok. Tabuľka @tab-time zobrazuje priemernú časovú náročnosť generovania útokov.

#figure(
  table(
    columns: (1.5fr, 1.2fr, 1.2fr, 1.2fr, 1.2fr, 1.2fr),
    align: (left, center, center, center, center, center),
    stroke: 0.5pt + gray,
    fill: (x, y) => if y == 0 { rgb(240, 240, 240) } else { none },
    table.header([*Model*], [*FGSM*], [*PGD*], [*BIM*], [*MI-FGSM*], [*C&W*]),
    [FaceNet], [4.55 ms], [37.03 ms], [36.57 ms], [37.06 ms], [166.28 ms],
    [ArcFace], [10.35 ms], [113.16 ms], [113.32 ms], [113.28 ms], [552.15 ms],
    [AdaFace], [11.26 ms], [116.23 ms], [116.60 ms], [116.68 ms], [568.95 ms],
    [BenchmarkCNN], [4.46 ms], [47.14 ms], [47.03 ms], [47.10 ms], [229.05 ms],
  ),
  caption: [Priemerná časová náročnosť vygenerovania útokov pre jeden obrázok.],
  kind: table,
) <tab-time>

- *Vplyv hĺbky modelu:* Čas potrebný na spätný prechod (backpropagation) a výpočet gradientov je priamo úmerný počtu vrstiev a hĺbke modelu. FaceNet (chrbticová sieť InceptionResnetV1) a vlastný BenchmarkCNN spracujú jeden obrázok pomocou PGD za $37$ až $47$ ms. ArcFace a AdaFace (využívajúce masívnejší IResNet50) potrebujú na jeden krok trojnásobok času - $113$ až $116$ ms.
- *Výpočtová náročnosť C&W:* Útok C&W je preukázateľne najpomalší ($166$ až $568$ ms na snímku). Tento výrazný rozdiel je spôsobený nutnosťou riešiť komplexný optimalizačný problém pomocou algoritmu Adam s veľkým počtom iterácií a výpočtom nelineárnej transformácie v priestore $text("tanh")$ v každom jednom kroku.

Zhrnutím možno konštatovať, že kým rýchle gradientové metódy (ako PGD) sú ideálne na okamžité vyhodnocovanie a penetračné testovanie robustnosti v reálnom čase, optimalizačný útok C&W zostáva zlatým štandardom pre sofistikované útoky s najvyššou prioritou minimalizácie detekovateľnosti ľudským okom.


#bibliography("citations.bib", style: "iso690-author-date-sk.csl")

#pagebreak(weak: true)
#v(2cm)
#align(center)[
  #block(width: 90%)[
    #set align(left)
    #set text(size: 10pt)
    #set par(first-line-indent: 0pt)
    #strong[Čestné vyhlásenie o použití generatívnej umelej inteligencie]
    #v(10pt)
    V súlade s metodickými usmerneniami a etickými štandardmi pre vypracovanie záverečných prác čestne vyhlasujem, že pri príprave tejto priebežnej správy o riešení diplomovej práce (DP I) boli využité nástroje generatívnej umelej inteligencie (AI).

    #v(8pt)
    #strong[1. Deklarované nástroje:]
    - *antigravity-cli a model Gemini 3.5 Flash* – konverzačný a vývojový asistent použitý počas párového programovania a sadzby dokumentu.

    #v(8pt)
    #strong[2. Rozsah a spôsob využitia:]
    - *Úprava štylistiky a jazyková korekcia:* Nástroj bol využitý na optimalizáciu štylistiky slovenského textu, odstránenie gramatických nepresností a spresnenie terminologických prekladov z anglického jazyka (slovakizácia odborných pojmov).
    - *Optimalizácia a vizualizácia:* Pomoc pri návrhu high-level diagramu softvérovej architektúry (blokový grid v Typste) a prenos experimentálnych dát z CSV súborov do štruktúrovaných Typst tabuliek.
  ]
]
#pagebreak(weak: true)

// #resume()[
// #lorem(250)
// ]

// start the appendices section with this line
#show: section-appendices

= Source code <source-code>


= Plán práce <plan-of-work>

Tento plán práce jasne vymedzuje ciele a očakávané výstupy jednotlivých semestrálnych etáp riešenia diplomovej práce. Prvý semester (DP I) bol zameraný na teoretickú analýzu a vybudovanie robustného testovacieho a trénovacieho prostredia pre digitálne white-box útoky, čo bolo úspešne dokončené v rámci tejto správy. Nasledujúce semestrálne etapy (DP II a DP III) sa zamerajú na praktické testovanie fyzických a presentation útokov a na samotný návrh nového robustného hybridného a adaptívneho útoku vrátane jeho experimentálnej evaluácie voči pokročilým formám obrán.

== Semester I: DP I
#table(
  columns: (1fr, 2fr, 2fr),
  align: (left, left, left),
  stroke: 0.5pt + gray,
  fill: (x, y) => if y == 0 { rgb(240, 240, 240) } else { none },
  table.header([*Fáza*], [*Úloha*], [*Očakávané výstupy*]),

  // Dáta (používajú predvolený štýl 0.5pt + gray)
  [Teória],
  [Vypracovanie podrobného prehľadu aktuálnych techník rozpoznávania tváre pomocou AI (Deep Learning modely).],
  [Dokončená textová časť k Sekcii 2 (Prehľad súčasného stavu) a jej rozšírenie o naj
    novšie práce.],

  [Útoky],
  [Klasifikácia a analýza digitálnych, fyzických a Presentation Attacks (PAD) relevantných pre modely tvárovej biometrie.],
  [Detailný popis vybraných útokov (napr. FGSM, PGD, print attacks, masky) na implementáciu.],

  [Model],
  [Návrh a implementácia vlastného, jednoduchšieho modelu rozpoznávania tváre (benchmark model).],
  [Funkčný vlastný CNN model pre porovnanie s open-source modelmi (Téza B).],

  [Prostredie],
  [Príprava experimentálneho prostredia a integrácia open-source modelov (FaceNet, ArcFace, AdaFace).],
  [Experimentálne prostredie pripravené na testovanie digi
    tálnych útokov.],
)

== Semester II: DP II
#table(
  columns: (1fr, 2fr, 2fr),
  align: (left, left, left),
  stroke: 0.5pt + gray,
  fill: (x, y) => if y == 0 { rgb(240, 240, 240) } else { none },
  table.header([*Fáza*], [*Úloha*], [*Očakávané výstupy*]),

  // Dáta (používajú predvolený štýl 0.5pt + gray)
  [Testovanie A],
  [Systematické otestovanie a porovnanie účinnosti vybraných digitálnych útokov (FGSM, PGD) na open-source i vlastnom modeli.],
  [Kvantitatívne výsledky (úspešnosť, transferabilita) pre digitálne útoky (podpora Téz A a B).],

  [Testovanie B],
  [Overenie efektivity fyzických a presentation attacks (simulovaných alebo reálnych) na všetkých vybraných modeloch.],
  [Dáta o účinnosti fyzických/PAD útokov a detekčných mechanizmov (podpora Tézy C).],

  [Návrh],
  [Návrh nového hybridného adversariálneho útoku, ktorý kombinuje digitálne a fyzické perturbácie.],
  [Detailný teoretický návrh útoku s popisom implementácie (podpora Tézy D).],

  [Implementácia útoku], [Implementácia navrhnutého hybridného útoku.], [Funkčná kódová báza nového útoku.],
)

== Semester III: DP III
#table(
  columns: (1fr, 2fr, 2fr),
  align: (left, left, left),
  stroke: 0.5pt + gray,
  fill: (x, y) => if y == 0 { rgb(240, 240, 240) } else { none },
  table.header([*Fáza*], [*Úloha*], [*Očakávané výstupy*]),

  // Dáta (používajú predvolený štýl 0.5pt + gray)
  [Vyhodnotenie útoku],
  [Experimentálne vyhodnotenie efektívnosti navrhnutého hybridného útoku proti obranným mechanizmom (napr. adversariálne trénovanie, PAD).],
  [Experimentálne overenie útoku a jeho účinnosti (podpora Tézy D).],

  [Analýza a Doporučenia],
  [Komplexná analýza všetkých experimentálnych výsledkov a sformulovanie doporučení pre zvýšenie odolnosti tvárových biometrických systémov.],
  [Záverečné doporučenia pre robustnosť a smerovanie ďalšieho výskumu.],

  [Dokumentácia],
  [Spracovanie finálnej textovej verzie diplomovej práce, revízia a formalizácia.],
  [Diplomová práca v súlade s pokynmi fakulty.],
)

#pagebreak()

