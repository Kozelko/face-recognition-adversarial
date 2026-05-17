#import "@preview/vintage-fiit-thesis:1.1.0": *

#show: fiit-thesis.with(
  title: "Odolnosť modelov AI pre tvárovú biometriu voči adversariálnym útokom",
  thesis: "dp1",
  author: "Bc. Peter Brandajský",
  supervisor: "Mgr. Ing. Emma Macháčová",
  abstract: (
    sk: lorem(150),
    en: lorem(150),
  ), // abstract
  id: "FIIT-12345-123456",
  lang: "sk", // this controls how the layout is presented, be careful!
  // remove the argument or made the value none to hide
  acknowledgment: [I would like to thank my supervisor for all the help and
    guidance I have received. I would also like to thank my friends and family
    for supporting during this work.],
  // remove the argument or leave the array empty to hide the list of
  // abbreviations
  abbreviations-outline: (),
  figures-outline: true,
  tables-outline: true,
  style: "regular",
)

= Úvod <introduction>

= Prehľad súčasného stavu problematiky <sota>
== Hlboké modely pre tvárovú biometriu
Moderné systémy tvárovej biometrie sú dnes prevažne formulované ako úloha overovania identity (face verification), pri ktorej model neurčuje iba triedu zo známej množiny osôb, ale vytvára vektorovú reprezentáciu tváre v embedding priestore a následne porovnáva podobnosť dvoch vzoriek. Typická pipeline takéhoto systému pozostáva z detekcie tváre, geometrického zarovnania, extrakcie embeddingu a rozhodnutia na základe podobnostnej metriky alebo prahovej hodnoty. Z pohľadu bezpečnosti je rozhodujúca najmä kvalita embedding modelu, pretože práve jeho reprezentácia určuje, ako citlivo bude systém reagovať na prirodzené zmeny aj na zámerné manipulácie vstupu @sohairkilany_2025_a.

Z architektonického hľadiska sú pre súčasnú tvárovú biometriu dôležité najmä hlboké konvolučné backbone siete, ktoré zabezpečujú extrakciu diskriminačných čŕt z obrazu tváre. Architektúra ResNet patrí medzi najčastejšie používané základy moderných systémov, pretože reziduálne spojenia umožňujú efektívne trénovanie hlbších modelov a podporujú tvorbu kvalitných príznakov pre následné overovanie identity. Naopak, MobileNet predstavuje ľahšiu alternatívu optimalizovanú pre výpočtovo obmedzené prostredia, kde je prioritou nižšia latencia a menšie hardvérové zaťaženie, hoci typicky za cenu kompromisu medzi efektivitou a presnosťou. V kontexte výskumu bezpečnosti AI je toto porovnanie kľúčové, pretože robustnosť voči adversariálnym útokom je priamo závislá od kapacity a architektonického návrhu backbone siete @sohairkilany_2025_a @fadel_2025_facial @wang_2026_a.

Popri backbone architektúre zohráva kľúčovú úlohu aj spôsob učenia embedding priestoru, teda použitá stratová funkcia. Model FaceNet je reprezentantom prístupu založeného na triplet loss, kde sa optimalizuje vzdialenosť medzi ukotvenou vzorkou, pozitívnou vzorkou tej istej identity a negatívnou vzorkou inej identity. Modernejšie modely ako ArcFace využívajú margin-based formuláciu, konkrétne uhlový margin v normalizovanom črtovom priestore, čím zvyšujú separovateľnosť identít a zlepšujú diskriminačnú schopnosť modelu. Model AdaFace tento princíp ďalej rozširuje o adaptívny margin závislý od kvality vzorky, čo je dôležité najmä pri nekvalitných alebo degradovaných tvárových snímkach. Pri analýze zraniteľností je preto nevyhnutné skúmať nielen topológiu konvolučných sietí, ale predovšetkým vzťah medzi stratovou funkciou a výslednou odolnosťou formovaného embedding priestoru @sohairkilany_2025_a.

== Klasifikácia a formalizácia adversariálnych útokov
Adversariálne útoky predstavujú rastúcu hrozbu pre spoľahlivosť systémov počítačového videnia, pričom ich cieľom je zaviesť model k nesprávnej predikcii pomocou cielenej úpravy vstupných dát. V kontexte tvárovej biometrie, z hľadiska zámeru útočníka, sa tieto hrozby delia na dva základné scenáre: dodging a impersonation. Kým dodging útoky sa zameriavajú na zlyhanie správnej identifikácie (útočník sa snaží skryť vlastnú identitu, čo znamená zväčšenie vzdialenosti vo feature priestore medzi jeho tvárou a referenčnou vzorkou), impersonation útoky sú výrazne náročnejšie, pretože vyžadujú manipuláciu vstupu tak, aby sa zhodoval s konkrétnou identitou obete v databáze @wang_2026_a @zhou_2024_ppr. Z hľadiska aplikačného priestoru sa následne tieto hrozby klasifikujú na čisto digitálne útoky, fyzické útoky v reálnom svete a ich hybridné formy @sohairkilany_2025_a.

=== Princíp a matematika digitálnych útokov
Digitálne adversariálne útoky modifikujú obrazové dáta priamo v digitálnej vrstve predtým, ako sú spracované neurónovou sieťou. Matematicky možno generovanie takéhoto útoku na biometrický systém formalizovať ako hľadanie optimálnej perturbácie $delta$. Cieľom je nájsť takú perturbáciu, ktorá po pripočítaní k originálnemu vstupu $x$ vedie k zlyhaniu verifikačného modelu, pričom zmena je ohraničená konkrétnou $L_p$ normou (najčastejšie využívané sú $L_(infinity)$ pre obmedzenie maximálnej zmeny jedného pixelu alebo $L_2$ pre celkovú energetickú vzdialenosť), aby zostala vizuálne nepozorovateľná pre človeka @wang_2026_a @zhou_2024_improving @carlini_2017_towards @mao_2023_boosting @wang_2022_boosting @zhang_2023_boosting.

Štandardom pre analýzu robustnosti modelov rozpoznávania tváre sú metódy založené na gradientoch, ktoré využívajú spätnú propagáciu (backpropagation) na maximalizáciu chybovosti. Medzi najznámejšie patrí Fast Gradient Sign Method (FGSM), ktorú prvýkrát predstavil Goodfellow a kol., čo je jednokroková technika generujúca útok v smere gradientu stratovej funkcie @goodfellow_2014_explaining. Tento útok sa riadi nasledujúcou rovnicou @dong_2018_boosting:

$ x_"adv" = x + epsilon dot text("sign")(nabla_x L(theta, x, y)) $

Hoci je výpočtovo veľmi rýchla, v súčasnom biometrickom výskume často zlyháva voči komplexným nelineárnym architektúram, čo viedlo k posunu k iteratívnym prístupom @musa_2021_attack. Na systematické hodnotenie hlbokých modelov sa dnes ako zlatý štandard používa Projected Gradient Descent (PGD). Ide o iteratívny variant, ktorý aplikuje princíp FGSM vo viacerých krokoch, pričom v každom kroku projektuje vygenerovaný šum späť do dovoleného $epsilon$-okolia pôvodnej vzorky. Matematicky sa krok tohto iteratívneho procesu zapisuje ako @wang_2022_boosting @chen_2025_boosting:

$ x_"adv"^(t+1) = Pi_(x+S) (x_"adv"^t + alpha dot text("sign")(nabla_x L(theta, x_"adv"^t, y))) $

Týmto postupom dokáže PGD efektívnejšie hľadať lokálne minimá vo funkčnom priestore vysoko-dimenzionálnych biometrických modelov @sohairkilany_2025_a @zhou_2024_improving. Okrem týchto prístupov sa často využíva aj Basic Iterative Method (BIM) @kurakin_2016_adversarial, čo je základná iteračná metóda veľmi podobná PGD. Pre predchádzanie problémom s uviaznutím útoku v lokálnom minime bola vyvinutá metóda Momentum Iterative FGSM (MI-FGSM) @dong_2018_boosting. Tá na stabilizáciu gradientu využíva „momentum“, čím zabezpečuje hladší a spoľahlivejší postup optimalizácie pri generovaní útoku @dong_2018_boosting @kurakin_2016_adversarial @carlini_2017_adversarial @wei_2026_physical. Moderný posun v digitálnych útokoch predstavujú metódy založené na generatívnych sieťach (GAN), ako napríklad framework AdvFaces @deb2020advfaces. Tento prístup sa nespolieha na priamy výpočet gradientu pre každý pixel, ale učí sa generovať minimálne, vizuálne nepozorovateľné perturbácie špecificky v oblastiach tváre, ktoré sú pre biometrický model najvýznamnejšie (oči, nos, ústa).


=== Fyzické adversariálne útoky
Na rozdiel od digitálnych manipulácií si fyzické adversariálne útoky nevyžadujú prístup do infraštruktúry systému (tzv. white-box útok priamo na dáta). Útočník manipuluje svoj reálny vzhľad pred kamerovým senzorom pomocou špeciálne navrhnutých artefaktov @wang_2026_a. V tejto oblasti je kritické rozlišovať medzi klasickým podvrhom a skutočnými adversariálnymi hrozbami. Zatiaľ čo Presentation Attacks (Spoofing) spočívajú napríklad v ukázaní vytlačenej fotografie (print attack) alebo v prehraní videa na mobilnom zariadení pred kamerou (replay attack), fyzické adversariálne artefakty fungujú odlišne. 

Medzi priekopnícke metódy v tejto oblasti patrí útok AdvHat @komkov2021advhat, ktorý využíva špecifickú farebnú nálepku umiestnenú na šilte baseballovej čiapky. Pri návrhu takejto perturbácie autori modelujú nelineárne zakrivenie nálepky v priestore, aby zabezpečili jej účinnosť aj po vytlačení a fyzickej deformácii. Iným typom sú reálne objekty, ako napríklad špeciálne vzorované rámy okuliarov, ktoré útočník priamo nosí a ktoré sú optimalizované tak, aby vytvorili presne ten šum, ktorý dokáže oklamať embedding model @cortellazzi_2019_intriguing @zhang_2024_adversarial @wang_2024_sustainable @mao_2023_boosting.

Aktuálne štúdie kategorizujú tieto hrozby do niekoľkých skupín, medzi ktoré patria špeciálne potlačené rámy okuliarov, adversariálne masky (2D a 3D), make-up či dokonca optické útoky prostredníctvom infračervených a laserových svetelných lúčov. Príkladom sofistikovaného optického útoku je metóda Agile (Invisible Polyjuice Potion) @wang_2024_the, ktorá využíva miniatúrne infračervené lasery zabudované v okuliaroch. Tieto lasery vytvárajú interferenčný vzor priamo na senzore kamery, ktorý je pre ľudské oko neviditeľný, ale pre biometrický systém predstavuje deštruktívnu perturbáciu. Najnovším smerom vo výskume je testovanie robustnosti voči neočakávaným fyzickým zmenám oblečenia, ako napríklad nosenie tričiek s potlačou ľudskej tváre, ktoré môžu zmiasť detektory a verifikačné systémy zamerané na priestorovú konzistenciu @ibsen2026detection. Hlavnou výzvou pri návrhu takýchto útokov je ich citlivosť na podmienky fyzického prostredia - napríklad zmenu uhla snímania, ohniskovej vzdialenosti či osvetlenia scény. Na prekonanie týchto rozdielov medzi ideálnym digitálnym priestorom a reálnym svetom výskumníci v posledných rokoch vo veľkom integrujú techniku Expectation over Transformation (EoT). Táto metóda počas trénovania adversariálneho vzoru simuluje rôzne fyzikálne podmienky a geometrické transformácie, čím zabezpečuje, že vygenerovaný útok bude po zachytení reálnym senzorom stále účinný a povedie k úspešnému obídeniu biometrickej verifikácie @zheng_2022_robust @wang_2026_a @wang_2025_boosting @wang_2021_boosting @mao_2023_boosting.

=== Koncept hybridných útokov a ich význam
V aktuálnom výskume bezpečnosti tvárovej biometrie sa do popredia dostávajú hybridné útoky. Tento prístup je dôležitý pre realistické posúdenie zraniteľností, keďže prepája presnosť digitálnych útokov s reálnou aplikovateľnosťou fyzických hrozieb @wang_2026_a. Čisto digitálne PGD útoky vykazujú v laboratórnych podmienkach vysokú mieru úspešnosti, no zlyhávajú na nutnosti priameho prístupu k dátam. Fyzické útoky sú zase prakticky nasediteľné, ale vyznačujú sa nízkou schopnosťou prenosu medzi rôznymi modelmi (tzv. transferability), čo znamená, že útok optimalizovaný na jeden model nedokáže oklamať iný model @zhou_2024_improving @wang_2026_a.

Hybridné útoky riešia tento rozpor kombinovaním fyzických nosičov (napríklad špecifického artefaktu na tvári) s digitálne vypočítanými perturbáciami zameranými výlučne na tento priestor. Návrh a systematické overenie takýchto hybridných modelov hrozieb v experimentálnych podmienkach je preto nevyhnutným krokom k odhaleniu skrytých nedostatkov moderných architektúr. Len pochopením správania týchto komplexných vektorov útokov je možné navrhovať adekvátne obranné stratégie do reálnych komerčných aplikácií @sohairkilany_2025_a.

== Obranné mechanizmy a ich limity

S narastajúcou sofistikovanosťou adversariálnych útokov sa výskum v oblasti počítačového videnia intenzívne zameriava na vývoj robustných obranných mechanizmov pre systémy tvárovej biometrie. Tieto mechanizmy možno vo všeobecnosti rozdeliť do dvoch hlavných kategórií: prístupy zamerané na vnútornú úpravu samotného verifikačného modelu (architektonické a trénovacie zmeny) a externé moduly zamerané na filtrovanie vstupov či detekciu lživých prezentácií @sohairkilany_2025_a.

=== Metódy zvyšovania robustnosti modelov

Najvyužívanejšou a doteraz najefektívnejšou stratégiou obrany na úrovni samotnej neurónovej siete je adversariálne trénovanie (Adversarial Training). Tento proces spočíva v zámernom obohacovaní trénovacej množiny o vopred vygenerované adversariálne príklady (typicky pomocou metódy PGD), čím je model nútený učiť sa robustnejšie reprezentácie a ignorovať umelo pridaný šum v embedding priestore @brindha_2025_face. Ďalšou bežne nasadzovanou metódou je predspracovanie vstupného obrazu (input preprocessing), ktoré zahŕňa techniky ako Gaussovská filtrácia, kompresia obrazu alebo priestorová normalizácia s cieľom zničiť vysokofrekvenčný adversariálny šum ešte pred tým, než vstúpi do konvolučnej siete @brindha_2025_face.

Tieto proaktívne obrany však majú preukázateľné teoretické aj praktické limity. Adversariálne trénovanie je výpočtovo extrémne náročné a spravidla vedie k fenoménu zvanému robustness-accuracy trade-off, pri ktorom sa so zvyšovaním odolnosti voči útokom znižuje celková presnosť modelu na čistých, nemanipulovaných dátach. Navyše, modely chránené touto metódou vykazujú nízku schopnosť generalizácie, čo znamená, že robustnosť voči jednému typu útoku (napr. PGD) neposkytuje ochranu voči novým, nepredvídaným typom útokov alebo fyzickým zmenám @jootremoo_2025_adversarial @sohairkilany_2025_a.

=== Presentation Attack Detection (PAD) a jeho limity

Na obranu voči fyzickým hrozbám (ako sú vytlačené masky, fotografie či prehrávané videá na displejoch) sa do biometrickej pipeline štandardne nasadzujú systémy Presentation Attack Detection (PAD), často označované aj ako Face Anti-Spoofing (FAS) modely. PAD moduly fungujú ako bezpečnostná brána pred samotnou extrakciou identity a využívajú techniky analýzy textúry, detekcie živosti (liveness detection) alebo dáta z multispektrálnych senzorov na odlíšenie reálnej ľudskej tváre od fyzického falzifikátu @matinehpooshideh_2024_presentation @riaz_2025_improving.

Z hľadiska adversariálnej bezpečnosti sa však práve na úrovni PAD systémov ukazuje kritická zraniteľnosť. Najnovšie výskumy demonštrujú, že hoci sú aktuálne State-of-the-Art PAD algoritmy efektívne proti bežným fyzickým podvrhom, sú vysoko náchylné na zlyhanie pri strete s multimodálnymi (hybridnými) útokmi. Ak útočník aplikuje optimalizovaný adversariálny šum priamo na fyzický artefakt (napríklad na rám okuliarov), dokáže takouto fúziou oklamať nielen samotný model rozpoznávania identity, ale súčasne úplne vyradiť z činnosti aj predsadený PAD systém, ktorý takúto anomáliu nedokáže správne klasifikovať @agarwal_2025_on @zhou_2025_adversarial.

=== Adaptívne útoky a potreba systematického testovania

Limity súčasných obrán sa najvýraznejšie prejavujú pri tzv. adaptívnych útokoch (adaptive attacks). V tomto scenári útočník nenasadzuje statický útok, ale pri návrhu perturbácie priamo zohľadňuje a matematicky modeluje nasadený obranný mechanizmus. Ak napríklad biometrický systém využíva konkrétny typ filtra na čistenie obrazu, útočník zakomponuje parametre tohto filtra do svojej optimalizačnej funkcie, čím obranu v štádiu generovania útoku úplne obíde @jootremoo_2025_adversarial.

Vzhľadom na tieto zistenia sa v aktuálnom výskume bezpečnosti upúšťa od izolovaného testovania obrán. Výskumná komunita zdôrazňuje potrebu systematického testovania odolnosti prostredníctvom štandardizovaných benchmarkov a komplexných experimentálnych prostredí. Integrácia otvorených produkčných modelov (ako sú ArcFace či AdaFace) s vlastnými referenčnými architektúrami, a ich následné vystavenie hybridným hrozbám, je dnes považovaná za jediný relevantný spôsob, ako identifikovať zraniteľnosti (medzery v stave poznania) a navrhnúť skutočne odolné biometrické systémy @jootremoo_2025_adversarial @brindha_2025_face.

== Metodika hodnotenia robustnosti a metriky
Aby bolo možné objektívne a systematicky porovnávať úspešnosť rôznych typov útokov voči rozdielnym biometrickým architektúram, výskum sa spolieha na kvantitatívne metriky. Základným a najčastejšie uvádzaným indikátorom je Attack Success Rate (ASR), čo je percentuálny podiel útokov, ktoré úspešne oklamali model. Výpočet ASR sa priamo viaže na typ útoku: pri dodging scenároch sa za úspech považuje, ak vzdialenosť medzi embeddingami klesne (resp. stúpne) pod/nad stanovenú prahovú hodnotu (threshold) podobnosti, zatiaľ čo pri impersonation musí útočníkov embedding prekonať prahovú hodnotu voči zhluku cudzej identity @jootremoo_2025_adversarial @zhou_2024_ppr.

Druhým kritickým parametrom je miera prenositeľnosti (Transferability Rate). Zatiaľ čo ASR primárne hodnotí úspešnosť útoku na modeli, pre ktorý bol priamo optimalizovaný (white-box scenár), transferabilita meria, do akej miery je vygenerovaný adversariálny vzor účinný proti úplne inému modelu (black-box scenár). Táto metrika je obzvlášť kľúčová pri fyzických a hybridných útokoch, pretože v reálnom prostredí útočník spravidla nemá prístup k architektúre nasadeného obranného systému @zhou_2024_improving.

== Súvisiaca práca a identifikácia medzier vo výskume
Analýza súčasného stavu (State-of-the-Art) poukazuje na intenzívny, no často fragmentovaný výskum v oblasti biometrickej bezpečnosti. Komplexné štúdie, ako napríklad systematický prehľad Kilany & Mahfouz (2025), potvrdzujú, že hoci deep-learningové modely (ArcFace, AdaFace) dosahujú v ideálnych podmienkach presnosť nad 99%, ich robustnosť voči adversariálnym perturbáciám zostáva kritickým problémom @sohairkilany_2025_a.

Z hľadiska útokov sa ukazuje posun od čisto digitálnych metód, akými sú FGSM či PGD, k štúdiu fyzických útokov, pri ktorých sa pozornosť sústreďuje na prekonávanie reálnych fyzikálnych transformácií (EoT) @wang_2026_a @zheng_2022_robust. Napriek tomu viacerí autori upozorňujú na limitovanú prenosnosť týchto fyzických artefaktov medzi modelmi @zhou_2024_improving. V oblasti obranných mechanizmov zas výskum Boutrosa a kol. (2025) či štúdie zamerané na PAD systémy @agarwal_2025_on @zhou_2025_adversarial demonštrujú, že súčasné obrany nedokážu efektívne čeliť hybridným, multimodálnym hrozbám.

=== Medzery v stave poznania a východiská pre návrh riešenia
Na základe vyššie uvedenej analýzy možno konštatovať, že v súčasnom výskume chýba systematické porovnanie vplyvu hybridných útokov naprieč rôznymi trénovacími paradigmami (Triplet loss vs. Margin-based loss), najmä v kontraste s kontrolnými "baseline" modelmi. Súčasné práce sa spravidla zameriavajú na vylepšenie útoku pre jeden špecifický produkčný model, pričom izolujú vplyv predspracovania a samotnej topológie siete.

Táto diplomová práca preto nadväzuje na identifikovanú medzeru. Na základe poznatkov z teoretickej časti bol navrhnutý systematický postup, ktorého cieľom je vytvorenie kontrolovaného experimentálneho prostredia (s integráciou ArcFace, FaceNet, AdaFace a vlastnej baseline CNN) pre porovnanie digitálnych a fyzických hrozieb, čo následne vytvorí fundament pre návrh a testovanie nového hybridného adversariálneho útoku. Detailný návrh tohto prostredia a vybraných architektúr je predmetom nasledujúcej kapitoly.


= Implementácia <implementation>
V rámci praktickej časti diplomovej práce bolo navrhnuté a implementované komplexné experimentálne prostredie určené na trénovanie biometrických modelov, generovanie adversariálnych útokov a vyhodnocovanie robustnosti. Celé riešenie je napísané v jazyku Python s využitím knižnice PyTorch, ktorá poskytuje potrebnú flexibilitu pre prácu s tenzormi, výpočet gradientov a definíciu architektúr neurónových sietí.

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
- *Carlini & Wagner (C&W) L2 Attack:* Výpočtovo najnáročnejší útok, ktorý priamo optimalizuje kompromis medzi zmenou obrázka ($L_2$ normou) a úspešnosťou oklamania modelu. Vzhľadom na požiadavku, aby modifikované pixely zostali v rozsahu $[-1, 1]$, je optimalizácia vykonávaná v priestore funkcie $text("arctanh")$, ktorá zabezpečuje plynulé mapovanie bez nutnosti orezávania (clippingu) v každom kroku optimalizácie.

== Vyhodnocovacie a používateľské rozhranie
Na masové testovanie bola implementovaná skriptovacia logika (`evaluate_batched.py`), ktorá spúšťa útoky nad veľkými sadami (batche) z datasetu. Systém automaticky agreguje a ukladá do CSV štruktúry rozšírenú sadu metrík pre detailnú analýzu: Attack Success Rate (percentuálny podiel úspešných útokov pri thresholde 0.5), priemernú kosínusovú podobnosť, energetickú náročnosť šumu (normy $L_2$ a $L_infinity$) a časovú náročnosť prepočítanú na jeden obrázok v milisekundách.

Na demonštráciu a manuálne testovanie zraniteľností bola vytvorená aj interaktívna webová aplikácia pomocou knižnice Gradio (`app.py`). Rozhranie je rozdelené na dve hlavné časti. Prvá slúži na vizualizáciu útokov, kde môže používateľ nahrať vlastnú fotografiu, vybrať si cieľový model, typ útoku a jeho parametre (napr. veľkosť šumu $epsilon$). Systém následne v reálnom čase vygeneruje adversariálny obrázok, zobrazí vizualizáciu zosilneného šumu a reportuje, či sa model podarilo oklamať. Druhá časť rozhrania umožňuje zber dát cez webkameru v režime nepretržitého snímania (streaming), spracovanie tváre detektorom MTCNN a spustenie fine-tuning procesu priamo z prehliadača. Tento streaming mód výrazne urýchľuje proces tvorby custom datasetov, keďže umožňuje snímať viaceré zábery v rýchlom slede bez nutnosti manuálneho reštartovania kamery. Prepojenie týchto modulov dovoľuje rýchle prispôsobenie `BenchmarkCNN` na nové identity priamo počas demonštrácie.

// has the right format, goes before appendices
= Experimentálne výsledky a diskusia <results>
V tejto časti sú prezentované a analyzované výsledky systematického testovania robustnosti vybraných modelov tvárovej biometrie voči digitálnym adversariálnym útokom. Experimenty boli vykonané na podmnožine 2000 obrázkov z datasetu CASIA-WebFace, pričom každý model bol vystavený piatim typom útokov: FGSM, PGD, BIM, MI-FGSM a C&W.

== Analýza úspešnosti útokov
Získané dáta potvrdzujú teoretické predpoklady o nízkej účinnosti jednokrokových útokov v porovnaní s iteratívnymi metódami. Útok FGSM vykazoval najnižšiu mieru úspešnosti (Attack Success Rate - ASR), ktorá sa pohybovala v rozmedzí od 5,8 % (BenchmarkCNN) po 23,95 % (ArcFace). Tento výsledok naznačuje, že hlboké modely trénované na rozsiahlych datasetoch disponujú prirodzenou mierou robustnosti voči jednoduchému zašumeniu v smere gradientu.

Naopak, iteratívne útoky (PGD, BIM, MI-FGSM) dosiahli takmer 100 % úspešnosť na všetkých testovaných architektúrach. Tieto metódy dokážu efektívne hľadať lokálne minimá v embedding priestore a systematicky vzďaľovať adversariálnu vzorku od pôvodnej identity. Útok PGD s 20 iteráciami a parametrom $epsilon = 8/255$ sa ukázal ako mimoriadne efektívny "dodging" útok, ktorý úplne zlikvidoval schopnosť modelov správne verifikovať identitu.

== Analýza pridávaného šumu a neviditeľnosti
Kritickým bodom diskusie je porovnanie útokov PGD a Carlini & Wagner (C&W) z pohľadu vizuálnej kvality a energetickej náročnosti perturbácie.

- *PGD útok:* Hoci dosahuje 100 % úspešnosť, jeho perturbácia je plošná a ohraničená normou $L_infinity$. Priemerný šum $L_2$ sa pohyboval na úrovni 4,6 až 5,7. Tento útok mení každý pixel až po povolený limit $epsilon$, čo pri vyšších hodnotách môže byť vizuálne postrehnuteľné.
- *C&W útok:* Tento útok demonštruje vysokú precíznosť. Napriek tomu, že jeho $L_infinity$ norma bola v niektorých prípadoch vyššia (0,09 až 0,14), celková energetická vzdialenosť $L_2$ bola výrazne nižšia (približne 1,7 až 1,9). To znamená, že C&W útok koncentruje zmeny len do vybraných oblastí, ktoré sú pre model najdôležitejšie, pričom celkovo pridáva do obrázka menej šumu. Vďaka tomu je útok C&W z pohľadu človeka "neviditeľnejší" a sofistikovanejší.

== Časová náročnosť a vplyv architektúry
Výsledky jasne ukazujú priamu úmeru medzi hĺbkou modelu a časom potrebným na vygenerovanie útoku.

1. *Ľahké modely:* FaceNet a BenchmarkCNN spracovali jeden obrázok pomocou PGD útoku v priemere za 37 až 47 ms. Tieto modely sú vhodné pre aplikácie vyžadujúce nízku latenciu, no ich menšia kapacita môže byť nevýhodou pri obrane.
2. *Ťažké modely:* ArcFace a AdaFace (využívajúce architektúru IResNet50) potrebovali na rovnaký útok 113 až 116 ms na obrázok.
3. *Algoritmická zložitosť:* Útok C&W je rádovo pomalší (166 až 568 ms na obrázok) v porovnaní s PGD. Tento časový rozdiel je spôsobený nutnosťou optimalizácie pomocou algoritmu Adam a prácou v priestore $text("arctanh")$, čo je daňou za vyššiu kvalitu adversariálneho obrázka.

Zhrnutím možno konštatovať, že kým iteratívne metódy ako PGD predstavujú vynikajúci nástroj na rýchle testovanie robustnosti, útok C&W zostáva zlatým štandardom pre scenáre, kde je prioritou minimalizácia detekovateľnosti útočníka.

#bibliography("citations.bib", style: "iso690-author-date-sk.csl")
#pagebreak(weak: true)

// #resume()[
// #lorem(250)
// ]

// start the appendices section with this line
#show: section-appendices

= Source code <source-code>


= Plán práce <plan-of-work>
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

