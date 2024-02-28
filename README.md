# masinsko-ucenje-radovi

Mašinsko učenje (ML) je oblast proučavanja veštačke inteligencije koja se bavi razvojem i proučavanjem statističkih algoritama koji mogu efikasno da generalizuju i na taj način izvršavaju zadatke bez eksplicitnih uputstava. Pristupi mašinskom učenju su primenjeni na velike jezičke modele, kompjuterski vid, prepoznavanje govora, filtriranje e-pošte, poljoprivredu, medicinu...

Mašinsko učenje je usko povezano sa drugim metodama:

*   Data mining - fokusira se na otkrivanje (ranije) nepoznatih svojstava u podacima
*   Optimizacija - koristi se pri minimizaciji neslaganja između predviđanja modela koji se obučava i stvarnih instanci problema
*   Generalizacija
*   Statistika

Deep learning je podoblast metoda mašinskog učenja zasnovanih na veštačkim **neuronskim mrežama** sa učenjem reprezentacije. Pridev „duboki“ se odnosi na upotrebu više slojeva u mreži.

Neuronska mreža je jedan oblik implementacije sistema veštačke inteligencije, koji predstavlja sistem koji se sastoji od određenog broja međusobno povezanih procesora ili čvorova, ili procesnih elemenata koje nazivamo veštačkim neuronima.

Telo neurona naziva se čvor ili jedinica. Svaki od neurona ima lokalnu memoriju u kojoj pamti podatke koje obrađuje. Podaci koji se obrađuju su lokalni podaci kao i oni koji se primaju preko veze. Podaci koji se ovim kanalima razmenjuju su obično numerički.

Neuronske mreže se koriste koriste za, na primer, prepoznavanje objekata na osnovu njihovih osobina.

### Nadgledano

Kompjuteru se prikazuju primeri ulaza i njihovih željenih izlaza, koje mi dajemo (labeled dataset), a cilj je naučiti opšte pravilo koje mapira ulaze u izlaze. Primeri:
*   Sve rađeno na časovima
*   Prediktivna analiza
*   Spam detection

*   Klasifikacija - precizno dodeljivanje testnih podataka u određene kategorije: knn, linearni klasifikatori
*   Regresija - koristi se za razumevanje odnosa između zavisnih i nezavisnih varijabli: linearna, logistička...

### Nenadgledano

Analiza, grupisanje i obrasci među podacima neoznačenih skupova podataka.
Njegova sposobnost da otkrije sličnosti i razlike u informacijama čini ga idealnim rešenjem za istraživačku analizu podataka, segmentaciju kupaca i prepoznavanje slika.


*   Clustering - grupisanje podataka prema sličnosti ili razlikama
*   Redukcija dimenzionalnosti - smanjuje broj karakteristika (dimenzija) u skupu podataka

Nedostaci:
*   Računarska složenost zbog velikog obima podataka za obuku
*   Duže vreme treninga
*   Veći rizik od netačnih rezultata
*   Ljudska intervencija za validaciju izlaznih varijabli

### Reinforcment

Metoda mašinskog učenja zasnovana na nagrađivanju željenih ponašanja i kažnjavanju neželjenih

## Skup podataka

Ono bez čega ne mogu funkcionisati algoritmi mašinskog učenja jesu podaci, i to u velikim količinama.
Skupovi podataka mogu da budu u različitim oblicima, ali razlikujemo dve različite grupe:

1. Struktuirani podaci (datumi, imena, kraci tekstovi, tabelarni, svaki element ima istu strukturu) - najčešće susrećemo u tabelarnom obliku. Kolone tabele predstavljaju promenjive, dok redovi predstavjaju jedan slučaj (reprezentacija primera entiteta) iz skupa podatka - parametri pacijenta, nekretnine, transakcije u banci...

2. Nestruktuirani podaci (stream-ovi bajtova, veći fajlovi, audio, video, slike, loše ili nepostojeće strukture)

3. Polustruktuirani podaci (CSV, JSON)

### Tipovi promenjivih

Glavna podela tipova promenjivih je sledeća:

- Kategorijske - ne mogu se kvantifikovati
  - Nominalne (bez definisanog poretka - kategorija, ime...)
  - Ordinalne (sa definisanim poretkom - lose - bolje - najbolje)
    
- Numeričke - kvantifikovane
  - Diskretne - konacan broj vrednosti
  - Kontinualne - beskonacan

### Preprocesiranje podataka

#### Pretvaranje tipova
Postoje algoritmi mašinskog učenja koji zahtevaju da sve promenjive budu numeričke.

Nominalne promenjive se pretvaraju u numeričke tako što je primenjuje postupak koji se zove **one-hot encoding**.
Svaka moguća vrednost nominalne promenjive postaje jedna kolona, a jedinice se nalaze u onoj koloni koja reprezentuje originalnu vrednost nominalne promenjive.

| Color | Is_Red | Is_Green | Is_Blue |
| ----- | ------ | -------- | ------- |
| Red   | 1      | 0        | 0       |
| Green | 0      | 1        | 0       |
| Blue  | 0      | 0        | 1       |

Ordinalne promenjive nemaju problem sa pretvaranjem u numeričke jer postoji prirodan poredak.

| StressLevel | SLNumeric |
| ----------- | --------- |
| Low         | 1         |
| High        | 3         |
| Medium      | 2         |

#### Čišćenje podataka

Podaci iz realnog sveta su često nesavršeni i nepogodni za neposrednu primenu u mašinskom učenju.
Stoga je potrebno izvršiti čišćenje.
Najčešće čišćenje podrazumeva:

- Rešavanje nepostojećih vrednosti
- Otklanjanje suvišnih promenjivih

Nepostojeće vrednosti se mogu rešiti na 3 glavna načina:

1. Uklanjanje nepostojeće vrednosti
   1. Uklanjanje cele promenjive tj. kolone
   2. Uklanjanje samo tog reda
      
2. Zamena nepostojećih vrednosti
   1. Koristeći mod
   2. Koristeći medijanu
   3. Koristeći srednju vrednost
      
3. Nepostojanje vrednosti kao vrednost

#### Normalna raspodela

![Alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Standard_deviation_diagram_micro.svg/375px-Standard_deviation_diagram_micro.svg.png)

mi - prosek promenljivih
sigma - standardna devijacija (odstupanje)

#### Centralna granicna teorema
Teorema tvrdi da je suma velikog broja nezavisnih i identično raspoređenih slučajnih promenljivih teži **normalnoj raspodeli verovatnoće**.

#### Normalizacija

Dešava se da različite numeričke promenjive imaju značajno različite domene.
Uzmimo za primer GodišnjiPrihod i GodineStarosti.
Godišnji prihod može varirati od 0 do nekoliko miliona, dok GodineStarosti od 0 do 100 recimo.
Kada se ove dve promenjive prebace u neki vektorski prostor (što se veoma često dešava u algoritmima mašinskog učenja), GodišnjiPrihod bi znatno više uticao na distancu, uglove medju vektorima i sl. 
Da bismo izbegli ovaj problem pribegavamo procesu normalizacije.

Dva tipa normalizacije su:
- Linearna koja koristi linearnu interpolaciju od 0 do 1 gde 0 predstavlja minimalnu vrednost a 1 maksimalnu.
- z-score koja svaku vrednost predstavlja kao odstupanje od srednje vrednosti izraženo u standardnim devijacijama (left i right skew u odnosu na normalnu raspodelu)
- z = (data point - mean)/standard deviation

Prednosti linearne normalizacije su:
1. Jednostavnost
2. Očuvanje relativnog odstojanja

Mane su:
1. Osetljivost na ekstremne vrednost

Prednosti z-score normalizacije su:
1. Otpornost na ekstremne vrednosti
2. Očuvanje oblika raspodele

Mane su:
1. Potencijalni gubitak interpretabilnosti
2. Pretpostavlja da je raspodela normalna

## Obucavanje 

Ukoliko smo uspešno pripremili podatke, možemo pristupiti procesu obuke.
Proces obuke se sastoji od:
1. Podele podataka
2. Definicija funkcije gubitka i validacione metrike
3. Faze obuke
4. Evaluacije modela

### Podela podataka

Prvo i osnovno je da podelimo podatke na trening i testni skup.
Obično trening skup sadrži 80% podataka a testni 20%.

![Alt text](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*Nv2NNALuokZEcV6hYEHdGA.png)


**Trening skup** je skup podataka koji se koristi za **optimizaciju parametara** modela.

**Validacioni skup** podataka se koristi za **praćenje** kvaliteta modela tokom faze obuke na neviđenom skupu podataka.

**Testni skup** služi za procenu kvaliteta modela na do sada neviđenom skupu podataka i radi se nakon faze obuke.

### Definicija funkcije gubitka i validacione metrike

Da bismo mogli da obučavamo modele mašinskog učenja, potrebno je odrediti numeričku vrednost koja predstavlja kvalitet modela i nju zovemo **funkcija gubitka** ili **ciljna funkcija**. 
Ta vrednost će zavisiti od:

1. Skupa podataka (posto se on ne menja - uglavnom ne utice)

2. Hiperparametara modela  (parametri viseg nivoa) - to su vrednosti koje uticu na proces ucenja ili samu arhitekturu modela i ne mogu se menjati kada otpočne faza obuke.
Primeri hiperparametara:
- Trening test split
- Maksimalan broj epoha
- Veličina mini-batch-a
- Broj slojeva u neuronskoj mreži
- Stopa učenja
- Broj suseda koji se koristi u knn algoritmu

3. Parametara modela - uce se i/li procenjuju direktno iz podataka iz treninga i menjaju tokom procesa ucenja
Primeri parametara su koeficijenti linearne i logisticke regresije

Vrednost funkcije gubitka izražava se kao funkcija parametara modela.
**Cilj ucenja jeste da se nađu optimalne vrednosti parametara modela, tj minimizovati fju gubitka.
Optimalne vrednosti parametara modela su one vrednosti za koje je funkcija gubitka optimalna.**

Validaciona metrika takođe može biti ista kao funkcija gubitka, ali ne nužno.

## Faza obuke

Faza obuke predstavlja proces koji se odnosi na **optimizaciju parametara** modela.
U zavisnosti od modela, faza obuke može biti različita.
Kod proste linearne regresije, faza obuke predstavlja rešavanje sistema linearnih jednačina.
Kod nekih drugih modela se primenjuju optimizacione metode ciljne funkcije.

Obuka se vrši na trening skupu, a praćenje procesa obuke na validacionom skupu podataka.

### Evaluacija modela

Često nije poznato unapred koji hiperparametri mogu iznedriti najbolji model niti je moguće sa sigurnošću utvrditi.
Zato se često obučava nekoliko modela sa različitim hiperparametrima i biraju se oni koji imaju najbolje testne metrike.

Ukoliko model ima značajno bolje rezultate na trening skupu nego na testnom skupu, onda kažemo da model nedovoljno dobro generalizuje i kažemo da je model **preučen (overfitted)**.

Ukoliko model nema dovoljno dobre performanse ni na trening ni na testnom skupu onda se kaže da je model **nedoučen (underfitted)**.

## Metrike

Metrike su numericke vrednosti koje imaju ulogu da **kvantifikuju valjanost modela**.

### Regresione metrike

#### MSE (Mean Squared Error)

Srednja kvadratna greška. Predstavlja srednju vrednost sume kvadrata odstupanja predikcije od vrednosti ciljne promenjive. 
Manje vrednosti ukazuju na bolji model.
Može se koristiti kao ciljna funkcija za problem regresije.

![Alt text](https://miro.medium.com/v2/resize:fit:640/format:webp/1*-e1QGatrODWpJkEwqP4Jyg.png)

#### RMSE (Root Mean Squared Error)

Koren srednje kvadratne greške. Isto važi kao i za MSE.

![Alt text](https://editor.analyticsvidhya.com/uploads/56967RMSE.png)

#### MAE (Mean Absolute Error)

Srednja apsolutna greška. Srednja vrednost apsolutnog odstupanja predikcije od vrednosti ciljne promenjive. 
Manje vrednosti ukazuju na bolji model.
Nije preporučljivo da se koristi kao ciljna funkcija za problem regresije jer **nije diferencijablina** na celom svom domenu. Može se koristiti kao validacona metrika.

![Alt text](https://editor.analyticsvidhya.com/uploads/42439Screenshot%202021-10-26%20at%209.34.08%20PM.png)

### Klasifikacione metrike

#### Matrica konfuzije


Matrica konfuzije je alat koji se koristi u evaluaciji performansi klasifikacionih modela u mašinskom učenju

| Actual/Predicted | Positive | Negative |
| ---------------- | -------- | -------- |
| Positive         | TP       | FN       |
| Negative         | FP       | TN       |

-TP - True Positive - Broj tačnih pozitivnih predikcija
-FP - False Positive - Broj netacnih pozitivnih predikcija
-TN - True Negative - Broj netacnih negativnih predikcija
-FN - False Negative - Broj tačnih negativnih predikcija

#### Tačnost (Accuracy)

$$accuracy = \frac{TP + TN}{TP+TN+FP+FN}$$

Udeo tačnih predikcija u ukupnom broju predikcija.

#### Preciznost (Precision, Positive Predictive Value - PPV)

$$precision = \frac{TP}{TP+FP}$$

Udeo tačnih predikcija u ukupnom broju pozitivnih predikcija.

#### Senzitivnost (Sensitivity, Recall, True Positive Rate - TPR)

$$recall = \frac{TP}{TP+FN}$$

Udeo tačno prepoznatih stvarnih pozitivnih slučajeva.

#### Specifičnost (Specificity, True Negative Rate - TNR)

$$specificity = \frac{TN}{TN+FP}$$

Udeo tačno prepoznatih stvarnih negativnih slučajeva.

#### F1 Score

$$F1 = 2 \times\frac{precision \times recall}{precision + recall}$$

Harmonijska sredina preciznosti i senzitivnosti.

Za sve ove vrednosti važi što veće to bolje. Sve vrednosti su u rasponu od 0 do 1.

## Optimizacione metode

Onda kada imamo ciljnu funkciju takvu da je analitičko traženje optimuma ili previše računski skupo ili potpuno neizvodljivo, tada pribegavamo optimizacionim metodama za nalaženje optimuma.

### Gradijentni spust

Gradijentni spust je optimizaciona metoda koja koristi gradijent funkcije za njenu optimizaciju. 
Gradijent je n-dimenzioni vektor čija n-ta komponenta predstavlja parcijalni izvod n-tog parametra u odnosu na ciljnu funkciju.
Gradijent predstavlja vektor u čijem pravcu i smeru **funkcija najviše raste**.

Ideja je da trenutno stanje modela predstavimo kao tačku u n-dimenzionom prostoru (n je broj parametara modela, vrednost svakog parametra je jedna komponenta), a valjanost modela kao funkciju u nad tim prostorom. 

Trenutnu minimizaciju funkcije postižemo tako što pomerimo parametre modela u suprotnom smeru od gradijenta za neki realni umnožak $\eta ( 0 < \eta < 1)$.
Ponavljajuci ovaj postupak postepeno nalazimo parametre modela sa sve boljim performansama. 
Gradijentni spust ne garantuje nalazenje globalnog optimuma.


Postoje tri varijante gradijentnog spusta:

1. Paketni gradijentni spust (Mini-Batch Gradient Descent - MBGD)
2. Gradijentni spust (Gradient Descent - GD)
3. Stohastički gradijentni sipust (Stochastic Gradient Descent - SGD)

**Mini batch** vrši optimizaciju na podskupu skupa podataka koji zovemo **paket** i najčešće je broj slučajeva u paketu stepen dvojke. Gradijent se računa na datom podskupu. Najčešće je najbolji od sva tri jer dovoljan broj puta u jednoj epohi primeni gradijente na model, a svaki gradijent liči dovoljno na gradijent na celom skupu zbog centralne granične teoreme.

**Gradijentni spust** vrši optimizaciju na celom skupu podataka, tj. veličina paketa je ista kao veličina trening skupa. Dobar je na skupovima podataka koji su dovoljno mali. Sporije napreduje od MBGD.

**Stohasticki** vrši optimizaciju na jednom slučaju, tj. veličina paketa je 1.
Nije dobar jer je moguće da različiti slučajevi proizvedu gradijente koji se na kraju potru pa stoga učenje teče najsporije.

#### Early stopping

Kada se koriste optimizacione metode za obučavanje modela mašinskog učenja, ne zna se tačno posle koliko vremena će proces obuke stagnirati i ući u lokalni minimum. 
Kako bismo izbegli **nepotrebno trošenje resursa poput energije i procesorskih ciklusa**, primenjuje se tehnika zvana **rano zaustavljanje**.
Definišu se tolerancija i strpljenje.

**Tolerancija** predstavlja **minimalno poboljšanje validacione metrike** koje se smatra značajnim. 
**Strpljenje** predstavlja **broj epoha** koji mora da prodje bez značajnog poboljšanja da bi se proces obuke završio ranije nego predviđeno.

## Modeli

### Linearna regresija

Linearna regresija predstavlja model mašinskog učenja koji je pogodan za modelovanje linearnih odnosa medju promenjivama.
Cilj nam je da, sa što manjom greškom, predvidimo vrednost zavisne na osnovu vrednosti nezavisne promenjive.
S obzirom da imamo samo dva parametra od kojih direktno zavisi suma kvadrata reziduala, možemo analitički izračunati optimalne parametre alfa i beta.

### Logisticka regresija
