# Izbor metode za isecanje regiona usana (ROI) na AI-SPEAK korpusu

> Materijal za izveštaj (task 6), sekcija *Metode i materijali → Preprocesiranje*.
> Empirijski potvrđeno na 4 govornika iz AI-SPEAK baze (2026-08-21).

## Problem

AV-HuBERT očekuje na ulazu niz sivih slika regiona usana dimenzija **96×96**,
tesno centriranih oko usta (isti format kao LRS3 na kome je model pretreniran).
AI-SPEAK frontalni snimci su, međutim, kadrovi rezolucije **1080×1920** koji su
**pretežno crni** — vidljiv je samo anonimizovani „prozor” lica (usta oštra, gornji
deo lica pikselizovan, ostatak kadra maskiran crnom bojom). Položaj tog prozora
**varira od govornika do govornika** (izmereni centri sadržaja kretali su se od
frakcije (0.41, 0.41) do (0.60, 0.35) kadra), pa fiksno isecanje nije moguće, a
naivno skaliranje celog kadra 1080×1920 → 96×96 bi izobličilo sliku i svelo usta
na nekoliko piksela. Potrebno je **isecanje po kadru** koje pronalazi usta i
proizvodi standardizovani 96×96 ROI.

## Razmatrane metode

**(A) dlib — detekcija 68 tačaka lica (metoda samog AV-HuBERT-a).**
Zvanični AV-HuBERT preprocesing koristi dlib detektor lica + prediktor 68
orijentira, iz kojih se izdvajaju tačke usana i formira centriran isečak. Prednost:
identično onome što je model „video” na LRS3.

**(B) Content-bbox heuristika (naša metoda).**
Pošto je kadar pretežno crn sa licem kao svetlim „ostrvom”, pitanje „gde je lice?”
svodi se na „gde su ne-crni pikseli?”. Algoritam (numpy + OpenCV, bez dodatnih
zavisnosti i bez modela):
1. maska = (sivi_intenzitet > 15) → koordinate ne-crnih piksela;
2. granični okvir [x0, x1, y0, y1] oko njih;
3. centar isečka: cx = (x0+x1)/2, cy = y0 + 0.66·(y1−y0) (usta su ~2/3 niz okvir);
4. veličina isečka: pola-stranice = 0.34·(x1−x0);
5. isecanje kvadrata i skaliranje na 96×96 sivih tonova.

Konstante 0.66 i 0.34 su empirijski podešene i pokazale su se stabilnim.

## Rezultat poređenja (4 govornika: spk01, spk03, spk06, spk11)

| Govornik | dlib | content-bbox |
|---|---|---|
| spk01 | ✗ lice nije detektovano | ✓ čist, centriran ROI |
| spk03 | ✗ lice nije detektovano | ✓ čist, centriran ROI |
| spk06 | ✓ | ✓ |
| spk11 | ✓ | ✓ |

- **dlib je zakazao na 2 od 4 govornika (50%).** Pikselizacija gornjeg dela lica i
  crna pozadina razbijaju dlib-ov detektor frontalnog lica, pa on ne vraća nijednu
  detekciju. Gubitak polovine govornika je neprihvatljiv.
- **Content-bbox je uspešno obradio svih 4/4 govornika**, uz **stabilnost kroz
  vreme**: testirano na po 3 vremenska trenutka po klipu (usta u različitim
  položajima dok govornik priča) — usta ostaju centrirana kroz ceo iskaz.
- Kvalitet content-bbox isečaka je uporediv sa dlib isečcima tamo gde dlib radi
  (spk06, spk11), uz blagu varijaciju u „zumu” po govorniku (spk11 nešto šire,
  spk01 tešnje) — svi upotrebljivi.

## Odluka i obrazloženje

**Usvojena je content-bbox metoda (B).**

Obrazloženje: dlib je moćniji, ali *pretpostavlja normalno, celo lice* i zato je
**krhak** na anonimizovanom formatu AI-SPEAK-a — zakazuje na polovini govornika.
Content-bbox pravi mnogo *jednostavniju* pretpostavku („ne-crno = lice”) koja je za
ovaj konkretni format podataka **pouzdana i robusna** (100% govornika), bez ikakvog
dodatnog modela ili zavisnosti. Ovo je primer principa da jednostavnija metoda,
prilagođena stvarnoj strukturi podataka, može nadmašiti generičku „pametniju”
metodu kada su ulazni podaci van distribucije za koju je ta metoda građena.

Ograničenje: heuristika zavisi od toga da je kadar zaista pretežno crn sa jednim
svetlim regionom lica — što jeste slučaj kod AI-SPEAK anonimizacije. Zbog toga se
posle preprocesiranja radi vizuelna i automatska validacija isečaka nad svih 30
govornika (kontakt-list uzoraka + provera udela crne površine po ROI-u).
