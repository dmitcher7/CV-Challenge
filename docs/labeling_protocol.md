# 📋 Team Data Annotation Protocol: Voetbal & Bal-detectie

**Team:** Git Push and Pray  
**Datum:** Oktober 2026  
**Doel:** Strikte standaardisatie van de data-annotatie om *label noise* te elimineren en reproduceerbaarheid tussen teamleden te garanderen.

---

## 1. Klassenstructuur & Doel (The Data Gap)
We gebruiken een **pre-trained YOLO model (COCO weights)** als backbone voor de spelers (`person`, class 0). De doelgerichte fijnregeling focust primair op het grote data-tekort: de **bal** (`ball`).

| Klasse ID | Naam | Beschrijving | Pre-trained vs Custom |
| :---: | :---: | :--- | :--- |
| **0** | `ball` | De wedstrijdbal (voetbal) | **Custom fine-tuned (onze dataset)** |
| *(optioneel)* | `player A` | Veldspelers | Pre-trained COCO `person` | *Donkerblauw -> PSG*
| *(optioneel)* | `player B` | Veldspelers | Pre-trained COCO `person` | *Geel -> Inter Milan"
| *(optioneel)* | `referee` | Scheidsrechter / grensrechters | Pre-trained COCO `person` | *Fluo*
---

## 2. Gouden Labelregels voor het Team

### A. Tightness (Strakheid van de box)
* De bounding box moet **exact de fysieke randen** van de bal omsluiten.
* **Schaduwen op het gras worden NOOIT meegelabeld.** Een schaduw verandert mee met de zonnestand en leidt tot false positives op donkere plekken in het veld.

### B. Motion Blur (Harde schoten en snelle passes)
* Bij hoge balsnelheid ontstaat er een uitgerekte vlek of veeg (motion blur streak).
* **Afspraak:** Teken de bounding box om de **volledige zichtbare blur streak**. Het model moet leren dat deze elliptische vervaging op topsnelheid de bal representeert.

### C. Occlusions (Verstopt achter een been, speler of paal)
* **$\ge 30\%$ zichtbaar:** Wel annoteren. Teken de box strak om het **zichtbare gedeelte** van de bal.
* **$< 30\%$ zichtbaar of compleet verborgen:** **NIET labelen.** Het forceren van een label op onzichtbare pixels ("hallucineren") leidt tot gradients op achtergrondpixels en verslechtert de precisie.

### D. Truncation (Rand van het camerabeeld)
* Als de bal gedeeltelijk buiten het camerabeeld valt: alleen labelen als minimaal de helft ($>50\%$) zichtbaar is.

---

## 3. Dataset Samenstelling & Train/Val Split

* **Doelomvang:** 40 tot 80 frames geëxtraheerd uit de 20-seconden videoclip.
* **Selectiecriteria voor frames (Diversiteit):**
  * Minimaal 20% frames met stilstaande bal (aftrap, vrije trap, rustige fase).
  * Minimaal 40% frames met dynamische actie (passes, harde schoten, bal in de lucht).
  * Minimaal 20% frames met occlusies / duels (meerdere spelers dicht bij de bal).
  * 10% lege frames zonder bal (ter evaluatie van False Positives).
* **Split Strategie:**
  * **80% Train** / **20% Validation**.
  * **Belangrijk:** Om data-leakage te voorkomen tussen opeenvolgende frames, splitsen we op tijdblokken (chronologisch of chunk-based) en niet puur willekeurig.

---

## 4. Kwaliteitscontrole (Peer Review)
Voordat we gaan trainen, valideert elk teamlid een batch labels van een ander teamlid (cross-validation van labels) om inconsistenties direct op te sporen en te corrigeren.
