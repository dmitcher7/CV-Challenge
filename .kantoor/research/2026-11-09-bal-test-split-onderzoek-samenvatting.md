# Samenvatting onderzoek: bal 0 keer correct in test-split

- **Datum:** 2026-11-09
- **Type:** Onderzoek (losse opdracht, geen plan, uitgevoerd door drie Sonnet-agents)
- **Onderwerp:** Waarom de bal in de test-split van de controlecel (cel 17) van `kobe/notebooks/minimap/02_minimap_video.ipynb` 0 keer correct wordt gedetecteerd.

## Belangrijkste bevindingen

1. **Geen koppelingsfout:** Test-afbeeldingen in `kobe/notebooks/dataset/test` zijn exact dezelfde frames als `data/frames` (zelfde index, 1920x1080, geen resize/crop); coördinaten en homografie kloppen.
2. **Labels zijn juist:** Visueel gecontroleerd; het label ligt op de kleine witte bal bij de voeten van de Inter-keeper aan de doellijn.
3. **Niet modelspecifiek:** Alle drie de modellen scoren 0 op test (per frame correct train/val/test: eigen `yolo26l_1920` 13/51, 8/23, 0/15; Brent oud 17/51, 17/23, 0/15; `brent_v2` 29/51, 19/23, 0/15). `brent_v2` presteert het best op train en val.
4. **Oorzaak:** De test-split (frames 600-659, 15 frames) heeft slechts 5 frames met een bal-label (604, 636, 640, 644, 648), afkomstig uit dezelfde korte scène. De bal is daar circa 5 pixels groot, bevindt zich tegen de keeper en het witte net, en wordt door geen enkel model herkend. Modellen verwarren de strafschopstip (terecht weggefilterd in cel 15) en witte schoenen/benen van spelers met de bal (4x 'wrong place', 4x 'false ball').
5. **Documentatiefout:** De tekst in sectie 9 van `kobe/notebooks/01_split_images.ipynb` is onjuist: er wordt vermeld dat de bal in frames 600-659 niet zichtbaar is, terwijl deze in 5 frames wel degelijk zichtbaar is.

## Aanbevelingen (open)
- Test-frames verspreid over de hele video kiezen in plaats van enkel het laatste stuk.
- Meer trainingsvoorbeelden toevoegen met een kleine bal in de buurt van doel en keeper.
- In het verslag expliciet toelichten dat een score van 0 op de test-split neerkomt op 0 uit 5 moeilijke frames uit één specifieke scène.
- Er is niets in het project aangepast tijdens dit onderzoek.
