import os
import base64
import subprocess
from pathlib import Path

# Paths
REPO_ROOT = Path(r"c:\Users\dimit\Documents\01 - Thomas More\Thomas More Jaar 3\01 - Semester 1\Deep Learning\Project 2 - CV\CV-Challenge")
USER_IMG_PATH = Path(r"C:\Users\dimit\.gemini\antigravity\brain\27447295-8834-4623-a1e2-436d23b49c80\.user_uploaded\media_1794673222159_5f345b91.jpg")
IMG_BEELDGROOTTE_PATH = REPO_ROOT / "final_upload" / "uitleg" / "img" / "04_effect_beeldgrootte.png"
IMG_HEATMAP_PATH = REPO_ROOT / "final_upload" / "uitleg" / "img" / "07_heatmaps.png"
IMG_SMOOTH_PATH = REPO_ROOT / "final_upload" / "uitleg" / "img" / "07_ruw_vs_afgevlakt.png"
CHROME_PATH = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

PDF_DIR = REPO_ROOT / "final_upload" / "pdf"
PDF_DIR.mkdir(parents=True, exist_ok=True)
TARGET_PDF = PDF_DIR / "CV_Challenge_Report_Team3.pdf"

def get_base64(path):
    if path.exists():
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

user_img_b64 = get_base64(USER_IMG_PATH)
img_beeldgrootte_b64 = get_base64(IMG_BEELDGROOTTE_PATH)
img_heatmap_b64 = get_base64(IMG_HEATMAP_PATH)
img_smooth_b64 = get_base64(IMG_SMOOTH_PATH)

html_template = """<!DOCTYPE html>
<html lang="nl">
<head>
<meta charset="UTF-8">
<title>Computer Vision Challenge - Team 3 Report</title>
<style>
  @page {
    size: A4;
    margin: 16mm 19mm 15mm 19mm;
    @bottom-center {
      content: counter(page);
      font-size: 8.5pt;
      color: #64748b;
    }
  }

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #1e293b;
    background-color: #ffffff;
    font-size: 8.7pt;
    line-height: 1.50;
  }

  /* Page Break Helpers */
  .page-break {
    page-break-after: always;
    break-after: page;
  }
  .avoid-break {
    page-break-inside: avoid;
    break-inside: avoid;
  }

  /* Professional Header */
  .header-box {
    border-bottom: 2px solid #1e3a5f;
    padding-bottom: 9px;
    margin-bottom: 12px;
  }

  .institution-kicker {
    font-size: 7.9pt;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #0369a1;
    font-weight: 700;
    margin-bottom: 3px;
  }

  .main-doc-title {
    font-size: 14.2pt;
    font-weight: 800;
    color: #0f2942;
    line-height: 1.2;
    margin-bottom: 3px;
    white-space: nowrap;
    letter-spacing: -0.2px;
  }

  .sub-doc-title {
    font-size: 9.6pt;
    color: #475569;
    font-weight: 400;
  }

  /* Team Card */
  .team-panel {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 4px solid #1e3a5f;
    border-radius: 5px;
    padding: 8px 12px;
    margin-bottom: 11px;
    break-inside: avoid;
    page-break-inside: avoid;
  }

  .team-panel-title {
    font-size: 9.8pt;
    font-weight: 700;
    color: #0f2942;
    margin-bottom: 3px;
  }

  .team-members-grid {
    display: flex;
    gap: 16px;
    font-size: 8.8pt;
    color: #334155;
    flex-wrap: wrap;
    margin-bottom: 3px;
  }

  .team-members-grid span strong {
    color: #0f2942;
  }

  .team-meta-note {
    font-size: 7.9pt;
    color: #64748b;
    border-top: 1px solid #e2e8f0;
    padding-top: 3px;
    margin-top: 3px;
    white-space: nowrap;
  }

  /* Hero Standalone Image (Border-free, clean, no caption) */
  .hero-image-box {
    margin: 11px 0 13px 0;
    text-align: center;
    border: none;
    background: transparent;
    padding: 0;
    break-inside: avoid;
    page-break-inside: avoid;
  }

  .hero-image-box img {
    width: 100%;
    max-height: 165px;
    object-fit: contain;
    border: none;
    border-radius: 4px;
    display: block;
    margin: 0 auto;
  }

  /* Headings Hierarchy */
  h1 {
    font-size: 12.2pt;
    font-weight: 700;
    color: #1e3a5f;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 3px;
    margin-top: 16px;
    margin-bottom: 8px;
    break-after: avoid;
    page-break-after: avoid;
  }

  h1.project-intro-title {
    margin-top: 28px;
    margin-bottom: 9px;
  }

  h1.section-spacious {
    margin-top: 26px;
  }

  h1.section-extra-spacious {
    margin-top: 32px;
  }

  h2 {
    font-size: 9.9pt;
    font-weight: 700;
    color: #0f766e;
    margin-top: 13px;
    margin-bottom: 5px;
    break-after: avoid;
    page-break-after: avoid;
  }

  h3 {
    font-size: 9pt;
    font-weight: 700;
    color: #334155;
    margin-top: 10px;
    margin-bottom: 4px;
    break-after: avoid;
    page-break-after: avoid;
  }

  p {
    margin-bottom: 7px;
    text-align: justify;
  }

  /* Figure Card (for later pages) */
  .figure-card {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    padding: 6px;
    margin: 10px 0;
    text-align: center;
    break-inside: avoid;
    page-break-inside: avoid;
  }

  .figure-card img {
    width: 100%;
    max-height: 190px;
    object-fit: cover;
    border-radius: 3px;
    display: block;
  }

  .figure-caption {
    font-size: 7.6pt;
    color: #475569;
    margin-top: 5px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 4px;
  }

  /* KPI Summary Bar */
  .kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 9px;
    margin: 11px 0;
    break-inside: avoid;
    page-break-inside: avoid;
  }

  .kpi-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #0284c7;
    border-radius: 4px;
    padding: 5px 9px;
  }

  .kpi-card.teal {
    border-left-color: #0f766e;
  }

  .kpi-card.amber {
    border-left-color: #d97706;
  }

  .kpi-label {
    font-size: 6.9pt;
    text-transform: uppercase;
    color: #64748b;
    font-weight: 700;
  }

  .kpi-value {
    font-size: 11.2pt;
    font-weight: 800;
    color: #0f2942;
  }

  .kpi-note {
    font-size: 6.8pt;
    color: #64748b;
  }

  /* Callout */
  .info-callout {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 4px solid #1e3a5f;
    border-radius: 4px;
    padding: 8px 12px;
    margin: 12px 0;
    font-size: 8.6pt;
    break-inside: avoid;
    page-break-inside: avoid;
  }

  .info-callout-title {
    font-weight: 700;
    color: #0f2942;
    margin-bottom: 4px;
    font-size: 8.9pt;
  }

  /* Side-by-side Images */
  .grid-two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 12px 0;
    break-inside: avoid;
    page-break-inside: avoid;
  }

  /* Lists */
  ul {
    margin-left: 17px;
    margin-bottom: 7px;
  }

  li {
    margin-bottom: 2px;
    font-size: 8.6pt;
  }

  li strong {
    color: #0f2942;
  }

  /* Tables */
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 8.2pt;
    break-inside: avoid;
    page-break-inside: avoid;
  }

  th, td {
    padding: 5px 8px;
    border: 1px solid #e2e8f0;
    text-align: left;
  }

  th {
    background: #f1f5f9;
    color: #0f2942;
    font-weight: 700;
  }

  tr:nth-child(even) {
    background: #f8fafc;
  }
</style>
</head>
<body>

  <!-- ==================== PAGE 1 ==================== -->
  <div class="header-box">
    <div class="institution-kicker">Thomas More Geel &bull; Toegepaste Informatica &bull; Deep Learning &amp; Computer Vision</div>
    <div class="main-doc-title">Computer Vision Challenge: Sports Vision &amp; Tactical Mapping</div>
    <div class="sub-doc-title">Automatische objectdetectie, 2D-homografie en tactische match-analyse van PSG vs. Inter</div>
  </div>

  <div class="team-panel">
    <div class="team-panel-title">Team 3: Git Push &amp; Pray</div>
    <div class="team-members-grid">
      <span><strong>Dmitrij Tchervonisjenko</strong></span>
      <span><strong>Kobe Vandenberk</strong></span>
      <span><strong>Cisse Vandeweyer</strong></span>
      <span><strong>Brent Van Gelder</strong></span>
    </div>
    <div class="team-meta-note">
      Video: PSG vs. Inter (660 frames, 30 FPS) &bull; Model: YOLO26l (26.2M) &bull; PyTorch, CUDA &amp; OpenCV &bull; Deployment: Streamlit
    </div>
  </div>

  <div class="hero-image-box">
    <img src="data:image/jpeg;base64,__USER_IMG_B64__" alt="Match detectie en 2D-minimap">
  </div>

  <div class="kpi-row">
    <div class="kpi-card">
      <div class="kpi-label">Spelers F1-Score</div>
      <div class="kpi-value">0.99</div>
      <div class="kpi-note">Gemeten op 28 GT-frames</div>
    </div>
    <div class="kpi-card teal">
      <div class="kpi-label">Rolaccuraatheid</div>
      <div class="kpi-value">99.2%</div>
      <div class="kpi-note">321 van 324 rollen correct</div>
    </div>
    <div class="kpi-card amber">
      <div class="kpi-label">Homografiefout</div>
      <div class="kpi-value">0.16 m</div>
      <div class="kpi-note">Mediane fout via RANSAC</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Bal Terugvinding</div>
      <div class="kpi-value">19 / 21</div>
      <div class="kpi-note">Op 1280px testresolutie</div>
    </div>
  </div>

  <h1 class="project-intro-title">Projectomschrijving &amp; Doelstelling</h1>
  <p>
    Voor deze Computer Vision Challenge ontwikkelen we een toepassing die voetbalbeelden automatisch analyseert en omzet naar een overzichtelijke tweedimensionale tactische weergave van het voetbalveld. De toepassing vertrekt vanuit een vooraf opgenomen voetbalvideo van 22 seconden (660 frames, 30 beelden per seconde, 1920 &times; 1080) van de topaffiche tussen Paris Saint-Germain en Inter Milan.
  </p>
  <p>
    Het model herkent automatisch alle belangrijke objecten op het veld: de spelers van beide teams, de doelman, de scheidsrechters en de voetbal. De gedetecteerde objecten worden rechtstreeks op de oorspronkelijke videobeelden aangeduid met behulp van bounding boxes en bijhorende labels. Naast de geannoteerde videobeelden voorzien we een tweede weergave in de vorm van een 2D-bovenaanzicht van het veld waarop de posities synchroon worden geprojecteerd.
  </p>
  <p>
    Het hoofddoel van dit project is om aan te tonen hoe computer vision ruwe uitzendbeelden kan transformeren naar bruikbare ruimtelijke informatie. Hierdoor krijgen trainers en analisten direct inzicht in formaties, onderlinge afstanden, dekkingsgraden en balbezit.
  </p>

  <div class="page-break"></div>

  <!-- ==================== PAGE 2 ==================== -->
  <h1>1. Aanpak, Obstakels &amp; Oplossingen</h1>

  <h2>Targeted Data Annotation &amp; Labelprotocol</h2>
  <p>
    De eerste stap was het samenstellen van een kwalitatieve dataset. We extraheerden alle 660 frames uit de video en brachten deze onder in Roboflow. In totaal hebben we 136 representatieve frames gelabeld, goed voor <strong>3.305 bounding boxes</strong> verdeeld over vijf klassen: <code>ball</code> (105 stuks), <code>player_A</code> (PSG donkerblauw, 1.351 stuks), <code>player_B</code> (Inter geel, 1.348 stuks), <code>player_B_Goalkeeper</code> (doelman oranje, 134 stuks) en <code>referee</code> (turquoise/groen, 367 stuks).
  </p>
  <p>
    Om consistent te labelen hebben we vooraf duidelijke afspraken vastgelegd in een <strong>annotatieprotocol</strong>:
  </p>
  <ul>
    <li><strong>Geen grondschaduw:</strong> De bounding box sluit strak aan op het lichaam en de bal, zonder schaduwen mee te nemen.</li>
    <li><strong>Motion blur meenemen:</strong> Bij snelle acties werd de volledige bewegingsveeg van de bal in de box opgenomen om het netwerk te leren dat een onscherpe veeg ook de bal representeert.</li>
    <li><strong>Occlusie en beeldrand:</strong> Een bal werd enkel gelabeld als minstens 30% zichtbaar was; aan de rand van het beeld moest meer dan de helft zichtbaar zijn.</li>
    <li><strong>Peer review:</strong> Elk teamlid controleerde het labelwerk van een ander teamlid om inconsistenties meteen te corrigeren.</li>
  </ul>

  <h2>Datasetvoorbereiding en Preventie van Data Leakage</h2>
  <p>
    Omdat opeenvolgende videoframes sterk op elkaar lijken, zou een willekeurige splitsing leiden tot <strong>data leakage</strong>: het netwerk herkent dan naburige beelden uit het geheugen in plaats van generaliseerbare patronen te leren. We hebben daarom gekozen voor een <strong>temporele opsplitsing in blokken</strong> met bufferzones tussen train, validatie en test. Daarnaast werden statische duplicaatframes (veroorzaakt door de conversie van 25 naar 30 FPS) verwijderd.
  </p>

  <h2>Model Fine-Tuning &amp; GPU-Acceleratie</h2>
  <p>
    Tijdens de eerste trainingen duurde het proces op de CPU te lang. We hebben PyTorch geconfigureerd met <strong>CUDA op dedicated NVIDIA GeForce RTX GPU's</strong>, waardoor de training automatisch versnelde en een volledige run van 100 epochs binnen circa 25 minuten afgerond was. Hierdoor konden we negen trainingsruns vergelijken.
  </p>
  <p>
    Het finale model is <strong>YOLO26l</strong> (26,2 miljoen parameters), getraind vanaf COCO-basisgewichten over 100 epochs (batch size 2, gradient accumulation tot 64, AdamW met cosinus learning rate decay en early stopping na 25 epochs). Belangrijke keuzes bij de data-augmentatie: <strong>Mosaic uitgeschakeld</strong> (zodat de kleine bal niet kromp tot minder dan 8 pixels), lichte Mixup (0.15) en scaling voor robuustheid tegen overlappende spelers, en beperkte HSV-kleuraugmentatie (tintvariatie slechts 0.015) om de shirtkleuren betrouwbaar te houden.
  </p>

  <div class="figure-card">
    <img src="data:image/png;base64,__IMG_BEELDGROOTTE_B64__" style="max-height: 95px; object-fit: contain;" alt="Effect van beeldgrootte">
    <div class="figure-caption">
      <span><strong>Figuur 2:</strong> Effect van inferentiebeeldgrootte op baldetectie: 1280px vindt 19 van de 21 ballen terug (tegenover 16 op 960px).</span>
    </div>
  </div>

  <h2>Vastgestelde Foutanalyse &amp; Gerichte Oplossingen</h2>
  <div class="info-callout">
    <div class="info-callout-title">Belangrijkste Foutbronnen en Oplossingen</div>
    <ul>
      <li><strong>Spelersoverlappingen:</strong> Bij drukte rond de middellijn en het doelgebied zag het model soms één gecombineerde bounding box. Dit losten we op door Non-Maximum Suppression (NMS) strak af te stellen en een dubbelenfilter (IoU &gt; 0.6) toe te passen.</li>
      <li><strong>Baldetectie &amp; Valse Positieven:</strong> De bal is klein (ongeveer 16 pixels) en werd soms verward met witte schoenen of de penaltystip. We hebben dit opgelost door de inferentie uit te voeren op <strong>1280 pixels</strong> in plaats van 960 (waardoor 19 van de 21 ballen werden gevonden, zie Figuur 2), de confidence-drempel voor de bal op 0.20 te zetten en een ruimtelijke uitsluiting toe te passen rond de penaltystippen.</li>
    </ul>
  </div>

  <div class="page-break"></div>

  <!-- ==================== PAGE 3 ==================== -->
  <h1>Homography &amp; 2D Minimap Projection</h1>
  <p>
    Om de spelers en de bal vanuit het schuine perspectief van de tv-camera nauwkeurig om te rekenen naar een tweedimensionaal bovenaanzicht van 105 &times; 68 meter, maken we gebruik van een <strong>homografietransformatie</strong> via OpenCV (<code>cv2.findHomography</code>). Hierbij geldt de <strong>planar assumption</strong>: we beschouwen het speelveld als een plat tweedimensionaal vlak (Z = 0).
  </p>
  <p>
    Een veelvoorkomend probleem bij homografie is perspectivische parallaxfout: wanneer het middelpunt van een spelerbox wordt getransformeerd, belandt de positie door de hoogte van het lichaam meters te ver naar achteren (een spelerhoofd belandt gemiddeld 6,8 meter te ver). We lossen dit structureel op door niet het boxmidden te projecteren, maar het exacte contactpunt met het gras bij de <strong>voeten van de speler</strong> (<code>x_center, y_bottom</code>).
  </p>
  <p>
    Voor de continu meebewegende en zoomende uitzendcamera hebben we <strong>305 veldpunten gelabeld over 39 keyframes</strong> (hoeken van het strafschopgebied, doelgebied, doellijn, penaltystippen en middellijn). Tussenliggende frames worden temporeel geïnterpoleerd. Door uitschieters en foute kliks automatisch weg te filteren via het <strong>RANSAC-algoritme</strong>, behalen we een mediane herprojectiefout van slechts <strong>0,16 meter</strong> op de geteste veldpunten.
  </p>

  <h1 class="section-extra-spacious">2. Lokale Interface &amp; Deployment (Streamlit Dashboard)</h1>
  <p>
    Voor de eindgebruiker bouwden we een performante, interactieve webapplicatie met behulp van <strong>Streamlit</strong> (op te starten via <code>streamlit run app/app.py</code>). De interface biedt een intuïtief dashboard waarin de gebruiker de videoclip kan laden, drempelwaarden kan aanpassen en de analyse realtime kan volgen.
  </p>
  <p>
    De interface bestaat uit twee gesynchroniseerde panelen: enerzijds de originele videoweergave met bounding boxes, teamidentificatie, scheidsrechters en de bal, en anderzijds een meelopende 2D-tactische minimap waarop alle spelersposities en camerabegrenzingen (Field of View) synchroon worden geprojecteerd.
  </p>

  <div class="info-callout">
    <div class="info-callout-title">Detectie-Caching &amp; Kwaliteitsborging</div>
    <p>
      Om de applicatie responsief te houden op apparaten zonder dedicated GPU, worden de YOLO-detecties eenmalig berekend op een lage confidence-drempel (conf 0.05) en opgeslagen in een cachebestand op basis van een unieke MD5-inhoudshash van de videoclip en de modelgewichten. Hierdoor kan de gebruiker in de Streamlit-sidebar live spelen met drempelwaarden zonder wachttijd.
    </p>
    <p style="margin-bottom: 0;">
      De codebasis is strikt modulair gestructureerd (scheiding van detectie, homografie, tracking en visualisatie) en wordt bewaakt door <strong>77 automatische unittests</strong> via pytest, waarmee wiskundige transformaties, matrixinterpolaties en gegevensformaten continu gevalideerd worden.
    </p>
  </div>

  <div class="page-break"></div>

  <!-- ==================== PAGE 4: THE EXTRA MILE ONLY ==================== -->
  <h1 class="section-spacious">3. The Extra Mile: Technische Uitbreidingen (Extra Credits)</h1>
  <p>
    Bovenop de basisvereisten van de challenge heeft ons team vier geavanceerde technische extensies geïmplementeerd om de robuustheid, consistentie en analytische waarde van het platform naar een professioneel niveau te tillen:
  </p>

  <h2>Tracking en Trajectorie van de Bal (ByteTrack &amp; Kalman-filter)</h2>
  <p>
    Spelers worden over opeenvolgende frames stabiel gevolgd met <strong>ByteTrack</strong> via het Hongaarse algoritme op basis van bounding box overlapping en bewegingsvoorspelling. Elke speler behoudt hierdoor een uniek ID doorheen de clip. De uiteindelijke teamtoewijzing wordt bepaald door een confidence-gewogen meerderheidsstemming over alle gedetecteerde frames van dat track-ID, wat resulteert in een <strong>rolconsistentie van 99.04%</strong> (geen flikkerende teamkleuren).
  </p>
  <p>
    Voor de bal ontwikkelden we een op maat gemaakt <strong>Kalman-filter in veldmeters</strong> (toestand vector: x, y, vx, vy) met een dynamische zoekpoort (2 m basispoort + 1.5 m per frame zonder meting). Dit filter overbrugt occlusies en gemiste detecties tot 15 frames (0.5 s) via lineaire toestandsextrapolatie en negeert statische reflecties zoals de penaltystip.
  </p>

  <h2>Ruimtelijke Veldbezetting en Warmtekaarten (KDE)</h2>
  <p>
    We genereren tactische bezettingskaarten met behulp van <strong>tweedimensionale Gaussiaanse Kernel Density Estimation (KDE)</strong> op de getransformeerde veldposities. Hiermee worden de ruimtelijke dominantiezones en de formaties van PSG en Inter direct zichtbaar gemaakt voor tactische coaches (zie Figuur 3).
  </p>

  <h2>Geautomatiseerde Wedstrijdstatistieken &amp; Trajectorie-afvlakking</h2>
  <p>
    Het platform berekent continu de afgelegde afstanden, snelheden en balbezit. Omdat lichte trillingen van de tv-camera leidden tot een overschatting van de loopafstanden (zonder correctie bedroeg de cumulatieve meetafwijking +59%), passen we een <strong>afvlakking toe via een voortschrijdend gemiddelde over 0,5 seconden</strong> (zie Figuur 4). Daarnaast analyseert het systeem continu het balbezit met hysterese (PSG 42.3%, Inter 27.1%, Betwist 30.7%) en herkent het schoten op doel.
  </p>

  <div class="grid-two-col">
    <div class="figure-card" style="margin: 0;">
      <img src="data:image/png;base64,__IMG_HEATMAP_B64__" style="max-height: 105px; object-fit: contain;" alt="Heatmaps">
      <div class="figure-caption">
        <span><strong>Figuur 3:</strong> Territoriale bezettingsheatmaps (PSG links, Inter rechts) via 2D Gaussiaanse KDE.</span>
      </div>
    </div>
    <div class="figure-card" style="margin: 0;">
      <img src="data:image/png;base64,__IMG_SMOOTH_B64__" style="max-height: 105px; object-fit: contain;" alt="Trajectorie-afvlakking">
      <div class="figure-caption">
        <span><strong>Figuur 4:</strong> Ruwe vs. afgevlakte trajectorie (elimineert 59% meetruis door cameratrilling).</span>
      </div>
    </div>
  </div>

  <div class="page-break"></div>

  <!-- ==================== PAGE 5 ==================== -->
  <h1 class="section-spacious">4. Gebruik van Generatieve AI Tools (GenAI)</h1>
  <p>
    Tijdens het ontwerpproces heeft ons team doelgericht en transparant samengewerkt met Generatieve AI-assistenten (Claude Code, Gemini en GitHub Copilot). De AI-tools werden ingezet als sparringpartner en productiviteitsversneller binnen duidelijke kaders:
  </p>
  <ul>
    <li><strong>Omgevingsconfiguratie &amp; CUDA-optimalisatie:</strong> Assistentie bij het correct inrichten van PyTorch met CUDA FP16-ondersteuning op NVIDIA RTX GPU's en het beheren van videogeheugen.</li>
    <li><strong>Modulaire Architectuur &amp; Testautomatisatie:</strong> Ondersteuning bij het ontwerpen van een schaalbare datastructuur en het schrijven van de testsuite van 77 pytest unittests.</li>
    <li><strong>Wiskundige Verificatie:</strong> Controle van de algebraïsche stappen bij de Direct Linear Transform (DLT), homogene coördinaten en de toestandsovergangen van het Kalman-filter in meters.</li>
    <li><strong>Kritische Menselijke Evaluatie:</strong> Elk gegenereerd script en voorstel werd handmatig gecontroleerd, door het team doorgelicht en lokaal getest op de dataset voordat het in de repository werd opgenomen.</li>
  </ul>

  <h1 class="section-extra-spacious">5. Conclusie &amp; Final Remarks</h1>
  <p>
    Met dit project tonen we aan dat een zorgvuldig geannoteerde dataset van 136 beelden en een goed doordachte computer vision-pijplijn volstaan om professionele wedstrijdanalyses uit te voeren. De combinatie van YOLO26l, ByteTrack, homografische veldprojectie en Kalman-filtering levert een betrouwbaar en reproduceerbaar eindresultaat:
  </p>

  <table>
    <thead>
      <tr>
        <th>Pijplijncomponent</th>
        <th>Gebruikte Technologie</th>
        <th>Gevalideerd Resultaat &amp; Metriek</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Objectdetectie</strong></td>
        <td>YOLO26l op 1280px inferentie</td>
        <td>F1-score 0.99 op spelers, 19/21 ballen teruggevonden</td>
      </tr>
      <tr>
        <td><strong>Spelerstracking</strong></td>
        <td>ByteTrack + Majority Voting</td>
        <td>41 unieke tracks, 99.04% rol- en teamconsistentie</td>
      </tr>
      <tr>
        <td><strong>Bal State Estimation</strong></td>
        <td>Kalman Filter in meters + Gating</td>
        <td>Occlusie-overbrugging tot 15 frames, stabiel traject</td>
      </tr>
      <tr>
        <td><strong>Veldhomografie</strong></td>
        <td>RANSAC met 305 punten over 39 keyframes</td>
        <td>Mediane projectiefout van 0.16 m op het veld</td>
      </tr>
      <tr>
        <td><strong>Tactische Analytics</strong></td>
        <td>2D Gaussiaanse KDE + 0.5s filter</td>
        <td>Afstandsafwijking -59% gecorrigeerd, live balbezitanalyse</td>
      </tr>
      <tr>
        <td><strong>Stabiliteit &amp; CI</strong></td>
        <td>Detectiecache + 77 pytest unittests</td>
        <td>100% slaging op tests, directe lokale start zonder GPU</td>
      </tr>
    </tbody>
  </table>

  <p>
    De belangrijkste lessen uit dit project zijn de noodzaak van een temporele split ter voorkoming van data leakage, het uitschakelen van mosaic-augmentatie bij zeer kleine objecten, en de grote winst van inferentie op 1280px voor baldetectie. Als mogelijke vervolgstap kan geautomatiseerde keypointdetectie (bijvoorbeeld via YOLO Pose) worden onderzocht om manuele keyframe-selectie volledig te automatiseren.
  </p>

</body>
</html>
"""

html_content = html_template.replace("__USER_IMG_B64__", user_img_b64)
html_content = html_content.replace("__IMG_BEELDGROOTTE_B64__", img_beeldgrootte_b64)
html_content = html_content.replace("__IMG_HEATMAP_B64__", img_heatmap_b64)
html_content = html_content.replace("__IMG_SMOOTH_B64__", img_smooth_b64)

# Write HTML
html_file = REPO_ROOT / "scripts" / "report_temp_clean.html"
with open(html_file, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"HTML generated at: {html_file}")

# Run Chrome to generate PDF
cmd = [
    str(CHROME_PATH),
    "--headless=new",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={TARGET_PDF}",
    str(html_file)
]

print("Rendering clean PDF via Chrome...")
res = subprocess.run(cmd, capture_output=True, text=True)
print("Chrome exit code:", res.returncode)

if TARGET_PDF.exists():
    print(f"Clean PDF successfully generated at: {TARGET_PDF} ({TARGET_PDF.stat().st_size} bytes)")
    html_file.unlink(missing_ok=True)
else:
    print("Error: PDF could not be created!")
