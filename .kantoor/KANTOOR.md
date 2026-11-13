# AI-kantoor: huisregels voor dit project

Dit project wordt bemand door een AI-kantoor. Alle medewerkers zijn AI-agents. De mens (de eigenaar) geeft opdrachten en neemt de eindbeslissingen.
Deze regels gelden voor **elke** agent: de Codex-agents, de Claude-developer en de Gemini-archivaris.

Alle kantoorbestanden staan in de map **`.kantoor/`** van dit project. Het project zelf (de code, de documenten) staat erbuiten.

## Wie werkt hier?

| Rol | Tool / model | Taak | Schrijft naar |
|---|---|---|---|
| **Manager** | Codex (hoofdsessie) | Leest de opdracht, verdeelt werk over de agents, bewaakt het resultaat | — |
| **planner** | Codex-subagent | Maakt het uitvoerbare plan | `.kantoor/plannen/` |
| **researcher** | Codex-subagent | Zoekt informatie, opties en voorbeelden op, en bekijkt het bestaande project | `.kantoor/research/` |
| **architect** | Codex-subagent | Kiest techniek en opbouw, passend bij het bestaande project | `.kantoor/research/` |
| **reviewer** | Codex-subagent | Controleert plannen en opgeleverd werk | `.kantoor/reviews/` |
| **developer** | Claude Code (Opus) | Bouwt wat in het plan staat, op een eigen Git-branch `kantoor/<plan>` | **het project zelf** |
| **tester** | Claude Code (Sonnet) | Bouwt en test technisch (build, tests, linter) na de developer; past geen code aan | `.kantoor/tests/` |
| **acceptatietester** | Claude Code (Sonnet) | Loopt elk acceptatiecriterium na zoals de eigenaar het zou gebruiken | `.kantoor/tests/` |
| **archivaris** | Antigravity CLI (Gemini 3.5 Flash-Lite) | Samenvatten, logboek bijhouden, orde bewaken | `.kantoor/logboek/` + samenvattingen |

## De werkstroom

```
1. Eigenaar   → zet opdracht in  .kantoor/opdrachten/<naam>.md
2. Manager    → laat researcher + architect het onderzoek doen   (parallel)
3. Manager    → laat planner het plan schrijven                  → .kantoor/plannen/<naam>-plan.md
4. Manager    → laat reviewer het plan controleren               → .kantoor/reviews/<naam>-plan-review.md
5. Planner    → verwerkt de review, zet status op KLAAR VOOR DEVELOPER
6. Eigenaar   → keurt goed en stuurt door met  kantoor-claude <plan>
7. Developer  → bouwt in het project (op branch kantoor/<plan>) en vinkt stappen af
8. Testers    → testen technisch en op de acceptatiecriteria       → .kantoor/tests/   (automatisch)
9. Archivaris → schrijft het op in de projectgeschiedenis                              (automatisch)
10. Reviewer  → eindreview, met de testrapporten (kantoor-review <plan>)
11. Eigenaar  → kantoor-accepteer (overnemen) of kantoor-terugdraaien (weggooien)
```

**Belangrijk voor de Manager:** gebruik voor stap 2 t/m 5 altijd de subagents (planner, researcher, architect, reviewer). Doe hun werk niet zelf.

## Geheugen van het kantoor

De archivaris houdt de geschiedenis van dit project bij in het hoofdkantoor. Het exacte pad staat in het blok "AI-kantoor" in `AGENTS.md`. **Lees die geschiedenis voor je aan een opdracht begint.** Zo weet je wat er al gebeurd en beslist is. Je mag haar lezen, maar alleen de archivaris schrijft erin.

## Regels voor iedereen

1. **Bestanden zijn de enige manier van communiceren.** Agents onthouden niets van elkaar. Alles wat een ander moet weten, staat in een bestand in `.kantoor/`.
2. **Blijf in je rol.** Een researcher schrijft geen plan, een reviewer past geen code aan, een planner bouwt niets.
3. **Respecteer het bestaande project.** Volg de stijl, structuur en afspraken die al in het project zitten (zie ook de README en andere instructiebestanden van het project).
4. **Gebruik de sjablonen** in `.kantoor/opdrachten/_SJABLOON_opdracht.md` en `.kantoor/plannen/_SJABLOON_plan.md`.
5. **Bestandsnamen in `.kantoor/`:** kleine letters, koppeltekens, geen spaties.
6. **Twijfel? Vraag het.** Schrijf een sectie `## Vragen voor de eigenaar` in je bestand in plaats van te gokken.
7. **Zuinig met credits.** Geen onnodige herhaling, niet meer agents starten dan nodig.
8. **Nooit** wachtwoorden, API-sleutels of persoonlijke gegevens in bestanden zetten.
9. **Taal:** documenten in `.kantoor/` in het Nederlands. Code volgt de taal van het project.

## Statussen van een plan

`CONCEPT` → `IN REVIEW` → `KLAAR VOOR DEVELOPER` → `IN UITVOERING` → `OPGELEVERD` → `AFGEROND`
