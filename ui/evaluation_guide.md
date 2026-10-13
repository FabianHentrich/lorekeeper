Das **Golden Set** ist ein Katalog von Fragen mit der Angabe, welche Datei (und ggf. welcher
Abschnitt) die Antwort enthält. Die Tabs messen daran, ob die Suche das Richtige findet — so wird
jede Änderung an Suche, Chunking oder Prompts an Zahlen statt am Bauchgefühl entschieden.

| Tab | Was es tut | Wann |
|---|---|---|
| **Golden Set** | Fragen pflegen: erwartete Datei, Abschnitt (z.B. `4.10`), Stichwörter für die Antwort | Neue Inhalte oder eine Frage, die im Chat schiefging |
| **Frage testen** | Eine Frage durch die Suche schicken (ohne LLM): welche Chunks mit welchem Score | Ursachensuche bei einer einzelnen Frage |
| **Retrieval Eval** | Alle Fragen durch die Suche, ohne LLM (~1 min) | Nach Änderungen an Suche oder Chunking (Chunking: vorher neu indizieren) |
| **End-to-End** | Alle Fragen durch die ganze Kette inkl. LLM — langsam | Nach Prompt- oder Modelländerungen |
| **Ergebnisse** | Gespeicherte Läufe (je Typ die letzten 3) vergleichen | Vorher/Nachher einer Änderung |
| **Feedback** | 👍/👎 aus dem Chat; 👎-Fragen ins Golden Set übernehmen | Regelmäßig — so wächst das Golden Set an echten Problemen |
| **Chunks** | Alle gespeicherten Chunks einer Datei, so wie die Suche sie sieht | Wenn eine Datei nie gefunden wird oder seltsam zerteilt ist |

**Die Zahlen**
- **Hit Rate**: Anteil der Fragen, bei denen die erwartete Datei unter den gefundenen Chunks ist.
- **Abschnitt-Treffer**: zusätzlich der richtige Abschnitt (wichtig beim Regelbuch, wo fast jeder
  Treffer aus derselben Datei kommt). ⚠️ in den Details = richtige Datei, falscher Abschnitt.
- **Answer Contains** (End-to-End): Anteil der Antworten, die alle erwarteten Stichwörter enthalten.
- **A/B (Vektor vs. Hybrid)**: zwei Läufe hintereinander; die Tabelle zeigt nur die Fragen, deren
  Ergebnis kippt.
- **🎯 Suchgenauigkeit** (Tab Retrieval Eval): vergleicht die schnelle Vektorsuche mit einer exakten
  Suche. „Exakt“ heißt: die Ergebnisse sind reproduzierbar und nichts wird verpasst. Wächst der Index
  stark und die Prüfung meldet Abweichungen, `vectorstore.hnsw_ef_search` erhöhen.

Läufe sind reproduzierbar: zwei Läufe ohne Änderung liefern dieselben Zahlen. Weicht ein Lauf ab,
hat sich am Index, an den Einstellungen oder am Golden Set etwas geändert.
Ausführlich: `docs/evaluation.md` im Repository.
