# Provider-Strategie

Welche LLM-Provider LoreKeeper unterstützt, wie man sie konfiguriert und wechselt, wie Fallback
und Condense-Modell funktionieren und wie Tokens gezählt werden. Alle Schlüssel:
[configuration.md](configuration.md#llm).

## Inhaltsverzeichnis
1. [Überblick](#überblick)
2. [Vergleich](#vergleich)
3. [Einrichtung](#einrichtung)
   - [Ollama](#ollama)
   - [Gemini](#gemini)
4. [Provider zur Laufzeit wechseln](#provider-zur-laufzeit-wechseln)
5. [Fallback-Provider](#fallback-provider)
6. [Condense-Modell](#condense-modell)
7. [Token-Zählung](#token-zählung)
8. [Neuen Provider hinzufügen](#neuen-provider-hinzufügen)

---

## Überblick

Der LLM-Zugriff liegt hinter einer einheitlichen Schnittstelle `BaseLLMProvider`. Unterstützt:
**Ollama** (lokal, Hauptweg) und **Gemini** (Cloud, Ausweichlösung).

```python
class BaseLLMProvider(ABC):
    provider: str = "unknown"
    model: str                       # im __init__ setzen — kein Default

    async def generate(self, prompt: str, **kwargs) -> LLMResponse: ...
    def generate_stream(self, prompt: str, system_prompt: str = "",
                        stream_result: StreamResult | None = None,
                        **kwargs) -> AsyncGenerator[str, None]: ...
    async def health_check(self) -> bool: ...
```

Implementierungen schreiben `generate_stream` als `async def` mit `yield`. `StreamResult` ist ein
Kontext pro Aufruf: der Provider füllt `usage`, der `Generator` setzt `model` auf das Modell, das
tatsächlich geantwortet hat.

---

## Vergleich

| Kriterium | Ollama (lokal) | Gemini (Cloud) |
|-----------|----------------|----------------|
| **Antwortzeit** | erster Token ~1 s, gesamt ~3–4 s (qwen3:14b auf RX 9070 XT, warm) | 2–8 s |
| **Kosten** | Strom + GPU | Pro Token — Budgets in Google Cloud warnen nur, sie stoppen nicht |
| **Datenschutz** | Komplett lokal | Daten verlassen den Rechner |
| **Kontext** | Über `OLLAMA_CONTEXT_LENGTH` (LoreKeeper: 16384) | 1 Mio. Tokens (Flash) |
| **Qualität** | Gut auf Deutsch, modellabhängig | Sehr gut |
| **Offline** | ✅ | ❌ |
| **Einrichtung** | Ollama installieren + Modell holen | API-Key in `.env` oder der UI |

Deshalb ist Ollama der Hauptweg und Gemini nur Fallback.

---

## Einrichtung

### Ollama

`llm.ollama` in `settings.yaml` (Standard: `qwen3:14b`, `base_url` `http://127.0.0.1:11434`).

| VRAM | Empfehlung |
|------|------------|
| < 8 GB | `qwen3:4b` |
| 8–12 GB | `qwen3:8b` |
| ≥ 16 GB | `qwen3:14b` |

**Kontextfenster:** Ollamas Standardkontext (4096 Tokens) ist zu klein — eine Regelfrage mit
expandierter Sektion braucht schon ~3600 Eingabe-Tokens plus bis zu `max_tokens` für die Antwort,
und ein Überlauf schneidet still den Anfang des Prompts (den Systemprompt) ab. Der OpenAI-
kompatible Endpoint ignoriert `num_ctx` pro Anfrage, also wird die Größe auf Ollama-Seite gesetzt:

```powershell
setx OLLAMA_CONTEXT_LENGTH 16384   # danach Ollama im Tray beenden und neu starten
ollama ps                          # Spalte CONTEXT muss 16384 zeigen
```

16384 kostet bei qwen3:14b ~2,6 GB VRAM (11 GB gesamt, passt komplett in 16 GB). Startet das
LoreKeeper-Backend `ollama serve` selbst (Ollama lief beim Start nicht), erbt dieser Prozess die
Umgebung des Backends — nach dem Ändern der Variable erst die Ollama-App starten. Das gilt auch für
den **Modellordner**: Die Ollama-App kann ihre Modelle in einem eigenen Ordner halten (App-Einstellung);
ein selbst gestartetes `ollama serve` findet sie nur, wenn `OLLAMA_MODELS` auf denselben Ordner zeigt
— sonst steht „Ollama lists no models at all“ im Log. Seine Ausgabe schreibt das selbst gestartete
Ollama nach `logs/ollama.log` ([operations.md](operations.md)).

**Mehrere Fragen gleichzeitig — Entscheidung (13.10.): `OLLAMA_NUM_PARALLEL` bleibt 1.** Ollama
beantwortet gleichzeitige Fragen dann nacheinander. Gemessen mit 1–4 gleichzeitigen `/query/stream`-
Fragen (gemischte Rollen, je 3 Runden, qwen3:14b auf der RX 9070 XT, echter Index; 84 Fragen, keine
Fehler):

| `OLLAMA_NUM_PARALLEL` | gleichzeitig | erstes Token Median / Max | Tempo pro Antwort | Gesamtdurchsatz |
|---|---|---|---|---|
| **1** | 1 | 0,2 / 0,3 s | 162 Zeichen/s | 151 Z/s |
| 1 | 2 | 1,7 / 2,9 s | 185 Z/s | 148 Z/s |
| 1 | 3 | 2,5 / 15,8 s | 166 Z/s | 149 Z/s |
| 1 | 4 | 5,0 / 17,0 s | 161 Z/s | 146 Z/s |
| 2 | 1 | 0,3 / 0,3 s | 119 Z/s | 114 Z/s |
| 2 | 2 | 0,6 / 0,6 s | 95 Z/s | 137 Z/s |
| 2 | 3 | 0,9 / 4,7 s | 88 Z/s | 140 Z/s |
| 2 | 4 | 2,5 / 12,5 s | 87 Z/s | 147 Z/s |

Begründung: Der Engpass ist das LLM, nicht LoreKeeper (Retrieval bei 4 gleichzeitig ~450 ms statt
135 ms). Mit 2 parallelen Slots kommt das erste Token für Gleichzeitige schneller, aber jede Antwort
läuft langsamer — **auch wenn nur einer fragt (−27 %)** — ohne Gewinn beim Durchsatz, bei 14,4 statt
11,8 GB VRAM (KV-Cache 5,1 statt 2,6 GB) neben dem Reranker auf derselben GPU. Der häufige Fall
(einer fragt) bleibt so am schnellsten; Wartende sehen im Chat „Das LLM beantwortet gerade eine
andere Frage …“ (SSE-Event `status`). Ungemessen: Fragen während eines laufenden Ingests.

**Keep-alive:** Ollama entlädt ein Modell nach 5 Minuten Leerlauf; die nächste Frage wartet dann
~16 s (gemessen: erster Token nach 18 s statt 1,1 s). `setx OLLAMA_KEEP_ALIVE 1h` hält es bis eine
Stunde nach der letzten Frage geladen (`ollama ps` → „59 minutes from now“), belegt solange aber
~11 GB VRAM; `-1` entlädt nie.

**Qwen3-Denken:** abgeschaltet über `extra_body={"reasoning_effort": "none"}`
(`OllamaProvider._thinking_off`). Aktuelle Ollama-Versionen ignorieren das alte Suffix `/no_think`
und streamen das Denken in einem eigenen Feld `reasoning` statt in `<think>`-Tags — gemessen ~7 s
vor dem ersten Antwort-Token bei qwen3:14b. `think: false` wird ebenfalls ignoriert. Was trotzdem
ankommt (andere Denkmodelle, ältere Server), wird nie angezeigt und als `tokens_thinking` gezählt —
ein Wert ungleich 0 heißt also: das Denken ist wieder an.

### Gemini

`llm.gemini` in `settings.yaml` (Standard: `gemini-2.5-flash`), Key in `.env`:

```ini
GEMINI_API_KEY=dein-key
```

oder zur Laufzeit in der Chat-Sidebar (nur im Prozess, nie auf Platte —
[configuration.md](configuration.md#llm)).

**Rate-Limit:** LoreKeeper hält mindestens 4 s Abstand zwischen Anfragen und wiederholt bei 429
mit exponentiellem Backoff (bis zu 3×).

---

## Provider zur Laufzeit wechseln

Über das Auswahlfeld in der Chat-Sidebar (hinter dem Admin-Passwort) oder per API:

```powershell
curl.exe -X POST http://127.0.0.1:8000/provider -H "Content-Type: application/json" -d '{\"provider\": \"gemini\"}'
```

Der Wechsel baut Provider, Condense-Provider und `Generator` (mit Fallback) zusammen neu
(`ProviderFactory.build_all()` über `install_llm()` in `src/api/system_routes.py`) und frischt danach den
LLM-Health-Status auf; scheitert er (z.B. fehlender Key), bleibt der alte Satz. Derselbe Weg läuft
beim Gemini-Key und beim Umschalten von `llm.fallback_enabled` auf der Settings-Seite — der
`Generator` legt seinen Fallback beim Bau fest, ein bloßes Ändern der Einstellung erreichte ihn
früher nicht (Roadmap 11.0). Der Wechsel gilt bis zum nächsten Neustart und schreibt nicht in
`settings.yaml`.

---

## Fallback-Provider

`llm.fallback_enabled: true` (Standard `false`) mit `fallback_provider: gemini`: Fällt der
Hauptprovider aus (Timeout, Ollama nicht erreichbar), antwortet der Fallback mit seiner eigenen
Konfiguration.

Beim Streaming (`/query/stream`) greift der Fallback nur **vor dem ersten Token** — sind schon
Tokens unterwegs, würde ein Neustart die Ausgabe doppeln, dann kommt stattdessen ein Fehler. Das
`done`-Event nennt in `model_used` das Modell, das tatsächlich geantwortet hat.

---

## Condense-Modell

`conversation.condense_model` (Standard `null` = Hauptprovider): optional ein eigenes, schnelleres
Modell nur fürs Umschreiben von Folgefragen, z.B. `qwen3:4b`. Immer ein Ollama-Modell (gleiche
`base_url`); Gemini wird dafür nicht unterstützt. Spart Zeit bei Folgefragen — eigenständige
Fragen brauchen ohnehin keinen Condense-Aufruf.

---

## Token-Zählung

Beide Provider liefern Token-Zahlen pro Anfrage; LoreKeeper gibt sie im `done`-Event aus (`usage`)
und summiert sie pro Session (`session_usage`). Die UI zeigt sie unter jeder Antwort und im Kopf
der Seite ([ui-ux.md](ui-ux.md)).

| Feld | Ollama (Qwen3) | Gemini 2.5 |
|---|---|---|
| `tokens_in` | ✅ `prompt_tokens` | ✅ `usage_metadata.prompt_token_count` |
| `tokens_out` | ✅ `completion_tokens` | ✅ `usage_metadata.candidates_token_count` |
| `tokens_thinking` | Reasoning-Chunks im Feld `reasoning` — normalerweise `0` | ✅ `usage_metadata.thoughts_token_count` (wenn Denken an ist) |

**Ollama:** Die OpenAI-kompatible Streaming-API liefert die Zählung nur mit
`stream_options={"include_usage": True}` — sonst kämen Nullen. Gesetzt in
`OllamaProvider.generate_stream`; die Werte landen im `StreamResult.usage` des Aufrufs (kein
geteilter Zustand, parallele Streams mischen sich nicht).

**Gemini:** `google-genai` hängt `usage_metadata` an Stream-Chunks, meist nur an den letzten; der
letzte vorhandene Wert gilt. `thoughts_token_count` ist bei Gemini 2.5 das **kostenrelevante**
Feld: Denk-Tokens werden als Ausgabe berechnet, stehen aber nicht in `candidates_token_count` und
können die sichtbare Ausgabe um das 2–10-fache übersteigen. Die UI zeigt sie getrennt als
`🧠 N think`. **Diese Zahl im Blick behalten** — sie ist die häufigste Ursache für eine höhere
Rechnung als erwartet.

Die Session-Summen im Seitenkopf geben eine grobe Live-Schätzung, was eine Session bei einem
bezahlten Provider kosten würde; bei Ollama sind sie ein Lastindikator (Kontextgröße,
Antwortlänge).

---

## Neuen Provider hinzufügen

1. `src/generation/providers/mein_provider.py` anlegen, `BaseLLMProvider` implementieren
   (`generate_stream` als `async def` mit `yield`, Tokens in `stream_result.usage`).
2. Klassenattribut `provider = "mein_provider"` und `self.model` im `__init__` setzen.
3. In der `ProviderFactory` (`provider_factory.py`) registrieren.
4. Config-Klasse in `src/config/manager.py` ergänzen.
5. `settings.yaml`, `.env.example` und [configuration.md](configuration.md) nachziehen.
6. Unit-Tests mit gemocktem Client — nie echte API-Aufrufe.
