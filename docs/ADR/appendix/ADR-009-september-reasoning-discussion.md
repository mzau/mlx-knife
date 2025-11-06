## ADR-009 Appendix: Reasoning Scope Notes (September 2025)

**Status:** Archived exploration – captures early reasoning API considerations now owned by future ADRs

## Meine Fragen im ChatGPT-Dialog (GPT5-Thinking):

```plaintext
wie sollte ein OpenAI kompatibler Server optimal mit gpt-oss umgehen?
...
ich frage, wie das mlx-knife handhaben soll - eine MLX version ist verfügbar mit MXFP4 unterstützung.
````

## Antwort

**Kurz gesagt:**

Erwarte Reasoning nie „heimlich“ (keine impliziten Chain-of-Thought-Leaks). Erzeuge Reasoning explizit durch das Modell (structured output / function call / separate pass) und mache die Freigabe/Anzeige kontrollierbar (flags, audit, encrypt). Diese Herangehensweise ist kompatibel mit dem, was etablierte Anbieter empfehlen (Responses API → reasoning summaries), und lässt sich lokal für MLX-Modelle sauber umsetzen.  ￼ ￼

⸻

1 — Designprinzipien (high level)
	1.	Explizit statt implizit: Reasoning ist nur dann Teil der Ausgabe, wenn du das Modell explizit so aufforderst (Prompt oder separate Reasoning-Pass). Default: kein CoT.
	2.	Strukturierte Antwort bevorzugen: Fordere das Modell an, eine JSON-Struktur zu liefern: { "answer": "...", "reasoning": "...", "score": ... }. Server parst und trennt sicher.
	3.	Feinsteuerung per Flag: Client gibt include_reasoning=none|summary|full (oder reasoning=summary) an. Server liefert nur das, wozu Client berechtigt/angefragt hat.
	4.	Streaming mit Kanal-Metadaten: Beim Streamen markiere Reasoning-Deltas ausdrücklich (z. B. delta.reasoning vs delta.content), statt sie als normalen Text zu mischen.
	5.	Validierung & Audit: Prüfe/validiere JSON (jsonschema). Aufbewahrung von kompletten Reasoning-Traces optional & verschlüsselt (audit).
	6.	Fallbacks: Wenn kein strukturiertes Format kommt, suche nach Marker-Patterns (z. B. <<REASONING>>...<</REASONING>>) oder versuche post-hoc einen reasoning_summary via zweitem Prompt zu erzeugen.

(Praktisch relevant für MLX-Modelle / MXFP4 — mlx-knife kann Tokenizer/chat_template aus dem Modell verwenden, siehe MF-Modelcards und MLX-ecosystem).  ￼ ￼

⸻

2 — API-Contract (proposal)

/v1/chat/completions (OpenAI-compatible) — zusätzliche, optionale Felder:

Request (relevante Felder)

{
  "model":"mlx-community/gpt-oss-20b-MXFP4-Q4",
  "messages":[ ... ],
  "stream": true,
  "reasoning": "none" | "summary" | "full",      // NEW: server honoriert das
  "reasoning_encrypt": true | false,            // optional: server speichert verschlüsselt
  "response_schema": { "type":"json", "schema": { ... } } // optional zwingt JSON-Ausgabe
}

Response (streamed SSE chunks): klassische chat.completion.chunk mit erweitertem Delta-Shape:

// Normaler answer-delta
{
  "object":"chat.completion.chunk",
  "model":"...",
  "choices":[{"index":0,"delta":{"content":"Hello, "},"finish_reason":null}]
}

// Reasoning-delta (klar gekennzeichnet)
{
  "object":"chat.completion.chunk",
  "model":"...",
  "choices":[{"index":0,"delta":{"reasoning":"step1: consider X..."},"finish_reason":null}]
}

Am Stream-Ende wie gewohnt data: [DONE].
(Alternative OpenAI-kompatible Variante: benutze choices[].delta mit delta.content und delta.meta.reasoning=true — aber ein dedicated delta.reasoning Feld ist klarer für Clients.)

Wichtig: Wenn reasoning=none, streamet der Server ausschließlich delta.content. Wenn reasoning=summary, erzeugt der Server (während oder nach der Antwort) eine kompakte natural-language reasoning_summary und streamt diese in einem eigenen delta.reasoning_summary-Event oder liefert sie als separate API-Antwort.  ￼

⸻

3 — Prompting / Tokenizer-Handling (praktisch für MLX)
	•	System prompt steuert Erzeugung: wenn du response_schema verlangst, erzeugt dein prompt explizit die JSON-Ausgabe. Nutze HF-chat_template und tokenizer.apply_chat_template(...) wie gehabt (wichtig für MLX-models).  ￼ ￼
	•	Beispiel-System-Prompt (verkürzt):

System: You must return a JSON object with keys "answer" (string) and "reasoning" (string). 
If the client requested "reasoning": "none", put reasoning = "".
Do not include extra text outside the JSON.

	•	Wenn Streaming: instruct model to emit JSON in streamable chunks (choose safe delimiters) or use incremental JSON tokenization conventions (clients must reassemble). Consider response_schema to force well-formed JSON.

⸻

4 — Streaming: technische Details & client expectations
	•	Server-side: empfange backend-tokenstream; klassifiziere jedes Token/chunk als answer vs reasoning basierend auf:
	1.	explicit JSON keys (best), oder
	2.	markers (<<REASONING>>), oder
	3.	function-call events (siehe unten).
	•	Client: erwartet SSE events mit delta.content (answer) und optional delta.reasoning (reasoning). Client UI zeigt standardmäßig answer inkrementell. reasoning wird verdeckt/optional angezeigt (z. B. “Show reasoning” button) oder in dev/debug mode automatisch expanded.

Beispiel SSE event payload:

data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"The result is 42."}}]}

data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning":"I computed 6*7 because..."} } ]}

data: [DONE]


⸻

5 — Drei praktikable Implementationsmuster (mit Vor-/Nachteilen)

A) Structured JSON output (empfohlen)
	•	Wie: Prompt zwingt JSON {answer, reasoning}. Server json.loads() und trennt Felder.
	•	Pro: Robust, leicht zu validieren, kein Heuristik-Chaos.
	•	Contra: JSON-Stream-Parsing kann knifflig; erfordert gute prompt-engineering.

B) Function-Call / Tool pattern
	•	Wie: Fordere Modell, call einer Pseudo-Funktion report_reasoning({ ... }) auszuführen (ähnlich OpenAI function call). Server fängt function_call ab — behandelt arguments als reasoning.
	•	Pro: Natürliche Trennung; streaming-fähig (arguments können über Chunks kommen).
	•	Contra: Erfordert funktionale Unterstützung im Backend (MLX wrappers können das aber handhaben).  ￼

C) Markers in plain text (Fallback)
	•	Wie: Model schreibt <<REASONING>> ... <</REASONING>> vor/zwischen den Antworten. Server sucht Marker.
	•	Pro: Einfach umzusetzen bei Models ohne JSON-Disziplin.
	•	Contra: Brüchig (Modelle vergessen Marker).

⸻

6 — Security / Policy / UX Regeln (wichtig)
	•	Default: reasoning=none. Reasoning nur auf explizite Nachfrage. Logge/versichere Consent, wenn Reasoning gespeichert wird (es kann sensible Interna enthalten).
	•	Biete reasoning_summary (automatisch generiert) statt voller CoT als Standard; das ist oft genug für debug/trace und weniger riskant. OpenAI empfiehlt genau so eine Trennung (Responses API bietet reasoning summaries).  ￼ ￼

⸻

7 — Fallbacks & Heuristics (wenn Modell nicht kooperiert)
	1.	Try parse JSON → success → separate.
	2.	Else: look for markers <<REASONING>> → split.
	3.	Else: run post-hoc prompt: Given the answer above, summarize the reasoning steps that justify it. → attach reasoning_summary. (Das ist sicherer als rohe CoT-Leaks.)
	4.	Wenn client requested full CoT and model refuses or output undecodable → return warning + reasoning_unavailable flag.

⸻

8 — Mini-FastAPI-Sketch (Streaming + JSON-separation)

Nachfolgend ein kurzes Beispiel wie mlx-knife serve das trennen könnte — konzeptionell, anpassbar an Deine Backend-Adapter (MLX/mlx_lm).

# server_reasoning.py (sketch)
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json, time

app = FastAPI()

# backend stream generator -> yields raw token chunks (strings)
def backend_stream_generate(prompt, model_id):
    # >>> adapt to mlx_lm streaming API
    for token in ["{","\"answer\":","\"42\"",",","\"reasoning\":\"","I multiplied...","\"","}"]:
        yield token
        time.sleep(0.01)

def assemble_stream(prompt, model_id, reasoning_mode):
    """
    Reads backend token stream and yields SSE events with typed deltas:
     - delta.content for answer
     - delta.reasoning for reasoning
    Uses simple JSON reassembly for demo.
    """
    buf = ""
    for tok in backend_stream_generate(prompt, model_id):
        buf += tok
        # try to parse JSON once complete-ish
        try:
            obj = json.loads(buf)
            # we assume model returned full JSON obj
            if "answer" in obj:
                # stream the answer as one chunk (or split further)
                yield json.dumps({"choices":[{"index":0,"delta":{"content":obj["answer"]}}]})
            if reasoning_mode != "none" and "reasoning" in obj:
                yield json.dumps({"choices":[{"index":0,"delta":{"reasoning": obj["reasoning"]}}]})
            buf = ""  # reset
        except json.JSONDecodeError:
            # not yet complete: do nothing (or stream tokens as best_effort)
            continue
    # finish
    yield "[DONE]"

@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    reasoning = body.get("reasoning","none")  # none|summary|full
    prompt = "<built-from-messages>"  # build via HF tokenizer in real impl
    model_id = body.get("model")
    def sse():
        for event in assemble_stream(prompt, model_id, reasoning):
            if event == "[DONE]":
                yield "data: [DONE]\n\n"
            else:
                yield f"data: {event}\n\n"
    return StreamingResponse(sse(), media_type="text/event-stream")


⸻

9 — Concrete recommendations für mlx-knife (konkret & priorisiert)
	1.	Implementiere --reasoning flag für mlx-knife run / serve (none/summary/full). Default none.
	2.	Support response_schema/json output im run path — benutze HF chat_template + system prompt, parse JSON, separate fields. (Das ist die robusteste Methode für MLX runs — siehe HF model pages wie mlx-community/gpt-oss-20b-MXFP4-Q4 die tokenizer.chat_template beschreiben).  ￼
	3.	Streaming: add field types to SSE (delta.reasoning), so UIs can decide whether to render reasoning inline.
	4.	Tool/Function approach: wenn dein backend (mlx_lm) das kann, support function-call style report_reasoning(...) so you can get reasoning as a function result.
	5.	Post-hoc summary: wenn full CoT not available/undesired, always offer --reasoning summary that triggers a small second pass to produce a concise reasoning summary. This is lightweight & safe.  ￼
	6.	Logging & audit: store full CoT only with explicit opt-in and encryption.

⸻

Quellen / weiterlese (relevant)
	•	MLX model example (MXFP4 model card, shows mlx-lm usage & chat_template).  ￼
	•	MLX / mlx-lm docs (how to load/generate on Apple silicon).  ￼
	•	OpenAI Responses API / Reasoning summaries (official guidance why reasoning should be a separate capability).  ￼ ￼
	•	Community guidance re: how to handle raw CoT in Chat Completions (advice and conventions).  ￼
