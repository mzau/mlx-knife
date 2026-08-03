# Chat Templates, Stop Tokens & Reasoning: Konsolidierte Erkenntnisse

**Status:** Historical background – Stop-token findings inform ADR-009; reasoning/API notes preserved for future ADRs  
**Dokumentiert:** 2025-09-08 (Initial), 2025-10-21 (Konsolidierung)  
**Related:** Issue #32, ADR-004

---

## Executive Summary

**Problem:** Models generieren andere Stop-Tokens als konfiguriert, Reasoning-Content nicht API-konform
**Root Cause (gefunden 2025-10-21):** HuggingFace tokenizer configs unvollständig + unser Code nutzt falsche API
**Lösung:** 2-Phasen Approach (Beta.6: Stop-Token-Fix, 2.1+: Reasoning-API)

---

## Die Kernfrage (September 2025)
**Welches End-Token gilt für welches Modell?**

## Was wir gelernt haben (September → Oktober)

### 1. Chat Templates sind NICHT Protokolle
- Chat Templates sind **Jinja2-Formatierungsanweisungen**
- Sie konvertieren strukturierte Messages zu Token-Sequenzen
- Sie replizieren das Format aus dem Training
- Sie definieren NICHT das Stop-Verhalten

### 2. End-Token Verwirrung

#### MXFP4 Modell Beispiel:
- **EOS Token**: `<|return|>` (tokenizer config)
- **Generiert aber**: `<|end|>` nach Messages
- **Problem**: `<|end|>` wird nicht als Stop-Token erkannt
- **Test erwartet**: `<|end|>` sollte gefiltert werden

#### Token-Typen:
1. **Control Tokens** (aus Training):
   - `<|end|>` - Message-Ende Marker (MXFP4)
   - `<|im_end|>` - Message-Ende (Qwen)
   
2. **Stop Tokens** (Generation beenden):
   - `<|return|>` (MXFP4)
   - `</s>` (Llama)
   - `<|endoftext|>` (GPT)

3. **Template Tokens** (nur Formatierung):
   - `<|start|>`, `<|message|>` etc.

### 3. Das eigentliche Problem

Modelle generieren verschiedene Tokens als "ich bin fertig":
- Manche nutzen ihr definiertes EOS Token
- Manche nutzen gelernte Pattern aus dem Training
- Manche nutzen beides

**MLX Knife muss wissen**: 
- Was ist das offizielle EOS Token? (aus tokenizer config)
- Was generiert das Modell tatsächlich? (empirisch)
- Was sollte gefiltert werden? (beide?)

### 4. Unsere bisherige Implementierung

```python
# Aktuell in mlx_runner.py:
- Extrahiert EOS aus tokenizer
- Sucht nach "end"-ähnlichen Tokens
- ABER: Verpasst modell-spezifische Patterns wie <|end|>
```

### 5. Server-Test Failures

- **MXFP4**: Generiert `<|end|>`, wird nicht gefiltert → Test fail
- **Qwen3**: Self-conversation (vermutlich andere Ursache)

## Offene Fragen

1. Sollten wir ALLE "end-like" Tokens aus dem Training als Stop-Tokens behandeln?
2. Oder nur die explizit als EOS definierten?
3. Wie gehen andere Implementierungen (Ollama, vLLM) damit um?
4. Brauchen wir modell-spezifische Stop-Token Listen?
5. **Legacy-Modelle**: Was ist mit alten Modellen ohne Chat Templates?
   - Sind sie mit der neuen Implementation kompatibel?
   - Brauchen wir einen Fallback auf Human:/Assistant:?
   - Oder verweigern wir Support für template-lose Modelle?

## Legacy-Modell Kompatibilität

### Aktuelle Implementation
```python
# mlx_runner.py _format_conversation():
if use_chat_template and hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
    # Use chat template
else:
    # Fallback to _legacy_format_conversation (Human:/Assistant:)
```

### Fragen zur Klärung:
- Gibt es überhaupt MLX-Modelle ohne Chat Templates?
- Wenn ja, funktioniert Human:/Assistant: für diese?
- Sollten wir sie überhaupt unterstützen?

## Nächste Schritte

1. **Inventur**: Welche Modelle haben keine Chat Templates?
2. **Empirisch testen**: Welche Tokens generieren die Modelle tatsächlich?
3. **Stop-Token Strategie**: Klare Regeln definieren
4. **Legacy-Strategie**: Fallback oder Deprecation?
5. **Implementation**: Robuste Token-Erkennung
6. **Tests anpassen**: Realistische Erwartungen

## Neue Erkenntnisse (Oktober 2025)

### Root Cause gefunden: HuggingFace + mlx_knife Code Bugs

**MXFP4 Tokenizer Config (HuggingFace):**
```json
{
  "eos_token": "<|return|>",       // ID 200002
  "eos_token_id": 200002,           // SINGLE ID (falsch!)
  "extra_special_tokens": {}        // Leer!
}
```

**Was richtig wäre (wie Llama 3):**
```json
{
  "eos_token_id": [200002, 200007]  // ARRAY: <|return|> UND <|end|>
}
```

**Unser Code Bug:**
```python
# mlxk2/core/runner/__init__.py:468, 589
if token_id == self.tokenizer.eos_token_id:  # SINGULAR (falsch!)
    break
```

**mlx-lm macht es richtig:**
```python
# mlx_lm/generate.py:stream_generate()
if token in tokenizer.eos_token_ids:  # SET (korrekt!)
    break
```

### mlx-lm Architektur-Analyse

**Pattern:** Keine model-spezifischen Workarounds in `mlx_lm/models/*.py`
- `gpt_oss.py`, `qwen2.py`, `llama.py` - Reine Architektur (forward pass)
- Stop-Token Handling: Nur in `generate.py` (generisch via tokenizer metadata)
- API: `tokenizer.add_eos_token(token)` für Runtime-Additions

**Erkenntnis:** mlx-lm vertraut auf korrekte HuggingFace configs. Broken configs → broken generation.

### Reasoning-Token Analyse

**OpenAI o1 / Responses API:**
- Reasoning bleibt **hidden** (nur token count sichtbar)
- Reasoning summaries via `reasoning.summary: "auto"`
- Keine `reasoning_content` im Chat Completions API

**DeepSeek R1 API:**
```python
response.choices[0].message.reasoning_content  # Separates Feld!
response.choices[0].message.content            # Final answer
```

**Status Quo (mlx_knife):**
- Inline filtering via `StreamingReasoningParser`
- `hide_reasoning` Parameter (bereits vorhanden)
- Marker-basiert: `<|channel|>analysis<|message|>...` → entfernt

**Problem:** Nicht API-standard-konform, Client kann Reasoning nicht separat rendern

## Roadmap: 2-Phasen Approach

### Phase 1: Beta.6 - Stop Token Fix (BLOCKER)

**Scope:** Generische Mechanismen implementieren (KEIN Workaround-Gefrickel)

**Changes:**
1. ✅ **Fix Runner Stop-Check:**
   ```python
   # Vorher (broken):
   if token_id == self.tokenizer.eos_token_id:

   # Nachher (correct):
   if token_id in self.tokenizer.eos_token_ids:
   ```

2. ✅ **Add Stop Tokens via API:**
   ```python
   # In _extract_stop_tokens():
   for stop_token in self._stop_tokens:
       self.tokenizer.add_eos_token(stop_token)
   ```

3. ✅ **Defense-in-Depth behalten:**
   - String-based filtering (Issue #20) bleibt als Fallback
   - Reasoning parser bleibt wie ist

**Non-Scope (Beta.6):**
- ❌ KEINE Reasoning-API Changes (breaking)
- ❌ KEINE HuggingFace Issues melden (noch nicht)
- ❌ KEINE model-spezifischen Workarounds (erst nach Real-Model Tests)

**Test Strategy:**
- Real-Model Test Suite (MXFP4, Qwen3, Llama3.2)
- Validate stop token detection
- Measure before/after behavior

### Phase 2: 2.0.1+ - Reasoning API (Enhancement)

**Goal:** API-standard-konforme Reasoning-Unterstützung

**Design:** DeepSeek-Style (Option B)
```python
# Response structure:
{
  "choices": [{
    "message": {
      "content": "Final answer",           # Existing
      "reasoning_content": "CoT...",       # NEW
      "role": "assistant"
    }
  }]
}
```

**Streaming:**
```python
# SSE chunks:
data: {"choices":[{"delta":{"content":"Hello"}}]}
data: {"choices":[{"delta":{"reasoning":"step 1..."}}]}
```

**Client Benefits:**
- Web UI kann Reasoning optional einblenden (wie GPT-5 chat)
- Lokale Clients haben klare API-Struktur
- Runner code als Vorlage für broke cluster

**Implementation Tasks:**
1. Extend `ChatCompletionResponse` model
2. Modify `StreamingReasoningParser` → separate output streams
3. Add `include_reasoning` request parameter
4. Update server endpoints
5. Write API docs + examples

**Breaking Changes:**
- Opt-in: Default `include_reasoning=false` (backward compat)
- Existing clients funktionieren weiter

## Issue #32 Status Update

**Original Problem (September):** Hardcodiertes Human:/Assistant: Format
- ✅ **Gelöst:** Chat Templates werden verwendet

**Problem 1 (Oktober):** Stop-Token Detection
- 🔄 **Beta.6:** Generischer Fix (eos_token_ids Set)
- 📅 **Status:** Implementierung anstehend

**Problem 2 (Future):** Reasoning API
- 📋 **2.0.1+:** Separate `reasoning_content` field
- 📅 **Status:** Konzept definiert, Implementation später

## Offene Fragen (für später)

1. **HuggingFace Issues melden?**
   - MXFP4 tokenizer config fix (`eos_token_id` → array)
   - Erst nach Validation mit Real-Model Tests

2. **mlx-lm Enhancement vorschlagen?**
   - Warning wenn chat_template tokens nicht in `eos_token_ids`
   - Bessere Docs für `--extra-eos-token`
   - Erst nach Beta.6 Validation

3. **Legacy-Modelle ohne Chat Templates?**
   - Inventur durchführen (gibt es überhaupt welche?)
   - Fallback behalten oder deprecaten?

## Referenzen

- **mlx-lm Source:** https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py
- **DeepSeek API:** https://api-docs.deepseek.com/guides/reasoning_model
- **OpenAI Responses API:** https://cookbook.openai.com/examples/responses_api/reasoning_items

---

**Next Session:**
- [ ] Implement stop token fix (Phase 1)
- [ ] Run Real-Model Test Suite (validation)
- [ ] Create Issue for Phase 2 (Reasoning API)
- [ ] Consider upstream issue reports (after validation)
