# MLX-Knife Model Naming Specification

## Fundamental Mapping Rules

### Basic Conversion
**Universal conversion:** `--` ↔ `/` (all occurrences)

**External → Internal:** `org/sub/model` becomes `models--org--sub--model`  
**Internal → External:** `models--org--sub--model` becomes `org/sub/model`

### Character Constraints (Clean Names)

**External names (clean):**
- ✅ Maximum **one `-`** consecutive (single dashes allowed)
- ✅ `/` as path separators
- ❌ Never `--` (double dashes forbidden)

**Internal cache (clean):**  
- ✅ Maximum **two `-`** consecutive (`--` as separators only)
- ✅ Single `-` within names
- ❌ Never `---` or more (triple+ dashes forbidden)

### Why These Rules?

```
✅ Clean conversion:
External: org-name/model-v1
Internal: models--org-name--model-v1

❌ Rule violation creates chaos:  
External: org--invalid/model  (double dash = forbidden!)
Internal: models--org----model  (quadruple dash = chaos!)
```

## Examples (Clean Names)

| External | Internal Cache Directory |
|----------|--------------------------|
| `microsoft/DialoGPT-small` | `models--microsoft--DialoGPT-small` |
| `org/sub/model` | `models--org--sub--model` |
| `single-model` | `models--single-model` |

## MLX-Knife Implementation: Tolerant Handling

### Robustness Philosophy
**"Be liberal in what you accept"** - MLX-Knife handles rule violations gracefully.

### Error Handling for Corrupted Cache
**When reading entries that violate rules:** Mechanical 1:1 conversion without validation

```
Cache: models--microsoft--DialogGPT---small  (3 dashes = rule violation)
↓ Mechanical conversion: ALL "--" → "/" 
External: microsoft/DialogGPT/-small  (empty path segment visible)
```

**Benefits:**
- ✅ System remains functional (no crashes)
- ⚠️ Problems become visible (user sees `DialogGPT/-small`)  
- 🔍 User can identify and fix corrupted entries
- 🛠️ No complex error handling required

## Compatibility

✅ **HuggingFace Hub:** Compatible with standard `org/model` format  
✅ **Future-proof:** Supports deeper hierarchies like `org/sub/model`  
✅ **Robust:** Converts corrupted cache entries without failing