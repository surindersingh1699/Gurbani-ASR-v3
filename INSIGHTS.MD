# Fast Whisper Large for Kirtan Transcription Pipeline

## Overview

We use **Whisper Large** to transcribe kirtan audio and map it to the **canonical Gurbani text (STTM)**.

Whipser Setup 

OpenAI Whisper large-v2 (import whisper, whisper.load_model('large-v2'))
language="pa", beam_size=3, word_timestamps=True, fp16=True
VAD off
initial_prompt="ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ ॥ ਜਪੁ ॥ ਆਦਿ ਸਚੁ ਜੁਗਾਦਿ ਸਚੁ ॥ ਹੈ ਭੀ ਸਚੁ ਨਾਨਕ ਹੋਸੀ ਭੀ ਸਚੁ ॥"
Pin datasets==2.21.0

Goal:
- Handle **repetition in kirtan**
- Map imperfect Whisper output → **exact Gurbani lines**
- Produce **clean training data + accurate display**

---

# Key Challenge

Kirtan often includes repetition and variation.

### Example

**Canonical line:**
```
ਤੁਧੁ ਆਪੇ ਜਗਤੁ ਉਪਾਇ ਕੈ ਆਪਿ ਖੇਲੁ ਰਚਾਇਆ ॥
```

**Kirtan (Whisper output):**
```
ਤੁਧ ਤੁਧ ਤੁਧ ਆਪ ਜਗਤ ਉਪ
```

Problems:
- Repetition (`ਤੁਧ ਤੁਧ ਤੁਧ`)
- Partial words (`ਉਪ`)
- Missing suffixes (`ਆਪ` vs `ਆਪੇ`)

---

# Core Idea

👉 Use **canonical Gurbani as ground truth dictionary**

Pipeline:
1. Whisper gives **approximate text**
2. Identify **correct canonical line**
3. Map Whisper words → canonical words
4. Reconstruct output using canonical text

---

# Full Pipeline

## Step 1 — Whisper Transcription
```
Audio → Fast Whisper Large → Raw text
```

Example:
```
ਤੁਧ ਤੁਧ ਤੁਧ ਆਪ ਜਗਤ ਉਪ
```

---

## Step 2 — Remove Repetition (for matching only)

```
ਤੁਧ ਤੁਧ ਤੁਧ ਆਪ → ਤੁਧ ਆਪ
```

⚠️ Important:
- Only for **matching**
- Final output will **restore repetition**

---

## Step 3 — Find Correct Canonical Line

Instead of matching across full shabad:
- Match **one exact line first**

### Strategy

Use:
- First letters
- Optional consonant similarity
- Sequential filtering

---

### Example Matching

Whisper:
```
ਢਧਾ ਕਰਤਿਆ ਨਿਹਫਲੁ
```

Canonical candidates:
```
ਮਾਇਆ ਮਮਤਾ...
ਮਨਮੁਖ ਖਾਧੇ...
ਧੰਧਾ ਕਰਤਿਆ ਨਿਹਫਲੁ ਜਨਮੁ...
```

Matching logic:
1. Extract first letters:
```
ਢ ਕ ਨ
```

2. Search lines:
- If one match → ✅ done
- If multiple → refine using second letters
- If none → apply consonant similarity

---

## Step 4 — Consonant Similarity Map

Handle pronunciation / ASR errors:

| Group | Letters |
|------|--------|
| Dental ↔ Retroflex | ਨ ↔ ਣ |
| Stops | ਤ ↔ ਟ |
| Aspirated | ਥ ↔ ਠ |
| Voiced | ਦ ↔ ਡ |
| Breath | ਧ ↔ ਢ |

Example:
```
ਢਧਾ → ਧੰਧਾ
```

---

## Step 5 — Prefix Mapping

Match partial words:

| Whisper | Canonical |
|--------|----------|
| ਤੁਧ | ਤੁਧੁ |
| ਆਪ | ਆਪੇ / ਆਪਿ |
| ਜਗਤ | ਜਗਤੁ |

---

## Step 6 — Resolve Prefix Ambiguity

Problem:
```
ਆਪ → ਆਪੇ OR ਆਪਿ
```

### Solution: Positional Context

Example:
```
Whisper: ਆਪ ਜਗਤ
Canonical: ਆਪੇ ਜਗਤੁ
```

👉 Choose:
```
ਆਪੇ (because followed by ਜਗਤੁ)
```

---

## Step 7 — Handle Merged / Split Words

Problems:
- Merged: `ਹਰਬਿਨ → ਹਰਿ ਬਿਨੁ`
- Split: Whisper breaks one word into multiple parts

### Solutions:
1. **Merged words**
   - If matching fails, try splitting using canonical boundaries

2. **Split words (NEW ADDITION)**
   - If Whisper breaks a word:
   - Combine adjacent tokens and try matching

Example:
```
Whisper: ਹਰਿ ਬਿ ਨੁ
Try:
ਹਰਿ + ਬਿ → ਹਰਿਬਿ (no)
ਹਰਿ + ਬਿ + ਨੁ → ਹਰਿਬਿਨੁ → match → ਹਰਿ ਬਿਨੁ
```

👉 Strategy:
- Try combining current + next word
- If needed, skip next token(s) to form match
- Stop once match found

---

## Step 8 — Final Mapping

Map Whisper → Canonical using selected line:

```
ਤੁਧ → ਤੁਧੁ
ਆਪ → ਆਪੇ
ਜਗਤ → ਜਗਤੁ
```

---

## Step 9 — Restore Repetition

Original Whisper:
```
ਤੁਧ ਤੁਧ ਤੁਧ ਆਪੇ ਜਗਤੁ
```

Final Output:
```
ਤੁਧੁ ਤੁਧੁ ਤੁਧੁ ਆਪੇ ਜਗਤੁ
```

---

# Vishrams Handling

👉 Vishrams can be **ignored**

Reason:
- We match using **word sequence**
- Not dependent on pauses

---

# Key Design Decision

❌ Do NOT build dictionary from multiple lines  
✅ Always match **one canonical line first**

Benefits:
- Reduces ambiguity
- Improves prefix matching
- Simplifies mapping

---

# Debug / Dataset Logging

Log every step:

```
Whisper Raw:
ਤੁਧ ਤੁਧ ਤੁਧ ਆਪ ਜਗਤ ਉਪ

After Dedup:
ਤੁਧ ਆਪ ਜਗਤ ਉਪ

Matched Line:
ਤੁਧੁ ਆਪੇ ਜਗਤੁ ਉਪਾਇ ਕੈ ਆਪਿ ਖੇਲੁ ਰਚਾਇਆ ॥

Mapping:
ਤੁਧ → ਤੁਧੁ
ਆਪ → ਆਪੇ
ਜਗਤ → ਜਗਤੁ

Final Output:
ਤੁਧੁ ਤੁਧੁ ਤੁਧੁ ਆਪੇ ਜਗਤੁ
```

---

# Edge Cases Covered

## 1. Prefix Ambiguity
- ਆਪ → ਆਪੇ / ਆਪਿ  
✅ Solved using **context**

---

## 2. Consonant Confusion
- ਧ ↔ ਢ, ਤ ↔ ਟ  
✅ Solved using **similarity map**

---

## 3. Merged / Split Words
- ਹਰਬਿਨ → ਹਰਿ ਬਿਨੁ  
- ਹਰਿ ਬਿ ਨੁ → ਹਰਿ ਬਿਨੁ  
✅ Solved using **splitting + combining fallback**

---

## 4. Repetition
- ਤੁਧ ਤੁਧ ਤੁਧ  
✅ Removed for matching, restored later

---

# Final Summary

### Why this works

- Whisper gives **approximate phonetic text**
- Gurbani provides **perfect ground truth**
- Mapping aligns the two

---

### Final Pipeline

```
Audio
 → Whisper
 → Remove repetition (temp)
 → Match exact canonical line
 → Prefix + similarity mapping
 → Restore repetition
 → Final Gurbani output
```
