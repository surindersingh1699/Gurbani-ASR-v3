"""STTM ASCII → Unicode Gurmukhi converter.

The STTM database stores Gurbani in a legacy ASCII encoding based on the
GurmukhiAkhar font. This module converts that encoding to standard Unicode
Gurmukhi (U+0A00 block).

Canonical mapping from @shabados/gurmukhi-utils unicode.jsonc.

Usage:
    from src.data.unicode_convert import ascii_to_unicode
    text = ascii_to_unicode("<> siq nwmu krqw purKu")
    # → "ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ"
"""

import re

# ─── Multi-character replacements (applied first, longest match first) ───────

_ASCII_REPLACEMENTS = [
    ("<>", "\u0A74"),      # ੴ Ik Onkar
    ("<", "\u0A74"),       # ੴ
    (">", "\u262C"),       # ☬ Khanda
]

# ─── Single-character mapping ────────────────────────────────────────────────

_ASCII_MAP = {
    # Vowel carriers
    "a": "\u0A73",  # ੳ oora
    "A": "\u0A05",  # ਅ aira
    "e": "\u0A72",  # ੲ iri
    # Consonants
    "s": "\u0A38",  # ਸ
    "S": "\u0A36",  # ਸ਼ (sha)
    "h": "\u0A39",  # ਹ
    "H": "\u0A4D\u0A39",  # ੍ਹ pair ha
    "k": "\u0A15",  # ਕ
    "K": "\u0A16",  # ਖ
    "g": "\u0A17",  # ਗ
    "G": "\u0A18",  # ਘ
    "|": "\u0A19",  # ਙ
    "c": "\u0A1A",  # ਚ
    "C": "\u0A1B",  # ਛ
    "j": "\u0A1C",  # ਜ
    "J": "\u0A1D",  # ਝ
    "\\": "\u0A1E", # ਞ
    "t": "\u0A1F",  # ਟ
    "T": "\u0A20",  # ਠ
    "f": "\u0A21",  # ਡ
    "F": "\u0A22",  # ਢ
    "x": "\u0A23",  # ਣ
    "q": "\u0A24",  # ਤ
    "Q": "\u0A25",  # ਥ
    "d": "\u0A26",  # ਦ
    "D": "\u0A27",  # ਧ
    "n": "\u0A28",  # ਨ
    "p": "\u0A2A",  # ਪ
    "P": "\u0A2B",  # ਫ
    "b": "\u0A2C",  # ਬ
    "B": "\u0A2D",  # ਭ
    "m": "\u0A2E",  # ਮ
    "X": "\u0A2F",  # ਯ
    "r": "\u0A30",  # ਰ
    "l": "\u0A32",  # ਲ
    "L": "\u0A33",  # ਲ਼
    "v": "\u0A35",  # ਵ
    "V": "\u0A5C",  # ੜ
    # Vowel signs (matras)
    "w": "\u0A3E",  # ਾ kanna
    "W": "\u0A3E\u0A02",  # ਾਂ kanna+bindi
    "i": "\u0A3F",  # ਿ sihari
    "I": "\u0A40",  # ੀ bihari
    "u": "\u0A41",  # ੁ aunkar
    "U": "\u0A42",  # ੂ dulankar
    "y": "\u0A47",  # ੇ lavan
    "Y": "\u0A48",  # ੈ dulainkan
    "o": "\u0A4B",  # ੋ hora
    "O": "\u0A4C",  # ੌ kanaura
    "E": "\u0A13",  # ਓ
    # Nasals and modifiers
    "M": "\u0A70",  # ੰ tippi
    "N": "\u0A02",  # ਂ bindi
    "`": "\u0A71",  # ੱ adhak
    "~": "\u0A71",  # ੱ adhak (alt)
    "@": "\u0A51",  # ੑ udaat
    # Nukta forms
    "z": "\u0A5B",  # ਜ਼
    "Z": "\u0A5A",  # ਗ਼
    "^": "\u0A59",  # ਖ਼
    "&": "\u0A5E",  # ਫ਼
    # Subscript conjuncts
    "R": "\u0A4D\u0A30",  # ੍ਰ pair ra
    # Numbers
    "0": "\u0A66",  # ੦
    "1": "\u0A67",  # ੧
    "2": "\u0A68",  # ੨
    "3": "\u0A69",  # ੩
    "4": "\u0A6A",  # ੪
    "5": "\u0A6B",  # ੫
    "6": "\u0A6C",  # ੬
    "7": "\u0A6D",  # ੭
    "8": "\u0A6E",  # ੮
    "9": "\u0A6F",  # ੯
    # Punctuation
    "[": "\u0964",  # । single danda
    "]": "\u0965",  # ॥ double danda
    # Extended (from GurmukhiAkhar font)
    "\u00e6": "\u0A3C",  # ਼ nukta (æ)
    "\u00a1": "\u0A74",  # ੴ (¡)
    "\u0192": "\u0A28\u0A42\u0A70",  # ਨੂੰ (ƒ)
    "\u0153": "\u0A4D\u0A24",  # ੍ਤ (œ)
    "\u00cd": "\u0A4D\u0A35",  # ੍ਵ (Í)
    "\u00cf": "\u0A75",  # ੵ yakash (Ï)
    "\u00d2": "\u0965",  # ॥ (Ò)
    "\u00da": "\u0A03",  # ਃ visarga (Ú)
    "\u02c6": "\u0A02",  # ਂ bindi (ˆ)
    "\u02dc": "\u0A4D\u0A28",  # ੍ਨ (˜)
    "\u00a7": "\u0A4D\u0A39\u0A42",  # ੍ਹੂ (§)
    "\u00a4": "\u0A71",  # ੱ adhak (¤)
    "\u00e7": "\u0A4D\u0A1A",  # ੍ਚ (ç)
    "\u2020": "\u0A4D\u0A1F",  # ੍ਟ (†)
    "\u00fc": "\u0A41",  # ੁ (ü)
    "\u00ae": "\u0A4D\u0A30",  # ੍ਰ (®)
    "\u00b4": "\u0A75",  # ੵ yakash (´)
    "\u00a8": "\u0A42",  # ੂ (¨)
    "\u00b5": "\u0A70",  # ੰ tippi (µ)
}

# Characters nullified (produce empty string)
_ASCII_NULLIFY = set("\u00c6\u00d8\u00ff\u0152\u2030\u00d3\u00d4")

# Unicode sanitization: carrier+vowel → standalone vowel
_UNICODE_SANITIZE = [
    ("\u0A73\u0A4B", "\u0A13"),  # ੳੋ → ਓ
    ("\u0A05\u0A3E", "\u0A06"),  # ਅਾ → ਆ
    ("\u0A72\u0A3F", "\u0A07"),  # ੲਿ → ਇ
    ("\u0A72\u0A40", "\u0A08"),  # ੲੀ → ਈ
    ("\u0A73\u0A41", "\u0A09"),  # ੳੁ → ਉ
    ("\u0A73\u0A42", "\u0A0A"),  # ੳੂ → ਊ
    ("\u0A72\u0A47", "\u0A0F"),  # ੲੇ → ਏ
    ("\u0A05\u0A48", "\u0A10"),  # ਅੈ → ਐ
    ("\u0A05\u0A4C", "\u0A14"),  # ਅੌ → ਔ
]

# Gurmukhi base letters for sihari reordering
_BASE_LETTERS = set(
    "\u0A15\u0A16\u0A17\u0A18\u0A19"  # ka-nga
    "\u0A1A\u0A1B\u0A1C\u0A1D\u0A1E"  # cha-nya
    "\u0A1F\u0A20\u0A21\u0A22\u0A23"  # tta-nna
    "\u0A24\u0A25\u0A26\u0A27\u0A28"  # ta-na
    "\u0A2A\u0A2B\u0A2C\u0A2D\u0A2E"  # pa-ma
    "\u0A2F\u0A30\u0A32\u0A33\u0A35"  # ya-va
    "\u0A36\u0A38\u0A39"              # sha-ha
    "\u0A59\u0A5A\u0A5B\u0A5C\u0A5E"  # nukta forms
    "\u0A05\u0A06\u0A07\u0A08\u0A09\u0A0A"  # vowels
    "\u0A0F\u0A10\u0A13\u0A14"              # vowels
    "\u0A72\u0A73\u0A74"                     # carriers + ik onkar
)

_SIHARI = "\u0A3F"  # ਿ


def ascii_to_unicode(text: str) -> str:
    """Convert STTM ASCII encoding to Unicode Gurmukhi.

    Handles the GurmukhiAkhar font-based encoding used in STTM database.
    Steps: multi-char replacements → single-char map → sihari reorder → sanitize.
    """
    if not text:
        return text

    # Step 1: Multi-character replacements (longest first)
    for ascii_seq, uni in _ASCII_REPLACEMENTS:
        text = text.replace(ascii_seq, uni)

    # Step 2: Single-character mapping
    result = []
    for ch in text:
        if ch in _ASCII_NULLIFY:
            continue
        result.append(_ASCII_MAP.get(ch, ch))
    text = "".join(result)

    # Step 3: Sihari reordering — move ਿ after the following base letter
    # In ASCII encoding, sihari appears BEFORE its consonant (visual order)
    # In Unicode, it must come AFTER (logical order)
    chars = list(text)
    i = 0
    while i < len(chars) - 1:
        if chars[i] == _SIHARI and chars[i + 1] in _BASE_LETTERS:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            i += 2
        else:
            i += 1
    text = "".join(chars)

    # Step 4: Unicode sanitization — carrier+vowel → standalone vowel
    for raw, clean in _UNICODE_SANITIZE:
        text = text.replace(raw, clean)

    # Step 5: Strip vishraam markers and normalise whitespace
    text = text.replace(";", "").replace(".", " ")
    text = re.sub(r"\s+", " ", text).strip()

    return text
