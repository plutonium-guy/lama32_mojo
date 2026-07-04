"""Pure-Mojo port of OmniVoice's RuleDurationEstimator.

Estimates target audio-token count from text via per-character phonetic
weights (1.0 = one Latin letter). Script classes come from Unicode ranges;
this is a close approximation of the Python original, which additionally
consults unicodedata categories for exotic punctuation/marks.
"""


def _char_weight(cp: Int) -> Float64:
    # ASCII fast paths
    if (cp >= 65 and cp <= 90) or (cp >= 97 and cp <= 122):
        return 1.0                                     # latin letter
    if cp == 32 or cp == 9:
        return 0.2                                     # space
    if cp >= 48 and cp <= 57:
        return 3.5                                     # digit
    if cp < 128:
        return 0.5                                     # ASCII punct/symbol
    if cp == 0x0640:
        return 0.0                                     # arabic tatweel
    # combining marks (common blocks)
    if (cp >= 0x0300 and cp <= 0x036F) or (cp >= 0x0483 and cp <= 0x0489) \
            or (cp >= 0x0591 and cp <= 0x05C7) or (cp >= 0x064B and cp <= 0x065F) \
            or (cp >= 0x0E31 and cp <= 0x0E3A) or (cp >= 0x1DC0 and cp <= 0x1DFF) \
            or (cp >= 0x20D0 and cp <= 0x20FF) or (cp >= 0xFE00 and cp <= 0xFE0F):
        return 0.0
    # general punctuation / CJK punctuation / fullwidth punct
    if (cp >= 0x2000 and cp <= 0x206F) or (cp >= 0x3000 and cp <= 0x303F) \
            or (cp >= 0xFE30 and cp <= 0xFE4F) or (cp >= 0xFF01 and cp <= 0xFF0F) \
            or (cp >= 0xFF1A and cp <= 0xFF20) or (cp >= 0xFF3B and cp <= 0xFF40) \
            or (cp >= 0xFF5B and cp <= 0xFF65):
        return 0.5
    # script ranges (subset of the Python table, same weights)
    if cp <= 0x02AF:
        return 1.0                                     # latin ext
    if cp <= 0x03FF:
        return 1.0                                     # greek
    if cp <= 0x052F:
        return 1.0                                     # cyrillic
    if cp <= 0x058F:
        return 1.0                                     # armenian
    if cp <= 0x05FF:
        return 1.5                                     # hebrew
    if cp <= 0x08FF:
        return 1.5                                     # arabic
    if cp <= 0x0DFF:
        return 1.8                                     # indic scripts
    if cp <= 0x0EFF:
        return 1.5                                     # thai/lao
    if cp <= 0x0FFF:
        return 1.8                                     # tibetan
    if cp <= 0x109F:
        return 1.8                                     # myanmar
    if cp <= 0x10FF:
        return 1.0                                     # georgian
    if cp <= 0x11FF:
        return 2.5                                     # hangul jamo
    if cp <= 0x139F:
        return 3.0                                     # ethiopic
    if cp <= 0x17FF and cp >= 0x1780:
        return 1.8                                     # khmer
    if cp >= 0x1E00 and cp <= 0x1EFF:
        return 1.0                                     # latin ext additional
    if cp >= 0x3040 and cp <= 0x30FF:
        return 2.2                                     # kana
    if cp >= 0x3105 and cp <= 0x312F:
        return 3.0                                     # bopomofo
    if cp >= 0x3130 and cp <= 0x318F:
        return 2.5                                     # hangul compat jamo
    if cp >= 0x3400 and cp <= 0x9FFF:
        return 3.0                                     # cjk
    if cp >= 0xA000 and cp <= 0xA4CF:
        return 3.0                                     # yi
    if cp >= 0xAC00 and cp <= 0xD7AF:
        return 2.5                                     # hangul syllables
    if cp >= 0xF900 and cp <= 0xFAFF:
        return 3.0                                     # cjk compat
    if cp >= 0xFF66 and cp <= 0xFFDC:
        return 2.2                                     # halfwidth kana/hangul
    if cp > 0x20000:
        return 3.0                                     # cjk ext planes
    return 1.0


def _codepoints(s: String) -> List[Int]:
    """Decode UTF-8 bytes into codepoints."""
    var out = List[Int]()
    var p = s.unsafe_ptr()
    var n = s.byte_length()
    var i = 0
    while i < n:
        var b0 = Int(p[i])
        if b0 < 0x80:
            out.append(b0)
            i += 1
        elif b0 < 0xE0:
            out.append(((b0 & 0x1F) << 6) | (Int(p[i + 1]) & 0x3F))
            i += 2
        elif b0 < 0xF0:
            out.append(((b0 & 0x0F) << 12) | ((Int(p[i + 1]) & 0x3F) << 6)
                       | (Int(p[i + 2]) & 0x3F))
            i += 3
        else:
            out.append(((b0 & 0x07) << 18) | ((Int(p[i + 1]) & 0x3F) << 12)
                       | ((Int(p[i + 2]) & 0x3F) << 6) | (Int(p[i + 3]) & 0x3F))
            i += 4
    return out^


def text_weight(s: String) -> Float64:
    var cps = _codepoints(s)
    var w = Float64(0)
    for i in range(len(cps)):
        w += _char_weight(cps[i])
    return w


def estimate_target_tokens(text: String, ref_text: String,
                           num_ref_tokens: Int, speed: Float64) -> Int:
    """Estimated audio-token count (25 Hz frames) for `text`.

    Mirrors OmniVoice._estimate_target_tokens: scale by the reference's
    tokens-per-weight rate (fallback: "Nice to meet you." = 25 tokens),
    with a power-curve boost below 50 tokens (boost_strength 3).
    """
    var rt = ref_text
    var rn = num_ref_tokens
    if rn <= 0 or rt.byte_length() == 0:
        rt = String("Nice to meet you.")
        rn = 25
    var ref_w = text_weight(rt)
    if ref_w <= 0:
        return 1
    var est = text_weight(text) * Float64(rn) / ref_w
    var low = Float64(50)
    if est < low:
        est = low * ((est / low) ** (1.0 / 3.0))
    if speed > 0 and speed != 1.0:
        est = est / speed
    if est < 1:
        return 1
    return Int(est)
