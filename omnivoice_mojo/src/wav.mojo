"""Minimal 16-bit PCM mono WAV writer."""


def _le32(mut b: List[UInt8], v: Int):
    b.append(UInt8(v & 0xFF))
    b.append(UInt8((v >> 8) & 0xFF))
    b.append(UInt8((v >> 16) & 0xFF))
    b.append(UInt8((v >> 24) & 0xFF))


def _le16(mut b: List[UInt8], v: Int):
    b.append(UInt8(v & 0xFF))
    b.append(UInt8((v >> 8) & 0xFF))


def _tag(mut b: List[UInt8], s: String):
    var p = s.unsafe_ptr()
    for i in range(s.byte_length()):
        b.append(p[i])


def write_wav(path: String, wav: List[Float32], sample_rate: Int) raises:
    var n = len(wav)
    var b = List[UInt8](capacity=44 + 2 * n)
    _tag(b, "RIFF")
    _le32(b, 36 + 2 * n)
    _tag(b, "WAVE")
    _tag(b, "fmt ")
    _le32(b, 16)
    _le16(b, 1)                     # PCM
    _le16(b, 1)                     # mono
    _le32(b, sample_rate)
    _le32(b, sample_rate * 2)       # byte rate
    _le16(b, 2)                     # block align
    _le16(b, 16)                    # bits per sample
    _tag(b, "data")
    _le32(b, 2 * n)
    for i in range(n):
        var x = wav[i]
        if x > 1.0:
            x = 1.0
        if x < -1.0:
            x = -1.0
        var v = Int(x * 32767.0)
        _le16(b, v & 0xFFFF)
    var f = open(path, "w")
    f.write_bytes(b)
    f.close()
