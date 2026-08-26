#!/usr/bin/env python3
"""Render the placeholder photographs the image figures use.

The figures previously drew outline glyphs, which read as clip art. These are
small procedurally rendered scenes, written straight to PNG with zlib so there is
no image dependency. They look like photographs at thumbnail size without
depicting anything real, which is what a moderation example needs.

Writes tiles.js, which defines window.SS_TILES.
"""
import base64
import json
import math
import pathlib
import struct
import zlib

W = H = 168


def png(rgb):
    raw = b''.join(b'\x00' + bytes(row) for row in rgb)

    def chunk(tag, data):
        return (struct.pack('>I', len(data)) + tag + data
                + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff))

    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', W, H, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw, 9))
            + chunk(b'IEND', b''))


def clamp(v):
    return 0 if v < 0 else (255 if v > 255 else int(v))


def smooth(t):
    return t * t * (3 - 2 * t)


def scene(kind, seed=1):
    """Build one image as a list of rows of packed RGB bytes."""
    rows = []
    for y in range(H):
        row = bytearray()
        v = y / H
        for x in range(W):
            u = x / W
            if kind == 'landscape':
                # sky gradient, sun glow, hills, foreground
                r, g, b = 244 - 120 * v, 220 - 70 * v, 198 - 10 * v
                d = math.hypot(u - 0.68, v - 0.26)
                glow = max(0.0, 1 - d * 2.6) ** 2
                r += 60 * glow; g += 44 * glow; b += 12 * glow
                hill = 0.62 + 0.07 * math.sin(u * 5.1 + seed) + 0.03 * math.sin(u * 11.0)
                if v > hill:
                    k = smooth(min(1.0, (v - hill) * 4))
                    r = r * (1 - k) + 72 * k
                    g = g * (1 - k) + 96 * k
                    b = b * (1 - k) + 74 * k
                hill2 = 0.78 + 0.05 * math.sin(u * 3.3 + 2.1)
                if v > hill2:
                    k = smooth(min(1.0, (v - hill2) * 5))
                    r = r * (1 - k) + 38 * k
                    g = g * (1 - k) + 48 * k
                    b = b * (1 - k) + 40 * k
                if abs(v - hill) < 0.006:
                    r *= .72; g *= .78; b *= .74
                for tx in (0.18, 0.27, 0.79):
                    th = hill - 0.10 - 0.03 * math.sin(tx * 30)
                    if abs(u - tx) < 0.016 and th < v < hill + 0.005:
                        r, g, b = 54, 66, 52
            elif kind == 'interior':
                # a warm room: wall, window light, table edge
                r, g, b = 206 - 30 * v, 190 - 34 * v, 172 - 36 * v
                if 0.12 < u < 0.46 and 0.14 < v < 0.56:
                    k = 0.85
                    r = r * (1 - k) + 246 * k; g = g * (1 - k) + 240 * k; b = b * (1 - k) + 222 * k
                beam = max(0.0, 1 - abs((u - 0.34) - (v - 0.3) * 0.5) * 3.0) * max(0.0, 1 - v)
                r += 46 * beam; g += 40 * beam; b += 24 * beam
                if v > 0.72:
                    k = smooth(min(1.0, (v - 0.72) * 6))
                    r = r * (1 - k) + 122 * k; g = g * (1 - k) + 92 * k; b = b * (1 - k) + 64 * k
            else:  # 'objects', a flat lay of shapes on a surface
                r, g, b = 214 - 18 * v, 206 - 20 * v, 192 - 20 * v
                for (cx, cy, rad, col) in ((0.34, 0.44, 0.17, (176, 92, 62)),
                                           (0.62, 0.56, 0.13, (86, 118, 106)),
                                           (0.50, 0.30, 0.09, (120, 106, 152))):
                    d = math.hypot(u - cx, v - cy)
                    if d < rad:
                        k = smooth(min(1.0, (rad - d) * 14))
                        sh = 1 - 0.35 * ((v - cy) / rad)
                        r = r * (1 - k) + col[0] * sh * k
                        g = g * (1 - k) + col[1] * sh * k
                        b = b * (1 - k) + col[2] * sh * k
            # film grain and a soft vignette, which is what sells it as a photo
            n = ((x * 7349 + y * 9151 + seed * 5417) % 23) / 23.0 - 0.5
            vig = 1 - 0.42 * math.hypot(u - 0.5, v - 0.5) ** 2 * 2
            q = 6   # posterise, which costs nothing visually and compresses far better
            row += bytes((clamp(round((r + n * 7) * vig / q) * q),
                          clamp(round((g + n * 7) * vig / q) * q),
                          clamp(round((b + n * 7) * vig / q) * q)))
        rows.append(row)
    return rows


tiles = {}
for name, kind, seed in (('landscape', 'landscape', 1),
                         ('interior', 'interior', 3),
                         ('objects', 'objects', 5)):
    data = png(scene(kind, seed))
    tiles[name] = 'data:image/png;base64,' + base64.b64encode(data).decode()
    print(f"  {name:10s} {len(data)/1024:.1f} KB")

out = pathlib.Path(__file__).resolve().parent / 'tiles.js'
out.write_text('/* Procedurally rendered placeholder photographs. See make_tiles.py. */\n'
               'window.SS_TILES = ' + json.dumps(tiles) + ';\n', encoding='utf-8')
print("wrote tiles.js", round(out.stat().st_size / 1024), "KB")
