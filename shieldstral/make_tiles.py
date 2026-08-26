#!/usr/bin/env python3
"""Ray trace the placeholder images the moderation figures use.

Earlier versions drew outline glyphs, which read as clip art, then procedural
gradients, which read as blur. These are actually ray traced: spheres on a
checkered plane, a sky gradient, one area light sampled for soft shadows,
Lambert plus Blinn-Phong shading and a reflection bounce. Pure Python written
straight to PNG with zlib, so there is no image dependency, the output is
deterministic, and nothing real is depicted.

Writes tiles.js, which defines window.SS_TILES.
"""
import base64
import json
import math
import pathlib
import random
import struct
import zlib

W = H = 200
SHADOW_SAMPLES = 6
MAXD = 1e9


def norm(v):
    l = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) or 1.0
    return (v[0] / l, v[1] / l, v[2] / l)


def sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def mul(a, s): return (a[0] * s, a[1] * s, a[2] * s)
def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def hit_sphere(o, d, c, r):
    oc = sub(o, c)
    b = dot(oc, d)
    cc = dot(oc, oc) - r * r
    disc = b * b - cc
    if disc < 0:
        return MAXD
    s = math.sqrt(disc)
    t = -b - s
    if t > 1e-3:
        return t
    t = -b + s
    return t if t > 1e-3 else MAXD


def hit_box(o, d, lo, hi):
    """Slab test. Returns (t, normal) or (MAXD, None)."""
    tmin, tmax, nrm = -MAXD, MAXD, (0, 0, 0)
    for i in range(3):
        if abs(d[i]) < 1e-9:
            if o[i] < lo[i] or o[i] > hi[i]:
                return MAXD, None
            continue
        inv = 1.0 / d[i]
        t1, t2 = (lo[i] - o[i]) * inv, (hi[i] - o[i]) * inv
        sgn = -1.0
        if t1 > t2:
            t1, t2 = t2, t1
            sgn = 1.0
        if t1 > tmin:
            tmin = t1
            n = [0.0, 0.0, 0.0]
            n[i] = sgn
            nrm = tuple(n)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return MAXD, None
    return (tmin, nrm) if tmin > 1e-3 else (MAXD, None)


def scene_for(kind):
    """Return (spheres, boxes, ground, sky_top, sky_bot).

    sphere = (centre, radius, colour, gloss, stripe)
    box    = (lo, hi, colour, gloss)
    """
    if kind == 'blocks':
        # a stack of toy building blocks
        return ([],
                [((-0.62, 0.00, 2.55), (0.10, 0.62, 3.27), (0.82, 0.30, 0.20), 0.20),
                 ((-0.52, 0.62, 2.66), (0.00, 1.18, 3.16), (0.24, 0.48, 0.62), 0.20),
                 ((-0.42, 1.18, 2.76), (-0.10, 1.62, 3.06), (0.92, 0.72, 0.24), 0.20),
                 ((0.28, 0.00, 2.30), (0.94, 0.50, 2.96), (0.34, 0.56, 0.38), 0.20)],
                (0.66, 0.63, 0.57), (0.44, 0.60, 0.80), (0.92, 0.88, 0.78))
    if kind == 'ball':
        # a striped beach ball beside a smaller one
        return ([((-0.30, 0.66, 2.75), 0.66, (0.90, 0.88, 0.84), 0.42, 1),
                 ((0.95, 0.28, 2.20), 0.28, (0.86, 0.44, 0.18), 0.35, 0)],
                [],
                (0.68, 0.65, 0.60), (0.46, 0.62, 0.82), (0.94, 0.90, 0.80))
    # a little toy car: body, cabin, two wheels
    return ([((-0.42, 0.20, 2.32), 0.20, (0.16, 0.16, 0.18), 0.30, 0),
             ((0.46, 0.20, 2.32), 0.20, (0.16, 0.16, 0.18), 0.30, 0)],
            [((-0.72, 0.20, 2.06), (0.76, 0.56, 2.58), (0.84, 0.26, 0.20), 0.34),
             ((-0.34, 0.56, 2.12), (0.38, 0.86, 2.52), (0.40, 0.62, 0.74), 0.40)],
            (0.66, 0.63, 0.58), (0.44, 0.60, 0.80), (0.92, 0.88, 0.78))


def trace(kind, seed):
    rnd = random.Random(seed)
    spheres, boxes, ground, sky_top, sky_bot = scene_for(kind)
    eye = (0.0, 0.85, 0.0)
    light = (-2.2, 3.4, 1.2)
    lr = 0.6
    rows = []
    for y in range(H):
        row = bytearray()
        for x in range(W):
            u = (x + 0.5) / W * 2 - 1
            v = -((y + 0.5) / H * 2 - 1) * 0.75
            d = norm((u, v, 1.0))

            best, what, sp, bn = MAXD, None, None, None
            for s in spheres:
                t = hit_sphere(eye, d, s[0], s[1])
                if t < best:
                    best, what, sp = t, 'sphere', s
            for bx in boxes:
                t, nb = hit_box(eye, d, bx[0], bx[1])
                if t < best:
                    best, what, sp, bn = t, 'box', bx, nb
            if d[1] < -1e-4:
                tg = -eye[1] / d[1]
                if 1e-3 < tg < best:
                    best, what, sp = tg, 'plane', None

            if what is None:
                k = max(0.0, min(1.0, (d[1] + 0.25) / 1.1))
                col = add(mul(sky_bot, 1 - k), mul(sky_top, k))
            else:
                p = add(eye, mul(d, best))
                if what == 'sphere':
                    n = norm(sub(p, sp[0]))
                    base, gloss = sp[2], sp[3]
                    if len(sp) > 4 and sp[4]:
                        # beach ball panels, from the angle around the vertical axis
                        ang = math.atan2(n[0], n[2])
                        seg = int((ang + math.pi) / (2 * math.pi) * 6) % 6
                        base = ((0.86, 0.30, 0.24), (0.94, 0.92, 0.88), (0.22, 0.46, 0.68),
                                (0.94, 0.92, 0.88), (0.96, 0.74, 0.22),
                                (0.94, 0.92, 0.88))[seg]
                elif what == 'box':
                    n = bn
                    base, gloss = sp[2], sp[3]
                else:
                    n = (0.0, 1.0, 0.0)
                    chk = (int(math.floor(p[0] * 1.3)) + int(math.floor(p[2] * 1.3))) & 1
                    base = mul(ground, 1.0 if chk else 0.84)
                    gloss = 0.10
                    fade = max(0.0, min(1.0, (8.0 - p[2]) / 6.0))
                    base = add(mul(base, fade), mul(sky_bot, 1 - fade))

                lit = 0
                for _ in range(SHADOW_SAMPLES):
                    lp = (light[0] + (rnd.random() - .5) * lr,
                          light[1] + (rnd.random() - .5) * lr,
                          light[2] + (rnd.random() - .5) * lr)
                    ld = norm(sub(lp, p))
                    blocked = False
                    for s2 in spheres:
                        if hit_sphere(p, ld, s2[0], s2[1]) < MAXD:
                            blocked = True
                            break
                    if not blocked:
                        for b2 in boxes:
                            if hit_box(p, ld, b2[0], b2[1])[0] < MAXD:
                                blocked = True
                                break
                    if not blocked:
                        lit += 1
                shade = lit / SHADOW_SAMPLES

                ld = norm(sub(light, p))
                lam = max(0.0, dot(n, ld)) * shade
                hv = norm(sub(ld, d))
                spec = max(0.0, dot(n, hv)) ** 42 * gloss * shade
                amb = 0.26 + 0.13 * max(0.0, n[1])
                col = add(mul(base, amb + 0.85 * lam), (spec, spec, spec))
                if gloss > 0.2:
                    r = norm(sub(d, mul(n, 2 * dot(d, n))))
                    k = max(0.0, min(1.0, (r[1] + 0.25) / 1.1))
                    sky = add(mul(sky_bot, 1 - k), mul(sky_top, k))
                    col = add(mul(col, 1 - gloss * 0.32), mul(sky, gloss * 0.32))

            cx, cy = (x / W - .5), (y / H - .5)
            vig = 1 - 0.34 * (cx * cx + cy * cy) * 2
            g = ((x * 7349 + y * 9151 + seed * 5417) % 17) / 17.0 - 0.5
            for c in col:
                c = (c ** 0.85) * vig + g * 0.016
                q = 6
                row.append(max(0, min(255, int(round(c * 255 / q) * q))))
        rows.append(row)
    return rows


def png(rows):
    raw = b''.join(b'\x00' + bytes(r) for r in rows)

    def chunk(tag, data):
        return (struct.pack('>I', len(data)) + tag + data
                + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff))

    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', W, H, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw, 9))
            + chunk(b'IEND', b''))


if __name__ == '__main__':
    tiles = {}
    for name, seed in (('blocks', 11), ('ball', 23), ('car', 37)):
        data = png(trace(name, seed))
        tiles[name] = 'data:image/png;base64,' + base64.b64encode(data).decode()
        print(f"  {name:10s} {len(data)/1024:5.1f} KB")
    out = pathlib.Path(__file__).resolve().parent / 'tiles.js'
    out.write_text('/* Ray traced placeholder images. See make_tiles.py. */\n'
                   'window.SS_TILES = ' + json.dumps(tiles) + ';\n', encoding='utf-8')
    print("wrote tiles.js", round(out.stat().st_size / 1024), "KB")
