#!/usr/bin/env python3
"""Parse the Robostral Navigate paper text into structured JSON.

Nothing downstream may hand-type a number. Run:

    curl -sL -o /tmp/robostral.pdf https://arxiv.org/pdf/2607.20785
    python3 -c "from pypdf import PdfReader; ..."   # -> /tmp/robostral.txt
    python3 parse_robostral.py

The PDF extractor drops the spaces between some table cells, so
'3.470.7510.687' has to be read as 3.47 / 0.751 / 0.687. Navigation error is
always d.dd and every other metric is a 0.ddd fraction, which is enough to
tokenise the run unambiguously.
"""
import json
import pathlib
import re

SRC = pathlib.Path("/tmp/robostral.txt")
OUT = pathlib.Path(__file__).resolve().parent / "robostral_data.json"
txt = SRC.read_text(encoding="utf-8")

# Order matters. Try the 0.ddd fraction first so that '0.490' is one cell and
# not a 0.49 navigation error with a stray 0 after it. Navigation error is the
# only d.dd cell, so whatever is left over is unambiguous.
NUM = re.compile(r"0\.\d{3}|\d{1,2}\.\d{2}")


def numbers(s):
    """Read a concatenated run of table cells left to right."""
    out, i = [], 0
    while i < len(s):
        m = NUM.match(s, i)
        if m:
            out.append(float(m.group()))
            i = m.end()
        else:
            i += 1
    return out


# ---------------------------------------------------------------- table 1
# 'NaVid [Zhang et al., 2024b]✓5.47 0.490 0.370 0.350 5.72 0.457 0.382'
ROW = re.compile(r"^(?P<name>[A-Za-z][\w\-\. ()]*?)\s*"
                 r"(?:\[[^\]]*\])?\s*"
                 r"(?P<cam>[✓✗])\s*"
                 r"(?P<nums>[\d\. ]+)$")

rows, group = [], None
for line in txt.splitlines():
    line = line.strip()
    if line.startswith("Single RGB camera"):
        group = "single"
        continue
    if line.startswith("Depth / multi-camera"):
        group = "depth"
        continue
    m = ROW.match(line)
    if not m or group is None:
        continue
    v = numbers(m.group("nums"))
    if len(v) != 7:
        raise SystemExit(f"expected 7 metrics, got {len(v)} in: {line!r}")
    # NE is metres, the other five are fractions. If that is not true the run
    # was tokenised wrong and every downstream figure would be silently off.
    if not (1.0 < v[0] < 12.0 and 1.0 < v[4] < 12.0):
        raise SystemExit(f"navigation error out of range in: {line!r} -> {v}")
    if not all(0.0 < x < 1.0 for x in (v[1], v[2], v[3], v[5], v[6])):
        raise SystemExit(f"fraction out of range in: {line!r} -> {v}")
    name = m.group("name").strip()
    rows.append({
        "model": name,
        "ours": "ours" in name.lower(),
        "single_camera": m.group("cam") == "✓",
        "group": group,
        # the paper reports NE in metres and the rest as fractions; the guide
        # shows percentages, so convert once here rather than in every figure
        "r2r": {"NE": v[0], "OS": round(v[1] * 100, 1),
                "SR": round(v[2] * 100, 1), "SPL": round(v[3] * 100, 1)},
        # RxR-CE has no oracle-success column in table 1
        "rxr": {"NE": v[4], "SR": round(v[5] * 100, 1), "SPL": round(v[6] * 100, 1)},
    })

if not rows:
    raise SystemExit("table 1 did not parse")
ours = [r for r in rows if r["ours"]]
if len(ours) != 1:
    raise SystemExit(f"expected exactly one 'ours' row, got {len(ours)}")
ours = ours[0]
# the results section calls it single-camera, so the table glyph must agree
assert ours["single_camera"], "our row must be marked single camera"
# Our row is printed at the foot of the table, below the depth block, so the
# section walker inherits 'depth' for it. It is a single-camera system; give it
# its own group so figures that colour by sensing class do not miscolour it.
ours["group"] = "ours"


def best(group_key, bench, metric, lower_is_better=False):
    pool = [r for r in rows if r["group"] == group_key and not r["ours"]]
    return (min if lower_is_better else max)(pool, key=lambda r: r[bench][metric])


# ------------------------------------------------------------- scalars
# The PDF wraps sentences mid-phrase, so every scalar pattern would otherwise
# need its own \s*\n?. Match against a whitespace-flattened copy instead and
# the whole class of line-break misses goes away.
flat = re.sub(r"\s+", " ", txt)


def grab(pattern, cast=float):
    m = re.search(pattern, flat)
    if not m:
        raise SystemExit(f"missing from paper: {pattern}")
    return cast(m.group(1).replace(",", ""))


facts = {
    "vlm_params_b": grab(r"a large (\d+)B parameter vision-language model", int),
    "policy_params_m": grab(r"approximately (\d+)M parameters", int),
    "trajectories_m": grab(r"approximately ([\d.]+) million trajectories", float),
    "scenes_k": grab(r"across (\d+)k scenes", int),
    "token_saving_x": grab(r"reduces the number of training tokens by (\d+)×", int),
    "hz_vlm": grab(r"predicts the next waypoint at ([\d.]+) hertz", float),
    "hz_policy": grab(r"dense trajectory at (\d+) hertz", int),
    "hz_motor": grab(r"into (\d+) Hz motor commands", int),
    "chunk_steps": grab(r"a sequence of (\d+) delta coordinates", int),
    "invisible_pct": grab(r"approximately (\d+)% of our training data features invisible", int),
    "hard_tasks_k": grab(r"curate ahard subsetof (\d+)k tasks", int),
    "reward_clip_m": grab(r"clipping the distance penalty at (\d+) meters", int),
    "success_radius_m": grab(r"stopped within (\d+) m of the goal", int),
    "height_min": grab(r"robot height between ([\d.]+) - [\d.]+m", float),
    "height_max": grab(r"robot height between [\d.]+ - ([\d.]+)m", float),
    "radius_min": grab(r"robot radius between ([\d.]+) - [\d.]+m", float),
    "radius_max": grab(r"robot radius between [\d.]+ - ([\d.]+)m", float),
    "cam_h_min": grab(r"camera placement height between (\d+) - \d+% ", int),
    "cam_h_max": grab(r"camera placement height between \d+ - (\d+)% ", int),
    "pitch_min": grab(r"camera pitch between (\d+) - \d+", int),
    "pitch_max": grab(r"camera pitch between \d+ - (\d+)", int),
}

# ---------------------------------------------------- the RL ablation prose
rl = {
    "r2r_sft_seen": grab(r"R2R-CE SFT baseline \(([\d.]+)% seen", float),
    "r2r_sft_unseen": grab(r"R2R-CE SFT baseline \([\d.]+% seen, ([\d.]+)% unseen", float),
    "r2r_rl_seen": grab(r"a success rate of ([\d.]+)% on seen environments", float),
    "r2r_rl_unseen": grab(r"and ([\d.]+)% on unseen environments \(\+", float),
    "r2r_peak_seen": grab(r"validation performance during the run reached ([\d.]+)% on seen", float),
    "r2r_peak_unseen": grab(r"on seen and ([\d.]+)% on unseen", float),
    "rxr_sft": grab(r"improves from ([\d.]+)% to a state-of-the-art success", float),
    "rxr_rl": grab(r"to a state-of-the-art success rate of ([\d.]+)%", float),
}
rl["r2r_gain_seen"] = round(rl["r2r_rl_seen"] - rl["r2r_sft_seen"], 2)
rl["r2r_gain_unseen"] = round(rl["r2r_rl_unseen"] - rl["r2r_sft_unseen"], 2)
rl["rxr_gain"] = round(rl["rxr_rl"] - rl["rxr_sft"], 1)

best_single = best("single", "r2r", "SR")
best_depth = best("depth", "r2r", "SR")

data = {
    "meta": {
        "arxiv": "2607.20785",
        "title": "Robostral Navigate",
        "webpage": "https://mistral.ai/news/robostral-navigate",
    },
    "table1": rows,
    "ours": ours,
    "facts": facts,
    "rl": rl,
    "headline": {
        "r2r_sr": ours["r2r"]["SR"], "r2r_spl": ours["r2r"]["SPL"],
        "r2r_ne": ours["r2r"]["NE"], "r2r_os": ours["r2r"]["OS"],
        "rxr_sr": ours["rxr"]["SR"], "rxr_spl": ours["rxr"]["SPL"],
        "rxr_ne": ours["rxr"]["NE"],
        "best_single_model": best_single["model"],
        "best_single_sr": best_single["r2r"]["SR"],
        "gain_vs_single": round(ours["r2r"]["SR"] - best_single["r2r"]["SR"], 1),
        "best_depth_model": best_depth["model"],
        "best_depth_sr": best_depth["r2r"]["SR"],
        "gain_vs_depth": round(ours["r2r"]["SR"] - best_depth["r2r"]["SR"], 1),
    },
    "metrics": {
        "NE": "Geodesic distance in metres between where the agent stopped and the goal. Lower is better.",
        "SR": "Share of episodes where the agent stopped within %d m of the goal." % facts["success_radius_m"],
        "OS": "Share of episodes that came within %d m of the goal at any point, whether or not the agent stopped there." % facts["success_radius_m"],
        "SPL": "Success weighted by path length. Scales success by shortest-path length over actual path length, so detours are penalised.",
    },
}

OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"wrote {OUT.name}")
print(f"  table 1 rows: {len(rows)}  "
      f"({sum(r['group']=='single' for r in rows)} single / "
      f"{sum(r['group']=='depth' for r in rows)} depth)")
print(f"  ours: R2R SR {ours['r2r']['SR']} SPL {ours['r2r']['SPL']} NE {ours['r2r']['NE']} "
      f"OS {ours['r2r']['OS']} | RxR SR {ours['rxr']['SR']} SPL {ours['rxr']['SPL']} NE {ours['rxr']['NE']}")
print(f"  vs best single ({best_single['model']} {best_single['r2r']['SR']}): "
      f"+{data['headline']['gain_vs_single']}")
print(f"  vs best depth  ({best_depth['model']} {best_depth['r2r']['SR']}): "
      f"+{data['headline']['gain_vs_depth']}")
print(f"  facts: {facts}")
print(f"  rl: {rl}")
