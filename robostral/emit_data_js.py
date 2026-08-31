#!/usr/bin/env python3
"""robostral_data.json -> data.js, exposing window.RN to the figures.

Every figure reads its numbers from here. A scene that hardcodes a score is a
defect you can grep for, which is the entire point of keeping this step.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
data = json.loads((HERE / "robostral_data.json").read_text(encoding="utf-8"))

# Table 1 model names carry citation noise and inconsistent casing. Canonicalise
# once, here, rather than in each figure.
SHORT = {
    "Robostral Navigate (ours)": "Robostral Navigate",
    "InternVLA-N1 (S1+S2)": "InternVLA-N1 S1+S2",
    "Qwen-VLA-Base": "Qwen-VLA-Base",
    "Qwen-VLA-Instruct": "Qwen-VLA-Instruct",
}
# Three systems appear in BOTH sensing blocks. Left alone, a sorted chart shows
# two rows called "Qwen-RobotNav-8B" separated only by bar colour.
names = {}
for r in data["table1"]:
    names.setdefault(r["model"], []).append(r)
for r in data["table1"]:
    short = SHORT.get(r["model"], r["model"])
    if len(names[r["model"]]) > 1 and not r["ours"]:
        short += " (depth)" if r["group"] == "depth" else " (RGB)"
    r["short"] = short

# rank on each benchmark so no figure has to hand-assert a placing
for bench in ("r2r", "rxr"):
    order = sorted(data["table1"], key=lambda x: -x[bench]["SR"])
    for i, r in enumerate(order):
        r.setdefault("rank", {})[bench] = i + 1
# data["ours"] is a separate copy after the json round trip, so read the rank
# back off the table row and keep both in step
ours_row = [r for r in data["table1"] if r["ours"]][0]
data["ours"]["rank"] = ours_row["rank"]
data["ours"]["short"] = ours_row["short"]
data["ours_rank"] = dict(ours_row["rank"])
print("  rank: R2R #%d, RxR #%d of %d" % (ours_row["rank"]["r2r"],
      ours_row["rank"]["rxr"], len(data["table1"])))

js = "window.RN = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n"
(HERE / "data.js").write_text(js, encoding="utf-8")
print(f"wrote data.js ({len(js):,} bytes, {len(data['table1'])} table rows)")
