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
for r in data["table1"]:
    r["short"] = SHORT.get(r["model"], r["model"])

js = "window.RN = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n"
(HERE / "data.js").write_text(js, encoding="utf-8")
print(f"wrote data.js ({len(js):,} bytes, {len(data['table1'])} table rows)")
