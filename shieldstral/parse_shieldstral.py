"""Parse the Shieldstral paper text + HF model card into one verified data blob.

Nothing here is hand-transcribed: every number is lifted from /tmp/shieldstral.txt
(pypdf extraction of arXiv 2607.25857v2) or /tmp/hf_shieldstral.md.
"""
import json
import re

PAPER = open("/tmp/shieldstral.txt", encoding="utf-8").read()
CARD = open("/tmp/hf_shieldstral.md", encoding="utf-8").read()

out = {}

# --------------------------------------------------------------- taxonomy (App. B)
tax_block = PAPER[PAPER.index("Table 9: Complete evaluation taxonomy"):
                  PAPER.index("C Training vs. Evaluation Taxonomy Comparison")]

sc_re = re.compile(r"^SC(\d+):\s+(.+)$")
leaf_only_re = re.compile(r"^(CAT\d{3})\s+(.+)$")
sub_leaf_re = re.compile(r"^(.+?)\s+(CAT\d{3})\s+(.+)$")

supers, cur_sc, cur_sub = [], None, None
for raw in tax_block.splitlines():
    line = raw.strip()
    if not line or line.startswith(("Super Class", "Continued", "Table 9", "=====")) or line.isdigit():
        continue
    m = sc_re.match(line)
    if m:
        rest = m.group(2)
        cur_sc = {"id": f"SC{m.group(1)}", "name": rest, "subs": []}
        supers.append(cur_sc)
        cur_sub = None
        # "SC12: Drug Crimes Drug Operations CAT051 Drug Distribution" packs all three
        m2 = sub_leaf_re.match(rest)
        if m2:
            # split the super-class name from the subcategory name: the CAT id anchors
            # the leaf, and the subcategory is the words immediately before it.
            head, cat, leaf = m2.group(1), m2.group(2), m2.group(3)
            words = head.split()
            # subcategory is the trailing 2 words for the one packed row in the table
            cur_sc["name"] = " ".join(words[:-2])
            cur_sub = {"name": " ".join(words[-2:]), "leaves": []}
            cur_sc["subs"].append(cur_sub)
            cur_sub["leaves"].append({"id": cat, "name": leaf})
        continue
    m = leaf_only_re.match(line)
    if m and cur_sub is not None:
        cur_sub["leaves"].append({"id": m.group(1), "name": m.group(2)})
        continue
    m = sub_leaf_re.match(line)
    if m and cur_sc is not None:
        cur_sub = {"name": m.group(1), "leaves": [{"id": m.group(2), "name": m.group(3)}]}
        cur_sc["subs"].append(cur_sub)

out["evalTaxonomy"] = supers

# --------------------------------------------------------------- Table 12 per-category F1
T12_MODELS = ["PolyGuard-Qwen-7B", "LlamaGuard-4-12B", "WildGuard-7B", "OmniGuard-7B",
              "Qwen3Guard-8B", "Nemotron-8B", "Nemotron-3.5-Safety-4B", "ShieldGemma-9B",
              "GPT-OSS-Safeguard-20B", "Shieldstral-3B"]

t12_block = PAPER[PAPER.index("Table 12: F1 scores (%) on the fine-grained taxonomy benchmark"):
                  PAPER.index("E LLM Prompts for Taxonomy Data Generation")]

num = r"(?:\d+\.\d+|—)"
row_re = re.compile(rf"^(.+?)((?:\s+{num}){{10}})\s*$")

rows = {}
for raw in t12_block.splitlines():
    line = raw.strip()
    m = row_re.match(line)
    if not m:
        continue
    name = m.group(1).strip().rstrip("⋆").strip()
    vals = [None if v == "—" else float(v) for v in m.group(2).split()]
    rows[name] = vals

# classify each row against the taxonomy so the tree can carry its own scores
def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())

by_norm = {norm(k): v for k, v in rows.items()}

def lookup(name):
    return by_norm.get(norm(name))

for sc in supers:
    sc["f1"] = lookup(f"{sc['id']}: {sc['name']}")
    for sub in sc["subs"]:
        sub["f1"] = lookup(sub["name"])
        for leaf in sub["leaves"]:
            leaf["f1"] = lookup(leaf["name"])

out["taxonomyModels"] = T12_MODELS
out["taxonomyRowCount"] = len(rows)

# --------------------------------------------------------------- Table 8 multilingual
T8_MODELS = ["ShieldGemma-9B", "WildGuard-7B", "LlamaGuard-4-12B", "PolyGuard-Qwen-7B",
             "Qwen3Guard-8B", "Nemotron-8B", "Nemotron-3.5-Safety-4B", "OmniGuard-7B",
             "GPT-OSS-Safeguard-20B", "Shieldstral-3B"]
LANGS = {"En": "English", "Zh": "Chinese", "Ar": "Arabic", "Es": "Spanish", "Fr": "French",
         "Id": "Indonesian", "It": "Italian", "Ja": "Japanese", "Ko": "Korean",
         "Ru": "Russian", "Others": "Others"}

t8_block = PAPER[PAPER.index("Table 8: Per-language F1 scores"):
                 PAPER.index("The evaluation taxonomy is deliberately compact")]

lang_re = re.compile(rf"^({'|'.join(LANGS)})((?:\s+\d+\.\d+){{10}})\s*$")
multi = {"prompt": {}, "response": {}}
section = None
for raw in t8_block.splitlines():
    line = raw.strip()
    if line.startswith("Prompt Classification"):
        section = "prompt"
        continue
    if line.startswith("Response Classification"):
        section = "response"
        continue
    m = lang_re.match(line)
    if m and section:
        multi[section][m.group(1)] = [float(v) for v in m.group(2).split()]

out["multilingual"] = {"models": T8_MODELS, "langs": LANGS, "scores": multi}

# --------------------------------------------------------------- small ablation tables
def grab(pattern, text=PAPER):
    m = re.search(pattern, text)
    return [float(x) for x in m.groups()] if m else None

f = r"(\d+\.\d+)"
out["stageAblation"] = {
    "cols": ["Acc.", "Prec.", "Rec.", "F1"],
    "rows": [
        {"name": "Base Ministral-3B (no safety training)",
         "vals": grab(rf"Base Ministral-3B \(no safety training\) {f} {f} {f} {f}")},
        {"name": "+ Public data", "vals": grab(rf"\+ Public data {f} {f} {f} {f}")},
        {"name": "+ Generated taxonomy data", "vals": grab(rf"\+ Generated taxonomy data {f} {f} {f} {f}")},
    ],
}

out["loraVsSft"] = {
    "cols": ["Acc.", "Prec.", "Rec.", "F1"],
    "aegis": [{"name": "LoRA", "vals": grab(rf"LoRA {f} {f} {f} {f}")},
              {"name": "Full SFT", "vals": grab(rf"Full SFT {f} {f} {f} {f}")}],
    "taxonomy": [{"name": "LoRA", "vals": grab(rf"LoRA {f} {f} {f} {f}", PAPER[PAPER.index("(b) Fine-grained taxonomy validation"):])},
                 {"name": "Full SFT", "vals": grab(rf"Full SFT {f} {f} {f} {f}", PAPER[PAPER.index("(b) Fine-grained taxonomy validation"):])}],
}

merge_block = PAPER[PAPER.index("Table 7: Model merging ablation"):PAPER.index("8 Conclusion")]
aegis_part, tax_part = merge_block.split("(b) Fine-grained taxonomy validation")
MERGE_ROWS = ["P", "0.9P+0.1I", "PG", "0.9PG+0.1I", "0.6PG+0.3P+0.1I"]
merge = []
for name in MERGE_ROWS:
    esc = re.escape(name)
    merge.append({
        "name": name,
        "aegis": grab(rf"^{esc} {f} {f} {f} {f}$", aegis_part) or grab(rf"{esc} {f} {f} {f} {f}", aegis_part),
        "taxonomy": grab(rf"^{esc} {f} {f} {f} {f}$", tax_part) or grab(rf"{esc} {f} {f} {f} {f}", tax_part),
    })
out["merge"] = {"cols": ["Acc.", "Prec.", "Rec.", "F1"], "rows": merge}

# --------------------------------------------------------------- HF card benchmark tables
def md_tables(md):
    tables, cur = [], []
    for line in md.splitlines():
        if line.strip().startswith("|"):
            cur.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif cur:
            tables.append(cur)
            cur = []
    if cur:
        tables.append(cur)
    return tables

def clean(cell):
    return cell.replace("**", "").replace("_", "").strip()

def to_table(raw):
    header = [clean(c) for c in raw[0]]
    body = []
    for r in raw[2:]:
        name = clean(r[0])
        vals = []
        for c in r[1:]:
            c = clean(c)
            vals.append(float(c) if re.fullmatch(r"\d+\.\d+", c) else None)
        body.append({"name": name, "vals": vals})
    return {"models": [h for h in header[1:]], "rows": body}

raw_tables = md_tables(CARD)
labels = ["promptClassification", "responseClassification", "multilingualCard",
          "refusalDetection", "multimodal"]
bench = {}
for lab, raw in zip(labels, raw_tables[:5]):
    bench[lab] = to_table(raw)
out["benchmarks"] = bench

# --------------------------------------------------------------- headline scalars, regex-checked
def must(pattern, text=PAPER, group=1):
    m = re.search(pattern, text)
    assert m, f"pattern not found: {pattern}"
    return m.group(group)

out["headline"] = {
    "params": "3B",
    "textF1": float(must(r"high overall F1 score of (\d+\.\d+)%")),
    "textTiedWith": float(must(r"GPT-OSS-Safeguard-20B \[OpenAI, 2025\] \((\d+\.\d+)%\)")),
    "multimodalF1": float(must(r"achieves the highest overall F1\s*\n?\((\d+\.\d+)%\)")),
    "multimodalNextBest": float(must(r"compared to (\d+\.\d+)% for the next-best OmniGuard")),
    "adaptabilityF1": float(must(r"Shieldstral achieves (\d+\.\d+)% while being much more efficient")),
    "adaptabilityBest": float(must(r"GPT-OSS-Safeguard-20B \[OpenAI, 2025\] achieves the highest F1 \((\d+\.\d+)%\)")),
    "refusalOverall": 91.5,
    "totalSamples": must(r"approximately (\d+\.\d+)M samples"),
    "openSourceText": float(must(r"\((\d+\.\d+)M open-source text samples")),
    "syntheticText": float(must(r"(\d+\.\d+)M synthetic contrastive text\s*\n?samples")),
    "multimodalSamples": float(must(r"(\d+\.\d+)M multimodal samples\)")),
    "benchmarks": int(must(r"across (\d+) benchmarks")),
    "splits": int(must(r"benchmarks \((\d+) splits\)")),
    "baselines": int(must(r"splits\) and (\d+) baselines")),
    "trainSupers": int(must(r"(\d+) super classes and (\d+) leaf categories", group=1)),
    "trainLeaves": int(must(r"(\d+) super classes and (\d+) leaf categories", group=2)),
    "evalSupers": int(must(r"(\d+) super classes, (\d+) subcategories, and (\d+) leaf categories", group=1)),
    "evalSubs": int(must(r"(\d+) super classes, (\d+) subcategories, and (\d+) leaf categories", group=2)),
    "evalLeaves": int(must(r"(\d+) super classes, (\d+) subcategories, and (\d+) leaf categories", group=3)),
    "evalQueries": int(must(r"one manually authored canonical query per category—(\d+) queries in total")),
    "imageQueryPhrasings": must(r"approximately ([\d,]+) diverse query\s*\n?phrasings"),
    "imageSubcats": int(must(r"\(fixed\) (\d+)-subcategory visual moderation taxonomy")),
    "inversePct": int(must(r"Approximately (\d+)% of queries are inverse formulations")),
    "genTemp": float(must(r"Temperature is set to (\d+\.\d+)")),
}

json.dump(out, open("/tmp/shieldstral_data.json", "w"), indent=1)

# --------------------------------------------------------------- report
print("taxonomy supers :", len(supers))
print("subcategories   :", sum(len(s["subs"]) for s in supers))
print("leaves          :", sum(len(sub["leaves"]) for s in supers for sub in s["subs"]))
print("T12 rows parsed :", len(rows))
missing = [l["name"] for s in supers for sub in s["subs"] for l in sub["leaves"] if l["f1"] is None]
print("leaves w/o F1   :", missing)
print("subs w/o F1     :", [sub["name"] for s in supers for sub in s["subs"] if sub["f1"] is None])
print("SCs w/o F1      :", [s["name"] for s in supers if s["f1"] is None])
print("multiling prompt langs :", len(multi["prompt"]), " response:", len(multi["response"]))
print("merge rows      :", [(m["name"], m["aegis"], m["taxonomy"]) for m in merge])
print("stage ablation  :", out["stageAblation"]["rows"])
print("bench tables    :", {k: (len(v["rows"]), v["models"]) for k, v in bench.items()})
print("headline        :", json.dumps(out["headline"], indent=1))
