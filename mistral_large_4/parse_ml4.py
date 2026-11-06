#!/usr/bin/env python3
"""Parse the Mistral Large 4 report into ml4_data.json.

Reads the text layer of the PDF (ml4_work/ref/ml4_v3.txt, made by fitz) and
pulls out:

  tables  the six results tables, every cell, "not reported" kept as null
  facts   every number quoted in the prose of this page, each one located in
          the paper text by a pattern; a pattern that does not match stops the
          build, so a mistyped number cannot reach the page

Values read off a figure (not printed anywhere as text) live in figread.json
instead, each with the figure it came from, and the page says so wherever one
is drawn.
"""
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent.parent / "ml4_work" / "ref" / "ml4_v3.txt"
raw = SRC.read_text(encoding="utf-8")
lines = [l.strip() for l in raw.splitlines()]

COLS = ["GLM-5.2", "GLM-5.3", "DeepSeek V4-Pro 0813", "DeepSeek V4.1-Flash 1017",
        "Kimi K3", "Muse Spark Open", "Mistral Large 4"]
VAL = re.compile(r"^(—|\d+(?:\.\d+)?%?)$")


def clean_name(s):
    s = re.sub(r"\s*\[[\d,\s–-]+\]\s*$", "", s)          # trailing citation
    s = re.sub(r"(\d)–(\d)", r"\1 to \2", s)               # 03/26-05/26 style ranges
    s = s.replace("–", " to ").replace("—", "")
    return s.strip()


def parse_tables():
    out = []
    starts = [i for i, l in enumerate(lines) if re.match(r"^Table [1-6]: ", l)]
    for k, i0 in enumerate(starts):
        title = lines[i0]
        cap = [title]
        j = i0 + 1
        while lines[j] != "Large 4":
            cap.append(lines[j])
            j += 1
        caption = " ".join(cap[:cap.index("Eval")]) if "Eval" in cap else title
        j += 1
        end = starts[k + 1] if k + 1 < len(starts) else lines.index("Production inference", j)
        rows, name, vals, raw_vals = [], None, [], []
        for l in lines[j:end]:
            if not l or l.startswith("====="):
                continue
            toks = l.split()
            if all(VAL.match(t) for t in toks):
                if name is None:          # a page number between two tables
                    continue
                for t in toks:
                    raw_vals.append(t)
                    vals.append(None if t == "—" else float(t.rstrip("%")))
                if len(vals) == len(COLS):
                    rows.append({"name": clean_name(name), "cite_name": name,
                                 "vals": vals, "raw": raw_vals})
                    name, vals, raw_vals = None, [], []
                elif len(vals) > len(COLS):
                    sys.exit(f"row overflow in {title}: {name}")
            else:
                if vals:
                    sys.exit(f"short row in {title}: {name} {raw_vals}")
                name = l if name is None else name + " " + l
        if name is not None:
            sys.exit(f"dangling row in {title}: {name}")
        num = int(re.match(r"^Table (\d)", title).group(1))
        cap_txt = re.sub(r"^Table \d: ", "", caption)
        out.append({"n": num, "title": cap_txt.split(". ")[0].rstrip("."),
                    "caption": re.sub(r"^Table \d: ", "", caption), "rows": rows})
    return out


tables = parse_tables()
for t in tables:
    print(f"table {t['n']}: {len(t['rows'])} rows  {t['title']}")

# ---------------------------------------------------------------- prose facts
body = "\n".join(l for l in lines if not l.startswith("====="))
norm = re.sub(r"\s+", " ", body)
facts = {}


def F(key, pattern, *names, cast=float):
    """Find pattern in the paper text; store each group under key.name."""
    m = re.search(pattern, norm)
    if not m:
        sys.exit(f"FACT NOT FOUND {key}: {pattern}")
    ctx = norm[max(0, m.start() - 40):m.end() + 40]
    for n, g in zip(names, m.groups()):
        v = g.replace(",", "")
        try:
            v = cast(v)
        except ValueError:
            v = g
        facts[f"{key}.{n}" if n else key] = v
    facts.setdefault("_ctx", {})[key] = ctx


F("size", r"a (\d+)B active, (\d+)T total parameter multimodal", "active_b", "total_t")
F("pre", r"pretrained on ([\d,]+) NVIDIA GB200 GPUs", "gpus")
F("pre", r"Pretraining took (\d+) days, or about ([\d.]+) million GPU-hours", "days", "gpu_hours_m")
F("pre", r"\(MFU\) of (\d+)% and a goodput of ([\d.]+)%", "mfu", "goodput")
F("mix", r"first phase \((\d+)%\).*?the second \((\d+)%\)", "p1_share", "p2_share")
F("mix", r"sequence length from (\d+)K to (\d+)K", "seq_from_k", "seq_to_k")
F("mix", r"Web ([\d.]+)% Docs ([\d.]+)% Code ([\d.]+)% Misc ([\d.]+)% "
         r"Web ([\d.]+)% Docs ([\d.]+)% Code ([\d.]+)% Misc ([\d.]+)%",
  "p1_web", "p1_docs", "p1_code", "p1_misc", "p2_web", "p2_docs", "p2_code", "p2_misc")
F("img", r"sustaining (\d+)% of text-only MFU", "mfu_kept")
F("gran", r"granularity of (\d+) for its mixture-of-experts", "g")
F("gran", r"from 4 to (\d+) reduces the FLOPs needed to reach a matched perplexity by about "
          r"([\d.]+)%, with a reported uncertainty of ([\d.]+) percentage points\. "
          r"The measured training slowdown was approximately (\d+)%",
  "g_alt", "flops_saved", "flops_unc", "slowdown")
F("adam", r"\(β1, β2\) = \(([\d.]+), ([\d.]+)\)", "b1", "b2")
F("batch", r"runs for the last (\d+)% of tokens, the penultimate batch size runs for (\d+)%",
  "last_share", "penult_share")
F("batch", r"ramp-up from ([\d.]+)× to (\d+)× the predicted batch size", "opt_from", "opt_to")
F("batch", r"ramp up from (\d+)× to (\d+)× based on wall-clock", "chosen_from", "chosen_to")
F("batch", r"produces the (\d+)M, (\d+)M, (\d+)M schedule", "b1_m", "b2_m", "b3_m")
F("vis", r"We pretrain a ([\d.]+)B-parameter vision encoder", "params_b")
F("vis", r"horizon of (\d+)T tokens", "tokens_t")
F("vis", r"maximum resolution of (\d+)×(\d+) pixels", "res_w", "res_h")
F("vis", r"roughly (\d+)B tokens\. The resulting", "sft_b")
F("vis", r"Perception Encoder ViT-L/14 \((\d+)M, frozen\)", "pe_m")
F("kv", r"window size of (\d+) tokens", "swa_window")
F("kv", r"One in every (\w+) layers uses full attention", "long_every", cast=str)
F("kv", r"two-stage prefilling, which is ([\d.]+)× faster at (\d+)M context", "yoco_prefill", "yoco_ctx_m")
F("kv", r"KV-cache size by a factor of (\d+)", "reduction")
F("surg", r"At the 8 × 10(22) FLOPs scale, we performed", "small_exp")
F("surg", r"conservative test at the 8 × 10(23) FLOPs scale", "big_exp")
F("surg", r"rescale the RoPE frequencies by (\d+)×", "yarn")
F("surg", r"increased by approximately ([\d.]+)× and ([\d.]+)× relative to YaRN, which shortened "
          r"their wavelengths by (\d+)% and (\d+)%", "f1", "f2", "wl1", "wl2")
F("surg", r"trained for approximately (two-thirds) of the original phase 2", "phase2_part", cast=str)
F("mass", r"using (\d+)k-token probes across (\w+) data sources", "probe_k", "sources", cast=str)
F("mass", r"around (\d+)% of the attention mass to far-away \(>(\d+) tokens\) content, "
          r"compared to (\d+)% for native long layers", "converted", "far", "native")
F("dsa", r"select the top-([\d,]+) key tokens", "topk")
F("dsa", r"approximately ([\d.]+)B tokens through KL-based distillation", "stage1_b")
F("dsa", r"an additional (\d+)B", "stage2_b")
F("dsa", r"groups starting at L(\d+)", "share_from")
F("cve", r"tens of thousands of CVEs across roughly (\d+) weakness classes", "classes")
F("mid", r"using 1/(\d+) of the peak pretraining LR and 1/(\d+) of pretraining rehearsal",
  "lr_div", "rehearsal_div")
F("sftlad", r"Optimal SFT learning rate \((\d+)M-token batch\)", "batch_m")
F("rlinf", r"ran (\d+)k concurrent rollouts using about (\d+)k sandboxes and ([\d,]+) GPUs "
           r"with a 1:([\d.]+) trainer-to-generator ratio", "rollouts_k", "sandboxes_k", "gpus", "gen_ratio")
F("rlinf", r"about (\d+)k GPUs were used simultaneously", "all_gpus_k")
F("rlinf", r"lasted about (\d+) hours and spanned (\d+) model updates", "longest_h", "updates")
F("esc", r"an agent chained (\w+) different vulnerabilities", "vulns", cast=str)
F("esc", r"deleted about (\d+)k sandboxes by mistake", "deleted_k")
F("rl", r"\((\d+)k for ML4\)", "step_traj_k")
F("rl", r"(\d+) by default, (\d+) for agentic coding, and between (\d+) and (\d+) overall",
  "group_default", "group_code", "group_min", "group_max")
F("rl", r"generated ([\d.]+)M rollouts across (\d+)k groups and (\d+)k instances, for a total "
        r"of ([\d.]+)B output tokens", "rollouts_m", "groups_k", "instances_k", "out_tokens_b")
F("rl", r"Code agent tasks accounted for (\d+)% of rollouts but ([\d.]+)% of output tokens",
  "code_roll", "code_tok")
F("rl", r"with ([\d.]+)% of rollouts and ([\d.]+)% of output tokens", "reason_roll", "reason_tok")
F("rl", r"min\(πθ\(yt\)/πgen\(yt\), (\d+)\)", "clip")
F("muon", r"Nesterov momentum ([\d.]+), quintic Newton.Schulz with (\d+) iterations", "mom", "ns_iters")
F("muon", r"constant learning rate of (\d+) × 10−(\d+) after a (\d+)-step warmup", "lr_m", "lr_e", "warmup")
F("muon", r"gradient-norm clipping at (\d+)", "clipnorm")
F("muon", r"k2 estimator \[40\] with coefficient (\d+\.\d+)", "kl")
F("mis", r"≤(\d+) × 10−(\d+) throughout training, with an average staleness of (\w+) steps",
  "k2_m", "k2_e", "stale", cast=str)
F("hack", r"enumerated (\w+) discrepancies", "discrepancies", cast=str)
F("hack", r"acknowledging “(I have the full reference implementa- ?tion from the board binaries\.)”",
  "quote", cast=str)
F("len", r"bins of width ∆= ([\d.]+)Lmin", "bin")
F("cp", r"approximately (\d+)M tokens of in-memory KV-cache space on (\d+)×GB300", "kv_m", "gpus")
F("cp", r"improve decode interactivity by (\d+)%", "dcp_gain")
F("cp", r"no more than (\d+)–(\d+)% communication overhead", "ovh_lo", "ovh_hi")
F("spec", r"by \+([\d.]+)% and \+([\d.]+)%, respectively", "head", "mask")
F("spec", r"We train the DSpark module on (\d+)B tokens", "train_b")
F("spec", r"DSpark: (\d+)% increase in decode interactivity at (\d+) concurrent", "gain", "conc")
F("spec", r"by \+(\d+)% compared to an EAGLE-3", "vs_eagle")
F("fig16", r"measured at (\d+)K input length", "input_k")
F("fig16", r"off-the-shelf CP implementation on (\d+)×GB300", "base_gpus")
F("q", r"precision checkpoint at (\d+) GB", "gb")
F("q", r"We use (\d+)M tokens from our SFT mixture for calibration", "calib_m")
F("ops", r"start time of new model replicas to below (\d+) seconds", "start_s")
F("f17", r"(\d+) μs (\d+) μs (\d+) μs Q/KV proj", "b_topk", "b_qg", "b_oc")
F("f17", r"Q gather (\d+) μs output combine (\d+) μs top-k exchange (\d+) μs (\d+) μs (\d+) μs −(\d+)%",
  "a_qg", "a_oc", "a_topk", "before", "after", "cut")
F("intro", r"(\d+\.\d+)% on FrontierCode 2\.0", "frontiercode")
F("intro", r"(\d+)% on Harvey’s Legal Agent Benchmark", "hlab")
F("intro", r"(\d+)% on Terminal- ?Bench 4\.0", "tb4")
F("intro", r"(\d+\.\d+)% on the Artificial Analysis Cyber Index", "aacyber")
F("intro", r"achieves a (\d+\.\d+)% unweighted average", "safety")

# ---------------------------------------------------------------- cross-checks
T = {t["n"]: t for t in tables}


def row(n, prefix):
    hits = [r for r in T[n]["rows"] if r["name"].startswith(prefix)]
    assert len(hits) == 1, (n, prefix, [r["name"] for r in T[n]["rows"]])
    return hits[0]


ml4 = COLS.index("Mistral Large 4")
assert [len(T[n]["rows"]) for n in range(1, 7)] == [4, 7, 5, 5, 5, 3]
assert row(3, "FrontierCode")["vals"][ml4] == facts["intro.frontiercode"]
assert row(3, "Terminal-Bench 4.0")["vals"][ml4] == facts["intro.tb4"]
assert row(4, "AA Cyber Index")["vals"][ml4] == facts["intro.aacyber"]
assert int(row(6, "HLAB")["vals"][ml4]) == facts["intro.hlab"]  # intro truncates 15.83 to 15
trio = [row(5, p)["vals"][ml4] for p in ("JailbreakBench", "StrongREJECT", "AgentHarm")]
assert round(sum(trio) / 3, 1) == facts["intro.safety"], trio
assert facts["f17.before"] == 240 and facts["f17.after"] == 153
assert round(100 * (1 - 153 / 240)) == facts["f17.cut"]
pies = [facts[f"mix.p{p}_{k}"] for p in (1, 2) for k in ("web", "docs", "code", "misc")]
assert abs(sum(pies[:4]) - 100) < 0.05 and abs(sum(pies[4:]) - 100) < 0.05

facts.pop("_ctx")
facts["hack.quote"] = facts["hack.quote"].replace("implementa- tion", "implementation")
out = {"source": "Mistral Large 4 technical report, v3 (alphaxiv 2611.mistral-large-4)",
       "cols": COLS, "ml4": ml4, "tables": tables, "facts": facts}
(HERE / "ml4_data.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"facts: {len(facts)}  cross-checks passed  -> ml4_data.json")
