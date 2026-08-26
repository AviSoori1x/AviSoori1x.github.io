/* Act IV, does it hold up. Teal, with red for every loss. */
window.SCENES = window.SCENES || {};

(function () {
  var K = window.KIT, C = K.C, T = K.TINT, G = C.teal;

  function sizeOf(SS, name) {
    var b = (SS.baselines || []).filter(function (x) {
      return name.toLowerCase().indexOf(String(x.model).toLowerCase().split(' ')[0]) === 0;
    })[0];
    return b ? b.size : '';
  }

  /* generic grouped bar block for a benchmark table */
  function benchBlock(s, tbl, y0, opts) {
    var models = tbl.models, rows = tbl.rows, OURS = 0;
    rows.forEach(function (r, ri) {
      var y = y0 + ri * 62;
      var best = -1, bv = -1;
      r.vals.forEach(function (v, i) { if (v != null && v > bv) { bv = v; best = i; } });
      K.mono(s, 0, y + 12, r.name, { size: 13, color: C.ink });
      var won = best === OURS;
      K.label(s, 640, y + 12, won ? 'best' : 'best ' + models[best].split('-')[0],
        { color: won ? G : C.red, anchor: 'end' });
      r.vals.forEach(function (v, i) {
        if (v == null) return;
        var bw = 560 * (v / 100);
        var by = y + 22 + i * 6;
        s.appendChild(K.n('rect', { x: 0, y: by, width: bw, height: 4.5, rx: 2,
          fill: i === OURS ? G : 'rgba(31,37,48,.22)' }));
        if (i === OURS) K.mono(s, bw + 8, by + 5, v.toFixed(1), { size: 11.5, color: G });
      });
    });
    return y0 + rows.length * 62;
  }

  /* 27 two taxonomies built to differ */
  window.SCENES.S_EVALTAX = function (root, api) {
    var s = K.board(root, { alt: 'Training and evaluation taxonomies differ by design.' });
    K.head(s, 'An evaluation built not to match', 'different names, different granularity, different groupings');
    var div = api.SS.divergence || [], h = api.SS.headline;
    K.label(s, 0, 14, 'two trees, built not to match');
    [['training', h.trainSupers + ' super classes, ' + h.trainLeaves + ' leaves', 'variable subcategory sizes', C.ink3],
     ['evaluation', h.evalSupers + ' / ' + h.evalSubs + ' / ' + h.evalLeaves, 'exactly two leaves per subcategory', G]
    ].forEach(function (t, i) {
      K.panel(s, i * 336, 26, 304, 84, { stroke: t[3] === G ? G : C.line, fill: t[3] === G ? T.teal : '#fff' });
      K.label(s, i * 336 + 16, 50, t[0], { color: t[3] });
      K.mono(s, i * 336 + 16, 74, t[1], { size: 14, color: C.ink });
      K.text(s, i * 336 + 16, 96, t[2], { size: 12.5, color: C.ink3 });
    });

    var body = K.n('g', {}); s.appendChild(body);
    K.label(s, 0, 142, 'how one harm domain gets carved up');
    K.switcher(s, 0, 154, div.map(function (d) { return d.domain; }), function (i) {
      body.innerHTML = '';
      var d = div[i];
      [['training side', d.train, C.ink3], ['evaluation side', d.eval, G]].forEach(function (p, j) {
        var y = 200 + j * 88;
        K.label(body, 0, y, p[0], { color: p[2] });
        K.panel(body, 0, y + 10, 640, 62, { stroke: p[2] === G ? G : C.line });
        K.para(body, 16, y + 34, p[1], 70, { size: 12.5, color: C.ink, lh: 17 });
      });
      K.text(body, 0, 396, d.why + '.', { size: 15, color: C.ink2 });
      s.appendChild(body);
    }, { tint: 'teal' });
    K.foot(s, 'Ten of the twelve evaluation super classes have a loose training counterpart, and no leaf maps one to one.');
  };

  /* 28 how one eval sample is made */
  window.SCENES.S_EVALGEN = function (root, api) {
    var s = K.board(root, { alt: 'Iso-query evaluation pairs, then cross verification.' });
    K.head(s, 'Both unsafe. One answer.', 'how a single evaluation sample gets made');
    var f4 = api.SS.fig4 || {}, h = api.SS.headline;
    K.label(s, 0, 14, 'one fixed question per category, ' + h.evalQueries + ' in total');
    K.panel(s, 0, 26, 640, 54, { fill: T.teal, stroke: G });
    K.para(s, 16, 50, f4.query || '', 68, { size: 13, color: C.ink, lh: 18 });

    K.label(s, 0, 110, 'two rewrites of one innocuous sentence');
    [[f4.positive, f4.category, 'yes', C.red], [f4.negative, f4.sibling, 'no', C.teal]].forEach(function (p, i) {
      var y = 126 + i * 108;
      K.panel(s, 0, y, 490, 92);
      K.label(s, 16, y + 22, p[1] || '', { color: C.ink3 });
      K.para(s, 16, y + 44, (p[0] || {}).document || '', 52, { size: 12, color: C.ink, lh: 16 });
      K.big(s, 520, y + 58, p[2], { size: 32, color: p[3] });
    });
    K.text(s, 0, 366, 'Both are unsafe. Only one matches the question, so detecting general unsafety gets you nowhere.',
      { size: 14, color: C.ink2 });
    K.arrow(s, 320, 380, 320, 404, { color: C.line });
    K.panel(s, 120, 410, 400, 44, { stroke: G });
    K.mono(s, 140, 438, 'a second llm verifies, mismatches discarded', { size: 12.5, color: G });
    K.foot(s, 'Training holds the document fixed and varies the query. Evaluation does the opposite.');
  };

  /* 29 text benchmarks */
  window.SCENES.S_TEXT = function (root, api) {
    var s = K.board(root, { alt: 'Text safety benchmarks against the baselines.' });
    K.head(s, 'Level with a model seven times its size', 'text safety benchmarks, and the rows it loses');
    var B = api.SS.benchmarks || {}, h = api.SS.headline;
    var tabs = [['prompt', B.promptClassification], ['response', B.responseClassification]];
    K.big(s, 0, 54, h.textF1 + '', { size: 58, color: G });
    K.text(s, 176, 46, 'mean F1 across the text benchmarks,', { size: 14.5, color: C.ink2 });
    K.text(s, 176, 66, 'level with GPT-OSS-Safeguard-20B at ' + h.textTiedWith, { size: 14.5, color: C.ink2 });
    var body = K.n('g', {}); s.appendChild(body);
    K.switcher(s, 0, 84, ['prompt classification', 'response classification'], function (i) {
      body.innerHTML = '';
      var t = tabs[i][1];
      K.label(body, 0, 136, 'shieldstral 3b in teal, five baselines behind it', { color: C.ink3 });
      benchBlock(body, t, 152);
      s.appendChild(body);
    }, { tint: 'teal' });
    K.foot(s, 'Rows where a baseline wins are labelled with the winner. Nothing is filtered out.');
  };

  /* 30 multimodal */
  window.SCENES.S_MM = function (root, api) {
    var s = K.board(root, { alt: 'Multimodal benchmarks, including the one it loses.' });
    K.head(s, 'The strongest result is on images', 'three multimodal benchmarks, one of them a loss');
    var h = api.SS.headline, t = (api.SS.benchmarks || {}).multimodal;
    K.big(s, 0, 54, h.multimodalF1 + '', { size: 58, color: G });
    K.text(s, 176, 46, 'mean F1 across the three image benchmarks,', { size: 14.5, color: C.ink2 });
    K.text(s, 176, 66, 'against ' + h.multimodalNextBest + ' for the next highest', { size: 14.5, color: C.ink2 });
    K.label(s, 0, 110, 'shieldstral 3b in teal');
    benchBlock(s, t, 126);
    K.text(s, 0, 330, 'It loses LlavaGuard to LlavaGuard-7B. Some of that benchmark\'s test images were',
      { size: 14, color: C.ink2 });
    K.text(s, 0, 352, 'unavailable, so every model there is scored on the available subset.',
      { size: 14, color: C.ink2 });
  };

  /* 31 adaptability: the headline, then the whole table underneath */
  window.SCENES.S_ADAPT = function (root, api) {
    var s = K.board(root, { alt: 'Per category F1 across the adaptability benchmark.' });
    K.head(s, 'Second, and the report says why',
           'every leaf category, every model, one grid');
    var h = api.SS.headline, SSd = api.SS;
    var models = SSd.taxonomyModels || [];
    var OURS = models.indexOf('Shieldstral-3B');
    var BEST = models.indexOf('GPT-OSS-Safeguard-20B');

    /* headline pair */
    [[models[BEST] || 'GPT-OSS-Safeguard-20B', h.adaptabilityBest, C.ink, '20B'],
     ['Shieldstral', h.adaptabilityF1, G, '']].forEach(function (r, i) {
      var y = 16 + i * 40;
      K.mono(s, 0, y + 12, r[0], { size: 13, color: r[2], weight: 700 });
      K.bar(s, 240, y + 2, 300, 14, r[1] / 100, { color: i ? G : 'rgba(31,37,48,.3)' });
      K.mono(s, 640, y + 14, r[1].toFixed(1), { size: 17, color: r[2], anchor: 'end' });
    });

    /* the whole table, 52 leaves against every model */
    var leaves = [];
    (SSd.evalTaxonomy || []).forEach(function (sc) {
      (sc.subs || []).forEach(function (sub) {
        (sub.leaves || []).forEach(function (lf) {
          if (lf.f1) leaves.push({ name: lf.name, f1: lf.f1, sc: sc.id });
        });
      });
    });

    var CW = 40, CH = 8.4, X0 = 172, Y0 = 132;
    K.label(s, 0, Y0 - 88, leaves.length + ' leaf categories, ' + models.length + ' models');
    // full names collide at 40px column pitch even when rotated, so tag them
    var SHORT = {
      'PolyGuard-Qwen-7B': 'PolyG', 'LlamaGuard-4-12B': 'Llama',
      'WildGuard-7B': 'Wild', 'OmniGuard-7B': 'Omni',
      'Qwen3Guard-8B': 'Qwen3', 'Nemotron-Safety-8B': 'Nemo',
      'Nemotron-3.5-Safety-4B': 'Nemo3.5', 'ShieldGemma-9B': 'Gemma',
      'GPT-OSS-Safeguard-20B': 'GPT-OSS', 'Shieldstral-3B': 'SHIELDSTRAL'
    };
    models.forEach(function (m, mi) {
      var cx = X0 + mi * CW + CW / 2 - 4;
      var lab = K.mono(s, cx, Y0 - 10, SHORT[m] || m.slice(0, 9),
        { size: 8.2, color: mi === OURS ? G : C.ink3, anchor: 'start',
          weight: mi === OURS ? 700 : 400 });
      lab.setAttribute('transform', 'rotate(-62 ' + cx + ' ' + (Y0 - 10) + ')');
    });

    leaves.forEach(function (lf, li) {
      var y = Y0 + li * CH;
      if (li % 2 === 0) {
        K.mono(s, X0 - 8, y + 6.8, lf.name.slice(0, 22), { size: 7.6, color: C.ink3, anchor: 'end' });
      }
      lf.f1.forEach(function (v, mi) {
        var frac = v == null ? null : Math.max(0, Math.min(1, (v - 30) / 68));
        s.appendChild(K.n('rect', {
          x: X0 + mi * CW, y: y, width: CW - 1.6, height: CH - 1.2, rx: 1,
          fill: v == null ? 'rgba(31,37,48,.05)'
              : (mi === OURS ? 'rgba(15,110,86,' + (0.13 + frac * 0.82) + ')'
                             : 'rgba(31,37,48,' + (0.06 + frac * 0.5) + ')')
        }));
      });
    });

    var gy = Y0 + leaves.length * CH + 22;
    var ours = 0, tot = 0;
    leaves.forEach(function (lf) {
      if (lf.f1[OURS] != null && lf.f1[BEST] != null) {
        tot++;
        if (lf.f1[OURS] > lf.f1[BEST]) ours++;
      }
    });
    K.callout(s, 0, gy, 640,
      'Shieldstral leads on ' + ours + ' of ' + tot + ' leaf categories against the 20B model, '
      + 'and trails on the rest. The column is darker where a model scores higher.',
      { color: G, tint: T.teal, cols: 74 });
    K.foot(s, 'Adaptability benchmark, per category. This is the test set, not the taxonomy validation set used for the ablations.');
  };

  /* 32 the language holes */
  window.SCENES.S_LANG = function (root, api) {
    var s = K.board(root, { alt: 'Shieldstral prompt and response scores by language.' });
    K.head(s, 'The holes are in the low-resource languages', 'prompt classification against response classification');
    var M = api.SS.multilingual || {}, models = M.models || [], sc = M.scores || {};
    var ours = models.indexOf('Shieldstral-3B');
    var codes = Object.keys(sc.prompt || {});
    var rows = codes.map(function (c) {
      return { c: c, name: (M.langs || {})[c] || c,
               p: sc.prompt[c][ours], r: sc.response[c][ours] };
    }).sort(function (a, b) { return a.p - b.p; });

    K.label(s, 0, 14, 'shieldstral by language, weakest first');
    rows.forEach(function (r, i) {
      var y = 34 + i * 34;
      var weak = r.p < 70;
      K.mono(s, 0, y + 12, r.name, { size: 12.5, color: weak ? C.red : C.ink2 });
      s.appendChild(K.n('rect', { x: 108, y: y + 2, width: 300 * r.p / 100, height: 7, rx: 3.5,
        fill: weak ? C.red : G }));
      s.appendChild(K.n('rect', { x: 108, y: y + 13, width: 300 * r.r / 100, height: 7, rx: 3.5,
        fill: 'rgba(31,37,48,.26)' }));
      K.mono(s, 424, y + 9, r.p.toFixed(1), { size: 12, color: weak ? C.red : C.ink });
      K.mono(s, 476, y + 9, r.r.toFixed(1), { size: 12, color: C.ink3 });
    });
    var ly = 34 + rows.length * 34 + 18;
    s.appendChild(K.n('rect', { x: 0, y: ly - 8, width: 22, height: 7, rx: 3.5, fill: G }));
    K.text(s, 30, ly, 'prompt classification', { size: 12.5, color: C.ink2 });
    s.appendChild(K.n('rect', { x: 200, y: ly - 8, width: 22, height: 7, rx: 3.5, fill: 'rgba(31,37,48,.26)' }));
    K.text(s, 230, ly, 'response classification', { size: 12.5, color: C.ink2 });
    K.text(s, 0, ly + 28, 'Indonesian is the widest split for Shieldstral, ' + rows[0].p.toFixed(1)
      + ' on prompts against ' + rows[0].r.toFixed(1) + ' on responses.', { size: 14, color: C.ink2 });
    K.foot(s, 'Red marks a prompt score under 70. Others is an aggregate bucket and this table does not name its languages.');
  };

  /* 33 running it */
  window.SCENES.S_RUN = function (root, api) {
    var s = K.board(root, { alt: 'The scoring helper and the serve command.' });
    K.head(s, 'Twelve lines to a verdict', 'the endpoint hands back logprobs, not a score');
    var sp = api.SS.systemPrompt || '';
    K.label(s, 0, 14, 'serve it');
    K.code(s, 0, 24, 640, [
      { t: '$ vllm serve mistralai/Shieldstral-1.0-3B --max-model-len 32768', c: '#9ecb8a' }
    ]);
    K.label(s, 0, 108, 'then read the two logprobs yourself');
    K.code(s, 0, 118, 640, [
      { t: 'SYSTEM = (', c: '#c8cfdb' },
      { t: '  "' + sp.slice(0, 52) + '"', c: '#d8a657' },
      { t: '  "' + sp.slice(52, 104) + '"', c: '#d8a657' },
      { t: '  "' + sp.slice(104) + '"', c: '#d8a657' },
      { t: ')', c: '#c8cfdb' },
      { t: '', c: '#c8cfdb' },
      { t: 'r = client.chat.completions.create(', c: '#c8cfdb' },
      { t: '    model=MODEL, messages=msgs,', c: '#c8cfdb' },
      { t: '    max_tokens=1, temperature=0.0,', c: '#7fb3c8' },
      { t: '    logprobs=True, top_logprobs=20)', c: '#7fb3c8' },
      { t: '', c: '#c8cfdb' },
      { t: 'top = r.choices[0].logprobs.content[0].top_logprobs', c: '#c8cfdb' },
      { t: 'z_yes = max(t.logprob for t in top if norm(t.token) == "yes")', c: '#c8cfdb' },
      { t: 'z_no  = max(t.logprob for t in top if norm(t.token) == "no")', c: '#c8cfdb' },
      { t: '', c: '#c8cfdb' },
      { t: 'score   = exp(z_yes) / (exp(z_yes) + exp(z_no))', c: '#e0a3a3' },
      { t: 'flagged = score > 0.5', c: '#e0a3a3' }
    ]);
    K.text(s, 0, 452, 'The endpoint hands back token logprobs, not a score. Renormalising over the two is the whole trick.',
      { size: 14, color: C.ink2 });
  };

  /* 34 limits */
  window.SCENES.S_LIMITS = function (root, api) {
    var s = K.board(root, { alt: 'What the model will not do for you.' });
    K.head(s, 'What it will not do for you', 'straight from the model card, unsoftened');
    var lim = (api.SS.limitations || []).slice();
    lim.push({ t: 'No rationale to inspect',
      d: 'A single token verdict arrives with nothing attached, so understanding a flag means going back to the content and the policy yourself.' });
    lim.push({ t: 'Context', d: 'Trained on sequences up to 32k tokens. The card recommends staying inside that range.' });
    K.label(s, 0, 14, 'what it will not do for you');
    var y = 34;
    lim.forEach(function (l) {
      K.panel(s, 0, y, 640, 4, { fill: C.red, stroke: 'none', r: 2 });
      K.mono(s, 0, y + 30, l.t, { size: 14, color: C.ink, weight: 700 });
      var p = K.para(s, 0, y + 52, l.d, 84, { size: 12.5, color: C.ink3, lh: 17 });
      y += 52 + p.h + 18;
    });
    var names = api.SS.coreContributors || [];
    K.label(s, 0, y + 10, 'core contributors');
    var line = names.join(', ');
    K.para(s, 0, y + 30, line, 88, { size: 11.5, color: C.ink3, lh: 16 });
  };
})();
