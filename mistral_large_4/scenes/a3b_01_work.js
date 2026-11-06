/* S_WORK: §4.1.4, how a knowledge-work environment is built. The paper has no
   figure here, so this is a schematic of the text, in the paper's order: a seed
   sampled from the knowledge-work taxonomy, (1) research, (2) authoring and
   (3) review build the file system, agents then fill the workplace
   applications, and long-horizon tasks are generated to run inside the result.
   Step with back and next, or click any stage. The paper gives no counts here,
   so nothing is counted. */
window.SCENES.S_WORK = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, n = K.n;
  var svg = K.board(host, { alt: 'Schematic of section 4.1.4: a seed sampled from a ' +
    'knowledge-work taxonomy, a research agent, an authoring agent that builds a file ' +
    'system, a multi-agent review, workplace applications filled with records, and ' +
    'long-horizon tasks grounded in a behavior taxonomy' });
  K.head(svg, 'Building a simulated workplace', '§4.1.4 · schematic · one environment, built step by step');
  var body = CH.body(svg);
  var cur = 0, LAST = 5;

  // a stage is done, the one in focus, or still to come
  function st(i) { return i < cur ? 'done' : i === cur ? 'on' : 'todo'; }
  function ink(i) { return st(i) === 'on' ? C.amber : C.ink3; }

  /* ---- small line icons, drawn rather than written, so they cost no text ---- */
  function P(g, d, col, o) {
    o = o || {};
    g.appendChild(n('path', { d: d, fill: o.fill || 'none', stroke: col,
      'stroke-width': o.w || 1.5, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
  }
  function R(g, x, y, w, h, col, o) {
    o = o || {};
    g.appendChild(n('rect', { x: x, y: y, width: w, height: h, rx: o.rx == null ? 2.5 : o.rx,
      fill: o.fill || '#fff', stroke: col, 'stroke-width': o.w || 1.5 }));
  }
  var ICON = {
    sheet: function (g, x, y, c) {
      R(g, x - 13, y - 14, 26, 28, c);
      R(g, x - 13, y - 14, 26, 7, c, { fill: c === C.amber ? K.TINT.amber : K.TINT.ink, rx: 2.5 });
      P(g, 'M' + (x - 13) + ' ' + y + ' H' + (x + 13) + ' M' + (x - 13) + ' ' + (y + 7) + ' H' + (x + 13) +
        ' M' + (x - 4) + ' ' + (y - 7) + ' V' + (y + 14) + ' M' + (x + 5) + ' ' + (y - 7) + ' V' + (y + 14), c, { w: 1.1 });
    },
    report: function (g, x, y, c) {
      P(g, 'M' + (x - 11) + ' ' + (y - 14) + ' H' + (x + 4) + ' L' + (x + 11) + ' ' + (y - 7) + ' V' + (y + 14) +
        ' H' + (x - 11) + ' Z', c, { fill: '#fff' });
      P(g, 'M' + (x + 4) + ' ' + (y - 14) + ' V' + (y - 7) + ' H' + (x + 11), c, { w: 1.1 });
      P(g, 'M' + (x - 6) + ' ' + (y - 2) + ' H' + (x + 6) + ' M' + (x - 6) + ' ' + (y + 3) + ' H' + (x + 6) +
        ' M' + (x - 6) + ' ' + (y + 8) + ' H' + (x + 2), c, { w: 1.1 });
    },
    slides: function (g, x, y, c) {
      R(g, x - 17, y - 13, 34, 21, c);
      P(g, 'M' + (x - 8) + ' ' + (y + 4) + ' V' + (y - 1) + ' M' + x + ' ' + (y + 4) + ' V' + (y - 8) +
        ' M' + (x + 8) + ' ' + (y + 4) + ' V' + (y - 4), c, { w: 3 });
      P(g, 'M' + x + ' ' + (y + 8) + ' V' + (y + 14) + ' M' + (x - 6) + ' ' + (y + 14) + ' H' + (x + 6), c);
    },
    mail: function (g, x, y, c) {
      R(g, x - 16, y - 11, 32, 22, c);
      P(g, 'M' + (x - 16) + ' ' + (y - 11) + ' L' + x + ' ' + (y + 2) + ' L' + (x + 16) + ' ' + (y - 11), c, { w: 1.2 });
    },
    chat: function (g, x, y, c) {
      R(g, x - 16, y - 13, 21, 14, c, { rx: 5 });
      P(g, 'M' + (x - 11) + ' ' + (y + 1) + ' L' + (x - 13) + ' ' + (y + 5) + ' L' + (x - 7) + ' ' + (y + 1), c, { w: 1.2 });
      R(g, x - 3, y - 2, 20, 14, c, { rx: 5, fill: c === C.amber ? K.TINT.amber : K.TINT.ink });
      P(g, 'M' + (x + 11) + ' ' + (y + 12) + ' L' + (x + 13) + ' ' + (y + 16) + ' L' + (x + 7) + ' ' + (y + 12), c, { w: 1.2 });
    },
    cal: function (g, x, y, c) {
      R(g, x - 14, y - 11, 28, 25, c);
      P(g, 'M' + (x - 14) + ' ' + (y - 4) + ' H' + (x + 14) + ' M' + (x - 7) + ' ' + (y - 15) + ' V' + (y - 8) +
        ' M' + (x + 7) + ' ' + (y - 15) + ' V' + (y - 8), c, { w: 1.3 });
      [[-8, 1], [-1, 1], [6, 1], [-8, 7], [-1, 7], [6, 7]].forEach(function (q, k) {
        g.appendChild(n('rect', { x: x + q[0], y: y + q[1], width: 4, height: 4, rx: 1,
          fill: k === 4 ? c : C.line }));
      });
    },
    board: function (g, x, y, c) {
      R(g, x - 16, y - 13, 32, 26, c);
      P(g, 'M' + (x - 5) + ' ' + (y - 13) + ' V' + (y + 13) + ' M' + (x + 6) + ' ' + (y - 13) + ' V' + (y + 13), c, { w: 1.1 });
      [[-14, -9], [-14, -2], [-3, -9], [8, -9], [8, -2], [8, 5]].forEach(function (q) {
        g.appendChild(n('rect', { x: x + q[0], y: y + q[1], width: 7, height: 4.5, rx: 1, fill: c, opacity: 0.55 }));
      });
    },
    person: function (g, x, y, c) {
      g.appendChild(n('circle', { cx: x - 8, cy: y - 6, r: 5, fill: '#fff', stroke: c, 'stroke-width': 1.5 }));
      P(g, 'M' + (x - 17) + ' ' + (y + 12) + ' Q' + (x - 8) + ' ' + (y - 4) + ' ' + (x + 1) + ' ' + (y + 12), c);
      P(g, 'M' + (x + 6) + ' ' + (y - 7) + ' H' + (x + 16) + ' M' + (x + 6) + ' ' + y + ' H' + (x + 16) +
        ' M' + (x + 6) + ' ' + (y + 7) + ' H' + (x + 13), c, { w: 1.2 });
    }
  };

  /* ---- what each step says, in the paper's words where it has them ---- */
  var NAME = ['the seed', 'research', 'authoring', 'review', 'applications', 'tasks'];
  var SAY = [
    'Each environment starts from a seed: a workflow description with example tasks, ' +
      'sampled from a broad taxonomy of enterprise and industrial knowledge work.',
    'For each workflow, the pipeline writes an open-ended research query, and a research ' +
      'agent gathers relevant public documents.',
    'An authoring agent turns that material into a coherent workspace, keeping or adapting ' +
      'useful sources and creating complementary files.',
    'A multi-agent setup inspects the workspace for realism, consistency and file ' +
      'integrity, then gives integrated feedback.',
    'Agents read the files and each application\'s database schema, then create records ' +
      'that match the workspace\'s people, projects and chronology.',
    'Long-horizon agentic tasks are generated to run inside each environment. Conditioned ' +
      'on the taxonomies, samples came out higher quality, more realistic and more diverse.'
  ];
  var FILES = [['sheet', 'spreadsheets'], ['report', 'reports'], ['slides', 'presentations'],
    ['mail', 'correspondence']];
  var APPS = [['chat', 'messages'], ['cal', 'calendar events'], ['board', 'project updates'],
    ['person', 'customer histories']];
  var TX = [98, 246, 394, 542];   // four tile centres across the frame

  // a card or zone: amber when in focus, a white panel once built, dashed until then
  function zone(g, i, x, y, w, h, o) {
    o = o || {};
    var s = st(i);
    if (s === 'done' && !o.flat) { K.panel(g, x, y, w, h, { r: 9 }); return; }
    g.appendChild(n('rect', { x: x, y: y, width: w, height: h, rx: 9,
      fill: s === 'on' ? K.TINT.amber : s === 'done' ? '#fff' : 'none',
      stroke: s === 'on' ? C.amber : s === 'done' ? C.line : CH.GREY2,
      'stroke-width': s === 'on' ? 1.8 : 1.2, 'stroke-dasharray': s === 'todo' ? '5 4' : null }));
  }
  function lab(g, i, x, y, s, o) {
    o = o || {};
    return K.label(g, x, y, s, { color: st(i) === 'on' ? C.ink : C.ink3, anchor: o.anchor });
  }
  function line(g, i, x, y, s) {
    return K.mono(g, x, y, s, { size: 12.5, color: st(i) === 'on' ? C.ink : C.ink2 });
  }
  // the connector into stage i takes that stage's state
  function into(g, i, x1, y1, x2, y2) {
    var s = st(i);
    K.arrow(g, x1, y1, x2, y2, { color: s === 'on' ? C.amber : s === 'done' ? C.ink3 : CH.GREY2,
      w: s === 'on' ? 2.2 : 1.4, headSize: 7, dash: s === 'todo' ? '4 4' : null });
  }
  // a row of four tiles: icons and names once built, empty dashed slots before
  function tiles(g, i, list, yc) {
    list.forEach(function (t, k) {
      if (st(i) === 'todo') {
        g.appendChild(n('rect', { x: TX[k] - 16, y: yc - 16, width: 32, height: 32, rx: 5,
          fill: 'none', stroke: CH.GREY2, 'stroke-dasharray': '3 3' }));
        return;
      }
      ICON[t[0]](g, TX[k], yc, ink(i));
      K.mono(g, TX[k], yc + 31, t[1], { size: 12.5, anchor: 'middle',
        color: st(i) === 'on' ? C.ink : C.ink2 });
    });
  }

  function draw() {
    CH.clear(body);
    var g = body, on;

    // progress: six dots beside the buttons, no text
    for (var k = 0; k <= LAST; k++) {
      if (k) g.appendChild(n('path', { d: 'M' + (232 + (k - 1) * 22) + ' 15 H' + (232 + k * 22),
        stroke: k <= cur ? C.ink3 : CH.GREY2, 'stroke-width': 1.3 }));
    }
    for (k = 0; k <= LAST; k++) {
      g.appendChild(n('circle', { cx: 232 + k * 22, cy: 15, r: k === cur ? 6.5 : 4.5,
        fill: k === cur ? C.amber : k < cur ? C.ink3 : '#fff',
        stroke: k === cur ? '#fff' : k < cur ? C.ink3 : CH.GREY2, 'stroke-width': k === cur ? 2 : 1.3 }));
    }

    // 0: the seed, sampled from the knowledge-work taxonomy
    zone(g, 0, 0, 46, 320, 72);
    lab(g, 0, 14, 67, 'seeds: knowledge-work taxonomy');
    line(g, 0, 14, 88, 'finance · law · engineering ·');
    line(g, 0, 14, 106, 'data science · business management');

    // 1: research query and research agent
    into(g, 1, 322, 82, 350, 82);
    zone(g, 1, 352, 46, 288, 72);
    lab(g, 1, 366, 67, 'research agent');
    var mc = st(1) === 'todo' ? CH.GREY2 : ink(1);
    g.appendChild(n('circle', { cx: 612, cy: 62, r: 6.5, fill: 'none', stroke: mc, 'stroke-width': 1.6 }));
    P(g, 'M617 67 L623 73', mc, { w: 2.2 });
    if (st(1) !== 'todo') {
      line(g, 1, 366, 88, 'takes an open-ended query and');
      line(g, 1, 366, 106, 'gathers relevant public documents');
    }

    // the workplace frame, drawn as a fieldset so its name sits in the border
    // everything from the workplace down sits 10 units lower, under the taller cards
    g = n("g", { transform: "translate(0,10)" });
    body.appendChild(g);
    var built = cur >= 2;
    g.appendChild(n('path', { d: 'M244 128 H630 Q640 128 640 138 V340 Q640 350 630 350 H10 ' +
      'Q0 350 0 340 V138 Q0 128 10 128 H14', fill: 'none', stroke: built ? C.ink3 : CH.GREY2,
      'stroke-width': built ? 1.3 : 1.2, 'stroke-dasharray': built ? null : '5 4' }));
    K.label(g, 22, 132.5, 'one simulated workplace', { color: built ? C.ink : C.ink3 });

    // 2: the authoring agent builds the file system from the research
    into(g, 2, 496, 110, 496, 140);
    zone(g, 2, 10, 142, 620, 80, { flat: true });
    lab(g, 2, 24, 161, 'file system, by the authoring agent');
    tiles(g, 2, FILES, 183);

    // 3: several agents inspect the workspace and give integrated feedback on it
    zone(g, 3, 10, 226, 620, 30, { flat: true });
    on = st(3) === 'todo' ? CH.GREY2 : ink(3);
    [30, 44, 58].forEach(function (x) {
      g.appendChild(n('circle', { cx: x, cy: 236.5, r: 3.4, fill: '#fff', stroke: on, 'stroke-width': 1.4 }));
      P(g, 'M' + (x - 5.5) + ' 248 Q' + x + ' 238.5 ' + (x + 5.5) + ' 248', on, { w: 1.4 });
    });
    if (st(3) !== 'todo') {
      [[86, 'realism'], [178, 'consistency'], [298, 'file integrity']].forEach(function (q) {
        P(g, 'M' + q[0] + ' 241 l3.5 4 l7 -8.5', on, { w: 2 });
        K.mono(g, q[0] + 16, 245.5, q[1], { size: 12.5, color: st(3) === 'on' ? C.ink : C.ink2 });
      });
      K.mono(g, 598, 245.5, 'integrated feedback', { size: 12.5, anchor: 'end',
        color: st(3) === 'on' ? C.ink : C.ink2 });
      P(g, 'M602 241 H610 Q616 241 616 235 V204', on, { w: st(3) === 'on' ? 2.2 : 1.4 });
      K.arrow(g, 616, 206, 616, 170, { color: on, w: st(3) === 'on' ? 2.2 : 1.4, headSize: 7 });
    }

    // 4: agents fill the workplace applications with matching records
    zone(g, 4, 10, 260, 620, 80, { flat: true });
    lab(g, 4, 24, 279, 'applications: records from files and schemas');
    tiles(g, 4, APPS, 301);

    // 5: long-horizon tasks, run inside the environment
    into(g, 5, 320, 373, 320, 353);
    zone(g, 5, 0, 375, 640, 54);
    lab(g, 5, 14, 396, 'long-horizon tasks · behavior taxonomy');
    if (st(5) !== 'todo') {
      line(g, 5, 14, 417, 'reasoning types, document skills, goal settings, user personas and behaviors');
    }

    K.callout(g, 0, 447, 640, SAY[cur], { cols: 74 });
    back.setAttribute('opacity', cur ? 1 : 0.35);
    next.lastChild.textContent = cur === LAST ? 'start over' : 'next';
    next.setAttribute('aria-label', cur === LAST ? 'start over' : 'next step');
  }
  function go(i) { cur = Math.max(0, Math.min(LAST, i)); draw(); }

  // reserve the tallest callout (three lines), so the foot never lands on it
  svg.appendChild(n('rect', { x: 0, y: 457, width: 640, height: 3 * 19 + 22, fill: 'none' }));

  // every stage is also a click target; these sit above the drawing and never move
  [[0, 46, 320, 72], [352, 46, 288, 72], [10, 152, 620, 80], [10, 236, 620, 30],
    [10, 270, 620, 80], [0, 385, 640, 54]].forEach(function (r, i) {
    var h = n('rect', { x: r[0], y: r[1], width: r[2], height: r[3], fill: 'transparent',
      tabindex: 0, role: 'button', 'aria-label': 'show step ' + (i + 1) + ': ' + NAME[i],
      style: 'cursor:pointer;outline:none' });
    h.addEventListener('click', function () { go(i); });
    h.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(i); }
    });
    svg.appendChild(h);
  });

  var back = CH.button(svg, 0, 0, 'back', function () { go(cur - 1); }, { w: 84 });
  back.firstChild.setAttribute('fill', '#fff');
  back.firstChild.setAttribute('stroke', C.amber);
  back.firstChild.setAttribute('stroke-width', 1.5);
  back.lastChild.setAttribute('fill', C.amber);
  var next = CH.button(svg, 92, 0, 'next', function () { go(cur === LAST ? 0 : cur + 1); }, { w: 112 });

  K.foot(svg, 'Schematic of the text of §4.1.4, which has no figure and reports no counts.');
  draw();
};
