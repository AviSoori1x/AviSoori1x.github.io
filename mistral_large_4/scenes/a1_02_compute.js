/* S_COMPUTE: the pretraining run in three numbers, and what each one counts. */
window.SCENES.S_COMPUTE = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var gpus = F['pre.gpus'], days = F['pre.days'], gh = F['pre.gpu_hours_m'];
  var mfu = F['pre.mfu'], good = F['pre.goodput'];
  var svg = K.board(host, { alt: 'The pretraining run: GPU-hours, MFU and goodput' });
  K.head(svg, 'The run, in three numbers', CH.comma(gpus) + ' GB200 GPUs · ' + days +
    ' days · a data center near Paris');
  var body = CH.body(svg);

  function gpuHours() {
    var exact = gpus * days * 24;
    K.big(body, 0, 104, gh + 'M', { size: 56, color: C.amber });
    K.label(body, 196, 80, 'GPU-hours');
    K.mono(body, 196, 102, 'every GPU, every hour, for the whole run', { size: 13.5 });
    var ax = CH.axes(body, 70, 160, 540, 220, { x0: 0, x1: days, y0: 0, y1: gpus,
      xt: [0, 15, 30, 45], yt: [0, gpus / 2, gpus], yf: CH.comma, xl: 'days', yl: 'GPUs' });
    body.appendChild(K.n('rect', { x: ax.sx(0), y: ax.sy(gpus), width: ax.sx(days) - ax.sx(0),
      height: ax.sy(0) - ax.sy(gpus), fill: K.TINT.amber, stroke: C.amber, 'stroke-width': 1.5 }));
    K.mono(body, ax.sx(days / 2), ax.sy(gpus / 2) + 5, 'area = GPU-hours',
      { size: 14, color: C.amber, anchor: 'middle', weight: 600 });
    K.callout(body, 0, 446, 640, CH.comma(gpus) + ' GPUs × ' + days + ' days × 24 hours = ' +
      CH.comma(exact) + ' GPU-hours, which the paper rounds to ' + gh + ' million.', { cols: 74 });
  }

  function util() {
    K.big(body, 0, 104, mfu + '%', { size: 56, color: C.amber });
    K.label(body, 150, 80, 'model FLOPs utilization');
    K.mono(body, 150, 102, 'useful arithmetic, as a share of what the chips could do', { size: 13.5 });
    K.label(body, 0, 160, 'peak BF16 FLOP/s of the hardware = 100%');
    K.bar(body, 0, 172, 640, 26, mfu / 100, { color: C.amber });
    K.mono(body, 640 * mfu / 100 + 10, 190, mfu + '%', { size: 13.5, weight: 600, color: C.ink });
    K.label(body, 0, 250, 'counted', { color: C.amber });
    K.mono(body, 0, 276, 'matrix-multiply FLOPs', { size: 14, color: C.ink });
    K.mono(body, 0, 300, 'attention FLOPs', { size: 14, color: C.ink });
    K.label(body, 340, 250, 'not counted');
    K.mono(body, 340, 276, 'recomputed activations', { size: 14, color: C.ink });
    K.mono(body, 340, 300, '(HFU would count them)', { size: 13, color: C.ink3 });
    K.callout(body, 0, 340, 640, 'Leaving recomputation out is the strict choice. Work done ' +
      'twice to save memory does not count as progress, so it does not inflate the ' + mfu + '%.',
      { cols: 74 });
  }

  function goodput() {
    var lost = 100 - good, lostDays = days * lost / 100;
    K.big(body, 0, 104, good + '%', { size: 56, color: C.amber });
    K.label(body, 196, 80, 'goodput');
    K.mono(body, 196, 102, 'share of the run that moved training forward', { size: 13.5 });
    K.label(body, 0, 160, 'the ' + days + ' days of the run');
    K.bar(body, 0, 172, 640, 26, good / 100, { color: C.amber, track: CH.GREY2 });
    K.mono(body, 0, 226, 'training progress: ' + good + '%', { size: 13.5, color: C.ink, weight: 600 });
    K.mono(body, 640, 226, 'lost: ' + CH.fmt(lost, 1) + '%', { size: 13.5, color: C.ink2, anchor: 'end' });
    K.mono(body, 640, 248, 'about ' + CH.fmt(lostDays, 1) + ' of ' + days + ' days',
      { size: 13, color: C.ink3, anchor: 'end' });
    K.callout(body, 0, 290, 640, 'The paper reports the goodput but does not break down the remaining ' +
      CH.fmt(lost, 1) + '%. The day ' +
      'count is my arithmetic: ' + CH.fmt(lost, 1) + '% of ' + days + ' days.', { cols: 74 });
  }

  var views = [gpuHours, util, goodput];
  K.switcher(svg, 0, 0, [gh + 'M GPU-hours', 'MFU ' + mfu + '%', 'goodput ' + good + '%'],
    function (i) { CH.clear(body); views[i](); }, { tint: 'amber' });
};
