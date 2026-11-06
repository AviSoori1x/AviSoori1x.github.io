/* S_GRAN: granularity 4 against 8. Fewer FLOPs, slower steps, about break-even.
   The 4.3% (±0.5) and the 5% are the paper's; the net is my arithmetic. */
window.SCENES.S_GRAN = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var g = F['gran.g'], g2 = F['gran.g_alt'], save = F['gran.flops_saved'], unc = F['gran.flops_unc'];
  var slow = F['gran.slowdown'];
  var svg = K.board(host, { alt: 'Granularity ' + g + ' against ' + g2 + ': work, speed and time to the same perplexity' });
  K.head(svg, 'Granularity ' + g + ', on purpose', 'granularity ' + g2 + ': ' + save + '% fewer FLOPs, ' +
    slow + '% slower to train');
  var body = CH.body(svg);

  var flops = 100 - save, net = flops * (100 + slow) / 100;
  var netLo = (100 - save - unc) * (100 + slow) / 100, netHi = (100 - save + unc) * (100 + slow) / 100;
  var ROWS = [
    ['FLOPs to reach the same perplexity', flops, [100 - save - unc, 100 - save + unc], 'paper'],
    ['training speed: time for the same work', 100 + slow, null, 'paper, measured'],
    ['time to reach the same perplexity', net, [netLo, netHi], 'my arithmetic']
  ];

  function draw(sel) {
    CH.clear(body);
    var fine = sel === 1, gg = fine ? g2 : g;
    // schematic: the same expert parameters cut into g pieces
    K.label(body, 0, 62, 'one expert block cut ' + gg + ' ways (schematic)');
    var W = 640, gap = 4, pw = (W - gap * (gg - 1)) / gg;
    for (var i = 0; i < gg; i++) {
      body.appendChild(K.n('rect', { x: i * (pw + gap), y: 74, width: pw, height: 40, rx: 5,
        fill: fine ? K.TINT.blue : K.TINT.amber, stroke: fine ? C.blue : C.amber, 'stroke-width': 1.4 }));
    }
    K.mono(body, 0, 140, fine ? 'more, smaller experts: finer routing, smaller matrix multiplies' :
      'the granularity Mistral Large 3 used, kept for ML4', { size: 13, color: C.ink2 });

    // three dot rows on one zoomed axis, granularity 4 = 100
    var ax = CH.axes(body, 16, 196, 608, 230, { x0: 90, x1: 110, y0: 0, y1: 3,
      xt: [90, 95, 100, 105, 110], xf: function (v) { return v + '%'; },
      xl: 'relative to granularity ' + g + ' = 100%, axis starts at 90%', noY: true });
    ROWS.forEach(function (r, i) {
      var y = 196 + 34 + i * 76;
      K.mono(body, 0, y - 14, r[0], { size: 13.5, color: C.ink, weight: 600 });
      K.mono(body, 640, y - 14, r[3], { size: 12.5, color: C.ink3, anchor: 'end' });
      body.appendChild(K.n('path', { d: 'M' + ax.sx(90) + ' ' + y + ' H' + ax.sx(110), stroke: C.line2, 'stroke-width': 6, 'stroke-linecap': 'round' }));
      if (r[2]) body.appendChild(K.n('rect', { x: ax.sx(r[2][0]), y: y - 6, width: ax.sx(r[2][1]) - ax.sx(r[2][0]),
        height: 12, rx: 6, fill: fine ? 'rgba(28,110,140,.22)' : 'rgba(138,144,156,.25)' }));
      var p4 = ax.sx(100), p8 = ax.sx(r[1]);
      body.appendChild(K.n('circle', { cx: fine ? p4 : p8, cy: y, r: 6, fill: '#fff', stroke: CH.GREY, 'stroke-width': 2 }));
      body.appendChild(K.n('circle', { cx: fine ? p8 : p4, cy: y, r: 7.5, fill: fine ? C.blue : C.amber,
        stroke: '#fff', 'stroke-width': 2 }));
      var v = fine ? r[1] : 100;
      K.mono(body, (fine ? p8 : p4) + (fine && r[1] < 100 ? -14 : 14), y + 26,
        (i === 2 && fine ? 'about ' : '') + CH.fmt(v, 1) + '%', { size: 13.5, weight: 600,
          color: fine ? C.blue : C.amber, anchor: fine && r[1] < 100 ? 'end' : 'start' });
    });
    K.callout(body, 0, 484, 640, fine ?
      CH.fmt(flops, 1) + '% of the work at ' + (100 + slow) + '% of the time per unit of work is ' +
      CH.fmt(netLo, 0) + '% to ' + CH.fmt(netHi, 0) + '% of the time, about break-even. The measured slowdown is what kept ' + g + '.' :
      'Granularity ' + g + ' is the baseline. Switch to ' + g2 + ' to see the trade it was weighed against.',
      { cols: 74, color: fine ? C.blue : C.amber, tint: fine ? K.TINT.blue : K.TINT.amber });
  }
  K.switcher(svg, 0, 0, ['granularity ' + g + ' (chosen)', 'granularity ' + g2], draw, { tint: 'amber' });
  K.foot(svg, 'The band on the first row is the paper\'s ±' + unc + ' point uncertainty. The last ' +
    'row multiplies the first two, which assumes the slowdown applies to the whole run.');
};
