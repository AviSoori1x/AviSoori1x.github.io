/* S_SPAM: query-reflection pages (§4.4.5). The paper's own example URL, then
   Figure 14, the share of BrowseComp trajectories that hit such a URL in their
   first ten searches. Figure 14 is raster, so its curve is read from pixels and is
   approximate (window.ML4.figread.fig14). The plotted curve is not monotone. */
window.SCENES.S_SPAM = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts, FR = (api.D.figread || {}).fig14;
  var svg = K.board(host, { alt: 'Figure 14: share of BrowseComp trajectories that hit a query-reflection URL, by month' });
  var YEAR = FR && FR.start_date_approx ? FR.start_date_approx.slice(0, 4) : '';
  K.head(svg, 'Other agents\' footprints', 'Figure 14 · internal BrowseComp traces · February to October ' + YEAR);
  if (!FR || !FR.curve) { K.mono(svg, 0, 40, 'Figure 14 values are not available.', { size: 14 }); return; }

  // the paper's example (footnote 1); the URL breaks across a line in the PDF
  var URL = F['a4c.url_a'] + F['a4c.url_b'];
  var cut = URL.indexOf('solver/') + 7, head = URL.slice(0, cut), slug = URL.slice(cut);
  var br = slug.lastIndexOf('-', 62) + 1;
  var CW = 0.58 * 13;   // advance of one mono character at 13px
  K.panel(svg, 0, 0, 640, 104, { r: 9 });
  K.label(svg, 16, 24, 'a query-reflection URL, the paper\'s example');
  K.mono(svg, 16, 48, head, { size: 13, color: C.ink3 });
  svg.appendChild(K.n('rect', { x: 12, y: 54, width: CW * br + 8, height: 21, rx: 4, fill: K.TINT.amber }));
  svg.appendChild(K.n('rect', { x: 12, y: 76, width: CW * (slug.length - br) + 8, height: 21, rx: 4,
    fill: K.TINT.amber }));
  K.mono(svg, 16, 69, slug.slice(0, br), { size: 13, color: C.ink, weight: 500 });
  K.mono(svg, 16, 91, slug.slice(br), { size: 13, color: C.ink, weight: 500 });
  var endX = 16 + CW * (slug.length - br) + 18;
  K.arrow(svg, endX + 34, 86, endX + 4, 86, { color: C.amber, w: 1.6 });
  K.mono(svg, endX + 42, 91, 'the URL spells out the query', { size: 13, color: C.ink2 });

  // Figure 14: x is months since 1 January, so 2 is 1 March
  var MON = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
    'October', 'November', 'December'];
  var DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  function when(t) {
    var m = Math.floor(t), day = (t - m) * DAYS[m] + 1;
    return (day <= 10 ? 'early ' : day <= 20 ? 'mid ' : 'late ') + MON[m];
  }
  var pts = FR.curve.map(function (q) { return [q.t, q.v]; });
  function at(t) {
    for (var i = 1; i < pts.length; i++) {
      if (t <= pts[i][0]) {
        var a = pts[i - 1], b = pts[i], f = (t - a[0]) / (b[0] - a[0]);
        return a[1] + f * (b[1] - a[1]);
      }
    }
    return pts[pts.length - 1][1];
  }
  var T0 = FR.start.t, T1 = FR.end.t;
  // 582 wide: the slider under the data range then keeps its hit area inside 640
  var ax = CH.axes(svg, 46, 150, 582, 182, { x0: 1.3, x1: 9.3, y0: 0, y1: 100,
    xt: [2, 3, 4, 5, 6, 7, 8, 9], yt: [0, 25, 50, 75, 100],
    xf: function (v) { return MON[v].slice(0, 3); }, yf: function (v) { return v + '%'; },
    yl: 'trajectories that hit one, first ' + F['a4c.searches'] + ' searches' });
  // the area under the curve, then the curve
  var area = 'M' + ax.sx(T0) + ' ' + ax.sy(0) + pts.map(function (q) {
    return ' L' + ax.sx(q[0]).toFixed(1) + ' ' + ax.sy(q[1]).toFixed(1); }).join('') +
    ' L' + ax.sx(T1) + ' ' + ax.sy(0) + ' Z';
  svg.appendChild(K.n('path', { d: area, fill: K.TINT.amber }));
  svg.appendChild(K.n('path', { d: 'M46 ' + ax.sy(50) + ' H' + (46 + 582), stroke: C.ink2, 'stroke-width': 1.2,
    'stroke-dasharray': '5 4' }));
  K.mono(svg, 54, ax.sy(50) - 8, 'half of all trajectories', { size: 12.5, color: C.ink3 });
  CH.line(svg, ax, pts, { color: C.amber, w: 2.8 });
  var X50 = FR.features && FR.features.crosses_50_percent;
  if (X50) {
    K.mono(svg, ax.sx(X50.t) - 12, ax.sy(50) - 8, 'crosses half in ' + when(X50.t),
      { size: 12.5, anchor: 'end', color: C.ink2 });
  }

  // the cursor, driven by the slider
  var cur = K.n('path', { d: '', stroke: C.ink, 'stroke-width': 1.2, 'stroke-dasharray': '3 3' });
  svg.appendChild(cur);
  var dot = K.n('circle', { r: 6.5, fill: C.amber, stroke: '#fff', 'stroke-width': 2.2 });
  svg.appendChild(dot);
  var out = CH.readout(svg, 0, 424, { size: 13 });
  function move(t) {
    var v = at(t);
    cur.setAttribute('d', 'M' + ax.sx(t) + ' ' + ax.sy(0) + ' V' + ax.sy(v));
    dot.setAttribute('cx', ax.sx(t));
    dot.setAttribute('cy', ax.sy(v));
    out.set(when(t) + ': about ' + Math.round(v) + '% of trajectories hit one in their first ' +
      F['a4c.searches'] + ' searches');
  }
  // the track spans exactly the plotted months, so the knob sits under the cursor line
  var SX0 = ax.sx(T0), SX1 = ax.sx(T1);
  CH.slider(svg, SX0, 390, SX1 - SX0, { min: T0, max: T1, step: 0.01, value: 9, label: 'month',
    fmt: function (t) { return when(t); }, onChange: move });

  K.callout(svg, 0, 446, 640, 'Control: SimpleQA Verified, on the same search stack, showed no ' +
    'equivalent increase. Excluding these domains did not change scores significantly, but runs ' +
    'varied more.',
    { cols: 74 });
  // the reading error is stated in the figread notes ("about +/-0.3 points"); taken from there
  var errm = (FR.notes || []).join(' ').match(/about \+\/-([\d.]+) points/);
  K.foot(svg, 'Figure 14 is a raster image, so the curve is read from pixels and approximate' +
    (errm ? ', to about ±' + errm[1] + ' points' : '') + '. The text says the share rose steadily; ' +
    'the plotted curve levels off around July and dips into early August before the September climb.');
};
