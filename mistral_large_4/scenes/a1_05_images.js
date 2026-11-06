/* S_IMAGES: why "no regret" and "modest regret" are both true. Schematic: the
   paper states the two outcomes without numbers, so bar lengths are illustrative.
   The 99% MFU figure is the paper's. */
window.SCENES.S_IMAGES = function (host, api) {
  var K = api.KIT, C = K.C, CH = window.CH, F = api.D.facts;
  var svg = K.board(host, { alt: 'Schematic: a text-only run against an image and text run, matched two ways' });
  K.head(svg, 'Images from the first step', 'interleaved image and text pretraining · schematic');
  var body = CH.body(svg);
  var X0 = 132, TXT = 330, IMG = 120;

  function run(y, name, text, img, tag) {
    K.mono(body, 0, y + 19, name, { size: 13.5, weight: 600, color: C.ink });
    body.appendChild(K.n('rect', { x: X0, y: y, width: text, height: 28, rx: 5, fill: CH.GREY }));
    K.mono(body, X0 + 12, y + 19, 'text', { size: 13, color: '#fff', weight: 600 });
    if (img) {
      body.appendChild(K.n('rect', { x: X0 + text + 2, y: y, width: img, height: 28, rx: 5, fill: C.amber }));
      K.mono(body, X0 + text + 14, y + 19, 'images', { size: 13, color: '#fff', weight: 600 });
    }
    if (tag) K.mono(body, X0 + text + img + 12, y + 19, tag, { size: 13, color: C.ink3 });
  }

  function draw(m) {
    CH.clear(body);
    var horizon = m === 1, t1 = horizon ? TXT + IMG : TXT;
    K.label(body, 0, 64, horizon ? 'both runs train on the same total tokens' :
      'both runs see the same number of text tokens');
    run(84, 'text only', t1, 0, '');
    run(136, 'text + images', TXT, IMG, '');
    var mx = X0 + (horizon ? TXT + IMG : TXT);
    body.appendChild(K.n('path', { d: 'M' + mx + ' 72 V180', stroke: C.blue, 'stroke-width': 2,
      'stroke-dasharray': '5 4' }));
    K.mono(body, mx + 8, 196, horizon ? 'matched: total tokens' : 'matched: text tokens',
      { size: 13, color: C.blue, weight: 600, anchor: horizon ? 'end' : 'start' });
    K.label(body, 0, 250, 'text perplexity, mixed run against text only');
    K.big(body, 0, 304, horizon ? 'modest regret' : 'no regret', { size: 40,
      color: horizon ? C.ink : C.amber });
    K.callout(body, 0, 330, 640, horizon ?
      'With the total fixed, the image tokens take room the text would have had, so the mixed ' +
      'run sees less text and its text perplexity is a little worse.' :
      'With the text fixed, the images come on top. The mixed run is longer, and its text ' +
      'perplexity is no worse at all.', { cols: 74 });
  }
  K.switcher(svg, 0, 0, ['match text tokens', 'match training horizon'], draw, { tint: 'amber' });

  // the systems side, which is the paper's own number
  K.label(svg, 0, 440, 'the encoder is frozen, so its outputs can be cached');
  K.chip(svg, 0, 452, 'prefetch N batches ahead', { w: 200, stroke: CH.GREY, color: C.ink2 });
  K.arrow(svg, 206, 464.5, 230, 464.5, { color: C.ink3 });
  K.chip(svg, 236, 452, 'cache encoder features', { w: 190, stroke: CH.GREY, color: C.ink2 });
  K.arrow(svg, 432, 464.5, 456, 464.5, { color: C.ink3 });
  K.big(svg, 466, 476, F['img.mfu_kept'] + '%', { size: 30, color: C.amber });
  K.mono(svg, 466, 500, 'of text-only MFU', { size: 13, color: C.ink2 });

  K.foot(svg, 'Schematic. The paper reports "no regret" and "modest regret" without numbers, so ' +
    'the bar lengths are illustrative. The ' + F['img.mfu_kept'] + '% is the paper\'s figure.');
};
