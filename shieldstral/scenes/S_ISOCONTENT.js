window.SCENES = window.SCENES || {};

/* Act I, scene 05. Iso-content.
   One document, held completely still, and two different queries about it.
   Swapping the query flips the ground-truth answer from yes to no.
   Every string and every label is read from window.SS.fig3 at runtime. */
window.SCENES['S_ISOCONTENT'] = function (root, api) {
  var SS = api.SS || {};
  var F = SS.fig3 || {};
  var POS = F.positive || {};
  var NEG = F.negative || {};
  var DOC = F.document == null ? '' : String(F.document);
  var INS = F.instruct == null ? '' : String(F.instruct);

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  /* FNV-1a, 32 bit. Cheap, deterministic, and recomputed on every swap so the
     figure can honestly claim the document bytes did not move. */
  function checksum(s) {
    var h = 0x811c9dc5, i;
    for (i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
    }
    return ('00000000' + h.toString(16).toUpperCase()).slice(-8);
  }

  var DOT = '·';
  var baseSum = checksum(DOC);
  var baseLen = DOC.length;

  function dotSvg(cls) {
    return '<svg class="dot ' + cls + '" viewBox="0 0 16 16" aria-hidden="true">'
      + '<circle class="ring" cx="8" cy="8" r="6.2"></circle>'
      + '<circle class="core" cx="8" cy="8" r="3.1"></circle></svg>';
  }

  function qBtn(side, o) {
    return '<button type="button" class="q ' + side + '" id="S_ISOCONTENT-q-' + side + '"'
      + ' role="radio" aria-checked="false" tabindex="-1">'
      + '<span class="qtop">' + dotSvg('qdot')
      + '<span class="qrole">' + esc(o.role || '') + '</span>'
      + '<span class="qkey">query</span></span>'
      + '<span class="qtxt">' + esc(o.query || '') + '</span></button>';
  }

  root.className = 'sc-s_isocontent';
  root.appendChild(api.frag(
    '<div class="wrap">'

    + '<div class="hd">'
    +   '<span class="eyeb">iso-content</span>'
    +   '<span class="hnote">one document, two questions, the answer flips</span>'
    + '</div>'

    + '<div class="fld ins">'
    +   '<span class="flab">instruct</span>'
    +   '<span class="ftxt">' + esc(INS) + '</span>'
    +   '<span class="pin">held</span>'
    + '</div>'

    + '<div class="doc" id="S_ISOCONTENT-doc">'
    +   '<div class="docbar">'
    +     '<span class="flab">document</span>'
    +     '<span class="stamp" id="S_ISOCONTENT-stamp" aria-live="off"></span>'
    +   '</div>'
    +   '<p class="doctxt">' + esc(DOC) + '</p>'
    +   '<div class="counts">'
    +     '<span class="ct"><i>query swaps</i><b id="S_ISOCONTENT-swaps">0</b></span>'
    +     '<span class="ct keep"><i>characters changed</i><b>0</b></span>'
    +     '<span class="ct keep drift"><i>checksum drift</i><b>none</b></span>'
    +   '</div>'
    + '</div>'

    /* the same document forks into both questions, drawn with plain borders
       so the hairlines stay exactly one pixel at any panel width */
    + '<div class="split" aria-hidden="true">'
    +   '<i class="bar"></i>'
    +   '<i class="stem"></i>'
    +   '<i class="drop dpos"></i>'
    +   '<i class="drop dneg"></i>'
    + '</div>'

    + '<div class="qs" role="radiogroup" aria-label="choose which query is asked of the document">'
    +   qBtn('pos', POS) + qBtn('neg', NEG)
    + '</div>'

    + '<div class="verd" id="S_ISOCONTENT-verd">'
    +   '<div class="vleft">'
    +     '<span class="vlab">ground-truth answer</span>'
    /* aria-live starts off so the idle auto cycle does not chatter at a screen
       reader, and is turned on once the reader is driving the toggle */
    +     '<span class="vbig" id="S_ISOCONTENT-big" aria-live="off"></span>'
    +     '<span class="vsub" id="S_ISOCONTENT-sub"></span>'
    +   '</div>'
    +   '<div class="vright">'
    +     '<span class="vlab">answer space</span>'
    +     '<span class="tok" id="S_ISOCONTENT-tok-yes">' + dotSvg('tdot') + '<b>yes</b></span>'
    +     '<span class="tok" id="S_ISOCONTENT-tok-no">' + dotSvg('tdot') + '<b>no</b></span>'
    +   '</div>'
    + '</div>'

    + '<div class="foot">'
    +   '<span class="gt">These are the paper\'s ground-truth labels for this pair of samples, '
    +   'not a live model call.</span>'
    +   '<span class="hint" id="S_ISOCONTENT-hint">cycling on its own, click or use arrow keys to take over</span>'
    + '</div>'

    + '</div>'
  ).firstChild);

  var q = {
    pos: root.querySelector('#S_ISOCONTENT-q-pos'),
    neg: root.querySelector('#S_ISOCONTENT-q-neg')
  };
  var stamp = root.querySelector('#S_ISOCONTENT-stamp');
  var swapsEl = root.querySelector('#S_ISOCONTENT-swaps');
  var big = root.querySelector('#S_ISOCONTENT-big');
  var sub = root.querySelector('#S_ISOCONTENT-sub');
  var verd = root.querySelector('#S_ISOCONTENT-verd');
  var hint = root.querySelector('#S_ISOCONTENT-hint');
  var docBox = root.querySelector('#S_ISOCONTENT-doc');
  var tok = {
    yes: root.querySelector('#S_ISOCONTENT-tok-yes'),
    no: root.querySelector('#S_ISOCONTENT-tok-no')
  };

  var swaps = 0;
  var cur = null;
  var auto = true;
  var flashT = 0;

  function setStamp(rehashed) {
    var sum = checksum(DOC);
    stamp.textContent = (rehashed ? 're-hashed, identical ' : 'unchanged ')
      + DOT + ' ' + baseLen + ' chars ' + DOT + ' fnv1a ' + sum;
    stamp.classList.toggle('hot', !!rehashed && sum === baseSum);
  }

  function setActive(side, fromUser) {
    if (side === cur) return;
    var other = side === 'pos' ? 'neg' : 'pos';
    var o = side === 'pos' ? POS : NEG;
    var lab = String(o.label || '').toLowerCase();
    var yes = lab === 'yes';

    q[side].setAttribute('aria-checked', 'true');
    q[side].tabIndex = 0;
    q[other].setAttribute('aria-checked', 'false');
    q[other].tabIndex = -1;

    root.querySelector('.split').classList.toggle('on-pos', side === 'pos');
    root.querySelector('.split').classList.toggle('on-neg', side === 'neg');

    big.textContent = lab ? lab.toUpperCase() : 'n/a';
    sub.textContent = yes ? 'flagged, this policy is met'
      : 'not flagged, this policy is not met';
    verd.classList.toggle('yes', yes);
    verd.classList.toggle('no', !yes);
    tok.yes.classList.toggle('lit', yes);
    tok.no.classList.toggle('lit', !yes);

    if (cur !== null) {
      swaps++;
      swapsEl.textContent = String(swaps);
      /* prove the document is untouched by hashing it again on every swap */
      setStamp(true);
      if (!api.reduce) {
        docBox.classList.remove('recheck');
        void docBox.offsetWidth;
        docBox.classList.add('recheck');
        verd.classList.remove('flip');
        void verd.offsetWidth;
        verd.classList.add('flip');
      }
      clearTimeout(flashT);
      flashT = setTimeout(function () { setStamp(false); }, 1100);
    } else {
      setStamp(false);
    }

    if (fromUser && auto) {
      auto = false;
      big.setAttribute('aria-live', 'polite');
      hint.textContent = 'manual, arrow keys swap the query';
    }
    cur = side;
  }

  ['pos', 'neg'].forEach(function (side) {
    q[side].addEventListener('click', function () { setActive(side, true); });
  });
  root.querySelector('.qs').addEventListener('keydown', function (e) {
    var k = e.key;
    if (k === 'ArrowLeft' || k === 'ArrowUp' || k === 'Home') {
      setActive('pos', true); q.pos.focus(); e.preventDefault();
    } else if (k === 'ArrowRight' || k === 'ArrowDown' || k === 'End') {
      setActive('neg', true); q.neg.focus(); e.preventDefault();
    }
  });

  setActive('pos', false);

  /* auto cycle, so a reader who only scrolls still sees the flip */
  var running = false, nextAt = null;
  return {
    start: function () { running = true; nextAt = null; },
    stop: function () { running = false; },
    tick: function (t) {
      if (!running || !auto || api.reduce) return;
      if (nextAt === null) { nextAt = t + 2.4; return; }
      if (t >= nextAt) {
        nextAt = t + 3.8;
        setActive(cur === 'pos' ? 'neg' : 'pos', false);
      }
    }
  };
};
