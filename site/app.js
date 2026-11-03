(function () {
  'use strict';

  var TASKS = ['annotation', 'integration', 'development', 'spatial_niche'];
  var TASK_LABEL = { annotation: '细胞注释', integration: '数据整合', development: '发育预测', spatial_niche: '空间生态位' };
  var PRIMARY = { annotation: 'f1_macro', integration: 'integration_composite', development: 'population_mse', spatial_niche: 'ari' };
  var METRIC_LABEL = {
    f1_macro: 'Macro F1', accuracy: 'Accuracy', balanced_accuracy: 'Balanced accuracy', mcc: 'MCC', cohen_kappa: "Cohen's kappa",
    top3_accuracy: 'Top-3 accuracy', f1_weighted: 'Weighted F1', f1_micro: 'Micro F1', precision_macro: 'Macro precision',
    recall_macro: 'Macro recall', precision_micro: 'Micro precision', recall_micro: 'Micro recall', precision_weighted: 'Weighted precision',
    recall_weighted: 'Weighted recall', auprc_macro: 'Macro AUPRC', nll: 'NLL', ece: 'ECE', brier_multiclass: 'Brier',
    integration_composite: 'Integration composite', biology_conservation: 'Biology conservation', batch_removal: 'Batch removal',
    celltype_asw: 'Cell-type ASW', batch_asw_inverse: 'Inverse batch ASW', graph_connectivity: 'Graph connectivity',
    isolated_label_f1: 'Isolated-label F1', ilisi_normalized: 'iLISI (normalized)',
    population_mse: 'Population MSE', gene_spearman: 'Gene Spearman', gene_pearson: 'Gene Pearson', mmd_rbf: 'MMD (RBF)',
    mean_error: 'Mean error', variance_error: 'Variance error', output_floor_fraction: 'Output floor fraction',
    ari: 'ARI', nmi: 'NMI', morans_i: "Moran's I", gearys_c: "Geary's C", neighbor_label_enrichment: 'Neighbor label enrichment'
  };
  var FAMILY_LABEL = { classical: '经典', foundation: '基础模型', deep: '深度', integration: '整合工具', development: '发育模型', ours: 'nanoBrain', submitted: '提交' };

  var state = { rows: [], task: 'annotation', metric: PRIMARY.annotation, frTask: 'annotation', evidence: null };

  function $(s, el) { return (el || document).querySelector(s); }
  function esc(v) { return String(v == null ? '' : v).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function num(v) { if (v === null || v === undefined || v === '') return null; var n = Number(v); return isFinite(n) ? n : null; }
  function fmt(v) { var n = num(v); return n === null ? '—' : (Math.abs(n) >= 100 ? n.toFixed(1) : n.toFixed(4)); }
  function mlabel(m) { return METRIC_LABEL[m] || m; }
  function isOurs(r) { return r.method_family === 'ours' || /^nanobrain/.test(r.method || ''); }

  function tabs(el, current, onPick) {
    el.innerHTML = TASKS.map(function (t) {
      return '<button role="tab" type="button" data-t="' + t + '" aria-selected="' + (t === current) + '">' + TASK_LABEL[t] + '</button>';
    }).join('');
    el.onclick = function (e) {
      var b = e.target.closest('button'); if (!b) return;
      onPick(b.getAttribute('data-t'));
    };
  }

  /* ---------- leaderboard ---------- */
  function boardRows() {
    return state.rows.filter(function (r) {
      return r.task === state.task && r.metric === state.metric && (!r.evaluation_split || r.evaluation_split === 'test') && num(r.mean) !== null;
    });
  }

  function renderBoard() {
    tabs($('#task-tabs'), state.task, function (t) { state.task = t; state.metric = PRIMARY[t]; renderBoard(); });

    var metrics = {};
    state.rows.forEach(function (r) { if (r.task === state.task) metrics[r.metric] = true; });
    var list = Object.keys(metrics).sort(function (a, b) { return (a === PRIMARY[state.task] ? -1 : 0) - (b === PRIMARY[state.task] ? -1 : 0) || mlabel(a).localeCompare(mlabel(b)); });
    var sel = $('#metric-sel');
    sel.innerHTML = list.map(function (m) {
      return '<option value="' + esc(m) + '"' + (m === state.metric ? ' selected' : '') + '>' + esc(mlabel(m)) + (m === PRIMARY[state.task] ? '（主指标）' : '') + '</option>';
    }).join('');
    sel.onchange = function () { state.metric = sel.value; renderBoard(); };

    var rows = boardRows();
    var dir = rows.length && rows[0].direction === 'min' ? 'min' : 'max';
    rows.sort(function (a, b) { return dir === 'min' ? num(a.mean) - num(b.mean) : num(b.mean) - num(a.mean); });
    var vals = rows.map(function (r) { return num(r.mean); });
    var lo = Math.min.apply(null, vals.concat([0])), hi = Math.max.apply(null, vals);
    if (dir === 'min') { lo = Math.min.apply(null, vals); hi = Math.max.apply(null, vals); }

    var rank = 0, prev = null;
    var html = rows.map(function (r, i) {
      var v = num(r.mean);
      if (v !== prev) { rank = i + 1; prev = v; }
      var frac = hi === lo ? 1 : (dir === 'min' ? (hi - v) / (hi - lo) : (v - lo) / (hi - lo));
      var w = Math.max(3, Math.round(6 + 94 * frac));
      var ours = isOurs(r);
      var fam = FAMILY_LABEL[r.method_family] || r.method_family || '';
      var sd = num(r.std);
      return '<tr class="row' + (ours ? ' is-ours' : '') + '" data-i="' + i + '" tabindex="0" aria-expanded="false">' +
        '<td class="r rank">' + rank + '</td>' +
        '<td class="m"><b>' + esc(r.display_name || r.method) + '</b><small>' + esc(r.method) + '</small></td>' +
        '<td class="fam hide-sm"><span class="chip' + (ours ? ' ours' : '') + '">' + esc(fam) + '</span>' + esc(String(r.training_regime || '').replace(/_/g, ' ')) + '</td>' +
        '<td><div class="score"><span class="bar"><i style="width:' + w + '%"></i></span><span class="v">' + fmt(v) + '</span></div></td>' +
        '<td class="r hide-sm">' + (sd === null ? '—' : fmt(sd)) + '</td>' +
        '<td class="r hide-sm">' + esc(r.n_runs || '—') + '</td></tr>';
    }).join('');
    var tbody = $('#board-table tbody');
    tbody.innerHTML = html || '<tr><td colspan="6" class="foot">暂无该指标的测试集结果。</td></tr>';

    tbody.onclick = function (e) { var tr = e.target.closest('tr.row'); if (tr) toggleDetail(tr, rows); };
    tbody.onkeydown = function (e) { if ((e.key === 'Enter' || e.key === ' ') && e.target.matches('tr.row')) { e.preventDefault(); toggleDetail(e.target, rows); } };

    $('#board-foot').textContent = mlabel(state.metric) + (dir === 'min' ? ' · 越低越好' : ' · 越高越好') +
      ' · ' + rows.length + ' 个方法 · 评估细胞 ' + (rows[0] && rows[0].eval_rows ? Number(rows[0].eval_rows).toLocaleString() : '—') +
      ' · 仅 overall 层，不跨指标合成总分';
  }

  function toggleDetail(tr, rows) {
    var next = tr.nextElementSibling;
    if (next && next.classList.contains('detail')) { next.remove(); tr.setAttribute('aria-expanded', 'false'); return; }
    var r = rows[Number(tr.getAttribute('data-i'))];
    var d = document.createElement('tr');
    d.className = 'detail';
    d.innerHTML = '<td colspan="6">' +
      '<div>' + esc(r.notes || '无额外说明') + '</div>' +
      '<div style="margin-top:6px"><code>split ' + esc(r.evaluation_split || 'test') + '</code> · <code>' + esc(r.split_id || '') + '</code> · 评估细胞 ' +
      esc(r.eval_rows ? Number(r.eval_rows).toLocaleString() : '—') + ' · 数值指标 ' + esc(r.n_numeric || '—') + ' / ' + esc(r.n_runs || '—') + ' runs</div></td>';
    tr.after(d);
    tr.setAttribute('aria-expanded', 'true');
  }

  /* ---------- frontier ---------- */
  function svg(tag, attrs, text) {
    var e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function renderFrontier() {
    var ev = state.evidence; if (!ev) return;
    tabs($('#fr-tabs'), state.frTask, function (t) { state.frTask = t; renderFrontier(); });
    var task = state.frTask, metric = PRIMARY[task];
    var pts = ev.runs.map(function (r) {
      var m = r.metrics && r.metrics[task] && r.metrics[task][metric];
      return m && num(r.parameter_count) ? { x: num(r.parameter_count), y: num(m.value), dir: m.direction, r: r } : null;
    }).filter(Boolean);
    var box = $('#frontier'); box.innerHTML = '';
    if (!pts.length) { box.textContent = '暂无数据'; return; }
    var minDir = pts[0].dir === 'min';

    var W = Math.max(420, Math.min(960, box.clientWidth || 960)), H = W < 600 ? 340 : 400, L = 72, R = 24, T = 24, B = 52;
    var xs = pts.map(function (p) { return Math.log10(p.x); }), ys = pts.map(function (p) { return p.y; });
    var x0 = Math.floor(Math.min.apply(null, xs) * 10) / 10 - 0.05, x1 = Math.ceil(Math.max.apply(null, xs) * 10) / 10 + 0.05;
    var yMin = Math.min.apply(null, ys), yMax = Math.max.apply(null, ys), pad = (yMax - yMin) * 0.12 || 0.01;
    var y0 = yMin >= 0 ? Math.max(0, yMin - pad) : yMin - pad, y1 = yMax + pad;
    function sx(v) { return L + (Math.log10(v) - x0) / (x1 - x0) * (W - L - R); }
    function sy(v) { return T + (1 - (v - y0) / (y1 - y0)) * (H - T - B); }

    var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': '参数量与 ' + mlabel(metric) + ' 散点图' });
    for (var i = 0; i <= 4; i++) {
      var yv = y0 + (y1 - y0) * i / 4, yy = sy(yv);
      s.appendChild(svg('line', { x1: L, x2: W - R, y1: yy, y2: yy, 'class': 'ax' }));
      s.appendChild(svg('text', { x: L - 10, y: yy + 4, 'text-anchor': 'end', 'class': 'tk' }, Math.abs(yv) >= 100 ? yv.toFixed(0) : yv.toFixed(3)));
    }
    [1e5, 2e5, 3e5, 5e5, 1e6].forEach(function (t) {
      var lx = Math.log10(t); if (lx < x0 || lx > x1) return;
      s.appendChild(svg('text', { x: sx(t), y: H - B + 20, 'text-anchor': 'middle', 'class': 'tk' }, (t / 1000) + 'k'));
    });
    s.appendChild(svg('text', { x: (L + W - R) / 2, y: H - 10, 'text-anchor': 'middle', 'class': 'lbl' }, '参数量（对数）'));
    s.appendChild(svg('text', { x: 16, y: T + (H - T - B) / 2, 'text-anchor': 'middle', 'class': 'lbl', transform: 'rotate(-90 16 ' + (T + (H - T - B) / 2) + ')' }, mlabel(metric) + (minDir ? ' ↓' : ' ↑')));

    // Pareto frontier: no other point with fewer-or-equal params and a better score.
    var sorted = pts.slice().sort(function (a, b) { return a.x - b.x || (minDir ? a.y - b.y : b.y - a.y); });
    var front = [], best = null;
    sorted.forEach(function (p) {
      if (best === null || (minDir ? p.y < best : p.y > best)) { front.push(p); best = p.y; p.front = true; }
    });
    if (front.length > 1) {
      s.appendChild(svg('polyline', { points: front.map(function (p) { return sx(p.x) + ',' + sy(p.y); }).join(' '), 'class': 'pf' }));
    }
    var tip = document.createElement('div'); tip.className = 'tip'; tip.hidden = true;
    pts.forEach(function (p) {
      var c = svg('circle', { cx: sx(p.x), cy: sy(p.y), r: 6, 'class': 'pt' + (p.front ? ' front' : ''), tabindex: 0 });
      var show = function () {
        tip.innerHTML = '<b>' + esc(p.r.label) + '</b><br>' + p.x.toLocaleString() + ' 参数 · ' + esc(p.r.steps) + ' 步 · seed ' + esc(p.r.seed) +
          '<br>' + esc(mlabel(metric)) + ' ' + fmt(p.y);
        var bb = box.getBoundingClientRect(), cb = c.getBoundingClientRect();
        tip.style.left = (cb.left - bb.left + cb.width / 2) + 'px'; tip.style.top = (cb.top - bb.top) + 'px'; tip.hidden = false;
      };
      c.addEventListener('mouseenter', show); c.addEventListener('focus', show);
      c.addEventListener('mouseleave', function () { tip.hidden = true; }); c.addEventListener('blur', function () { tip.hidden = true; });
      s.appendChild(c);
      // Several candidates share a parameter count; label only the frontier, the rest via tooltip.
      if (p.front) s.appendChild(svg('text', { x: sx(p.x) + 10, y: sy(p.y) - 9, 'class': 'pl' }, p.r.label));
    });
    box.appendChild(s); box.appendChild(tip);
  }

  /* ---------- hero: two batches drifting into shared cell-type clusters ---------- */
  function hero() {
    var el = $('#cells'); if (!el) return;
    var colors = ['#3fa9cf', '#4fc9a8', '#7088ee', '#a77bf3', '#8fb8f0'];
    var centers = [[110, 110], [300, 95], [210, 190], [105, 270], [315, 265]];
    var seed = 7; function rnd() { seed = (seed * 16807) % 2147483647; return seed / 2147483647; }
    function gauss() { return (rnd() + rnd() + rnd() - 1.5) * 1.4; }
    var dots = [];
    for (var i = 0; i < 230; i++) {
      var k = i % 5, batch = i % 2, c = centers[k];
      var jx = gauss() * 26, jy = gauss() * 26;
      var off = batch ? [26, -18] : [-26, 18];
      var d = svg('circle', { cx: c[0] + jx, cy: c[1] + jy, r: batch ? 4.2 : 3.4, fill: colors[k], 'fill-opacity': batch ? 0.85 : 0.5 });
      d.style.transform = 'translate(' + off[0] + 'px,' + off[1] + 'px)';
      dots.push({ el: d, off: off });
      el.appendChild(d);
    }
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    var merged = false;
    setInterval(function () {
      merged = !merged;
      dots.forEach(function (d) { d.el.style.transform = merged ? 'translate(0,0)' : 'translate(' + d.off[0] + 'px,' + d.off[1] + 'px)'; });
    }, 3200);
  }

  function stats() {
    var methods = {};
    state.rows.forEach(function (r) { if (!isOurs(r)) methods[r.method] = true; });
    var m = $('[data-stat="methods"]'), n = $('[data-stat="rows"]');
    if (m) m.textContent = Object.keys(methods).length;
    if (n) n.textContent = state.rows.length.toLocaleString();
  }

  function copy() {
    document.addEventListener('click', function (e) {
      var b = e.target.closest('.copy'); if (!b) return;
      var t = document.getElementById(b.getAttribute('data-copy'));
      if (!t || !navigator.clipboard) return;
      navigator.clipboard.writeText(t.innerText).then(function () { b.textContent = '已复制'; setTimeout(function () { b.textContent = '复制'; }, 1400); });
    });
  }

  function getJSON(url) {
    return fetch(url, { cache: 'no-store' }).then(function (r) { if (!r.ok) throw new Error(url + ' ' + r.status); return r.json(); });
  }

  hero();
  copy();
  getJSON('data/leaderboard.json').then(function (d) {
    state.rows = d.rows || [];
    stats(); renderBoard();
    if (d.generated_at) $('#gen-at').textContent = '数据更新 ' + String(d.generated_at).replace('T', ' ').replace('Z', ' UTC');
  }).catch(function (err) {
    $('#board-table tbody').innerHTML = '<tr><td colspan="6" class="foot">排行榜数据加载失败：' + esc(err.message) + '</td></tr>';
  });
  getJSON('data/frontier-evidence.json').then(function (d) { state.evidence = d; renderFrontier(); }).catch(function () { $('#frontier').textContent = '演化数据加载失败'; });
})();
