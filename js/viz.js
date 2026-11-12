/* ===========================================================
   viz.js — Canvas 시각화 (외부 차트 라이브러리 없음)
   -----------------------------------------------------------
   직접 그리는 이유
   · 로그 스케일 축 — Vanishing Gradient는 10^-6 ~ 10^0 범위를
     봐야 해서 선형축으로는 아무것도 안 보인다.
   · 두 모델(Sigmoid vs ReLU) 이중 막대
   · 결정 경계 격자를 ImageData로 한 번에 그려 속도 확보
   의존성이 없어 GitHub Pages에 파일만 올리면 끝난다는 장점도 있다.
   =========================================================== */
(function (root) {
  'use strict';

  const C = {
    c0:'#3b82f6', c1:'#ef4444',
    blue:'#2563eb', orange:'#f59e0b',
    grid:'#eef2f7', axis:'#cbd5e1', text:'#94a3b8', ink:'#475569',
    layer:['#2563eb','#f59e0b','#16a34a','#dc2626','#9333ea','#0891b2','#c2410c']
  };

  /* ---------- 캔버스 준비 (고해상도 대응) ----------
     숨겨진 탭이나 레이아웃 확정 전에는 너비가 0으로 잡힌다.
     그 상태로 그리면 1px 캔버스가 만들어져 그대로 굳어버리므로,
     너무 좁으면 그리지 않고 null을 돌려준다. (호출부가 건너뛴다) */
  const MIN_W = 40;
  function setup(cv, aspect = 0.78) {
    const dpr = window.devicePixelRatio || 1;
    const measured = Math.round(cv.getBoundingClientRect().width);
    if (measured < MIN_W) { cv.dataset.pending = '1'; return null; }
    delete cv.dataset.pending;
    const w = Math.max(1, measured);
    const h = Math.round(w * aspect);
    cv.style.height = h + 'px';
    cv.width  = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.font = '10px system-ui, sans-serif';
    return { ctx, w, h };
  }

  const nice = v => {
    if (v === 0) return '0';
    const a = Math.abs(v);
    if (a >= 1000 || a < 0.01) return v.toExponential(0).replace('e+', 'e');
    return String(Math.round(v * 100) / 100);
  };

  /* ---------- 축 그리기 ----------
     box = {l,r,t,b} 여백. 값→픽셀 변환 함수를 돌려준다. */
  function axes(ctx, w, h, box, xs, ys, opts = {}) {
    const L = box.l, R = w - box.r, T = box.t, B = h - box.b;
    const xv = v => L + (v - xs.min) / (xs.max - xs.min) * (R - L);
    const yv = opts.logY
      ? v => {
          const lv = Math.log10(Math.max(v, ys.min));
          const lo = Math.log10(ys.min), hi = Math.log10(ys.max);
          return B - (lv - lo) / (hi - lo) * (B - T);
        }
      : v => B - (v - ys.min) / (ys.max - ys.min) * (B - T);

    ctx.lineWidth = 1;
    // 가로 격자
    const yticks = opts.logY
      ? logTicks(ys.min, ys.max)
      : linTicks(ys.min, ys.max, opts.yCount || 5);
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const t of yticks) {
      const y = yv(t);
      if (y < T - 1 || y > B + 1) continue;
      ctx.strokeStyle = C.grid;
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(R, y); ctx.stroke();
      ctx.fillStyle = C.text;
      ctx.fillText(opts.logY ? expLabel(t) : nice(t), L - 6, y);
    }
    // 세로 격자
    if (xs.ticks !== false) {
      const xticks = xs.values || linTicks(xs.min, xs.max, opts.xCount || 5);
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const t of xticks) {
        const x = xv(t);
        if (x < L - 1 || x > R + 1) continue;
        ctx.strokeStyle = C.grid;
        ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, B); ctx.stroke();
        ctx.fillStyle = C.text; ctx.fillText(nice(t), x, B + 5);
      }
    }
    ctx.strokeStyle = C.axis;
    ctx.beginPath(); ctx.moveTo(L, T); ctx.lineTo(L, B); ctx.lineTo(R, B); ctx.stroke();

    // 축 이름
    ctx.fillStyle = C.ink; ctx.font = '10.5px system-ui';
    if (opts.xLabel) { ctx.textAlign='center'; ctx.textBaseline='bottom'; ctx.fillText(opts.xLabel, (L+R)/2, h-1); }
    if (opts.yLabel) {
      ctx.save(); ctx.translate(9, (T+B)/2); ctx.rotate(-Math.PI/2);
      ctx.textAlign='center'; ctx.textBaseline='top'; ctx.fillText(opts.yLabel, 0, 0); ctx.restore();
    }
    ctx.font = '10px system-ui';
    return { xv, yv, L, R, T, B };
  }

  function linTicks(min, max, n) {
    const out = [];
    for (let i = 0; i <= n; i++) out.push(min + (max - min) * i / n);
    return out;
  }
  function logTicks(min, max) {
    const out = [];
    for (let e = Math.floor(Math.log10(min)); e <= Math.ceil(Math.log10(max)); e++) out.push(10 ** e);
    return out;
  }
  function expLabel(v) {
    const e = Math.round(Math.log10(v));
    const sup = String(e).replace('-', '⁻').replace(/\d/g, d => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d]);
    return '10' + sup;
  }

  /** 로그축 범위를 데이터에 맞춰 10의 거듭제곱 경계로 잡는다 */
  function logRange(vals) {
    const pos = vals.filter(v => v > 0 && isFinite(v));
    if (!pos.length) return { min: 1e-6, max: 1 };
    const lo = Math.min(...pos), hi = Math.max(...pos);
    return {
      min: 10 ** Math.floor(Math.log10(lo)),
      max: 10 ** Math.ceil(Math.log10(hi))
    };
  }

  /* =========================================================
     1. 데이터 산점도
     ========================================================= */
  function scatter(cv, data) {
    const S = setup(cv, 0.82); if (!S) return; const { ctx, w, h } = S;
    const d = data.domain;
    const A = axes(ctx, w, h, { l:32, r:10, t:8, b:24 },
      { min:d.min, max:d.max, values:[-2,-1,0,1,2] },
      { min:d.min, max:d.max },
      { yCount:4, xLabel:'x₁', yLabel:'x₂' });
    points(ctx, A, data.X, data.y);
  }

  function points(ctx, A, X, y, r = 3) {
    for (let i = 0; i < X.length; i++) {
      ctx.beginPath();
      ctx.arc(A.xv(X[i][0]), A.yv(X[i][1]), r, 0, 6.2832);
      ctx.fillStyle = y[i] ? C.c1 : C.c0;
      ctx.fill();
      ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(255,255,255,.92)'; ctx.stroke();
    }
  }

  /* =========================================================
     2. 결정 경계 — 격자 확률을 히트맵으로 깔고 점을 얹는다
     ========================================================= */
  function decision(cv, data, net, res = 64) {
    const S = setup(cv, 0.82); if (!S) return; const { ctx, w, h } = S;
    const d = data.domain;
    const box = { l:32, r:46, t:8, b:24 };
    const A = axes(ctx, w, h, box,
      { min:d.min, max:d.max, values:[-2,-1,0,1,2] },
      { min:d.min, max:d.max },
      { yCount:4, xLabel:'x₁', yLabel:'x₂' });

    if (net) {
      const grid = net.decisionGrid(d, res);
      // 오프스크린에 res×res로 찍고 확대 — 부드러운 그라데이션
      const off = document.createElement('canvas');
      off.width = off.height = res;
      const octx = off.getContext('2d');
      const img = octx.createImageData(res, res);
      let k = 0;
      for (let r = 0; r < res; r++) {
        for (let c = 0; c < res; c++) {
          const [R, G, B] = colormap(grid[r][c]);
          img.data[k++] = R; img.data[k++] = G; img.data[k++] = B; img.data[k++] = 255;
        }
      }
      octx.putImageData(img, 0, 0);
      ctx.save();
      ctx.beginPath(); ctx.rect(A.L, A.T, A.R - A.L, A.B - A.T); ctx.clip();
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, A.L, A.T, A.R - A.L, A.B - A.T);
      ctx.restore();
      colorbar(ctx, A.R + 10, A.T, 11, A.B - A.T);
    }

    points(ctx, A, data.X, data.y);
    ctx.strokeStyle = C.axis; ctx.lineWidth = 1;
    ctx.strokeRect(A.L, A.T, A.R - A.L, A.B - A.T);
  }

  /** 0=파랑 · 0.5=흰색 · 1=빨강
      데이터 점이 묻히지 않도록 양 끝을 연하게 잡았다 */
  function colormap(t) {
    t = Math.min(1, Math.max(0, t));
    const lo = [132,172,238], mid = [248,250,252], hi = [240,146,146];
    const [a, b, s] = t < 0.5 ? [lo, mid, t * 2] : [mid, hi, (t - 0.5) * 2];
    return [0,1,2].map(i => Math.round(a[i] + (b[i] - a[i]) * s));
  }

  function colorbar(ctx, x, y, w, h) {
    for (let i = 0; i < h; i++) {
      const [R, G, B] = colormap(1 - i / (h - 1));
      ctx.fillStyle = `rgb(${R},${G},${B})`;
      ctx.fillRect(x, y + i, w, 1);
    }
    ctx.strokeStyle = C.axis; ctx.lineWidth = 1; ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText('1.0', x + w + 3, y + 4);
    ctx.fillText('0.0', x + w + 3, y + h - 4);
  }

  /* =========================================================
     3. 학습 곡선 (Loss는 로그축 / Accuracy는 0~1)
     ========================================================= */
  function curve(cv, history, mode = 'loss', totalEpochs) {
    const S = setup(cv, 0.52); if (!S) return; const { ctx, w, h } = S;
    const series = mode === 'loss' ? history.loss : history.accuracy;
    const n = series.length;
    const xmax = Math.max(totalEpochs || 0, n, 10);

    if (!n) { empty(ctx, w, h, '학습을 시작하면 표시됩니다'); return; }

    const isLoss = mode === 'loss';
    const ys = isLoss ? logRange(series) : { min: 0, max: 1 };
    const A = axes(ctx, w, h, { l:46, r:12, t:22, b:26 },
      { min:0, max:xmax }, ys,
      { logY:isLoss, yCount:5, xCount:5,
        xLabel:'Epoch', yLabel:isLoss ? 'Loss' : 'Accuracy' });

    ctx.beginPath();
    ctx.strokeStyle = C.blue; ctx.lineWidth = 1.6;
    // 점이 많으면 솎아 그린다
    const stepN = Math.max(1, Math.floor(n / 900));
    for (let i = 0; i < n; i += stepN) {
      const x = A.xv(i + 1), y = A.yv(series[i]);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.lineTo(A.xv(n), A.yv(series[n - 1]));
    ctx.stroke();

    // 마지막 값 표시
    ctx.fillStyle = C.blue; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
    ctx.font = 'bold 11px system-ui';
    const last = series[n - 1];
    ctx.fillText(isLoss ? last.toFixed(4) : (last * 100).toFixed(1) + '%', A.R, A.T - 5);
    ctx.font = '10px system-ui';
  }

  /* =========================================================
     4. Layer별 Gradient Magnitude — 두 모델 이중 막대 (로그축)
     ========================================================= */
  function gradBars(cv, names, gA, gB, labelA, labelB) {
    const S = setup(cv, 0.52); if (!S) return; const { ctx, w, h } = S;
    if (!gA || !gA.length) { empty(ctx, w, h, '학습을 시작하면 표시됩니다'); return; }

    const all = [...gA, ...(gB || [])].filter(v => v > 0);
    const ys = logRange(all.length ? all : [1e-6, 1]);
    const items = [[C.blue, labelA], gB ? [C.orange, labelB] : null].filter(Boolean);
    const lg = reserveLegend(ctx, w, { l:52, r:12, t:10, b:30 }, items);
    const A = axes(ctx, w, h, lg.box,
      { min:0, max:names.length, ticks:false }, ys,
      { logY:true, yLabel:'Mean |Gradient|' });

    const slot = (A.R - A.L) / names.length;
    const bw = Math.min(26, slot * 0.33);

    for (let i = 0; i < names.length; i++) {
      const cx = A.L + slot * (i + 0.5);
      const draw = (v, off, color) => {
        if (!(v > 0)) return;
        const y = A.yv(v);
        ctx.fillStyle = color;
        ctx.fillRect(cx + off - bw / 2, y, bw, A.B - y);
      };
      draw(gA[i], gB ? -bw * 0.57 : 0, C.blue);
      if (gB) draw(gB[i], bw * 0.57, C.orange);

      ctx.fillStyle = C.ink; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText(names[i], cx, A.B + 6);
    }
    legend(ctx, A.L, lg.rows);
  }

  /* =========================================================
     5. Layer별 Gradient 변화 (epoch 축, 로그)
     ========================================================= */
  function gradLines(cv, history, names, totalEpochs) {
    const S = setup(cv, 0.52); if (!S) return; const { ctx, w, h } = S;
    const G = history.gradPerLayer;
    if (!G.length) { empty(ctx, w, h, '학습을 시작하면 표시됩니다'); return; }

    const flat = [];
    for (const row of G) for (const v of row) if (v > 0) flat.push(v);
    const ys = logRange(flat);
    const xmax = Math.max(totalEpochs || 0, G.length, 10);

    const lg = reserveLegend(ctx, w, { l:52, r:12, t:10, b:26 },
      names.map((n, i) => [C.layer[i % C.layer.length], n]), true);
    const A = axes(ctx, w, h, lg.box,
      { min:0, max:xmax }, ys,
      { logY:true, xCount:5, xLabel:'Epoch', yLabel:'Mean |Gradient|' });

    const stepN = Math.max(1, Math.floor(G.length / 700));
    for (let l = 0; l < names.length; l++) {
      ctx.beginPath();
      ctx.strokeStyle = C.layer[l % C.layer.length];
      ctx.lineWidth = 1.4;
      let started = false;
      for (let i = 0; i < G.length; i += stepN) {
        const v = G[i][l];
        if (!(v > 0)) continue;
        const x = A.xv(i + 1), y = A.yv(v);
        started ? ctx.lineTo(x, y) : (ctx.moveTo(x, y), started = true);
      }
      ctx.stroke();
    }
    legend(ctx, A.L, lg.rows);
  }

  /* =========================================================
     6. 신경망 구조도
     ========================================================= */
  function network(cv, layerSizes) {
    const S = setup(cv, 0.82); if (!S) return; const { ctx, w, h } = S;
    const pad = { l:26, r:26, t:26, b:18 };
    const cols = layerSizes.length;
    const MAXDOT = 9;                       // 이보다 많으면 생략 표시

    const xs = cols === 1
      ? [w / 2]
      : layerSizes.map((_, i) => pad.l + i * (w - pad.l - pad.r) / (cols - 1));

    const coords = layerSizes.map((size, i) => {
      const shown = Math.min(size, MAXDOT);
      const gap = Math.min(26, (h - pad.t - pad.b) / Math.max(shown, 1));
      const y0 = (h + pad.t - pad.b) / 2 - (shown - 1) * gap / 2;
      return Array.from({ length: shown }, (_, k) => ({ x: xs[i], y: y0 + k * gap }));
    });

    // 연결선
    ctx.strokeStyle = 'rgba(148,163,184,.33)'; ctx.lineWidth = .6;
    for (let i = 0; i < coords.length - 1; i++) {
      for (const a of coords[i]) for (const b of coords[i + 1]) {
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
    }
    // 노드
    for (let i = 0; i < coords.length; i++) {
      const isIn = i === 0, isOut = i === coords.length - 1;
      const fill = isIn ? '#dbeafe' : isOut ? '#dcfce7' : '#fed7aa';
      const line = isIn ? '#60a5fa' : isOut ? '#4ade80' : '#fb923c';
      for (const p of coords[i]) {
        ctx.beginPath(); ctx.arc(p.x, p.y, 7.5, 0, 6.2832);
        ctx.fillStyle = fill; ctx.fill();
        ctx.strokeStyle = line; ctx.lineWidth = 1.4; ctx.stroke();
      }
      if (layerSizes[i] > MAXDOT) {
        const last = coords[i][coords[i].length - 1];
        ctx.fillStyle = C.text; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        ctx.fillText('⋮', last.x, last.y + 10);
      }
      // 라벨
      const name = isIn ? 'Input' : isOut ? 'Output' : `Hidden ${i}`;
      ctx.fillStyle = C.ink; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      ctx.font = '10.5px system-ui';
      ctx.fillText(name, xs[i], pad.t - 12);
      ctx.fillStyle = C.text;
      ctx.fillText(`(${layerSizes[i]})`, xs[i], pad.t - 1);
      ctx.font = '10px system-ui';
    }
  }

  /* =========================================================
     7. Activation 분포 — 층마다 작은 히스토그램
     ========================================================= */
  function activations(cv, stats, aspect = 0.52) {
    const S = setup(cv, aspect); if (!S) return; const { ctx, w, h } = S;
    if (!stats || !stats.length) { empty(ctx, w, h, '학습을 시작하면 표시됩니다'); return; }

    const n = stats.length;
    const gap = 10;
    const cw = (w - gap * (n + 1)) / n;
    const top = 34, bot = 26;

    for (let i = 0; i < n; i++) {
      const x0 = gap + i * (cw + gap);
      const s = stats[i];
      const lo = Math.min(0, s.min), hi = Math.max(1, s.max);
      const BINS = 18;
      const bins = new Array(BINS).fill(0);
      for (const v of s.values) {
        let k = Math.floor((v - lo) / (hi - lo) * BINS);
        k = Math.min(BINS - 1, Math.max(0, k));
        bins[k]++;
      }
      const mx = Math.max(...bins, 1);
      const H = h - top - bot;

      ctx.fillStyle = '#fafbfc'; ctx.fillRect(x0, top, cw, H);
      const bw = cw / BINS;
      for (let k = 0; k < BINS; k++) {
        const bh = bins[k] / mx * H;
        ctx.fillStyle = 'rgba(59,130,246,.72)';
        ctx.fillRect(x0 + k * bw, top + H - bh, Math.max(1, bw - 0.7), bh);
      }
      ctx.strokeStyle = C.axis; ctx.lineWidth = 1; ctx.strokeRect(x0, top, cw, H);

      ctx.fillStyle = C.ink; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      ctx.font = '10.5px system-ui';
      ctx.fillText(s.isOutput ? 'Output' : `Layer ${s.layer}`, x0 + cw / 2, top - 16);
      ctx.font = '9.5px system-ui'; ctx.fillStyle = C.text;
      ctx.fillText(`μ=${s.mean.toFixed(2)} σ=${s.std.toFixed(2)}`, x0 + cw / 2, top - 4);
      ctx.textBaseline = 'top';
      ctx.textAlign = 'left';  ctx.fillText(nice(lo), x0, top + H + 4);
      ctx.textAlign = 'right'; ctx.fillText(nice(hi), x0 + cw, top + H + 4);
      ctx.font = '10px system-ui';
    }
    ctx.fillStyle = C.ink; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText('Activation', w / 2, h - 1);
  }

  /* =========================================================
     8. 여러 모델의 곡선 겹쳐 그리기 (실험 탭용)
        series = [{label, values, color}]
     ========================================================= */
  function multiCurve(cv, series, opts = {}) {
    const S = setup(cv, opts.aspect || 0.46); if (!S) return; const { ctx, w, h } = S;
    const valid = series.filter(s => s.values && s.values.length);
    if (!valid.length) { empty(ctx, w, h, opts.emptyMsg || '실행하면 표시됩니다'); return; }

    const logY = opts.logY !== false;
    const flat = [];
    for (const s of valid) for (const v of s.values) if (!logY || v > 0) flat.push(v);
    const ys = logY ? logRange(flat)
                    : { min: Math.min(0, ...flat), max: Math.max(1, ...flat) };
    const xmax = Math.max(opts.xmax || 0, ...valid.map(s => s.values.length), 10);

    const lg = reserveLegend(ctx, w, { l:52, r:12, t:10, b:26 },
      valid.map((s, i) => [s.color || C.layer[i % C.layer.length], s.label]), true);
    const A = axes(ctx, w, h, lg.box,
      { min:0, max:xmax }, ys,
      { logY, xCount:5, yCount:5, xLabel:'Epoch', yLabel:opts.yLabel || 'Loss' });

    valid.forEach((s, i) => {
      const color = s.color || C.layer[i % C.layer.length];
      const n = s.values.length;
      const stepN = Math.max(1, Math.floor(n / 700));
      ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 1.5;
      let started = false;
      for (let k = 0; k < n; k += stepN) {
        const v = s.values[k];
        if (logY && !(v > 0)) continue;
        const x = A.xv(k + 1), y = Math.max(A.T, Math.min(A.B, A.yv(v)));
        started ? ctx.lineTo(x, y) : (ctx.moveTo(x, y), started = true);
      }
      ctx.stroke();
    });

    legend(ctx, A.L, lg.rows);
  }

  /* =========================================================
     9. 층 위치별 Gradient 프로파일 (Experiment 3 전용)
        층 수가 서로 다른 모델을 한 그래프에서 비교한다.
     ========================================================= */
  function gradProfile(cv, series, opts = {}) {
    const S = setup(cv, opts.aspect || 0.46); if (!S) return; const { ctx, w, h } = S;
    const valid = series.filter(s => s.grad && s.grad.length);
    if (!valid.length) { empty(ctx, w, h, '실행하면 표시됩니다'); return; }

    const flat = [];
    for (const s of valid) for (const v of s.grad) if (v > 0) flat.push(v);
    const ys = logRange(flat);
    const maxLen = Math.max(...valid.map(s => s.grad.length));
    const far = maxLen - 1;          // 출력층에서 가장 먼 거리

    // x = 출력층으로부터의 거리. 축을 뒤집어 출력층(0)을 오른쪽에 둔다.
    // 깊이가 다른 모델도 출력층이 한 점에 모여 비교가 맞아떨어진다.
    const lg = reserveLegend(ctx, w, { l:52, r:12, t:10, b:30 },
      valid.map(s => [s.color, s.label]), true);
    const A = axes(ctx, w, h, lg.box,
      { min: far, max: 0, values: Array.from({ length: maxLen }, (_, i) => far - i) },
      ys, { logY:true, yLabel:'Mean |Gradient|' });

    ctx.fillStyle = C.ink; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText('출력층으로부터의 거리 (0 = 출력층, 왼쪽일수록 입력에 가까움)', (A.L + A.R) / 2, h - 1);

    valid.forEach(s => {
      const L = s.grad.length;
      ctx.beginPath(); ctx.strokeStyle = s.color; ctx.lineWidth = 1.7;
      let started = false;
      s.grad.forEach((v, i) => {
        if (!(v > 0)) return;
        const x = A.xv(L - 1 - i), y = A.yv(v);
        started ? ctx.lineTo(x, y) : (ctx.moveTo(x, y), started = true);
      });
      ctx.stroke();
      ctx.fillStyle = s.color;
      s.grad.forEach((v, i) => {
        if (!(v > 0)) return;
        ctx.beginPath(); ctx.arc(A.xv(L - 1 - i), A.yv(v), 2.6, 0, 6.2832); ctx.fill();
      });
    });

    legend(ctx, A.L, lg.rows);
  }

  /* =========================================================
     10. 그룹 막대 (죽은 뉴런 비율 등 0~1 비율용, 선형 축)
         groups = [{label, values:[...], color}]
     ========================================================= */
  function groupBars(cv, categories, groups, opts = {}) {
    const S = setup(cv, opts.aspect || 0.42); if (!S) return;
    const { ctx, w, h } = S;
    if (!groups.length) { empty(ctx, w, h, '실행하면 표시됩니다'); return; }

    const lg = reserveLegend(ctx, w, { l:48, r:12, t:10, b:30 },
      groups.map((g, k) => [g.color || C.layer[k % C.layer.length], g.label]), true);
    const A = axes(ctx, w, h, lg.box,
      { min:0, max:categories.length, ticks:false },
      { min:0, max:opts.yMax || 1 },
      { yCount:5, yLabel:opts.yLabel || '' });

    const slot = (A.R - A.L) / categories.length;
    const bw = Math.min(22, slot / (groups.length + 1.4));

    categories.forEach((cat, i) => {
      const cx = A.L + slot * (i + 0.5);
      groups.forEach((g, k) => {
        const v = g.values[i];
        if (v == null || !isFinite(v)) return;
        const off = (k - (groups.length - 1) / 2) * (bw + 2);
        const y = A.yv(v);
        ctx.fillStyle = g.color || C.layer[k % C.layer.length];
        ctx.fillRect(cx + off - bw / 2, y, bw, A.B - y);
        if (v > 0.02) {                       // 값이 보이게 라벨
          ctx.fillStyle = C.ink; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
          ctx.font = '9px system-ui';
          ctx.fillText(Math.round(v * 100) + '%', cx + off, y - 1);
          ctx.font = '10px system-ui';
        }
      });
      ctx.fillStyle = C.ink; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText(cat, cx, A.B + 6);
    });
    legend(ctx, A.L, lg.rows);
  }

  /* ---------- 공통 보조 ---------- */
  /* 범례는 그래프 영역 위에 줄을 따로 잡아 그린다.
     안쪽에 그리면 위쪽에 붙는 선(발산한 Loss, 100% Accuracy 등)과 겹친다.
     폭이 모자라면 줄을 바꾼다. */
  const LEG_ROW = 14;
  function legendRows(ctx, items, maxW, compact) {
    ctx.font = '10px system-ui';
    const rows = [[]];
    let cx = 0;
    for (const [color, label] of items) {
      const iw = 11 + ctx.measureText(label).width;
      if (cx > 0 && cx + iw > maxW) { rows.push([]); cx = 0; }
      rows[rows.length - 1].push({ color, label, x: cx });
      cx += iw + (compact ? 9 : 14);
    }
    return rows;
  }
  // 범례 줄 수만큼 위 여백을 늘린 box 와 줄 배치를 돌려준다
  function reserveLegend(ctx, w, box, items, compact) {
    const rows = legendRows(ctx, items, w - box.l - box.r, compact);
    return { box: { ...box, t: box.t + rows.length * LEG_ROW }, rows };
  }
  function legend(ctx, x, rows) {
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.font = '10px system-ui';
    rows.forEach((row, r) => {
      const y = 2 + r * LEG_ROW;
      for (const it of row) {
        ctx.fillStyle = it.color; ctx.fillRect(x + it.x, y + 2, 8, 8);
        ctx.fillStyle = C.ink; ctx.fillText(it.label, x + it.x + 11, y + 1);
      }
    });
  }

  function empty(ctx, w, h, msg) {
    ctx.fillStyle = C.text; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(msg, w / 2, h / 2);
  }

  root.Viz = {
    setup, scatter, decision, curve, gradBars, gradLines,
    network, activations, colormap, multiCurve, gradProfile, groupBars,
    palette: C.layer
  };
})(typeof window !== 'undefined' ? window : globalThis);
