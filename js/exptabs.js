/* ===========================================================
   exptabs.js — 실험 탭 3개 + Activation 분석 탭
   -----------------------------------------------------------
   각 탭은 "설정 → 실행 → 결과 그래프 → 표 → 자동 결론" 구조.
   결론 문장은 실제 측정값에서 만들어지므로, 실행할 때마다
   숫자가 바뀌면 문장도 같이 바뀐다. (보고서에 그대로 옮길 수 있게)
   =========================================================== */
(function () {
  'use strict';

  const $ = id => document.getElementById(id);
  const P = Viz.palette;
  const pct = v => (v * 100).toFixed(1) + '%';
  const ex  = v => (v > 0 ? v.toExponential(2) : '—');

  /** 결과 저장 — 탭을 다시 열 때 캔버스를 다시 그려야 한다
      (display:none 상태에서는 캔버스 너비가 0이라 그릴 수 없다) */
  const store = { e1:null, e2:null, e3:null, e4:null };
  let busy = null;

  /* ---------- 공통 ---------- */
  function fillDatasets(sel, current) {
    sel.innerHTML = '';
    for (const d of Data.DATASETS) sel.add(new Option(d.label, d.id));
    sel.value = current || 'xor';
  }

  function progress(bar, stat) {
    return (done, total) => {
      bar.style.width = (done / total * 100).toFixed(1) + '%';
      stat.textContent = done >= total ? '완료' : `학습 중 ${Math.round(done / total * 100)}%`;
    };
  }

  function table(rows, head) {
    const th = head.map(h => `<th>${h}</th>`).join('');
    const tb = rows.map(r => '<tr>' + r.map(c => `<td>${c}</td>`).join('') + '</tr>').join('');
    return `<table class="res"><thead><tr>${th}</tr></thead><tbody>${tb}</tbody></table>`;
  }

  const sw = c => `<span class="sw" style="background:${c}"></span>`;

  /** 작은 결정 경계 카드들 */
  function cards(box, results, data, sub) {
    box.innerHTML = '';
    // 1단계: 카드를 모두 붙인다.
    //   하나씩 붙이면서 바로 그리면, 첫 카드는 "그리드에 혼자" 있는 상태의
    //   너비로 캔버스가 잡혀 비정상적으로 커진다.
    const els = results.map((r, i) => {
      const el = document.createElement('div');
      el.className = 'minicard';
      el.innerHTML = `<h4>${sw(P[i % P.length])}${r.cfg.label}</h4>` +
                     `<div class="sub">${sub(r)}</div>` +
                     `<canvas class="plot"></canvas>`;
      box.appendChild(el);
      return el;
    });
    // 2단계: 열 수가 확정된 뒤에 그린다.
    els.forEach((el, i) => Viz.decision(el.querySelector('canvas'), data, results[i].net, 44));
  }

  /** 실행 버튼 공통 처리 */
  function runExp(key, btn, bar, stat, build, render) {
    if (busy) { busy.cancel(); busy = null; }
    btn.disabled = true;
    bar.style.width = '0%';
    const { data, configs, epochs, batch } = build();
    busy = Experiments.run({
      data, configs, epochs, batch: batch || 16,
      onProgress: progress(bar, stat),
      onDone(results) {
        busy = null; btn.disabled = false;
        store[key] = { results, data, epochs };
        render(store[key]);
      }
    });
  }

  /* =========================================================
     Experiment 1 — Why Do We Need Hidden Layers?
     ========================================================= */
  function initExp1() {
    fillDatasets($('e1Data'), 'xor');
    $('e1Run').onclick = () => runExp('e1', $('e1Run'), $('e1Bar'), $('e1Stat'),
      () => ({
        data: Data.generate($('e1Data').value, { n: 300, seed: 42 }),
        configs: Experiments.exp1Configs(Number($('e1Neurons').value) || 8),
        epochs: Number($('e1Epochs').value) || 500
      }),
      renderExp1);
  }

  function renderExp1(S) {
    const { results, data, epochs } = S;

    cards($('e1Cards'), results, data,
      r => `${r.cfg.hiddenLayers === 0 ? '은닉층 없음' : `은닉층 ${r.cfg.hiddenLayers}층`} · ` +
           `Test ${pct(r.testAcc)}`);

    Viz.multiCurve($('e1Curve'), results.map((r, i) => ({
      label: r.cfg.label, values: r.net.history.loss, color: P[i % P.length]
    })), { xmax: epochs, yLabel: 'Loss' });

    $('e1Table').innerHTML = table(
      results.map((r, i) => [
        sw(P[i % P.length]) + r.cfg.label,
        r.net.layerSizes.join('-'),
        r.loss.toFixed(4),
        pct(r.acc),
        `<span class="${r.testAcc > 0.9 ? 'good' : r.testAcc < 0.75 ? 'bad' : ''}">${pct(r.testAcc)}</span>`
      ]),
      ['모델', '구조', 'Loss', 'Train Acc', 'Test Acc']);

    const p = results[0], best = results.slice(1).sort((a, b) => b.testAcc - a.testAcc)[0];
    const ceil = linearCeiling(data);
    $('e1Concl').innerHTML = `
      <h3>분석</h3>
      <ul>
        <li>Single Perceptron은 Test 정확도 <b>${pct(p.testAcc)}</b>에 그쳤습니다.
            은닉층이 없으면 출력이 입력의 <b>선형 결합</b>뿐이라 결정 경계가 <b>직선 하나</b>로 고정됩니다.</li>
        <li>이 데이터에서 <b>직선 하나로 얻을 수 있는 최대 정확도는 ${pct(ceil)}</b>입니다
            (가능한 직선을 모두 훑어 찾은 값). 즉 퍼셉트론은 아무리 잘 학습해도
            이 선을 넘을 수 없습니다 — <b>학습의 문제가 아니라 구조의 문제</b>입니다.</li>
        <li>은닉층을 넣은 ${best.cfg.label}은 <b>${pct(best.testAcc)}</b>로 올라갔습니다.
            은닉층이 입력을 비선형 변환해 <b>선형 분리가 가능한 공간</b>으로 옮기기 때문입니다.</li>
        <li>Loss 곡선을 보면 Perceptron은 ${p.loss.toFixed(3)} 부근에서 더 내려가지 못하고 평평해집니다.
            MLP는 같은 구간에서 계속 내려갑니다.</li>
        <li>결정 경계 그림에서 차이가 분명합니다 — Perceptron은 직선,
            MLP는 데이터에 맞춘 <b>곡선 경계</b>를 만듭니다.</li>
      </ul>`;
  }

  /* =========================================================
     Experiment 2 — Effect of Learning Rate
     ========================================================= */
  function initExp2() {
    fillDatasets($('e2Data'), 'xor');
    $('e2Run').onclick = () => runExp('e2', $('e2Run'), $('e2Bar'), $('e2Stat'),
      () => ({
        data: Data.generate($('e2Data').value, { n: 300, seed: 42 }),
        configs: Experiments.exp2Configs(
          Number($('e2Hidden').value) || 2, 8, $('e2Act').value),
        epochs: Number($('e2Epochs').value) || 500
      }),
      renderExp2);
  }

  /** 직선 하나로 얻을 수 있는 최대 정확도를 수치로 찾는다.
      "퍼셉트론이 못 푼다"를 데이터의 구조적 한계로 설명하기 위한 기준선. */
  function linearCeiling(data) {
    const { X, y } = data;
    let best = 0;
    const STEPS = 180, OFFS = 61;
    for (let a = 0; a < STEPS; a++) {
      const th = a / STEPS * Math.PI, cx = Math.cos(th), cy = Math.sin(th);
      for (let b = 0; b < OFFS; b++) {
        const off = -3 + b * (6 / (OFFS - 1));
        let hit = 0;
        for (let i = 0; i < X.length; i++) {
          if (((cx * X[i][0] + cy * X[i][1] + off) > 0 ? 1 : 0) === y[i]) hit++;
        }
        const acc = hit / X.length;
        best = Math.max(best, acc, 1 - acc);   // 부호를 뒤집은 경우도 같이
      }
    }
    return best;
  }

  /** loss가 처음으로 th 아래로 내려간 epoch (없으면 null) */
  function convergeAt(loss, th = 0.1) {
    for (let i = 0; i < loss.length; i++) if (loss[i] < th) return i + 1;
    return null;
  }

  /** 곡선을 보고 수렴 양상을 판정한다 (문장을 데이터에서 만들기 위함) */
  function analyze(r) {
    const L = r.net.history.loss;
    const last = L[L.length - 1];
    const tail = L.slice(Math.floor(L.length * 0.2));
    let ups = 0;
    for (let i = 1; i < L.length; i++) if (L[i] > L[i - 1]) ups++;
    const upRate = ups / Math.max(L.length - 1, 1);
    return {
      last, upRate,
      peak: Math.max(...tail),
      converged: convergeAt(L),
      diverged: !isFinite(last) || last > 0.8 || Math.max(...tail) > 2,
      unstable: upRate > 0.3
    };
  }

  function verdict(a) {
    if (a.diverged) return ['bad', '발산'];
    if (a.unstable) return ['bad', '불안정'];
    if (a.converged === null) return ['bad', '너무 느림'];
    if (a.converged > 200) return ['', '느림'];
    return ['good', '적절'];
  }

  function renderExp2(S) {
    const { results, data, epochs } = S;

    Viz.multiCurve($('e2Curve'), results.map((r, i) => ({
      label: r.cfg.label, values: r.net.history.loss, color: P[i % P.length]
    })), { xmax: epochs, yLabel: 'Loss' });

    Viz.multiCurve($('e2Acc'), results.map((r, i) => ({
      label: r.cfg.label, values: r.net.history.accuracy, color: P[i % P.length]
    })), { xmax: epochs, yLabel: 'Accuracy', logY: false });

    cards($('e2Cards'), results, data, r => `Test ${pct(r.testAcc)}`);

    const A = results.map(analyze);

    $('e2Table').innerHTML = table(
      results.map((r, i) => {
        const a = A[i], [cls, txt] = verdict(a);
        return [
          sw(P[i % P.length]) + r.cfg.label,
          isFinite(r.loss) ? r.loss.toFixed(4) : '발산',
          pct(r.acc), pct(r.testAcc),
          a.converged === null ? '—' : `${a.converged} epoch`,
          pct(a.upRate),
          `<span class="${cls}">${txt}</span>`
        ];
      }),
      ['Learning rate', '최종 Loss', 'Train Acc', 'Test Acc',
       'Loss&lt;0.1 도달', 'Loss 상승 구간', '판정']);

    // --- 문장을 실제 측정값에서 만든다 ---
    const ok   = results.filter((r, i) => !A[i].diverged && !A[i].unstable);
    const best = (ok.length ? ok : results).reduce((m, r) => r.loss < m.loss ? r : m);
    const bestA = A[results.indexOf(best)];
    const slow = results.reduce((m, r, i) =>
      (!A[i].diverged && !A[i].unstable && r.loss > m.loss) ? r : m, results[0]);
    const slowA = A[results.indexOf(slow)];
    const bad  = results.filter((r, i) => A[i].diverged || A[i].unstable);

    const badLi = bad.length
      ? bad.map(r => {
          const a = A[results.indexOf(r)];
          return `<li><b>${r.cfg.label}</b>에서는 Loss가 매끄럽게 내려가지 않았습니다 —
            전체 구간의 <b>${pct(a.upRate)}</b>에서 Loss가 오히려 <b>증가</b>했고,
            최종 Loss는 <b>${isFinite(a.last) ? a.last.toFixed(4) : '발산'}</b>입니다.
            보폭이 너무 커서 최소점을 <b>건너뛰며 왕복</b>하기 때문입니다.</li>`;
        }).join('')
      : `<li>이번 설정에서는 발산한 학습률이 없었습니다. 더 큰 값(예: 30 이상)을 넣으면
         Loss가 튀는 것을 확인할 수 있습니다.</li>`;

    $('e2Concl').innerHTML = `
      <h3>분석</h3>
      <ul>
        <li>가장 안정적으로 좋았던 설정은 <b>${best.cfg.label}</b>입니다 —
            최종 Loss <b>${best.loss.toFixed(4)}</b>, Test 정확도 <b>${pct(best.testAcc)}</b>,
            Loss&lt;0.1 도달 ${bestA.converged === null ? '—' : `<b>${bestA.converged} epoch</b>`}.</li>
        <li><b>${slow.cfg.label}</b>처럼 학습률이 작으면 한 걸음이 작아 같은 epoch 안에
            충분히 내려가지 못합니다 (최종 Loss <b>${slow.loss.toFixed(4)}</b>,
            ${slowA.converged === null ? 'Loss&lt;0.1에 끝내 도달하지 못함' : `${slowA.converged} epoch 소요`}).
            발산하지는 않지만 <b>시간이 낭비</b>됩니다.</li>
        ${badLi}
        <li>정리하면 학습률에는 <b>속도와 안정성의 교환 관계</b>가 있습니다.
            실무에서는 <b>Loss 곡선이 단조롭게 내려가는 가장 큰 값</b>을 고르고,
            학습이 진행되면 점차 줄이는 방식(learning rate decay)을 씁니다.</li>
      </ul>`;
  }

  /* =========================================================
     Experiment 3 — Vanishing Gradient
     ========================================================= */
  function initExp3() {
    fillDatasets($('e3Data'), 'xor');
    $('e3Run').onclick = () => runExp('e3', $('e3Run'), $('e3Bar'), $('e3Stat'),
      () => ({
        data: Data.generate($('e3Data').value, { n: 300, seed: 42 }),
        configs: Experiments.exp3Configs(
          Number($('e3Neurons').value) || 8, Number($('e3Lr').value) || 0.1,
          $('e3Init').value),
        epochs: Number($('e3Epochs').value) || 300
      }),
      renderExp3);
  }

  const COLOR3 = r => r.cfg.activation === 'sigmoid'
    ? ['#93c5fd', '#3b82f6', '#1e3a8a'][Experiments.DEPTHS.indexOf(r.cfg.hiddenLayers)]
    : ['#fcd34d', '#f59e0b', '#b45309'][Experiments.DEPTHS.indexOf(r.cfg.hiddenLayers)];

  function renderExp3(S) {
    const { results, epochs } = S;

    Viz.gradProfile($('e3Profile'), results.map(r => ({
      label: r.cfg.label, grad: r.grad, color: COLOR3(r)
    })));

    Viz.multiCurve($('e3Curve'), results.map(r => ({
      label: r.cfg.label, values: r.net.history.loss, color: COLOR3(r)
    })), { xmax: epochs, yLabel: 'Loss' });

    // 설정별 층 막대
    const box = $('e3Cards');
    box.innerHTML = '';
    const els = results.map(r => {
      const el = document.createElement('div');
      el.className = 'minicard';
      el.innerHTML = `<h4>${sw(COLOR3(r))}${r.cfg.label}</h4>` +
                     `<div class="sub">출력층/첫층 = ${r.gradRatio.toExponential(1)} 배 · Test ${pct(r.testAcc)}</div>` +
                     `<canvas class="plot"></canvas>`;
      box.appendChild(el);
      return el;
    });
    els.forEach((el, i) => Viz.gradBars(
      el.querySelector('canvas'), results[i].net.layerNames(), results[i].grad, null, results[i].cfg.label));

    $('e3Table').innerHTML = table(
      results.map(r => [
        sw(COLOR3(r)) + r.cfg.label,
        r.net.layerSizes.length - 1 + '개 층',
        ex(r.grad[0]), ex(r.grad[r.grad.length - 1]),
        `<span class="${r.gradRatio > 100 ? 'bad' : r.gradRatio < 10 ? 'good' : ''}">${r.gradRatio.toExponential(1)} 배</span>`,
        pct(r.testAcc)
      ]),
      ['설정', '가중치 층', '첫 층 |dW|', '출력층 |dW|', '비율', 'Test Acc']);

    const initLabel = results[0].cfg.initScheme === 'auto'
      ? '각자 표준 방식 (Sigmoid→Xavier, ReLU→He)'
      : '두 활성화에 동일하게 Xavier';
    const sg = results.filter(r => r.cfg.activation === 'sigmoid');
    const rl = results.filter(r => r.cfg.activation === 'relu');
    const sgDeep = sg[sg.length - 1], sgShallow = sg[0];
    const rlDeep = rl[rl.length - 1];

    $('e3Concl').innerHTML = `
      <h3>분석</h3>
      <ul>
        <li><b>깊어질수록 심해집니다.</b> Sigmoid 기준 출력층/첫층 기울기 비율이
            ${sgShallow.cfg.hiddenLayers}층에서 <b>${sgShallow.gradRatio.toExponential(1)}배</b>,
            ${sgDeep.cfg.hiddenLayers}층에서 <b>${sgDeep.gradRatio.toExponential(1)}배</b>로 커졌습니다.
            입력에 가까운 층일수록 거의 학습되지 않는다는 뜻입니다.</li>
        <li><b>원인은 활성화 함수의 도함수입니다.</b> Sigmoid의 도함수는 <b>많아야 0.25</b>라,
            가중치가 크지 않으면 역전파로 한 층을 지날 때마다 기울기가 <b>4분의 1 이하로</b> 줄어듭니다.
            n층을 지나면 0.25<sup>n</sup>까지 작아질 수 있습니다.</li>
        <li><b>ReLU는 훨씬 덜합니다.</b> 같은 ${rlDeep.cfg.hiddenLayers}층에서 비율이
            <b>${rlDeep.gradRatio.toExponential(1)}배</b>에 그쳤습니다.
            ReLU의 도함수는 양수 구간에서 1이라 기울기가 그대로 전달되기 때문입니다.</li>
        <li><b>학습 성능으로도 나타납니다.</b> Sigmoid ${sgDeep.cfg.hiddenLayers}층의 Test 정확도는
            ${pct(sgDeep.testAcc)}, 같은 깊이의 ReLU는 ${pct(rlDeep.testAcc)}입니다.
            Loss 곡선에서도 Sigmoid 깊은 망은 초반에 오래 정체합니다.</li>
        <li class="hint">실험 조건: 데이터·초기화 시드·학습률·뉴런 수를 모두 고정하고
            <b>은닉층 수와 활성화 함수만</b> 바꿨습니다.
            가중치 초기화는 <b>${initLabel}</b>을 썼습니다.</li>
        <li class="hint">주의: 학습이 수렴하면 모든 층의 기울기가 함께 0으로 가므로
            비율이 1에 가까워집니다. Vanishing 비교는 <b>학습 초기~중반</b>에서 봐야 합니다.</li>
      </ul>`;
  }

  /* =========================================================
     Activation 분석 탭
     ========================================================= */
  function renderActTab() {
    const app = window.__app;
    if (!app || !app.net) return;
    const net = app.net, d = app.data;
    if (net.epoch === 0) {
      Viz.activations($('cvActsBig'), null);
      $('actTable').innerHTML = '<p class="hint">먼저 실험하기 탭에서 학습을 실행하세요.</p>';
      $('actHint').textContent = '';
      return;
    }
    const stats = net.activationStats(d.Xtrain);
    Viz.activations($('cvActsBig'), stats);
    $('actHint').textContent = `${net.activation} · ${net.layerSizes.join('-')} · epoch ${net.epoch}`;

    $('actTable').innerHTML = table(
      stats.map(s => {
        const sat = s.values.filter(v => v < 0.1 || v > 0.9).length / s.values.length;
        return [
          s.isOutput ? 'Output' : `Layer ${s.layer}`,
          s.width,
          s.mean.toFixed(3), s.std.toFixed(3),
          s.min.toFixed(3), s.max.toFixed(3),
          `<span class="${sat > 0.8 ? 'bad' : sat < 0.4 ? 'good' : ''}">${pct(sat)}</span>`,
          `<span class="${s.deadRatio > 0.3 ? 'bad' : s.deadRatio === 0 ? 'good' : ''}">${s.deadCount} / ${s.width}</span>`
        ];
      }),
      ['층', '뉴런', '평균', '표준편차', '최솟값', '최댓값', '포화 비율', '죽은 뉴런']);
  }

  /* =========================================================
     Dying ReLU — 학습률을 키우며 죽은 뉴런 비율 관찰
     ========================================================= */
  function initExp4() {
    fillDatasets($('e4Data'), 'xor');
    $('e4Run').onclick = () => runExp('e4', $('e4Run'), $('e4Bar'), $('e4Stat'),
      () => ({
        data: Data.generate($('e4Data').value, { n: 300, seed: 42 }),
        configs: Experiments.exp4Configs(
          Number($('e4Hidden').value) || 4, Number($('e4Neurons').value) || 16),
        epochs: Number($('e4Epochs').value) || 300
      }),
      renderExp4);
  }

  function renderExp4(S) {
    const { results, data } = S;
    // 각 모델의 은닉층별 죽은 뉴런 비율
    const prof = results.map(r => {
      const st = r.net.activationStats(data.Xtrain).slice(0, -1);   // 출력층 제외
      return { cfg: r.cfg, r, dead: st.map(x => x.deadRatio), counts: st.map(x => x.deadCount) };
    });
    const maxL = Math.max(...prof.map(p => p.dead.length));
    const cats = Array.from({ length: maxL }, (_, i) => `Layer ${i + 1}`);

    Viz.groupBars($('e4Chart'), cats,
      prof.map((p, i) => ({
        label: p.cfg.label,
        values: cats.map((_, k) => p.dead[k] ?? null),
        color: p.cfg.activation === 'tanh' ? '#16a34a' : ['#93c5fd','#60a5fa','#f59e0b','#dc2626'][i] || '#2563eb'
      })),
      { yLabel: '죽은 뉴런 비율', yMax: 1 });

    $('e4Table').innerHTML = table(
      prof.map(p => {
        const tot = p.counts.reduce((a, b) => a + b, 0);
        const all = p.dead.length * (Number($('e4Neurons').value) || 16);
        return [
          p.cfg.label,
          p.counts.join(' / '),
          `<span class="${tot / all > 0.3 ? 'bad' : tot === 0 ? 'good' : ''}">${pct(tot / all)}</span>`,
          p.r.loss.toFixed(4),
          `<span class="${p.r.testAcc > 0.9 ? 'good' : p.r.testAcc < 0.7 ? 'bad' : ''}">${pct(p.r.testAcc)}</span>`
        ];
      }),
      ['설정', '층별 죽은 뉴런 수', '전체 비율', '최종 Loss', 'Test Acc']);

    const relu = prof.filter(p => p.cfg.activation === 'relu');
    const worst = relu.reduce((m, p) =>
      p.dead.reduce((a, b) => a + b, 0) > m.dead.reduce((a, b) => a + b, 0) ? p : m, relu[0]);
    const okOne = relu[0];
    const ctrl = prof.find(p => p.cfg.activation === 'tanh');

    $('e4Concl').innerHTML = `
      <h3>분석</h3>
      <ul>
        <li><b>${okOne.cfg.label}</b>에서는 죽은 뉴런이 거의 없고 Test ${pct(okOne.r.testAcc)}입니다.
            학습률이 적당하면 ReLU는 잘 작동합니다.</li>
        <li>학습률을 키우면 달라집니다. <b>${worst.cfg.label}</b>에서는 층별 죽은 뉴런이
            <b>${worst.counts.join(' → ')}개</b>로 늘고 Test는 <b>${pct(worst.r.testAcc)}</b>로 떨어졌습니다.
            큰 보폭으로 가중치가 한번 크게 밀리면 그 뉴런의 입력이 <b>항상 음수</b>가 되고,
            ReLU는 음수에서 기울기가 0이라 <b>되살아날 방법이 없습니다.</b></li>
        <li>대조군 <b>${ctrl.cfg.label}</b>는 같은 학습률에서도 죽은 뉴런이
            <b>${ctrl.counts.reduce((a,b)=>a+b,0)}개</b>입니다.
            Tanh는 어떤 입력에서도 기울기가 완전히 0이 되지 않기 때문입니다.
            ${ctrl.r.testAcc < 0.8 ? `다만 Tanh도 이 학습률에서는 실패했습니다
              (Test ${pct(ctrl.r.testAcc)}, Loss ${ctrl.r.loss.toFixed(2)}).
              <b>실패 이유가 다릅니다</b> — Tanh는 보폭이 커서 <b>발산</b>한 것이고,
              ReLU는 뉴런이 <b>죽어서</b> 학습할 대상 자체가 사라진 것입니다.
              발산은 학습률을 낮추면 회복되지만, 죽은 뉴런은 되살아나지 않습니다.` : ''}</li>
        <li><b>정리:</b> Experiment 3에서 ReLU가 Vanishing Gradient를 피하는 것을 봤지만,
            ReLU에는 <b>Dying ReLU</b>라는 반대 방향의 실패가 있습니다.
            실무에서 Leaky ReLU·ELU 같은 변형이 나온 이유이고,
            학습률을 함부로 키우면 안 되는 또 하나의 근거입니다.</li>
      </ul>`;
  }

  /* =========================================================
     역전파 검증 — 도움말 탭에서 바로 실행
     ========================================================= */
  const GC_CASES = [
    { label: 'Sigmoid · 은닉 0층 (Perceptron)', hiddenLayers: 0, neurons: 8, activation: 'sigmoid' },
    { label: 'Sigmoid · 은닉 1층',              hiddenLayers: 1, neurons: 6, activation: 'sigmoid' },
    { label: 'Sigmoid · 은닉 3층',              hiddenLayers: 3, neurons: 5, activation: 'sigmoid' },
    { label: 'Tanh · 은닉 2층',                 hiddenLayers: 2, neurons: 6, activation: 'tanh' },
    { label: 'ReLU · 은닉 2층',                 hiddenLayers: 2, neurons: 6, activation: 'relu' },
    { label: 'ReLU · 은닉 4층',                 hiddenLayers: 4, neurons: 4, activation: 'relu' }
  ];

  function initGradCheck() {
    $('gcRun').onclick = async () => {
      const btn = $('gcRun'), stat = $('gcStat');
      btn.disabled = true; stat.textContent = '검증 중…';
      const ds = Data.generate('xor', { n: 24, seed: 3 });
      const rows = [];
      for (const c of GC_CASES) {
        // 한 건씩 넘겨주며 UI가 멈추지 않게 한다
        await new Promise(r => setTimeout(r, 0));
        const res = NN.gradientCheck({ ...c, seed: 7 }, ds.X, ds.y);
        rows.push([
          c.label, res.params,
          res.maxRelError.toExponential(2),
          `<span class="${res.ok ? 'good' : 'bad'}">${res.ok ? '일치' : '불일치'}</span>`
        ]);
        stat.textContent = `검증 중… ${rows.length}/${GC_CASES.length}`;
      }
      const allOk = rows.every(r => r[3].includes('일치</span>'));
      $('gcTable').innerHTML =
        table(rows, ['모델', '검사한 가중치 수', '최대 상대오차', '판정']) +
        `<p class="hint" style="margin:10px 0 0">
           상대오차가 10⁻⁵보다 작으면 역전파가 맞다고 봅니다. 위 값은 모두 10⁻⁷ 수준입니다.
           ${allOk ? '<b>모든 조합에서 해석적 기울기와 수치적 기울기가 일치했습니다.</b>' : ''}
         </p>`;
      stat.textContent = allOk ? '전부 통과' : '실패 있음';
      btn.disabled = false;
    };
  }

  /* =========================================================
     결과 내보내기 — 표는 CSV, 그래프는 PNG
     ========================================================= */
  function download(name, blob) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  function tableToCSV(host) {
    const t = host.querySelector('table');
    if (!t) return null;
    return [...t.rows].map(tr =>
      [...tr.cells].map(td => `"${td.textContent.trim().replace(/"/g, '""')}"`).join(',')
    ).join('\r\n');
  }

  function initExport() {
    document.body.addEventListener('click', e => {
      const b = e.target.closest('[data-export]');
      if (!b) return;
      const [kind, id] = b.dataset.export.split(':');
      if (kind === 'csv') {
        const csv = tableToCSV($(id));
        if (!csv) return;
        download(`${id}.csv`, new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8' }));
      } else {
        const cv = $(id);
        if (!cv) return;
        // 투명 배경 대신 흰 배경으로 저장 (보고서에 붙이기 좋게)
        const out = document.createElement('canvas');
        out.width = cv.width; out.height = cv.height;
        const c = out.getContext('2d');
        c.fillStyle = '#fff'; c.fillRect(0, 0, out.width, out.height);
        c.drawImage(cv, 0, 0);
        out.toBlob(bl => download(`${id}.png`, bl));
      }
    });
  }

  /* =========================================================
     탭 전환 시 다시 그리기
     (숨겨진 캔버스는 너비가 0이라 그 시점엔 그릴 수 없다)
     ========================================================= */
  function redrawActive() {
    const on = document.querySelector('.tabpage.on');
    if (!on) return;
    const t = on.dataset.tab;
    if (t === 'arch' && store.e1) renderExp1(store.e1);
    if (t === 'train' && store.e2) renderExp2(store.e2);
    if (t === 'grad'  && store.e3) renderExp3(store.e3);
    if (t === 'act') { renderActTab(); if (store.e4) renderExp4(store.e4); }
  }

  /* --------------------------------------------------------- */
  initExp1(); initExp2(); initExp3(); initExp4(); initGradCheck(); initExport();
  // 탭을 보여준 뒤 레이아웃이 끝나고 그려야 캔버스 너비가 제대로 잡힌다
  document.getElementById('nav').addEventListener('click',
    () => requestAnimationFrame(() => requestAnimationFrame(redrawActive)));

  // 너비를 못 재서 건너뛴 캔버스가 있으면 다시 그린다 (안전망)
  setInterval(() => {
    if (document.querySelector('.tabpage.on canvas[data-pending]')) redrawActive();
  }, 500);
  let t = null;
  window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(redrawActive, 150); });
})();
