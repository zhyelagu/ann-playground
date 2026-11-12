/* ===========================================================
   app.js — 상태 관리 / 컨트롤 연결 / 학습 루프 / 화면 갱신
   -----------------------------------------------------------
   설계 메모
   · 모델을 둘 돌린다.
       net        : 사용자가 고른 활성화 함수 (화면의 "현재")
       netCompare : 비교군 (현재가 ReLU면 Sigmoid, 아니면 ReLU)
     같은 데이터·같은 시드·같은 LR로 나란히 학습시켜야
     Experiment 3(Sigmoid vs ReLU)의 비교가 공정해진다.
   · 학습은 매 epoch, 무거운 차트는 REDRAW_EVERY epoch마다.
     (결정 경계는 격자 전체를 순전파하므로 매번 그리면 멈춘다)
   =========================================================== */
(function () {
  'use strict';

  const $ = id => document.getElementById(id);
  const UI = {
    dataset:$('dataset'), thumbs:$('thumbs'),
    nSamples:$('nSamples'), nSamplesV:$('nSamplesV'),
    noise:$('noise'), noiseV:$('noiseV'),
    hidden:$('hidden'), hiddenV:$('hiddenV'),
    neurons:$('neurons'), neuronsV:$('neuronsV'),
    lr:$('lr'), batch:$('batch'), epochs:$('epochs'),
    run:$('btnRun'), pause:$('btnPause'), reset:$('btnReset'),
    sEpoch:$('sEpoch'), sLoss:$('sLoss'), sAcc:$('sAcc'),
    sTest:$('sTest'), sState:$('sState'),
    cvData:$('cvData'), cvBoundary:$('cvBoundary'), cvNetwork:$('cvNetwork'),
    cvCurve:$('cvCurve'), cvGradBars:$('cvGradBars'),
    cvGradLines:$('cvGradLines'), cvActs:$('cvActs'),
    curveToggle:$('curveToggle'), archHint:$('archHint'),
    cmpOn:$('cmpOn'), costWarn:$('costWarn'), share:$('btnShare'), realNote:$('realNote'),
    debug:$('debug'), nav:$('nav')
  };

  const MODEL_SEED = 11;      // 가중치 초기화 — 고정해야 실험이 재현된다
  const DATA_SEED  = 42;
  // 차트 갱신 주기 — 비용이 다르므로 나눠서 관리한다
  const REDRAW_LIGHT = 10;    // 곡선·막대 (싸다)
  const REDRAW_HEAVY = 40;    // 결정 경계·Activation (격자 전체 순전파)

  /** 결정 경계 해상도를 망 크기에 맞춰 낮춘다.
      큰 망에서는 격자 한 번 그리는 비용이 50ms를 넘는다. */
  function gridRes() {
    const cost = Math.max(1, state.hidden) * state.neurons * state.neurons;
    return cost > 4000 ? 32 : cost > 1200 ? 42 : 56;
  }

  const state = {
    dataset:'xor', nSamples:300, noise:0,
    hidden:3, neurons:8, activation:'sigmoid',
    lr:0.1, batch:16, epochs:1000,
    curveMode:'loss', compareOn:true,
    data:null, net:null, netCompare:null,
    running:false, timer:null, lastLight:-1, lastHeavy:-1,
    t0:0, e0:0
  };

  /* ---------------------------------------------------------
     설정을 URL에 담고/읽기
     발표 슬라이드에 "이 설정으로 보세요" 링크를 박아둘 수 있고,
     현장에서 슬라이더를 다시 맞추다 실수할 일이 없다.
     --------------------------------------------------------- */
  const URL_KEYS = {
    ds:'dataset', n:'nSamples', nz:'noise', h:'hidden', u:'neurons',
    act:'activation', lr:'lr', b:'batch', ep:'epochs', cmp:'compareOn'
  };

  function readURL() {
    const q = new URLSearchParams(location.search);
    for (const [k, prop] of Object.entries(URL_KEYS)) {
      if (!q.has(k)) continue;
      const raw = q.get(k);
      if (prop === 'dataset') {
        if (Data.DATASETS.some(d => d.id === raw)) state.dataset = raw;
      } else if (prop === 'activation') {
        if (NN.ACTIVATIONS[raw]) state.activation = raw;
      } else if (prop === 'compareOn') {
        state.compareOn = raw !== '0';
      } else {
        const v = Number(raw);
        if (isFinite(v)) state[prop] = v;
      }
    }
  }

  /** 읽은 설정을 컨트롤에 반영 */
  function applyToControls() {
    UI.dataset.value = state.dataset; syncDatasetUI();
    UI.nSamples.value = state.nSamples; UI.nSamplesV.textContent = state.nSamples;
    UI.noise.value = state.noise;       UI.noiseV.textContent = state.noise.toFixed(2);
    UI.hidden.value = state.hidden;     UI.hiddenV.textContent = state.hidden;
    UI.neurons.value = state.neurons;   UI.neuronsV.textContent = state.neurons;
    UI.lr.value = state.lr; UI.batch.value = state.batch; UI.epochs.value = state.epochs;
    UI.cmpOn.checked = state.compareOn;
    const r = document.querySelector(`input[name=act][value=${state.activation}]`);
    if (r) r.checked = true;
  }

  function shareURL() {
    const q = new URLSearchParams();
    for (const [k, prop] of Object.entries(URL_KEYS)) {
      q.set(k, prop === 'compareOn' ? (state[prop] ? '1' : '0') : state[prop]);
    }
    return location.origin + location.pathname + '?' + q.toString();
  }

  /* ---------------------------------------------------------
     데이터 / 모델
     --------------------------------------------------------- */
  function buildData() {
    state.data = Data.generate(state.dataset, {
      n: state.nSamples, noise: state.noise, seed: DATA_SEED
    });
  }

  function buildNets() {
    const cfg = {
      hiddenLayers: state.hidden,
      neurons: state.neurons,
      learningRate: state.lr,
      seed: MODEL_SEED,
      // 두 모델에 같은 초기화를 써야 "활성화 함수만 다른" 비교가 된다.
      // (Gradient 분석 탭의 기본 설정과 동일하게 맞춘다)
      initScheme: 'xavier'
    };
    state.net = new NN.NeuralNetwork({ ...cfg, activation: state.activation });
    state.netCompare = new NN.NeuralNetwork({
      ...cfg, activation: state.activation === 'relu' ? 'sigmoid' : 'relu'
    });
  }

  /** 실제 데이터는 샘플 수·노이즈가 적용되지 않으므로 컨트롤을 잠근다 */
  function syncRealDataControls() {
    const real = Data.isReal(state.dataset);
    for (const el of [UI.nSamples, UI.noise]) {
      el.disabled = real;
      el.closest('.field').style.opacity = real ? .45 : '';
    }
    if (real && state.data && state.data.source) {
      const s = state.data.source;
      UI.nSamplesV.textContent = state.data.X.length;
      UI.noiseV.textContent = '—';
      UI.realNote.innerHTML =
        `실제 데이터입니다. 샘플 수·노이즈는 적용되지 않습니다.<br>` +
        `<b>Breast Cancer Wisconsin</b> · ${state.data.X.length}건 (악성 ${s.positive}건)<br>` +
        `특성: ${s.feature_x} / ${s.feature_y}`;
    } else {
      UI.nSamplesV.textContent = state.nSamples;
      UI.noiseV.textContent = state.noise.toFixed(2);
      UI.realNote.innerHTML = '';
    }
  }

  function rebuildAll() {
    stop();
    buildData();
    syncRealDataControls();
    buildNets();
    state.lastLight = state.lastHeavy = -1;
    UI.debug.textContent = '학습을 시작하면 표시됩니다.';
    setState('대기');
    updateStats();
    drawAll();
    updateCostWarning();
  }

  /** 설정이 무거우면 미리 알려준다.
      기기·활성화 함수에 따라 실제 속도가 2~3배까지 달라져서
      초 단위로 예측하지 않고, 무거운 조합일 때만 정성적으로 경고한다.
      (정확한 남은 시간은 학습이 시작되면 실측으로 표시한다) */
  function updateCostWarning() {
    const n = state.data ? state.data.Xtrain.length : 0;
    const L = Math.max(1, state.hidden);
    const score = L * state.neurons * state.neurons * n
                  * (state.compareOn ? 2 : 1)
                  * (state.batch === 1 ? 3 : 1);
    UI.costWarn.textContent = score > 2e6
      ? '설정이 무거워 학습이 오래 걸립니다.' +
        (state.compareOn ? ' "비교 모델 동시 학습"을 끄면 약 2배 빨라집니다.' : '') +
        ' 샘플 수나 뉴런 수를 줄여도 됩니다.'
      : '';
  }

  /* ---------------------------------------------------------
     학습 루프
     계산은 setTimeout(0)으로 돌린다. requestAnimationFrame을 쓰면
     화면 주사율(또는 브라우저의 프레임 제한)에 학습 속도가 묶여서,
     창이 가려지거나 느린 환경에서 학습이 거의 멈춘다.
     한 덩어리를 BUDGET_MS로 잘라 UI 응답성은 유지한다.
     --------------------------------------------------------- */
  const BUDGET_MS = 24;

  function step() {
    if (!state.running) return;
    const d = state.data;
    const t0 = performance.now();

    while (performance.now() - t0 < BUDGET_MS) {
      if (state.net.epoch >= state.epochs) { finish(); return; }
      state.net.trainEpoch(d.Xtrain, d.ytrain, state.batch);
      if (state.compareOn) state.netCompare.trainEpoch(d.Xtrain, d.ytrain, state.batch);
    }

    updateStats();
    const e = state.net.epoch;
    // 실측 속도로 남은 시간 표시
    const elapsed = (performance.now() - state.t0) / 1000;
    const doneE = e - state.e0;
    if (elapsed > 0.8 && doneE > 0) {
      const left = (state.epochs - e) / (doneE / elapsed);
      setState(left > 1 ? `학습 중 · 약 ${Math.ceil(left)}초 남음` : '학습 중');
    }
    if (e - state.lastLight >= REDRAW_LIGHT) drawLight();
    if (e - state.lastHeavy >= REDRAW_HEAVY) drawHeavy();
    state.timer = setTimeout(step, 0);
  }

  function start() {
    if (state.running) return;
    if (state.net.epoch >= state.epochs) rebuildAll();
    state.running = true;
    state.t0 = performance.now();
    state.e0 = state.net.epoch;
    setState('학습 중');
    UI.run.textContent = '▶ 학습 중…';
    state.timer = setTimeout(step, 0);
  }

  function stop() {
    state.running = false;
    if (state.timer) clearTimeout(state.timer);
    state.timer = null;
    UI.run.textContent = '▶ 학습 시작';
  }

  function finish() {
    stop();
    updateStats();
    drawAll();
    setState('완료');
  }

  const setState = s => { UI.sState.textContent = s; };

  /** 상단 수치는 "실험하기/Activation 분석"의 모델 것이다.
      실험 탭에는 자체 결과표가 있으므로 숨겨서 혼동을 막는다. */
  function syncStatsVisibility(tab) {
    const show = (tab === 'play' || tab === 'act');
    document.querySelector('.stats').style.display = show ? '' : 'none';
  }

  /* ---------------------------------------------------------
     상단 수치 + 수치 보기
     --------------------------------------------------------- */
  function updateStats() {
    const net = state.net, d = state.data;
    UI.sEpoch.textContent = net.epoch;

    const h = net.history;
    if (h.loss.length) {
      UI.sLoss.textContent = h.loss[h.loss.length - 1].toFixed(4);
      UI.sAcc.textContent  = (h.accuracy[h.accuracy.length - 1] * 100).toFixed(1) + '%';
      UI.sTest.textContent = (net.evaluate(d.Xtest, d.ytest).accuracy * 100).toFixed(1) + '%';
    } else {
      UI.sLoss.textContent = UI.sAcc.textContent = UI.sTest.textContent = '—';
    }

    if (net.lastGradients) {
      const names = net.layerNames();
      const g1 = net.lastGradients, g2 = state.netCompare.lastGradients || [];
      let out = `층        ${net.activation.padEnd(10)} ${state.netCompare.activation}\n`;
      for (let i = 0; i < g1.length; i++) {
        out += `${names[i].padEnd(9)} ${g1[i].toExponential(2).padEnd(10)} ${(g2[i] ?? 0).toExponential(2)}\n`;
      }
      out += `\n출력층 / 첫 층 = ${(g1[g1.length - 1] / Math.max(g1[0], 1e-300)).toExponential(2)} 배`;
      UI.debug.textContent = out;
    }
  }

  /* ---------------------------------------------------------
     전체 화면 갱신
     --------------------------------------------------------- */
  /** 싼 차트 — 이미 들고 있는 기록만 그린다 */
  function drawLight() {
    const net = state.net, cmp = state.netCompare;
    if (!net) return;
    state.lastLight = net.epoch;

    Viz.curve(UI.cvCurve, net.history, state.curveMode, state.epochs);
    const names = net.layerNames();
    Viz.gradBars(UI.cvGradBars, names,
      net.lastGradients, state.compareOn ? cmp.lastGradients : null,
      `${net.activation} (현재)`, `${cmp.activation} (비교)`);
    Viz.gradLines(UI.cvGradLines, net.history, names, state.epochs);
  }

  /** 비싼 차트 — 격자/전체 데이터를 다시 순전파해야 한다 */
  function drawHeavy() {
    const net = state.net, d = state.data;
    if (!d || !net) return;
    state.lastHeavy = net.epoch;

    Viz.decision(UI.cvBoundary, d, net, gridRes());
    Viz.activations(UI.cvActs, net.epoch > 0 ? net.activationStats(d.Xtrain) : null);
  }

  /** 거의 안 변하는 것 — 데이터셋/구조가 바뀔 때만 */
  function drawStatic() {
    const net = state.net, d = state.data;
    if (!d || !net) return;
    Viz.scatter(UI.cvData, d);
    Viz.network(UI.cvNetwork, net.layerSizes);
    UI.archHint.textContent = net.layerSizes.join(' → ');
  }

  function drawAll() { drawStatic(); drawLight(); drawHeavy(); }

  /* ---------------------------------------------------------
     데이터셋 썸네일
     --------------------------------------------------------- */
  function buildThumbs() {
    UI.dataset.innerHTML = '';
    UI.thumbs.innerHTML = '';
    for (const d of Data.DATASETS) {
      UI.dataset.add(new Option(d.label, d.id));

      const box = document.createElement('div');
      box.className = 'thumb' + (d.id === state.dataset ? ' on' : '');
      box.dataset.id = d.id;
      box.innerHTML = `<canvas width="120" height="90"></canvas>` +
                      `<span>${d.id[0].toUpperCase() + d.id.slice(1)}</span>`;
      box.onclick = () => { state.dataset = d.id; syncDatasetUI(); rebuildAll(); };
      UI.thumbs.appendChild(box);

      const c = box.querySelector('canvas').getContext('2d');
      const prev = Data.generate(d.id, { n: 90, seed: 7 });
      const m = Data.DOMAIN;
      for (let i = 0; i < prev.X.length; i++) {
        const x = (prev.X[i][0] - m.min) / (m.max - m.min) * 112 + 4;
        const y = (m.max - prev.X[i][1]) / (m.max - m.min) * 82 + 4;
        c.beginPath(); c.arc(x, y, 2, 0, 6.2832);
        c.fillStyle = prev.y[i] ? '#ef4444' : '#3b82f6'; c.fill();
      }
    }
    UI.dataset.value = state.dataset;
  }

  function syncDatasetUI() {
    UI.dataset.value = state.dataset;
    [...UI.thumbs.children].forEach(el =>
      el.classList.toggle('on', el.dataset.id === state.dataset));
  }

  /* ---------------------------------------------------------
     컨트롤
     --------------------------------------------------------- */
  function bind() {
    UI.dataset.onchange = e => { state.dataset = e.target.value; syncDatasetUI(); rebuildAll(); };

    const slider = (el, label, key, fmt) => {
      el.oninput = () => {
        state[key] = Number(el.value);
        label.textContent = fmt ? fmt(state[key]) : state[key];
        rebuildAll();
      };
    };
    slider(UI.nSamples, UI.nSamplesV, 'nSamples');
    slider(UI.noise,    UI.noiseV,    'noise', v => v.toFixed(2));
    slider(UI.hidden,   UI.hiddenV,   'hidden');
    slider(UI.neurons,  UI.neuronsV,  'neurons');

    document.querySelectorAll('input[name=act]').forEach(r => {
      r.onchange = () => { if (r.checked) { state.activation = r.value; rebuildAll(); } };
    });

    UI.lr.onchange     = () => { state.lr = Number(UI.lr.value) || 0.1; rebuildAll(); };
    UI.batch.onchange  = () => { state.batch = Number(UI.batch.value); rebuildAll(); };
    UI.epochs.onchange = () => {
      state.epochs = Math.max(1, Number(UI.epochs.value) || 1000);
      drawAll();
    };

    UI.cmpOn.onchange = () => { state.compareOn = UI.cmpOn.checked; rebuildAll(); };

    UI.share.onclick = async () => {
      const url = shareURL();
      const flash = m => {
        UI.share.textContent = m;
        setTimeout(() => UI.share.textContent = '🔗 설정 링크 복사', 1800);
      };
      // 1순위: 클립보드 API (https 또는 localhost에서 동작)
      try {
        await navigator.clipboard.writeText(url);
        flash('✅ 복사됨');
        return;
      } catch (_) { /* 권한 거부 또는 미지원 */ }
      // 2순위: 임시 textarea + execCommand (구형 경로)
      try {
        const ta = document.createElement('textarea');
        ta.value = url;
        ta.style.cssText = 'position:fixed;left:-9999px;opacity:0';
        document.body.appendChild(ta);
        ta.select();
        const ok = document.execCommand('copy');
        document.body.removeChild(ta);
        if (ok) { flash('✅ 복사됨'); return; }
      } catch (_) { /* 이것도 막힌 환경 */ }
      // 3순위: 화면에 띄워 직접 복사 (prompt는 차단되는 환경이 있다)
      UI.realNote.innerHTML =
        '자동 복사가 차단됐습니다. 아래 주소를 복사하세요.' +
        '<input readonly onclick="this.select()" ' +
        'style="width:100%;margin-top:5px;font-size:10.5px" value="' + url + '">';
      flash('↓ 아래에 표시됨');
    };

    UI.run.onclick   = start;
    UI.pause.onclick = () => { stop(); setState('일시정지'); };
    UI.reset.onclick = rebuildAll;

    // 학습 곡선 Loss / Accuracy 토글
    UI.curveToggle.onclick = e => {
      const b = e.target.closest('button[data-mode]');
      if (!b) return;
      state.curveMode = b.dataset.mode;
      [...UI.curveToggle.children].forEach(x => x.classList.toggle('on', x === b));
      Viz.curve(UI.cvCurve, state.net.history, state.curveMode, state.epochs);
    };

    // 탭
    UI.nav.onclick = e => {
      const b = e.target.closest('button[data-tab]');
      if (!b) return;
      [...UI.nav.children].forEach(x => x.classList.toggle('on', x === b));
      document.querySelectorAll('.tabpage').forEach(p =>
        p.classList.toggle('on', p.dataset.tab === b.dataset.tab));
      syncStatsVisibility(b.dataset.tab);
      requestAnimationFrame(() => requestAnimationFrame(drawAll));
    };

    let t = null;
    window.addEventListener('resize', () => {
      clearTimeout(t); t = setTimeout(drawAll, 120);
    });
  }

  /* --------------------------------------------------------- */
  buildThumbs();
  readURL();          // 주소에 설정이 있으면 먼저 반영
  bind();
  applyToControls();
  syncStatsVisibility(document.querySelector('.tabpage.on')?.dataset.tab || 'play');
  rebuildAll();
  window.__app = state;   // 콘솔 디버깅용
})();
