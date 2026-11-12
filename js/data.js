/* ===========================================================
   data.js — 2D 이진분류 데이터셋 생성
   -----------------------------------------------------------
   과제 요구: Linear, XOR, Spiral (+ 레퍼런스의 Circle)
   모든 좌표는 대략 [-2.5, 2.5] 범위에 들어오도록 맞춰서
   결정 경계 격자를 하나의 고정 범위로 그릴 수 있게 한다.
   =========================================================== */
(function (root) {
  'use strict';

  const makeRNG = root.makeRNG || require('./rng.js').makeRNG;

  /** 화면에 그릴 때 쓰는 고정 좌표 범위 */
  const DOMAIN = { min: -2.5, max: 2.5 };

  /* --------------------------------------------------------
     각 생성기는 {X: [[x1,x2], ...], y: [0|1, ...]} 를 돌려준다.
     -------------------------------------------------------- */

  /** 직선 하나로 나뉘는 두 덩어리 */
  function linear(n, noise, rng) {
    const X = [], y = [];
    for (let i = 0; i < n; i++) {
      const label = i % 2;                       // 두 클래스를 균등하게
      const cx = label ? 1.0 : -1.0;
      const cy = label ? 0.8 : -0.8;
      X.push([
        cx + rng.normal(0, 0.45 + noise),
        cy + rng.normal(0, 0.45 + noise)
      ]);
      y.push(label);
    }
    return { X, y };
  }

  /** XOR: 네 사분면, 대각선끼리 같은 클래스 */
  function xor(n, noise, rng) {
    const X = [], y = [];
    for (let i = 0; i < n; i++) {
      const sx = (i & 1) ? 1 : -1;
      const sy = (i & 2) ? 1 : -1;
      const x1 = sx * 1.2 + rng.normal(0, 0.32 + noise);
      const x2 = sy * 1.2 + rng.normal(0, 0.32 + noise);
      X.push([x1, x2]);
      // 같은 부호끼리(1사분면, 3사분면) 클래스 1
      y.push(sx * sy > 0 ? 1 : 0);
    }
    return { X, y };
  }

  /** 동심원: 안쪽 원 vs 바깥 고리 */
  function circle(n, noise, rng) {
    const X = [], y = [];
    for (let i = 0; i < n; i++) {
      const label = i % 2;
      const r = label
        ? rng.uniform(1.5, 2.2)                  // 바깥 고리
        : rng.uniform(0.0, 0.8);                 // 안쪽 원
      const t = rng.uniform(0, 2 * Math.PI);
      X.push([
        r * Math.cos(t) + rng.normal(0, 0.12 + noise),
        r * Math.sin(t) + rng.normal(0, 0.12 + noise)
      ]);
      y.push(label);
    }
    return { X, y };
  }

  /** 두 갈래 나선 (가장 어려운 데이터) */
  function spiral(n, noise, rng) {
    const X = [], y = [];
    const per = Math.floor(n / 2);
    for (let label = 0; label < 2; label++) {
      for (let i = 0; i < per; i++) {
        const r = (i / per) * 2.1;
        const t = 1.75 * (i / per) * 2 * Math.PI + label * Math.PI;
        X.push([
          r * Math.sin(t) + rng.normal(0, 0.1 + noise),
          r * Math.cos(t) + rng.normal(0, 0.1 + noise)
        ]);
        y.push(label);
      }
    }
    return { X, y };
  }

  /** 실제 의료 데이터 (Breast Cancer Wisconsin, 특성 2개 추출)
      합성 데이터와 달리 겹치는 구간이 있어 100%가 나오지 않는다.
      n·noise·seed 설정은 무시하고 전체를 그대로 쓴다. */
  function medical(n, noise, rng) {
    const M = root.MEDICAL;
    if (!M) throw new Error('medical.js가 로드되지 않았습니다');
    return { X: M.X.map(p => p.slice()), y: M.y.slice() };
  }

  const GENERATORS = { linear, xor, circle, spiral, medical };

  const DATASETS = [
    { id: 'linear', label: 'Linear (선형 분류)' },
    { id: 'xor',    label: 'XOR (비선형 분류)' },
    { id: 'circle', label: 'Circle (동심원)' },
    { id: 'spiral', label: 'Spiral (나선)' },
    { id: 'medical', label: 'Medical (실제 의료 데이터)' }
  ];

  /** 합성이 아닌 실제 데이터인지 */
  const isReal = id => id === 'medical';

  /**
   * 데이터셋 생성.
   * @param {string} type   linear | xor | circle | spiral
   * @param {object} opts   {n, noise, seed, trainRatio}
   * @returns {{X, y, Xtrain, ytrain, Xtest, ytest, type, domain}}
   */
  function generate(type, opts = {}) {
    const n          = opts.n          ?? 300;
    const noise      = opts.noise      ?? 0;
    const seed       = opts.seed       ?? 42;
    const trainRatio = opts.trainRatio ?? 0.8;

    const gen = GENERATORS[type];
    if (!gen) throw new Error('알 수 없는 데이터셋: ' + type);

    const rng = makeRNG(seed);
    const { X, y } = gen(n, noise, rng);
    const real = isReal(type);

    // 학습/검증 분할 (섞은 뒤 자르기)
    const idx = rng.shuffle(X.map((_, i) => i));
    const cut = Math.floor(idx.length * trainRatio);
    const pick = (ids, src) => ids.map(i => src[i]);

    return {
      type, X, y, domain: DOMAIN, isReal: real,
      source: real ? (root.MEDICAL || {}) : null,
      Xtrain: pick(idx.slice(0, cut), X),
      ytrain: pick(idx.slice(0, cut), y),
      Xtest:  pick(idx.slice(cut), X),
      ytest:  pick(idx.slice(cut), y)
    };
  }

  const api = { generate, DATASETS, DOMAIN, isReal };
  root.Data = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
