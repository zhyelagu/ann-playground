/* ===========================================================
   nn.js — 다층 퍼셉트론 (직접 구현)
   -----------------------------------------------------------
   라이브러리를 쓰지 않고 순전파/역전파를 직접 구현한다.
   이유: 과제 필수 기능인 "Layer별 Gradient Magnitude"를 보려면
         역전파 도중의 각 층 기울기를 꺼내 쓸 수 있어야 한다.

   구조 : input(2) → hidden × L → output(1)
   출력 : sigmoid  (이진 분류)
   손실 : Binary Cross-Entropy
   =========================================================== */
(function (root) {
  'use strict';

  const makeRNG = root.makeRNG || require('./rng.js').makeRNG;

  /* ---------- 활성화 함수 ---------------------------------- */
  const ACTIVATIONS = {
    sigmoid: {
      label: 'Sigmoid',
      f:  z => 1 / (1 + Math.exp(-z)),
      // a = sigmoid(z) 를 이미 알고 있을 때의 도함수
      df: (z, a) => a * (1 - a),
      init: 'xavier'
    },
    tanh: {
      label: 'Tanh',
      f:  z => Math.tanh(z),
      df: (z, a) => 1 - a * a,
      init: 'xavier'
    },
    relu: {
      label: 'ReLU',
      f:  z => (z > 0 ? z : 0),
      df: (z) => (z > 0 ? 1 : 0),
      init: 'he'
    }
  };

  const sigmoid = z => 1 / (1 + Math.exp(-z));

  /* ---------- 작은 행렬 유틸 -------------------------------
     데이터가 작아서(샘플 수백, 뉴런 ≤32) 2차원 배열로 충분하다.
     속도보다 읽기 쉬움을 택했다.
     -------------------------------------------------------- */
  const zeros = (r, c) => Array.from({ length: r }, () => new Float64Array(c));

  /** A[n][p] · B[p][q] → [n][q] */
  function matmul(A, B) {
    const n = A.length, p = B.length, q = B[0].length;
    const out = zeros(n, q);
    for (let i = 0; i < n; i++) {
      const Ai = A[i], Oi = out[i];
      for (let k = 0; k < p; k++) {
        const a = Ai[k];
        if (a === 0) continue;
        const Bk = B[k];
        for (let j = 0; j < q; j++) Oi[j] += a * Bk[j];
      }
    }
    return out;
  }

  /** A^T[p][n] · B[n][q] → [p][q]  (A를 실제로 전치하지 않음) */
  function matmulAT(A, B) {
    const n = A.length, p = A[0].length, q = B[0].length;
    const out = zeros(p, q);
    for (let i = 0; i < n; i++) {
      const Ai = A[i], Bi = B[i];
      for (let k = 0; k < p; k++) {
        const a = Ai[k];
        if (a === 0) continue;
        const Ok = out[k];
        for (let j = 0; j < q; j++) Ok[j] += a * Bi[j];
      }
    }
    return out;
  }

  /** A[n][q] · B^T[q][p] → [n][p]  (B를 실제로 전치하지 않음) */
  function matmulBT(A, B) {
    const n = A.length, q = A[0].length, p = B.length;
    const out = zeros(n, p);
    for (let i = 0; i < n; i++) {
      const Ai = A[i], Oi = out[i];
      for (let k = 0; k < p; k++) {
        const Bk = B[k];
        let s = 0;
        for (let j = 0; j < q; j++) s += Ai[j] * Bk[j];
        Oi[k] = s;
      }
    }
    return out;
  }

  /* ========================================================
     NeuralNetwork
     ======================================================== */
  class NeuralNetwork {
    /**
     * @param {object} cfg
     *   hiddenLayers  은닉층 개수 (0이면 Single Perceptron)
     *   neurons       은닉층 당 뉴런 수
     *   activation    'sigmoid' | 'tanh' | 'relu'
     *   learningRate  학습률
     *   seed          가중치 초기화 시드 (같으면 동일 초기값)
     *   inputDim      입력 차원 (기본 2)
     *   initScheme    'auto' | 'xavier' | 'he'
     */
    constructor(cfg = {}) {
      this.hiddenLayers = cfg.hiddenLayers ?? 1;
      this.neurons      = cfg.neurons      ?? 8;
      this.activation   = cfg.activation   ?? 'sigmoid';
      this.learningRate = cfg.learningRate ?? 0.1;
      this.seed         = cfg.seed         ?? 1234;
      this.inputDim     = cfg.inputDim     ?? 2;
      this.initScheme   = cfg.initScheme   ?? 'auto';

      if (!ACTIVATIONS[this.activation]) {
        throw new Error('알 수 없는 활성화 함수: ' + this.activation);
      }

      // [2, n, n, ..., 1]  — 은닉층이 0이면 [2, 1] (= Single Perceptron)
      this.layerSizes = [this.inputDim];
      for (let i = 0; i < this.hiddenLayers; i++) this.layerSizes.push(this.neurons);
      this.layerSizes.push(1);

      this.numLayers = this.layerSizes.length - 1;   // 가중치 행렬 개수
      this.reset();
    }

    /** 가중치 재초기화 + 기록 초기화 */
    reset(seed) {
      if (seed !== undefined) this.seed = seed;
      const rng = makeRNG(this.seed);

      const scheme = this.initScheme === 'auto'
        ? ACTIVATIONS[this.activation].init
        : this.initScheme;

      this.W = [];
      this.b = [];
      for (let l = 0; l < this.numLayers; l++) {
        const fanIn  = this.layerSizes[l];
        const fanOut = this.layerSizes[l + 1];
        // He: sqrt(2/fanIn) — ReLU용 / Xavier: sqrt(1/fanIn) — sigmoid·tanh용
        const std = scheme === 'he'
          ? Math.sqrt(2 / fanIn)
          : Math.sqrt(1 / fanIn);

        const Wl = zeros(fanIn, fanOut);
        for (let i = 0; i < fanIn; i++)
          for (let j = 0; j < fanOut; j++) Wl[i][j] = rng.normal(0, std);

        this.W.push(Wl);
        this.b.push(new Float64Array(fanOut));
      }

      this.rng   = rng;        // 미니배치 셔플에 재사용
      this.epoch = 0;
      this.history = {
        loss: [], accuracy: [],
        gradPerLayer: []       // [epoch][layer] = mean|dW|
      };
      this.lastActivations = null;   // Activation 분포 패널용
      this.lastGradients   = null;   // Gradient Magnitude 패널용
      return this;
    }

    /* ---------- 순전파 ------------------------------------ */
    /**
     * @returns {{zs: Array, as: Array, output: Array}}
     *   as[0] = 입력, as[l] = l번째 층 활성값, as[L] = 출력(확률)
     */
    forward(X) {
      const act = ACTIVATIONS[this.activation];
      const zs = [];          // 각 층의 선형결합 결과
      const as = [X];         // 각 층의 활성값 (as[0]은 입력)

      for (let l = 0; l < this.numLayers; l++) {
        const z = matmul(as[l], this.W[l]);
        const bl = this.b[l];
        const isOutput = (l === this.numLayers - 1);

        const a = zeros(z.length, z[0].length);
        for (let i = 0; i < z.length; i++) {
          const zi = z[i], ai = a[i];
          for (let j = 0; j < zi.length; j++) {
            zi[j] += bl[j];
            // 출력층은 항상 sigmoid (이진 분류 확률)
            ai[j] = isOutput ? sigmoid(zi[j]) : act.f(zi[j]);
          }
        }
        zs.push(z);
        as.push(a);
      }
      return { zs, as, output: as[as.length - 1] };
    }

    /* ---------- 역전파 ------------------------------------ */
    /**
     * 기울기를 계산한다. 가중치는 갱신하지 않는다.
     * (수치적 gradient check에서 이 함수를 그대로 쓴다)
     * @returns {{dW, db, gradMag}}  gradMag[l] = mean|dW[l]|
     */
    computeGradients(X, y) {
      const act = ACTIVATIONS[this.activation];
      const { zs, as } = this.forward(X);
      const N = X.length;
      const L = this.numLayers;

      const dW = new Array(L);
      const db = new Array(L);

      // 출력층: BCE + sigmoid 조합이라 dz = (p - y) / N 으로 단순해진다
      let dz = zeros(N, 1);
      const out = as[L];
      for (let i = 0; i < N; i++) dz[i][0] = (out[i][0] - y[i]) / N;

      for (let l = L - 1; l >= 0; l--) {
        dW[l] = matmulAT(as[l], dz);              // [fanIn][fanOut]
        const dbl = new Float64Array(dz[0].length);
        for (let i = 0; i < N; i++)
          for (let j = 0; j < dbl.length; j++) dbl[j] += dz[i][j];
        db[l] = dbl;

        if (l > 0) {
          // 이전 층으로 전파: da = dz · W^T, 그다음 활성함수 도함수를 곱한다
          const da = matmulBT(dz, this.W[l]);
          const zPrev = zs[l - 1], aPrev = as[l];
          const next = zeros(N, da[0].length);
          for (let i = 0; i < N; i++) {
            const dai = da[i], zi = zPrev[i], ai = aPrev[i], ni = next[i];
            for (let j = 0; j < dai.length; j++) {
              ni[j] = dai[j] * act.df(zi[j], ai[j]);
            }
          }
          dz = next;
        }
      }

      // 층별 기울기 크기 = |dW|의 평균
      const gradMag = dW.map(M => {
        let s = 0, c = 0;
        for (const row of M) for (const v of row) { s += Math.abs(v); c++; }
        return c ? s / c : 0;
      });

      return { dW, db, gradMag, activations: as, zs };
    }

    /** 한 배치에 대해 기울기를 구하고 가중치를 갱신한다 */
    trainBatch(X, y) {
      const { dW, db, gradMag, activations } = this.computeGradients(X, y);
      const lr = this.learningRate;

      for (let l = 0; l < this.numLayers; l++) {
        const Wl = this.W[l], dWl = dW[l];
        for (let i = 0; i < Wl.length; i++) {
          const Wi = Wl[i], di = dWl[i];
          for (let j = 0; j < Wi.length; j++) Wi[j] -= lr * di[j];
        }
        const bl = this.b[l], dbl = db[l];
        for (let j = 0; j < bl.length; j++) bl[j] -= lr * dbl[j];
      }

      this.lastGradients   = gradMag;
      this.lastActivations = activations;
      return gradMag;
    }

    /* ---------- 한 epoch 학습 ------------------------------ */
    /**
     * @param {Array} X, {Array} y, {number} batchSize  (0이면 전체 배치)
     * @returns {{loss, accuracy, gradMag}}
     */
    trainEpoch(X, y, batchSize = 16) {
      const n = X.length;
      const bs = (!batchSize || batchSize >= n) ? n : batchSize;

      const order = this.rng.shuffle(X.map((_, i) => i));
      const gradSum = new Array(this.numLayers).fill(0);
      let batches = 0;

      for (let s = 0; s < n; s += bs) {
        const ids = order.slice(s, s + bs);
        if (!ids.length) continue;
        const Xb = ids.map(i => X[i]);
        const yb = ids.map(i => y[i]);
        const g = this.trainBatch(Xb, yb);
        for (let l = 0; l < g.length; l++) gradSum[l] += g[l];
        batches++;
      }

      const gradMag = gradSum.map(v => v / Math.max(batches, 1));
      const { loss, accuracy } = this.evaluate(X, y);

      this.epoch++;
      this.history.loss.push(loss);
      this.history.accuracy.push(accuracy);
      this.history.gradPerLayer.push(gradMag);
      this.lastGradients = gradMag;

      return { loss, accuracy, gradMag };
    }

    /* ---------- 평가 -------------------------------------- */
    /** 확률 예측 [N] */
    predict(X) {
      return this.forward(X).output.map(r => r[0]);
    }

    /** BCE 손실과 정확도 */
    evaluate(X, y) {
      const p = this.predict(X);
      const eps = 1e-12;
      let loss = 0, correct = 0;
      for (let i = 0; i < p.length; i++) {
        const pi = Math.min(1 - eps, Math.max(eps, p[i]));
        loss += -(y[i] * Math.log(pi) + (1 - y[i]) * Math.log(1 - pi));
        if ((pi >= 0.5 ? 1 : 0) === y[i]) correct++;
      }
      return { loss: loss / p.length, accuracy: correct / p.length };
    }

    /** 결정 경계용 격자 예측 → res×res 확률 행렬 */
    decisionGrid(domain, res = 60) {
      const { min, max } = domain;
      const step = (max - min) / (res - 1);
      const pts = [];
      for (let r = 0; r < res; r++)
        for (let c = 0; c < res; c++)
          pts.push([min + c * step, max - r * step]);   // 위에서 아래로

      const p = this.predict(pts);
      const grid = [];
      for (let r = 0; r < res; r++) grid.push(p.slice(r * res, (r + 1) * res));
      return grid;
    }

    /** 층별 활성값 (Activation 분포 패널용) */
    activationStats(X) {
      const { as } = this.forward(X);
      // as[0]은 입력이므로 제외, 은닉층 + 출력층만
      return as.slice(1).map((A, i) => {
        const width = A[0].length;
        const flat = [];
        for (const row of A) for (const v of row) flat.push(v);
        const mean = flat.reduce((s, v) => s + v, 0) / flat.length;
        const varc = flat.reduce((s, v) => s + (v - mean) ** 2, 0) / flat.length;

        // 죽은 뉴런: 모든 샘플에서 출력이 0인 뉴런.
        // ReLU는 음수 입력에서 기울기도 0이라 한번 죽으면 되살아나지 않는다.
        let dead = 0;
        for (let j = 0; j < width; j++) {
          let allZero = true;
          for (let r = 0; r < A.length; r++) {
            if (Math.abs(A[r][j]) > 1e-9) { allZero = false; break; }
          }
          if (allZero) dead++;
        }

        return {
          layer: i + 1,
          isOutput: i === this.numLayers - 1,
          values: flat, width,
          mean, std: Math.sqrt(varc),
          min: Math.min(...flat), max: Math.max(...flat),
          deadCount: dead, deadRatio: dead / width
        };
      });
    }

    /** 층 이름 ['Layer 1', ..., 'Output'] — 차트 라벨용 */
    layerNames() {
      const names = [];
      for (let l = 0; l < this.numLayers; l++) {
        names.push(l === this.numLayers - 1 ? 'Output' : `Layer ${l + 1}`);
      }
      return names;
    }
  }

  /**
   * 역전파 검증 — 해석적 기울기 vs 수치적 기울기
   * 각 가중치를 ±h 흔들어 손실 변화를 보고, 역전파 결과와 비교한다.
   * 상대오차가 1e-5보다 작으면 구현이 맞다고 본다.
   * @returns {{maxRelError, params, ok}}
   */
  function gradientCheck(cfg, X, y, h = 1e-5) {
    const net = new NeuralNetwork({ ...cfg, initScheme: cfg.initScheme || 'xavier' });
    const { dW } = net.computeGradients(X, y);
    let maxRel = 0, count = 0;

    for (let l = 0; l < net.numLayers; l++) {
      for (let i = 0; i < net.W[l].length; i++) {
        for (let j = 0; j < net.W[l][i].length; j++) {
          const orig = net.W[l][i][j];
          net.W[l][i][j] = orig + h;  const lp = net.evaluate(X, y).loss;
          net.W[l][i][j] = orig - h;  const lm = net.evaluate(X, y).loss;
          net.W[l][i][j] = orig;

          const numeric = (lp - lm) / (2 * h);
          const analytic = dW[l][i][j];
          const denom = Math.max(1e-8, Math.abs(numeric) + Math.abs(analytic));
          maxRel = Math.max(maxRel, Math.abs(numeric - analytic) / denom);
          count++;
        }
      }
    }
    return { maxRelError: maxRel, params: count, ok: maxRel < 1e-5 };
  }

  const api = { NeuralNetwork, ACTIVATIONS, gradientCheck };
  root.NN = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
