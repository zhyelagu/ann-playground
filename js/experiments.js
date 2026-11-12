/* ===========================================================
   experiments.js — 과제 실험 3개를 한 번에 돌리는 러너
   -----------------------------------------------------------
   여러 설정을 동시에 학습시켜 비교한다.
   · 모든 모델이 같은 데이터·같은 시드를 쓴다 → 비교가 공정해짐
   · 라운드로빈으로 한 epoch씩 돌려 진행률이 고르게 올라간다
   · 시간을 잘라 UI를 막지 않는다 (메인 학습 루프와 같은 방식)
   =========================================================== */
(function (root) {
  'use strict';

  const SEED = 11;          // 모든 실험 모델의 초기화 시드 (고정)

  /**
   * @param {object} opt
   *   data     Data.generate() 결과
   *   configs  [{label, hiddenLayers, neurons, activation, lr}]
   *   epochs   각 모델의 학습 epoch
   *   batch    배치 크기
   *   onProgress(done, total)
   *   onDone(results)   results[i] = {cfg, net, acc, loss}
   * @returns {{cancel: Function}}
   */
  function run(opt) {
    const { data, configs, epochs, batch = 16 } = opt;

    const nets = configs.map(c => new NN.NeuralNetwork({
      hiddenLayers: c.hiddenLayers,
      neurons:      c.neurons,
      activation:   c.activation,
      learningRate: c.lr,
      initScheme:   c.initScheme || 'auto',
      seed:         SEED
    }));

    const total = configs.length * epochs;
    let done = 0, cancelled = false, timer = null;

    function chunk() {
      if (cancelled) return;
      const t0 = performance.now();

      while (performance.now() - t0 < 24) {
        if (done >= total) return finish();
        // 라운드로빈: 모든 모델을 한 epoch씩
        for (const net of nets) {
          if (net.epoch < epochs) {
            net.trainEpoch(data.Xtrain, data.ytrain, batch);
            done++;
          }
        }
      }
      opt.onProgress && opt.onProgress(done, total);
      timer = setTimeout(chunk, 0);
    }

    function finish() {
      opt.onProgress && opt.onProgress(total, total);
      const results = configs.map((cfg, i) => {
        const net = nets[i];
        const tr = net.evaluate(data.Xtrain, data.ytrain);
        const te = net.evaluate(data.Xtest,  data.ytest);
        const g  = net.lastGradients || [];
        return {
          cfg, net,
          loss: tr.loss,
          acc:  tr.accuracy,
          testAcc: te.accuracy,
          grad: g,
          // 첫 층 대비 출력층 기울기 비율 — Vanishing 정도를 한 숫자로
          gradRatio: g.length > 1 ? g[g.length - 1] / Math.max(g[0], 1e-300) : 1
        };
      });
      opt.onDone && opt.onDone(results);
    }

    timer = setTimeout(chunk, 0);
    return { cancel() { cancelled = true; clearTimeout(timer); } };
  }

  /* ---------------------------------------------------------
     과제 28쪽의 실험 3개 설정
     --------------------------------------------------------- */

  /** Experiment 1 — Single Perceptron vs MLP (XOR)
      은닉층 수만 바꾸고 나머지(활성화·학습률·초기화)는 전부 고정한다.
      (은닉층이 0개면 은닉 활성화는 쓰이지 않지만, 표에 다르게 찍히면
       조건을 통제하지 않은 것처럼 보이므로 값 자체를 통일한다) */
  function exp1Configs(neurons = 8) {
    const base = { neurons, activation: 'tanh', lr: 0.5, initScheme: 'xavier' };
    return [
      { label: 'Single Perceptron', hiddenLayers: 0, ...base },
      { label: 'MLP (hidden 1층)',  hiddenLayers: 1, ...base },
      { label: 'MLP (hidden 2층)',  hiddenLayers: 2, ...base }
    ];
  }

  /** Experiment 2 — Learning Rate 효과
      너무 작은 값부터 발산하는 값까지 포함해야 교환 관계가 보인다 */
  const LR_LIST = [0.001, 0.01, 0.1, 1.0, 30.0];
  function exp2Configs(hiddenLayers = 2, neurons = 8, activation = 'tanh') {
    return LR_LIST.map(lr => ({
      label: `LR = ${lr}`, hiddenLayers, neurons, activation, lr,
      initScheme: 'xavier'          // 학습률만 변수로 두기 위해 고정
    }));
  }

  /** Experiment 3 — Vanishing Gradient (깊이 × 활성화함수) */
  const DEPTHS = [2, 4, 6];
  /**
   * @param initScheme 'xavier' = 두 활성화에 같은 초기화 (활성화 효과만 분리)
   *                   'auto'   = 각자 표준 초기화 (Sigmoid→Xavier, ReLU→He)
   */
  function exp3Configs(neurons = 8, lr = 0.1, initScheme = 'xavier') {
    const out = [];
    for (const d of DEPTHS) {
      for (const act of ['sigmoid', 'relu']) {
        out.push({
          label: `${act === 'sigmoid' ? 'Sigmoid' : 'ReLU'} · ${d}층`,
          hiddenLayers: d, neurons, activation: act, lr, initScheme
        });
      }
    }
    return out;
  }

  /** Dying ReLU — 학습률을 키우며 죽은 뉴런이 늘어나는지 본다.
      Tanh는 죽을 수 없으므로 대조군으로 함께 돌린다. */
  const DYING_LRS = [0.1, 1, 3, 5];
  function exp4Configs(hiddenLayers = 4, neurons = 16) {
    const base = { hiddenLayers, neurons, initScheme: 'xavier' };
    const out = DYING_LRS.map(lr => ({ label: `ReLU · LR ${lr}`, activation: 'relu', lr, ...base }));
    out.push({ label: 'Tanh · LR 5 (대조군)', activation: 'tanh', lr: 5, ...base });
    return out;
  }

  root.Experiments = { exp4Configs, DYING_LRS, run, exp1Configs, exp2Configs, exp3Configs, LR_LIST, DEPTHS, SEED };
})(typeof window !== 'undefined' ? window : globalThis);
