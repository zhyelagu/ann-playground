/* ===========================================================
   gradcheck.js — 역전파 검증 + 엔진 동작 확인
   실행:  node test/gradcheck.js
   -----------------------------------------------------------
   1) 해석적 기울기 vs 수치적 기울기 비교 (가장 중요)
   2) XOR: Single Perceptron 실패 / MLP 성공
   3) Vanishing Gradient: Sigmoid 깊은 망 vs ReLU
   =========================================================== */
'use strict';

const path = require('path');
const { makeRNG } = require(path.join(__dirname, '..', 'js', 'rng.js'));
globalThis.makeRNG = makeRNG;
require(path.join(__dirname, '..', 'js', 'medical.js'));   // 실제 의료 데이터
const Data = require(path.join(__dirname, '..', 'js', 'data.js'));
globalThis.Data = Data;
const { NeuralNetwork } = require(path.join(__dirname, '..', 'js', 'nn.js'));

let failed = 0;
const ok   = m => console.log('  ✓ ' + m);
const bad  = m => { failed++; console.log('  ✗ ' + m); };

/* -----------------------------------------------------------
   1. Gradient check
   각 가중치를 h만큼 흔들어 손실 변화를 보고, 역전파가 내놓은
   기울기와 비교한다. 상대오차가 1e-6 수준이면 구현이 맞다.
   ----------------------------------------------------------- */
function gradCheck(activation, hiddenLayers, neurons) {
  const net = new NeuralNetwork({
    hiddenLayers, neurons, activation, seed: 7, initScheme: 'xavier'
  });

  const ds = Data.generate('xor', { n: 24, seed: 3 });
  const X = ds.X, y = ds.y;

  const { dW } = net.computeGradients(X, y);
  const h = 1e-5;
  let maxRel = 0;

  for (let l = 0; l < net.numLayers; l++) {
    for (let i = 0; i < net.W[l].length; i++) {
      for (let j = 0; j < net.W[l][i].length; j++) {
        const orig = net.W[l][i][j];

        net.W[l][i][j] = orig + h;
        const lossPlus = net.evaluate(X, y).loss;
        net.W[l][i][j] = orig - h;
        const lossMinus = net.evaluate(X, y).loss;
        net.W[l][i][j] = orig;

        const numeric  = (lossPlus - lossMinus) / (2 * h);
        const analytic = dW[l][i][j];
        const denom = Math.max(1e-8, Math.abs(numeric) + Math.abs(analytic));
        maxRel = Math.max(maxRel, Math.abs(numeric - analytic) / denom);
      }
    }
  }

  const label = `${activation.padEnd(7)} hidden=${hiddenLayers} n=${neurons}`;
  if (maxRel < 1e-5) ok(`${label}  최대 상대오차 ${maxRel.toExponential(2)}`);
  else               bad(`${label}  최대 상대오차 ${maxRel.toExponential(2)}  ← 역전파 오류`);
}

console.log('\n[1] Gradient check (해석적 vs 수치적)');
gradCheck('sigmoid', 0, 8);     // Single Perceptron
gradCheck('sigmoid', 1, 6);
gradCheck('sigmoid', 3, 5);
gradCheck('tanh',    2, 6);
gradCheck('relu',    2, 6);
gradCheck('relu',    4, 4);

/* -----------------------------------------------------------
   2. Experiment 1 재현: XOR은 은닉층이 있어야 풀린다
   ----------------------------------------------------------- */
console.log('\n[2] Experiment 1 — XOR: Single Perceptron vs MLP');
{
  const ds = Data.generate('xor', { n: 300, seed: 42 });

  function train(hiddenLayers, activation, lr, epochs) {
    const net = new NeuralNetwork({
      hiddenLayers, neurons: 8, activation, learningRate: lr, seed: 11
    });
    for (let e = 0; e < epochs; e++) net.trainEpoch(ds.Xtrain, ds.ytrain, 16);
    return net.evaluate(ds.Xtest, ds.ytest);
  }

  const perceptron = train(0, 'sigmoid', 0.5, 300);
  const mlp        = train(2, 'tanh',    0.5, 300);

  console.log(`     Single Perceptron : acc ${(perceptron.accuracy * 100).toFixed(1)}%  loss ${perceptron.loss.toFixed(4)}`);
  console.log(`     MLP (hidden 2×8)  : acc ${(mlp.accuracy * 100).toFixed(1)}%  loss ${mlp.loss.toFixed(4)}`);

  if (perceptron.accuracy < 0.70) ok('Perceptron은 XOR을 못 푼다 (기대한 결과)');
  else bad(`Perceptron 정확도가 너무 높다 (${perceptron.accuracy.toFixed(2)}) — 데이터 확인 필요`);

  if (mlp.accuracy > 0.90) ok('MLP는 XOR을 푼다');
  else bad(`MLP 정확도가 낮다 (${mlp.accuracy.toFixed(2)})`);
}

/* -----------------------------------------------------------
   3. Experiment 3 재현: 깊어질수록 Sigmoid 기울기가 사라진다
   ----------------------------------------------------------- */
console.log('\n[3] Experiment 3 — Vanishing Gradient');
{
  const ds = Data.generate('xor', { n: 300, seed: 42 });

  function gradProfile(activation, hiddenLayers) {
    const net = new NeuralNetwork({
      hiddenLayers, neurons: 8, activation, learningRate: 0.1, seed: 11
    });
    for (let e = 0; e < 30; e++) net.trainEpoch(ds.Xtrain, ds.ytrain, 16);
    return { names: net.layerNames(), grad: net.lastGradients };
  }

  for (const L of [2, 4, 6]) {
    const s = gradProfile('sigmoid', L);
    const r = gradProfile('relu', L);
    console.log(`\n     hidden layers = ${L}`);
    console.log('       층        Sigmoid      ReLU');
    for (let i = 0; i < s.grad.length; i++) {
      console.log(`       ${s.names[i].padEnd(8)}  ${s.grad[i].toExponential(2)}   ${r.grad[i].toExponential(2)}`);
    }
    const ratio = s.grad[s.grad.length - 1] / Math.max(s.grad[0], 1e-300);
    console.log(`       Sigmoid 출력층/첫층 비율 = ${ratio.toExponential(2)} 배`);
  }

  // 깊을수록 첫 층 기울기가 더 작아져야 한다
  const g2 = gradProfile('sigmoid', 2).grad[0];
  const g6 = gradProfile('sigmoid', 6).grad[0];
  if (g6 < g2) ok(`깊어질수록 첫 층 기울기 감소 (${g2.toExponential(2)} → ${g6.toExponential(2)})`);
  else bad(`기울기 감소가 관찰되지 않음 (${g2.toExponential(2)} → ${g6.toExponential(2)})`);
}

/* -----------------------------------------------------------
   4. 데이터셋 기본 점검
   ----------------------------------------------------------- */
console.log('\n[4] 데이터셋 점검');
for (const d of Data.DATASETS) {
  const ds = Data.generate(d.id, { n: 200, seed: 1 });
  const pos = ds.y.filter(v => v === 1).length;
  const xs = ds.X.map(p => p[0]), ys = ds.X.map(p => p[1]);
  const maxAbs = Math.max(...xs.map(Math.abs), ...ys.map(Math.abs));
  const inRange = maxAbs < 3.2;
  const ratio = pos / ds.X.length;
  // 실제 데이터는 클래스 비율이 정확히 반반일 수 없다
  const balanced = Data.isReal(d.id) ? (ratio > 0.2 && ratio < 0.8)
                                     : (ratio > 0.3 && ratio < 0.7);
  if (inRange && balanced) ok(`${d.id.padEnd(8)} n=${ds.X.length} 양성=${pos}(${(ratio*100).toFixed(0)}%) 범위 OK`);
  else bad(`${d.id} 이상 — 양성비율=${ratio.toFixed(2)}, 최대절댓값=${maxAbs.toFixed(2)}`);
}

/* -----------------------------------------------------------
   5. 실제 의료 데이터에서도 학습이 되는가
   ----------------------------------------------------------- */
console.log('\n[5] 실제 의료 데이터 (Breast Cancer 2D)');
{
  const ds = Data.generate('medical', {});
  console.log(`     특성: ${ds.source.feature_x} / ${ds.source.feature_y}`);
  console.log(`     n=${ds.X.length} (학습 ${ds.Xtrain.length} / 검증 ${ds.Xtest.length}), 악성 ${ds.source.positive}건`);

  const net = new NeuralNetwork({
    hiddenLayers: 2, neurons: 8, activation: 'tanh',
    learningRate: 0.5, initScheme: 'xavier', seed: 11
  });
  for (let e = 0; e < 300; e++) net.trainEpoch(ds.Xtrain, ds.ytrain, 16);
  const te = net.evaluate(ds.Xtest, ds.ytest);
  console.log(`     MLP(2x8, tanh) -> Test 정확도 ${(te.accuracy * 100).toFixed(1)}%  loss ${te.loss.toFixed(4)}`);

  // 합성 데이터와 달리 100%가 나오면 오히려 이상하다
  if (te.accuracy > 0.80 && te.accuracy < 1.0) ok('실제 데이터에서 학습되지만 완벽하진 않다 (정상)');
  else bad(`실제 데이터 정확도가 이상 (${te.accuracy.toFixed(3)})`);
}

/* -----------------------------------------------------------
   6. Dying ReLU 가 재현되는가
   ----------------------------------------------------------- */
console.log('\n[6] Dying ReLU');
{
  const ds = Data.generate('xor', { n: 300, seed: 42 });
  const deadOf = lr => {
    const net = new NeuralNetwork({
      hiddenLayers: 4, neurons: 16, activation: 'relu',
      learningRate: lr, initScheme: 'xavier', seed: 11
    });
    for (let e = 0; e < 300; e++) net.trainEpoch(ds.Xtrain, ds.ytrain, 16);
    const st = net.activationStats(ds.Xtrain).slice(0, -1);
    return {
      counts: st.map(s => s.deadCount),
      total: st.reduce((a, s) => a + s.deadCount, 0),
      acc: net.evaluate(ds.Xtest, ds.ytest).accuracy
    };
  };
  const low = deadOf(0.1), high = deadOf(5);
  console.log(`     LR 0.1 -> 죽은 뉴런 ${low.counts.join('/')} (합 ${low.total}), Test ${(low.acc*100).toFixed(1)}%`);
  console.log(`     LR 5   -> 죽은 뉴런 ${high.counts.join('/')} (합 ${high.total}), Test ${(high.acc*100).toFixed(1)}%`);
  if (high.total > low.total) ok('학습률이 크면 죽은 뉴런이 늘어난다');
  else bad('Dying ReLU가 재현되지 않음');
}

console.log('\n' + (failed === 0 ? '전부 통과' : `실패 ${failed}건`));
process.exit(failed === 0 ? 0 : 1);
