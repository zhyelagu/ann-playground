/* ===========================================================
   rng.js — 시드 고정 난수 생성기
   -----------------------------------------------------------
   실험 재현성을 위해 Math.random() 대신 사용한다.
   같은 시드를 주면 데이터 생성과 가중치 초기화가 항상 동일하게
   재현되므로, Sigmoid vs ReLU 같은 비교 실험이 공정해진다.
   =========================================================== */
(function (root) {
  'use strict';

  /** mulberry32: 작고 품질이 충분한 32비트 PRNG */
  function makeRNG(seed) {
    let a = seed >>> 0;
    function next() {
      a |= 0;
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }

    return {
      /** [0, 1) 균등분포 */
      random: next,
      /** [min, max) 균등분포 */
      uniform(min, max) { return min + (max - min) * next(); },
      /** 표준정규분포 (Box-Muller) */
      normal(mean = 0, std = 1) {
        let u = 0, v = 0;
        while (u === 0) u = next();
        while (v === 0) v = next();
        return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      },
      /** 0 이상 n 미만 정수 */
      int(n) { return Math.floor(next() * n); },
      /** 배열을 제자리에서 섞는다 (Fisher-Yates) */
      shuffle(arr) {
        for (let i = arr.length - 1; i > 0; i--) {
          const j = Math.floor(next() * (i + 1));
          const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
        }
        return arr;
      }
    };
  }

  root.makeRNG = makeRNG;
  if (typeof module !== 'undefined' && module.exports) module.exports = { makeRNG };
})(typeof window !== 'undefined' ? window : globalThis);
