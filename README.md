# ANN Playground

웹 브라우저에서 실행되는 인공신경망 시뮬레이터입니다.
2D 이진분류 문제를 가지고 은닉층·활성화 함수·학습률을 바꿔가며
**결정 경계**, **학습 곡선**, **층별 Gradient Magnitude**를 관찰할 수 있습니다.

> 의료인공지능 HW #2 — Build Your Own ANN Playground with Vibe Coding

## 실행

별도 설치가 필요 없습니다.

- **배포본**: (배포 후 URL 기입)
- **로컬**: 저장소를 받아 `index.html`을 열거나, 간단한 정적 서버를 띄웁니다.
  ```bash
  python -m http.server 5500
  ```

## 구현 범위

| 구분 | 내용 |
|---|---|
| 데이터셋 | Linear · XOR · Circle · Spiral (샘플 수·노이즈 조절) + **Medical** (실제 의료 데이터) |
| 모델 구조 | 은닉층 **0~6**층 (0이면 Single Perceptron), 층당 뉴런 **1~32** |
| 활성화 함수 | Sigmoid · Tanh · ReLU (출력층은 이진분류이므로 Sigmoid 고정) |
| 학습 파라미터 | Learning rate · Batch size · Epoch |
| 시각화 | 데이터 분포, 결정 경계, 신경망 구조도, 학습 곡선(Loss/Accuracy),<br>층별 Gradient Magnitude, 층별 Gradient 변화, Activation 분포 |

### 실험

탭 하나가 과제의 실험 하나를 담당합니다. 실행하면 측정값으로 **결론 문장이 자동 생성**됩니다.

| 탭 | 실험 | 비교 대상 |
|---|---|---|
| 모델 구조 | Experiment 1 — Why Do We Need Hidden Layers? | Single Perceptron vs MLP(1층·2층) |
| 학습 과정 | Experiment 2 — Effect of Learning Rate | LR 0.001 / 0.01 / 0.1 / 1 / 30 |
| Gradient 분석 | Experiment 3 — Vanishing Gradient | {2·4·6층} × {Sigmoid·ReLU} |
| Activation 분석 | **Dying ReLU** — ReLU의 반대 문제 | ReLU LR {0.1·1·3·5} + Tanh 대조군 |

비교군은 **같은 데이터·같은 초기화 시드·같은 학습률**로 학습시켜 조건을 통제했습니다.
Experiment 1은 은닉층 수만, Experiment 2는 학습률만, Experiment 3은 깊이와 활성화 함수만 바꿉니다.

Experiment 3에서는 가중치 초기화를 고를 수 있습니다.

| 선택 | 의미 |
|---|---|
| **동일 (Xavier)** | 활성화 함수 효과만 분리하는 통제 비교 (기본값) |
| 각자 표준 (Xavier / He) | 실무에서 쓰는 방식 그대로의 비교 |

어느 쪽을 골라도 결론은 같습니다 — Sigmoid는 깊어질수록 첫 층 기울기가 10³~10⁴배 작아지고,
ReLU는 거의 평평합니다. **결과가 초기화 선택에 좌우되지 않는다**는 점도 함께 확인할 수 있습니다.

## 구조

```
index.html          레이아웃 · 스타일
js/
  rng.js            시드 고정 난수 (mulberry32) — 실험 재현성
  data.js           데이터셋 생성 4종
  nn.js             신경망 엔진 (순전파 · 역전파 · 층별 gradient 기록)
  medical.js        Breast Cancer Wisconsin에서 특성 2개 추출한 실제 데이터
  viz.js            Canvas 차트 (로그축 · 히트맵 · 다중 선그래프 · 그룹 막대)
  experiments.js    여러 설정을 동시에 학습시키는 러너
  app.js            상태 관리 · 컨트롤 · 학습 루프
  exptabs.js        실험 탭 렌더링 · 자동 결론 생성
test/
  gradcheck.js      역전파 검증 (node로 실행)
```

## 구현 메모

### 외부 의존성이 없습니다
라이브러리·CDN·API 호출이 하나도 없습니다. 차트도 Canvas로 직접 그립니다.
모든 계산은 사용자 브라우저에서 실행되며 서버로 데이터를 보내지 않습니다.

### 신경망을 직접 구현했습니다
과제 필수 기능인 **Layer별 Gradient Magnitude**를 보려면
역전파 도중 각 층의 기울기를 꺼내 쓸 수 있어야 하기 때문입니다.

```
출력층  dz = (p − y) / N          (BCE + sigmoid 조합)
각 층   dW = aᵀ · dz
        dz ← (dz · Wᵀ) ⊙ act′(z)
기록    층별 mean|dW|
```

### 역전파를 수치적으로 검증했습니다

```bash
node test/gradcheck.js
```

각 가중치를 ±h만큼 흔들어 얻은 수치적 기울기와 역전파 결과를 비교합니다.
Sigmoid·Tanh·ReLU, 은닉층 0~4층 조합에서 **상대오차 10⁻⁷ 수준**으로 일치합니다.
XOR 분류와 Vanishing Gradient 재현도 같은 스크립트에서 확인합니다.

### 성능
- 계산은 `setTimeout`, 렌더링은 주기적으로 분리했습니다.
  `requestAnimationFrame`에 묶으면 창이 가려졌을 때 학습이 멈춥니다.
- 비용이 다른 차트를 나눠 갱신합니다 — 곡선·막대는 10 epoch마다,
  격자를 다시 순전파해야 하는 결정 경계·Activation은 40 epoch마다.

### 재현성
데이터 시드와 가중치 초기화 시드를 고정했습니다. 같은 설정이면 항상 같은 결과가 나옵니다.
가중치 초기화는 활성화 함수에 맞춰 Xavier(Sigmoid·Tanh) / He(ReLU)를 씁니다.

## 이 구현만의 특징

| | 내용 |
|---|---|
| **의존성 0** | 라이브러리·CDN·API 호출 없음. 차트도 Canvas로 직접 그림 |
| **화면에서 검증** | 도움말 탭의 **[검증 실행]** 버튼으로 역전파를 그 자리에서 수치 검증 |
| **측정값 기반 결론** | 실험 결론 문장이 실제 측정값에서 생성됨 (실행할 때마다 숫자가 바뀌면 문장도 바뀜) |
| **조건 통제** | 비교군은 데이터·시드·학습률 고정. Experiment 3은 초기화 방식까지 선택 가능 |
| **이론 상한 계산** | Experiment 1에서 "직선 하나로 가능한 최대 정확도"를 탐색해 제시 |
| **Dying ReLU** | Vanishing(Sigmoid)과 Dying(ReLU) 두 실패 양상을 한 도구에서 비교 |
| **실제 의료 데이터** | 합성 데이터뿐 아니라 실제 데이터에서도 동작 (Test 약 91%) |
| **설정 공유 링크** | 현재 설정이 URL에 담겨, 발표 슬라이드에 "이 설정" 링크를 박을 수 있음 |
| **결과 내보내기** | 표는 CSV, 그래프는 PNG로 저장 (보고서 작성용) |

## 관찰되는 현상

- **Experiment 1** — Single Perceptron은 XOR에서 찍기 수준에 머뭅니다.
  은닉층이 없으면 결정 경계가 직선 하나로 고정되기 때문입니다.
- **Experiment 2** — 학습률이 작으면 수렴하지 못하고, 크면 Loss가 증가 구간을 오가며 발산합니다.
- **Experiment 3** — Sigmoid 6층에서 출력층 대비 첫 층 기울기가 **10³~10⁴배** 작아지고,
  같은 깊이의 ReLU는 거의 평평합니다. 학습 정확도도 함께 벌어집니다.
- **Dying ReLU** — 그런데 ReLU도 만능이 아닙니다. 학습률을 5로 올리면
  층별 죽은 뉴런이 **4 → 11 → 13 → 16개**로 늘고 정확도가 53%까지 떨어집니다.
  같은 학습률에서 Tanh는 죽은 뉴런이 0개입니다(대신 발산합니다).
  **실패 방식이 다르다**는 점 — 발산은 학습률을 낮추면 회복되지만 죽은 뉴런은 되살아나지 않습니다.
- **실제 의료 데이터** — 유방암 데이터 2D에서 Test 정확도 약 **91%**.
  합성 데이터처럼 100%가 나오지 않는 것이 정상입니다. 클래스가 겹치는 구간이 실제로 존재하기 때문입니다.

## 캐시 주의
`index.html`의 스크립트 태그에 `?v=` 를 붙여 두었습니다.
JS를 수정하고 배포할 때는 이 숫자를 올려야 브라우저가 옛 파일을 계속 쓰지 않습니다.
