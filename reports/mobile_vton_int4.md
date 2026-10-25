# Mobile-VTON INT4 실험 — 2026-10-23

## 최신 결과: 원본 해상도 GPU 검증 완료

기존 `vllm/vllm-openai` 이미지를 CUDA 실행 환경으로 사용해 **768×1024,
28스텝, seed 42, guidance 2.5, 배치 1**에서 BF16과 INT4를 순차 실행했다.
동일 입력과 동일한 텍스트 임베딩 캐시를 사용하고, 두 텍스트 인코더는 로드하지
않았다. 각 정밀도에 1스텝 워밍업 후 두 번 측정했다. 원본 해상도에서는 RoPE
크기 조정이 적용되지 않았다. GPU는 RTX 3080 Laptop 16GB다.

| 측정 항목 | BF16 + 텍스트 캐시 | INT4 + 텍스트 캐시 |
| --- | ---: | ---: |
| 파이프라인 시간 | 15.056 / 14.906초 | 36.896 / 41.330초 |
| 평균 시간 | **14.981초** | **39.113초** |
| 모델 텐서 저장량 | 1,947,473,298 bytes (1.947GB) | 973,108,626 bytes (0.973GB) |
| CUDA peak allocated | 3,376,178,688 bytes (**3.14GiB**) | 2,410,055,168 bytes (**2.24GiB**) |
| CUDA peak reserved | 4,215,275,520 bytes (3.93GiB) | 3,510,632,448 bytes (3.27GiB) |
| 호스트 프로세스 peak RSS | 2,042,138,624 bytes | 2,040,418,304 bytes |

모델 텐서는 **50.0%**, 추론 구간 CUDA peak allocated는 **28.6%** 감소했다.
INT4는 이 실행에서 오히려 느렸다. 공유 GPU의 소수 표본이므로 다른 장치의
성능으로 일반화하지 않는다. PyTorch allocated/reserved에는 드라이버·CUDA
컨텍스트 메모리와 다른 프로세스가 포함되지 않는다. 호스트 RSS와 GPU 할당량도
구분해야 하며 **2.24GiB가 휴대폰의 전체 RAM 요구량이라는 뜻은 아니다**.

CUDA에서는 `TensorCoreTiledLayout`으로 원본 가중치를 새로 양자화했다. CPU용
레이아웃과 저장 패딩이 달라 이전 CPU 결과의 0.895GB와 수치가 다르다.
Linear 734개, 가중치 716,898,304개를 모두 변환했다. CUDA denoiser 파일 두 개는
합계 **626,148,367 bytes (626.1MB)**이며 나머지 구성요소는 별도로 필요하다.
별도 프로세스에서 두 파일을 실제 재로딩해 28스텝을 완료했으며 최초 INT4 출력과
PNG 해시가 일치했다. 두 정밀도의 반복 출력도 각각 완전히 일치했다.

이 샘플은 INT4에서도 실루엣·V넥·민트색 테두리를 유지하지만 **가슴 프린트가
대부분 소실되고 전면 여밈의 질감이 변한다**. BF16에서도 원상품 그래픽을 완벽히
재현하지 못한다. BF16 대비 MAE **0.01279**, PSNR **32.19dB**는 정답 이미지에
대한 품질 점수가 아니며, 전체 픽셀 차이가 작아도 상품 디테일 손실은 크다.
이번 단순 INT4 설정은 메모리 절감은 확인했으나 **품질 통과는 보류**한다.

[원본 크기 비교 PNG](../experiments/2026-10-23-mobile_vton/outputs_vllm_comparison_native/bf16_vs_int4_native.png)
· [입력 포함 비교](../experiments/2026-10-23-mobile_vton/outputs_vllm_comparison_native/bf16_vs_int4.jpg)
· [의류 확대](../experiments/2026-10-23-mobile_vton/outputs_vllm_comparison_native/garment_detail.jpg)
· [비교 JSON](../experiments/2026-10-23-mobile_vton/outputs_vllm_comparison_native/comparison.json)

컨테이너 이미지는 digest로 고정했다. 기존 Python 환경을 읽기 전용으로 마운트해
PyTorch 2.7.1+cu126, torchao 0.10.0을 실행했으며 `vllm serve`는 시작하지 않았다.
재현 명령과 의존성은 [실행 가이드](../experiments/2026-10-23-mobile_vton/INT4.md)에 있다.
실측 환경·소스 revision 확인은
[환경 검증 JSON](../experiments/2026-10-23-mobile_vton/outputs_vllm_comparison_native/environment.json)에 기록한다.
아래 CPU 기록은 이 GPU 검증 이전의 탐색 실험이다.

## 양자화 범위와 저장량

공식 체크포인트의 TryonNet·GarmentNet **Linear 734개 레이어 전체**를
torchao 0.10.0의 group-wise, weight-only INT4(group size 128)로 변환했다.
양자화한 가중치는 **716,898,304개**다. Convolution, 정규화, 이미지 인코더와
VAE는 BF16, 경량 최종 디코더는 FP32를 유지한다. 전체 모델이 INT4라는 뜻은 아니다.
활성값도 BF16이며 학습·보정 데이터 없이 적용한 후처리 양자화다.

| 구성 | BF16 저장량 | INT4 적용 후 저장량 |
| --- | ---: | ---: |
| TryonNet 텐서 | 840,014,880 bytes | 285,387,808 bytes |
| GarmentNet 텐서 | 759,047,712 bytes | 260,730,400 bytes |
| 텍스트 인코더 제외 전체 런타임 텐서 | 1,947,473,298 bytes | 894,528,914 bytes |

표는 원래 768×1024 설정의 파라미터·버퍼에 대해 **실제 packed backing storage**를
합한 값이다. 논리적인 BF16 tensor shape로 INT4 메모리를 잘못 계산하지 않았다.
텍스트 인코더 제외 모델 텐서 저장량은 **54.1% 감소**했다. 이 값에는 추론 중
활성값·임시 버퍼·런타임 오버헤드가 없으므로 **휴대폰 peak RAM이 0.895GB라는 뜻은 아니다**.

텍스트 인코더 두 개를 포함한 원래 BF16 모델 텐서는 3,584,094,818 bytes였다.
텍스트 임베딩 사전 계산에 따른 감소와 INT4 양자화에 따른 감소를 분리해 해석해야 한다.

## 생성된 파일

- TryonNet packed state: **286,091,438 bytes**.
- GarmentNet packed state: **261,416,097 bytes**.
- 두 파일 합계: **547,507,535 bytes = 547.5MB**. 두 denoiser 외 모델은 별도 필요하다.
- 이 샘플의 BF16 텍스트 임베딩 캐시: **8,199,289 bytes**. 큰 zero padding도 포함한다.

파일 경로와 SHA256, 모든 양자화 레이어는
[양자화 메타데이터](../experiments/2026-10-23-mobile_vton/outputs_int4_cpu/metrics.json)에 있다.
파일은 `torchao Int4CPULayout`의 CPU용 체크포인트다. Android/iOS 배포 파일은 아니며,
GPU에서는 원본 BF16 가중치로 CUDA용 레이아웃을 다시 만들어야 한다.
원래 safetensors 로더 대신 실행 래퍼의 `--quantized-state-dir`로 읽는다.

## 이미지 비교 조건

초기 샌드박스에서는 NVIDIA 장치 접근이 불가능해 CPU에서 비교했다. 768×1024는
스텝당 약 40초가 관찰되어 첫 전체 해상도 실행을 중단했고, 그 전에 저장한
INT4 파일을 재로딩해 **384×512, 28스텝, seed 42, guidance 2.5**로 비교한다.
BF16·INT4는 같은 사람·의류 입력과 동일한 텍스트 임베딩 캐시를 사용한다.
양쪽 모두 upstream의 직사각형 RoPE 리사이즈 기능을 같은 방식으로 적용한다.
따라서 이번 결과는 축소 해상도의 양자화 영향 확인이며 원래 해상도의 품질 검증은 아니다.

두 CPU 생성 작업은 공유 호스트에서 겹쳐 실행된다. 기록된 소요 시간은 실행 진단이며
정밀도별 독점 성능 비교가 아니다. 이전 BF16 GPU의 17.1초와도 비교하지 않는다.
모바일 실기기의 속도·peak RAM·NPU 호환성은 아직 측정하지 않았다.

## 재현 및 검증

코드와 모델 revision은 기존 실험과 동일하게 고정했다.
INT4 연산이 실제 출력에 영향을 주는지, 저장·재로딩 후 출력이 완전히 같은지,
CPU INT4 커널이 지원하지 않는 shape를 정확히 건너뛰는지 확인하는 테스트 **2개가 통과**했다.
실제 모델의 734개 Linear 레이어는 모두 변환되어 건너뛴 레이어가 없다.
상세 명령은 [INT4 실행 가이드](../experiments/2026-10-23-mobile_vton/INT4.md)에 있다.

## 완료된 이미지·메모리 비교

동일 입력·임베딩·seed의 384×512, 28스텝 추론이 **두 정밀도 모두 성공**했다.
INT4 쪽은 저장한 packed state를 실제로 재로딩해 실행했다.

| 측정 항목 | BF16 + 텍스트 캐시 | INT4 + 텍스트 캐시 |
| --- | ---: | ---: |
| 384×512 모델 텐서 저장량 | 1,938,183,570 bytes (1.938GB) | 885,239,186 bytes (0.885GB) |
| 전체 프로세스 최대 RSS | 4,283,719,680 bytes (4.284GB / 3.99GiB) | 2,836,054,016 bytes (2.836GB / 2.64GiB) |
| CPU 파이프라인 실행 시간 | 484.877초 | 323.011초 |

축소 해상도에서 모델 텐서는 **54.3%**, 전체 프로세스 최대 RSS는 **33.8%** 감소했다.
RSS는 Linux `getrusage(RUSAGE_SELF).ru_maxrss`로 측정했으며, 로드부터 종료까지의
고점을 포함한다. INT4 이미지는 별도 변환 프로세스에서 만든 파일을 읽었으므로
이 RSS에는 오프라인 양자화 작업이 포함되지 않는다. 공유 라이브러리·메모리 매핑 등도
포함하는 데스크톱 프로세스 값이며 모바일 RAM 요구량으로 그대로 적용할 수 없다.
두 작업의 실행 구간이 겹쳤고 워밍업·반복 측정도 하지 않았으므로 속도 향상률을
주장하지 않는다. 크기가 작아졌다는 이유로 모바일 NPU 실행이 확보되는 것도 아니다.

BF16 출력을 기준으로 한 픽셀 MAE는 **0.02362**(0–1 범위), PSNR은 **27.99dB**였다.
이는 두 출력의 차이이며 실제 착용 정답 이미지에 대한 정확도 점수가 아니다.

시각적으로 회색 가디건 형태와 대략적인 포즈는 양쪽 모두 유지된다. 이번 INT4 결과는
BF16보다 목·가슴 경계의 밝은 얼룩과 녹색 전면 여밈의 왜곡이 더 두드러진다.
단추·그래픽·니트 질감은 BF16에서도 불완전하다. 현재의 단순 INT4 설정은 **메모리
감소 확인에는 성공했지만 품질 통과·최종 모델 선정은 보류**한다. 한 샘플·축소 해상도
결과이므로 다른 입력과 원래 해상도의 결과로 일반화하지 않는다.

[전체 비교 이미지](../experiments/2026-10-23-mobile_vton/outputs_comparison/bf16_vs_int4.jpg)
· [의류 확대](../experiments/2026-10-23-mobile_vton/outputs_comparison/garment_detail.jpg)
· [비교 JSON](../experiments/2026-10-23-mobile_vton/outputs_comparison/comparison.json)

원래 해상도의 GPU 비교는 위와 같이 완료했다. 남은 검증은 민감한 레이어를 BF16/INT8로 남긴 혼합 정밀도,
여러 사람·의류 샘플의 품질 평가, 목표 휴대폰 런타임으로의 변환과 실기기 메모리 측정이다.
