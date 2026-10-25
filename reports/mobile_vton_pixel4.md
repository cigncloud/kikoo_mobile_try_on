# Mobile-VTON Pixel 4 실기기 검증

## 현재 판정

2026-10-24, Pixel 4(Android 13, Snapdragon 855/Adreno 640)에서 MNN 3.6.1
OpenCL로 **전체 28스텝 착용 이미지 생성을 1회 완료했다.** 768×1024, CFG 2.5에서
1장에 **2,081초(34분 41초)**, 최대 프로세스 RSS **1.603GiB**였다.
현재 Pixel 4 OpenCL 구현은 속도 때문에 모바일 실사용 후보에서 제외한다.
VAE FP32는 입력 처리 수정 후 원본과 일치한다. 메모리 절약 실행기로 Tryon
denoiser 단일 forward는 완료했다. 최신 INT4/FP16 혼합 후보는 Pixel 4에서
평균 24.511초, 최대 RSS 1.589GiB, 상대 RMSE 0.08390을 기록했다.
기존 INT4의 0.29655보다 단일 forward 오차가 줄었다. 전체 이미지의 고정 입력
대조군 비교는 아래에 기록했으며, 여러 사람·의상에 대한 품질 평가는 남아 있다.
동일 모델의 OpenCL Normal은 평균 31.178초, RSS 1.602GiB, 상대 RMSE 0.08466으로
이득이 없어, 현재 컴포넌트 후속 검증 후보는 OpenCL Low 설정이다.

기기 총 RAM은 시스템이 보고한 5.34GiB다. ADB 재연결 직후 사용 가능 RAM은
1.25GiB였고, 실행 중 다른 앱과 시스템 상태에 따라 변했다. APK가 아닌
Android arm64 CLI 실험이며, 데스크톱 CUDA 측정과 구분한다.

## 전체 생성 결과

| 단계 | 로딩·세션 준비 포함 시간 | 실제 forward·출력 다운로드 합계 | 최대 RSS |
| --- | ---: | ---: | ---: |
| 사람 VAE, 경계 포함 타일 6개 | 292.35초 | 75.34초 | 0.975GiB |
| Tryon, 순차 CFG 56회 | 1,747.78초 | **1,647.47초(27분 27초)** | **1.603GiB** |
| 최종 디코더 | 37.50초 | 5.40초 | 0.897GiB |
| 전체 | **2,081초** | 단계별 합계와 전체 벽시계 시간은 범위가 다름 | **1.603GiB** |

전체 시간은 단계별 모델·세션 생성, 입력 업로드, 캐시 읽기, CPU의 CFG·Euler
계산과 VAE 타일 혼합, 출력 전송·중간 latent 저장, 모니터 대기를 포함한다.
오프라인 의상 사전 계산, ADB 전송, JPEG 리사이즈·정규화, 호스트 PNG 저장은
제외한다. APK·카메라 입력 경로의 측정은 아니다. 세션 준비 약 301초를 제외해도
대부분의 처리 시간이 남는다. denoiser 호출은 평균 29.42초, 범위 24.64~33.04초였다.

단계를 별도 프로세스로 실행해 이전 모델과 GPU 컨텍스트를 해제했다. 고정 형태
세션은 첫 forward 후 `releaseModel()`로 직렬화 모델 버퍼를 해제했다. denoising
중 forward 후 RSS는 1.317~1.320GiB였고, 시스템 MemAvailable 최저는
804,336KiB(약 785.5MiB), 관측 swap은 0이었다. 중간 `dumpsys meminfo`의
TOTAL PSS는 약 1.308GiB였다. RSS와 PSS는 전용 GPU VRAM 수치가 아니다.
관측 배터리 온도는 약 30.5~38.3°C였으며, 후반 호출 시간이 늘었다.
온도·DVFS의 인과관계를 분리하는 교차 반복 실험은 수행하지 않았다.

고정 의상의 FP32 GarmentNet 출력을 timestep 28개에 대해 미리 계산하고 FP16으로
저장했다. 이 특징 캐시만 **1,849,688,064바이트(1.723GiB)/의상**이다. 이 실험은
주 생성기 287.72MiB만으로 배포할 수 있다는 증거가 아니며, 이 캐시 방식도
상품 카탈로그에 그대로 적용하기에는 저장량이 크다. 기존 BF16 posterior·투영
임베딩을 공유했고, 명시적인 CPU FP32 난수 파일을 고정했다.

VAE posterior 상대 RMSE는 0.00000106으로 대조군과 일치했다. 첫 CFG·Euler
업데이트와 VAE 타일 혼합을 별도로 재계산한 결과 최대 절대 차이는 모두 0이었다.
모든 출력은 유한했다. 최종 디코더 입력 상대 RMSE는 0.09123, 이미지 텐서는
0.16192였다. 대조군은 같은 FP16 의상 특징 캐시를 사용하는 원본 FP32 CUDA
denoiser·decoder다. 이 수치는 픽셀 수치 차이이며 지각 품질 점수가 아니다.

시각적으로 회색 가디건·민트 테두리와 사람의 자세·배경이 유지됐다. 원상품의
프린트·단추·니트 질감은 대조군과 휴대폰 출력 모두에서 차이가 남는다.
한 사람·한 의상·한 시드 결과이므로 일반적인 품질 합격으로 해석하지 않는다.
신경망 단계는 OpenCL을 요청했고 출력 backend는 VAE 209개, Tryon 3,507개,
decoder 125개 모두 3이었다. CPU는 스케줄러·CFG·후처리를 수행하며, 이 기록은
모든 개별 커널의 GPU 전용 실행을 입증하지 않는다.

[전체 결과·시각 비교](../experiments/2026-10-24-pixel4/generation_hybrid_low/comparison.png)
· [원본 결과 PNG](../experiments/2026-10-24-pixel4/generation_hybrid_low/pixel4_result.png)
· [전체 측정 JSON](../experiments/2026-10-24-pixel4/generation_hybrid_low/comparison.json)
· [실행 절차](../experiments/2026-10-24-pixel4/generation_hybrid_low/README.md)

## 측정 결과

| 컴포넌트 | 입력/출력 범위 | 평균 추론 시간 | 프로세스 최대 RSS | 출력 비교 |
| --- | --- | ---: | ---: | --- |
| VAE encoder FP32, 입력 복원 | 사람 이미지 512×512 타일 1개 | 16.904초, 3회 | 0.943GiB | MAE 0.00000936, 상대 RMSE 0.00000107 |
| Garment UNet INT4, 입력 복원 | 128×96 의상 latent, 특징 28개 | 5.527초, 3회 | 0.727GiB | 특징별 상대 RMSE 0.105~0.392 |
| Tryon UNet INT4, 입력 복원 | 256×96 결합 latent, 단일 forward | 24.057초, 3회 | 1.464GiB | 상대 RMSE 0.2965 |
| Tryon UNet INT8, 호스트 CPU 대조군 | 256×96 결합 latent, 단일 forward | 16.400초, 1회 | 2.130GiB | 상대 RMSE 0.02690 |
| Tryon UNet INT8, Pixel 4 OpenCL Low | 256×96 결합 latent, 단일 forward | 25.463초, 1회 | 1.815GiB | 상대 RMSE 0.5144 |
| Tryon INT8 + FP32 제외 8개, OpenCL Low | 256×96 결합 latent, 단일 forward | 24.349초, 1회 | 1.809GiB | 상대 RMSE 0.5344; 제외 후보 |
| Tryon 구조 기반 INT4/FP16, 호스트 CPU | 256×96 결합 latent, 단일 forward | 16.496초, 1회 | 1.934GiB | 상대 RMSE 0.08112 |
| Tryon 구조 기반 INT4/FP16, OpenCL Low | 256×96 결합 latent, 단일 forward | 24.511초, 3회 | 1.589GiB | 상대 RMSE 0.08390 |
| Tryon 구조 기반 INT4/FP16, OpenCL Normal | 동일 모델·입력, 단일 forward | 31.178초, 3회 | 1.602GiB | 상대 RMSE 0.08466 |
| Tryon UNet FP16, Pixel 4 시도 | 256×96 결합 latent, 단일 forward | 메모리 압박으로 종료 | RSS 2.246GiB에서 종료 | 출력 없음 |
| Decoder FP16, 종전 실행 | 128×96 latent → 768×1024 이미지 | 12.158초, 3회 | 0.891GiB | 상대 RMSE 0.00633; 입력 복원 방식 재측정 전 |

시간은 워밍업 이후 forward와 출력 다운로드를 포함한다. 모델 로딩,
세션·커널 생성, 입력 업로드·복원, 워밍업, 파일 저장은 제외한다.
정확도는 마지막 측정 forward의 저장 출력과 FP32 PyTorch fixture를 비교한 값이다.
VAE 세션 생성 49.739초와 워밍업 16.894초는 별도다.
VAE 입력 복원은 회당 평균 약 0.00294초, 의상망은 약 0.01174초,
Tryon 입력 복원은 약 0.665초였다.

RSS에는 프로세스에 보이는 GPU 매핑이 포함되지만 드라이버 전체 메모리를
보장하지 않는다. MNN 자체 allocator 수치만으로 메모리를 판단하면 안 된다.
컴포넌트별 최대값을 합산해 전체 파이프라인 최대 메모리로 해석할 수 없다.
OpenCL 출력 텐서의 backend는 모두 3으로 보고됐으나, 이는 개별 커널의
GPU 전용 실행을 입증하는 추적 결과는 아니다.

## 출력 오차 판정 정정

초기 벤치마크는 입력을 한 번 업로드하고 워밍업·측정을 반복했다. 이 경로에서
첫 forward의 중간 출력은 원본과 일치했지만 이후 forward의 출력이 달라졌다.
입력을 매번 복원하자 호스트의 64×64 진단 입력과 Pixel 4의 512×512 입력
모두 VAE 출력이 원본과 일치했다. 따라서 **초기의 큰 VAE 오차를 MNN 변환
오류로 판단한 결론은 철회한다.** 내부 입력 버퍼 수명의 세부 원인은 확정하지 않았다.

기존 FP16·FP32·CPU·GroupNorm 치환 대조군도 입력 복원 없이 측정했으므로
변환 오류의 근거로 사용하지 않는다. 이전 Tryon의 27.07초와 큰 출력 오차도
유효한 고정 입력 정확도 결과로 사용하지 않는다. 원본 로그는 보존한다.

입력을 복원해도 의상망의 INT4 출력 차이는 남았다. 비양자화 MNN CPU 대조군은
특징별 상대 RMSE 0.000067~0.000281로 원본과 일치했고, INT4 CPU 대조군은
0.111~0.391로 차이가 재현됐다. 따라서 의상망 오차는 MNN OpenCL보다
양자화 후보의 영향으로 판단한다. MNN의 INT4는 Conv·MatMul 가중치를
양자화하므로 데스크톱 TorchAO Linear-only INT4와 다른 후보이다.

Tryon에서도 같은 경향을 확인하기 위해 ONNX에서 MNN INT8 모델(391MiB)을
생성해 호스트 CPU 대조군을 실행했다. 상대 RMSE는 0.02690으로 INT4의
0.29655보다 낮았고 출력은 모두 유한했다. 그러나 같은 INT8 모델의 Pixel 4
OpenCL 실행은 25.463초, peak RSS 1.815GiB였고 상대 RMSE가 0.5144로
증가했다. 이 결과는 `precision=2`(Low) 설정의 해당 후보를 제외하는 근거다.
다른 OpenCL 정밀도 설정까지 같은 결과라고 단정하지 않는다.

## 선택적 INT4/FP16 실험

초기 제외 8개 후보는 실제 저장 가중치 수 대신 `입력 채널×출력 채널`로
레이어 크기를 추정했다. depthwise group을 반영하지 못해 작은 레이어를
선택했고, 호스트 상대 RMSE 0.02889와 달리 Pixel 4에서는 0.53436이었다.
이는 민감도 보정이나 유효한 대형 레이어 선별 실험이 아니다.

이후 직렬화된 MNN에서 실제 가중치 수를 읽는 검사기를 추가했다. 최신 정책은
공간 convolution, 가중치 65,536개 이하, time embedding, self-attention key를
FP16으로 보존하고 나머지는 asymmetric INT4 block32로 저장한다. 이는 구조적
휴리스틱이며 데이터 기반 레이어 민감도 보정은 아니다. 여러 조건을 함께
바꿨으므로 개선을 특정 레이어나 block32 한 가지에 귀속할 수 없다.

MNN 3.6.1에서는 `bits=16`이 INT8로 제한된다. `bits=0`으로 제외한 채 나머지를
INT4로 변환하고, 그 결과를 별도로 `--fp16` 재변환했다. 모델 내부 검사 결과
97개 convolution은 FP16, 341개는 INT4이며 float 가중치는 남지 않았다.
모델 파일은 301,700,316바이트(287.72MiB)다. 기존 INT4보다 파일과 RSS는
커졌지만 상대 RMSE는 약 71.7% 감소했고 추론 시간 증가는 약 1.9%였다.
실행 중 시스템 MemAvailable 최저는 541,144KiB(약 528.5MiB)였다.

OpenCL Low는 FP16 저장·FP16 연산, Normal(`precision=0`)은 FP16 저장·FP32
연산으로 커널을 빌드한다. 이는 공통 커널의 컴파일 정의이며 모든 최적화 연산이
FP32로 실행됐다는 증거는 아니다. Normal 비교는 평균 31.178초, RSS 1.602GiB,
상대 RMSE 0.08466으로 Low보다 정확도가 좋아지지 않았다. 3회 시간은
29.698/31.609/32.228초이며, 온도와 DVFS 영향을 분리하는 교차 반복 실험은
하지 않았다. 결과는 별도 manifest와 출력에 저장했다. 이 설정은 가중치 파일의
INT4/FP16 형식과 구분한다.

## 메모리 부족과 남은 검증

수정 전 Tryon 실행은 시스템 MemAvailable이 89,624KiB로 내려가 모니터가
종료했다. 입력을 하나씩 읽고 복사본을 유지하지 않는 실행기로 재시도하자
MemAvailable 약 0.3GiB를 유지하며 단일 forward 3회가 완료됐다. 당시 최대
RSS는 약 1.464GiB였다. 이는 전체 생성 이전의 컴포넌트 측정이다.

수정 실행기는 입력을 매번 파일에서 하나씩 읽어 복사하고 즉시 해제한다.
이 경로는 Pixel 4에서 성공했고 모델 출력은 nonfinite 없이 완료됐다.
INT8 OpenCL도 같은 스트리밍 실행기로 완료됐지만, 실행 중 MemAvailable 최저가
180,016KiB(0.172GiB)까지 내려갔다. 메모리 여유가 작으므로 모델 교체 시에도
동일한 모니터를 유지해야 한다.

FP16 Tryon 대조군은 모델 파일 731MiB를 기기에 올리는 데 성공했지만, 실행 중
RSS 2,355,244KiB(2.246GiB)와 MemAvailable 30,436KiB에서 모니터가 종료했다. 따라서
Pixel 4에서는 FP16 전체 denoiser가 현재 메모리 예산에 맞지 않는다. INT8은
메모리 기준으로 실행됐지만 OpenCL 출력 오차가 커서 전체 scheduler·CFG·타일
합성 후보로 사용하지 않는다. 구조 기반 INT4/FP16은 단일 입력 오차가 개선된
후보였다. 이후 전체 scheduler·CFG·타일 합성을 완료했지만 속도가 실사용에
맞지 않았다. 여러 사람·의상·시드와 앱·배터리 경로 검증은 수행하지 않았다.

## 다음 검증 방향

현재 구현은 추가 양자화보다 추론 구조 변경이 우선이다. 모바일 오프라인
기준선은 반복 확산 대신 warping·generator 구조인 DM-VTON 같은 경량 모델로
다시 측정한다. [공식 DM-VTON](https://github.com/KiseKloset/DM-VTON)의 약 40fps·
37.8MB 보고는 Tesla T4 기준이며 Pixel 4 수치나 배포 파일 크기로 해석하지
않는다. 후속으로 CuPy correlation을 ATen으로 치환해 MNN 변환과 Pixel 4
실행을 완료했다. DM-VTON FP32, 192×256에서 반복 1.70초·RSS 347MiB,
커널 캐시 재사용 첫 결과 1.90초였다. 원본 CUDA에서도 목·어깨 합성 문제가
남아 화질은 미합격이다. [DM-VTON 실측 보고서](dm_vton_pixel4.md)를 참조한다. 연구용
CC BY-NC-SA 4.0 코드이며 상용 모델 선정과 구분한다. 고품질 생성은 서버 GPU
경로를 별도로 비교하고, 소수 스텝 증류·NPU 포팅은 독립적인 개발 과제로 둔다.

## 저장 위치와 재현

- [요약 JSON](../experiments/2026-10-24-pixel4/summary.json)
- [실험 안내와 실행 명령](../experiments/2026-10-24-pixel4/README.md)
- [VAE 재측정](../experiments/2026-10-24-pixel4/outputs_vae_encoder_opencl_buffer_fp32_refreshed/comparison.json)
- [INT4 의상망 재측정](../experiments/2026-10-24-pixel4/outputs_garment_opencl_buffer_refreshed/comparison.json)
- [Tryon 재측정](../experiments/2026-10-24-pixel4/outputs_tryon_opencl_buffer_streamed/comparison.json)
- [Tryon INT8 호스트 대조군](../experiments/2026-10-24-pixel4/outputs_tryon_host_int8_control/comparison.json)
- [Tryon INT8 Pixel 4 OpenCL](../experiments/2026-10-24-pixel4/outputs_tryon_int8_opencl_streamed/comparison.json)
- [INT4/FP16 Pixel 4 OpenCL Low](../experiments/2026-10-24-pixel4/outputs_tryon_structural_int4_fp16_block32_opencl/comparison.json)
- [INT4/FP16 Pixel 4 OpenCL Normal](../experiments/2026-10-24-pixel4/outputs_tryon_structural_int4_fp16_block32_opencl_normal/comparison.json)
- [혼합 정책과 변환 절차](../experiments/2026-10-24-pixel4/policies/README.md)
- [Tryon 이전 메모리 부족 로그](../experiments/2026-10-24-pixel4/outputs_tryon_opencl_buffer_refreshed/tryon_opencl_buffer_refreshed.monitor.log)

코드·모델 revision과 SHA256은 실험 환경 JSON에 기록했다. Mobile-VTON 원본
소스·가중치와 기존 AGENTS.md는 수정하지 않았다.
