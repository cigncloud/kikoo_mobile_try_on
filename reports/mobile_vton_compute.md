# Mobile-VTON 중간 계산 메모리 최적화

## 결과와 선택한 설정

**최종 디코더의 합성곱을 32행씩 계산하고 CFG 두 분기를 순차 실행해,
최대 CUDA allocated를 1.263 → 1.052GiB로 16.7% 더 줄였다.** 생성 구간의
최대 reserved는 1.469 → 1.137GiB로 22.6% 감소했다. 모델 가중치와
768×1024 출력 해상도, 28스텝은 유지했다.

이번 샘플의 선택 설정은
[`config_compute_decoder32_cfg.json`](../experiments/2026-10-23-mobile_vton/config_compute_decoder32_cfg.json)이다.
FFN 분할, native GQA, denoiser 합성곱 분할은 이 설정에 포함하지 않는다.

| 설정 | 로딩·워밍업 포함 최대 allocated | 생성 구간 최대 reserved | 평균 생성 시간 |
| --- | ---: | ---: | ---: |
| 이전 최저 설정 재측정 | 1.263GiB | 1.469GiB | 39.018초 |
| FFN 256토큰 분할 | 1.263GiB | 1.486GiB | 44.518초 |
| native GQA로 K/V 복제 제거 | 1.263GiB | 1.469GiB | 39.093초 |
| FFN + native GQA | 1.263GiB | 1.486GiB | 44.329초 |
| denoiser·디코더 합성곱 32행 분할 | 1.231GiB | 1.352GiB | 45.651초 |
| 위 합성곱 분할 + 순차 CFG | **1.052GiB** | **1.137GiB** | 42.310초 |
| 디코더 합성곱만 분할 + 순차 CFG | **1.052GiB** | **1.137GiB** | 39.688초 |

공통 조건은 seed 42, guidance 2.5, scheduler shift 3.0, 동일 텍스트 임베딩
캐시와 CUDA INT4 가중치, BF16 denoiser 활성값, FP16 최종 디코더다. 앞서
적용한 VAE 인코더 전용 로딩·512px 타일링, 모델 단계별 해제, packed 가중치
전송 순서 개선, 임시 텐서 해제, expandable segments도 유지했다.
각 설정은 별도 프로세스에서 1스텝 워밍업과 두 번의 전체 생성을 수행했다.
RTX 3080 Laptop 공유 GPU이므로 시간은 진단값이며 속도 개선을 확정하지 않는다.

## 실제 고점과 구현

별도 프로파일링 실행에서 CUDA 누적 최고값을 넘은 모듈을 기록했다.
forward 중간에 최고값 카운터를 초기화하거나 텐서를 보관하지 않았다.
계측 실행은 시간 비교에서 제외했고 출력 PNG는 기준 설정과 같았다.
가장 좁게 관찰한 고점 구간은 다음 두 합성곱이었다.

- denoiser: `up_blocks.2.resnets.0.conv2.pw_conv` — 누적 약 1.255GiB.
- 최종 디코더: `up_blocks.2.upsamplers.0.conv` — 누적 약 1.263GiB.

FFN·Attention 내부 텐서를 줄여도 이 합성곱 고점은 남았다. FFN 256토큰
분할은 upstream의 기존 기능을 사용했고, native GQA는 명시적인 K/V 헤드
복제를 제거했다. CUDA에서는 Flash SDPA만 허용해 math 경로의 내부 복제를
숨기지 않았다. 두 방법의 단독·결합 실행 모두 전체 최대값을 줄이지 못했다.

`scripts/mobile_vton_spatial.py`는 stride 1의 홀수 커널 합성곱만 행 단위로
분할한다. 3×3 커널은 위아래 한 행을 더 읽고 유효 결과만 복사해 이미지
경계와 타일 경계를 처리한다. 입력이 32MiB 미만인 연산은 기존 경로를 쓴다.
stride 2 다운샘플링은 유지한다. denoiser의 separable convolution을 분할하는
옵션은 depthwise·pointwise 연산을 타일마다 이어 실행해 큰 중간값을 피한다.

**GroupNorm은 전체 이미지에 그대로 적용한다.** 디코더 전체를 타일로 나눠
정규화 통계를 바꾸는 방식이 아니다. 가중치를 CPU로 옮기는 방식도 사용하지
않았다. upstream 원본 파일과 체크포인트는 수정하지 않았다.

이전 순차 CFG 실험은 큰 디코더 고점 때문에 전체 절감이 없었다. 이번에는
디코더 합성곱 고점도 함께 낮춰 CFG의 denoiser 중간값 절감이 전체 최대값에
반영됐다. 조건 입력·의류 특징의 배치 복제 자체는 여전히 유지하며, denoiser
호출 두 분기를 나눠 계산하는 기존 구현이다.

## 이미지 검증

FFN·GQA 세 설정은 기준 PNG와 완전히 같았다. 선택 설정은 기준과 **완전히
같지는 않다**. 전체 파이프라인 출력의 MAE는 0.00068345, PSNR은 54.75dB다.
평균 RGB 채널 차이는 약 0.174/255, 최대는 22/255였다. 모든 설정에서 두 번
생성한 PNG가 각각 동일했고 NaN·Inf는 없었다.

선택 설정과 denoiser까지 합성곱을 분할한 설정은 워밍업·전체 생성의 최종
latent 해시와 두 PNG가 모두 같았다. 따라서 이번 샘플에서는 디코더만
분할하는 설정을 선택했다. 선택 설정의 생성 PNG SHA256은 다음과 같다.

```text
1f421d9d90b576db84dd19a7033dc403a650ba682d69a553c28f6763ad6b9b99
```

디코더 분할 자체의 오차는 **동일한 저장 latent와 FP16 가중치**로 별도
재실행해 분리했다. 선택 설정의 latent에서 분할/비분할 디코더의 이미지
MAE는 0.0000000349, PSNR은 98.64dB, 최대 차이는 1/255였다. 각 재실행은
해당 실제 파이프라인 PNG도 재현했다. 전체 출력 차이에는 순차 CFG의 연산
경로 변화가 포함되며, 디코더 분할 오차만으로 전체 차이를 설명하지 않는다.

이 지표는 이전 INT4 결과와의 차이이며 정답 이미지에 대한 정확도가 아니다.
기존 INT4의 의류 프린트 손실과 VAE 타일링의 변화는 남아 있다. 한 쌍의
이미지로 상품 전체의 품질을 통과시킨 것은 아니다.

[선택 설정 비교](../experiments/2026-10-23-mobile_vton/outputs_compute_comparison/recommended_comparison.jpg)
· [의류 확대](../experiments/2026-10-23-mobile_vton/outputs_compute_comparison/garment_detail.jpg)
· [전체 비교 JSON](../experiments/2026-10-23-mobile_vton/outputs_compute_comparison/comparison.json)
· [프로파일 요약](../experiments/2026-10-23-mobile_vton/outputs_compute_comparison/profile_summary.json)
· [동일 latent 재검증](../experiments/2026-10-23-mobile_vton/outputs_compute_comparison/decoder_replay/spatial_replay.json)

## 재현과 검증

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash scripts/run_mobile_vton_in_vllm.sh --config experiments/2026-10-23-mobile_vton/config_compute_decoder32_cfg.json
.venv/bin/python scripts/compare_mobile_vton_memory.py --suite compute --experiment experiments/2026-10-23-mobile_vton --output experiments/2026-10-23-mobile_vton/outputs_compute_comparison
.venv/bin/python -m pytest -q tests/
```

전체 설정을 다시 비교하려면 첫 명령의 config를 `config_compute_baseline.json`,
`config_compute_ffn256.json`, `config_compute_gqa.json`, `config_compute_combined.json`,
`config_compute_spatial32.json`, `config_compute_spatial32_cfg.json`으로 각각
바꿔 실행한다. 프로파일링은 `config_compute_profile.json`을 사용한다.
동일 CUDA 컨테이너에서 `scripts/replay_mobile_vton_spatial.py`에
`--experiment experiments/2026-10-23-mobile_vton`과
`--output experiments/2026-10-23-mobile_vton/outputs_compute_comparison/decoder_replay`를
전달하면 디코더 오차를 재검증한다.

정규화·위치 임베딩을 포함한 GQA 비교, 합성곱 경계·입력 보존·전체 영역
GroupNorm, stride 2 제외 검증을 추가해 **24개 테스트가 통과**했다.
환경·revision·최종 코드 해시는
[환경 기록](../experiments/2026-10-23-mobile_vton/outputs_compute_comparison/environment.json)에 있다.

## 남은 병목과 모바일 한계

선택 설정에서 첫 VAE 인코딩이 전체 최대값 **1.052GiB**를 만들고 이후
denoising과 디코딩은 이를 넘지 않았다. 현재 1GiB 미만은 달성하지 못했다.
다음 절감 후보는 VAE 인코딩 타일 크기와 버퍼 수명이며, 타일 크기를
바꾸면 latent와 화질을 다시 비교해야 한다.

선택 설정의 호스트 peak RSS는 **2.228GiB**다. CUDA allocated/reserved는
드라이버·컨텍스트를 포함하지 않고 호스트 RSS와도 별개다. reserved 표는
생성 구간의 최고값이며 초기 로딩 전체의 최고값은 아니다. 위 결과는 기존
vLLM CUDA 이미지에서 실행한 PyTorch 실험으로, vLLM 서버는 사용하지 않았다.
**안드로이드 변환·실기기 RAM·속도는 아직 검증하지 않았다.**
