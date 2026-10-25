# Mobile-VTON 최종 디코더 FP16 검증 — 2026-10-23

## 결과

최종 경량 디코더를 FP16으로 실행하고 VAE 512 타일링과 로딩 순서 개선을
함께 적용하면 **시작부터 생성 완료까지 CUDA peak allocated가
1.900 → 1.428GiB로 24.8% 감소**했다. 이전 단계의 원본 해상도 INT4 결과
2.24GiB보다도 낮지만, 이 값은 안드로이드 전체 RAM 요구량이 아니다.

| 설정 | 생성 구간 최대 | 로딩·워밍업 포함 최대 | 평균 생성 시간 |
| --- | ---: | ---: | ---: |
| 인코더만 로드 + FP32 최종 디코더 | 1.899GiB | 1.900GiB | 40.561초 |
| 최종 디코더 FP16 | 1.811GiB | 1.811GiB | 43.363초 |
| FP16 + VAE 512 타일 인코딩 | 1.427GiB | 1.811GiB | 43.294초 |
| 위 설정 + INT4 로딩 순서 개선 | **1.428GiB** | **1.428GiB** | 35.354초 |

기존 로더는 packed INT4 파일을 읽기 전에 원본 BF16 denoiser 가중치를 GPU에
올려 초기 고점이 1.811GiB였다. 새 옵션은 원본 모델을 CPU에 두고
`load_state_dict(assign=True)`로 packed CUDA 상태를 직접 설치한다. 로더 개선
전후에는 latent 해시와 출력 PNG가 모두 같았다. CPU에서 모델을 구성하는
단계는 남아 있다.

공통 조건은 768×1024, 28스텝, seed 42, guidance 2.5, scheduler shift 3.0,
텍스트 임베딩 캐시, 기존 CUDA INT4 denoiser 체크포인트, BF16 활성값이다.
VAE·이미지 인코더·최종 디코더는 각 단계에서만 로드한다. 각 설정을 별도
프로세스에서 1스텝 워밍업 후 두 번 생성했다. GPU는 RTX 3080 Laptop 16GB,
실행 환경은 기존 vLLM CUDA 이미지 안의 PyTorch다. 공유 GPU이므로 시간
차이는 진단값이며 로더 변경으로 13% 빨라졌다고 해석하지 않는다.

## 품질 검증

FP32와 FP16 비교에는 **같은 최종 latent**를 사용했다. latent를 저장한 뒤
별도 프로세스에서 두 정밀도로 재실행했고, 각각의 PNG가 실제 파이프라인
출력과 정확히 일치했다. 이 검증은 타일링 입력과 타일링하지 않은 입력
각각에서 수행했다.

| 같은 latent에서 FP16 vs FP32 | 이미지 MAE (0–1) | PSNR | 최대 RGB 채널 차이 |
| --- | ---: | ---: | ---: |
| VAE 타일링 없음 | 0.0000919 | 64.43dB | 1/255 |
| VAE 512 타일링 사용 | 0.0000945 | 64.31dB | 2/255 |

이 샘플에서 FP16 자체의 차이는 작았고 NaN·Inf는 없었다. 모든 설정의
반복 생성도 각각 동일했다. **타일링까지 적용한 결과**를 타일링하지 않은
FP32 기준과 비교하면 MAE 0.00550, PSNR 42.51dB이며, 이는 FP16 오차보다
크다. 이전에 관찰한 타일 인코딩의 색감·질감 차이와 INT4의 의류 프린트
손실은 남아 있다. 이 지표는 정답 이미지에 대한 정확도가 아니며, 단일
샘플로 전체 상품 품질을 통과시킨 것은 아니다.

[비교 이미지](../experiments/2026-10-23-mobile_vton/outputs_decoder_comparison/comparison.jpg)
· [원본 크기](../experiments/2026-10-23-mobile_vton/outputs_decoder_comparison/comparison_native.png)
· [의류 확대](../experiments/2026-10-23-mobile_vton/outputs_decoder_comparison/garment_detail.jpg)
· [전체 비교 JSON](../experiments/2026-10-23-mobile_vton/outputs_decoder_comparison/comparison.json)
· [동일 latent 재검증](../experiments/2026-10-23-mobile_vton/outputs_decoder_comparison/precision_replay/precision_comparison.json)

## 구현과 재현

`scripts/mobile_vton_decoder.py`는 최종 디코더 입력만 FP16으로 변환하고,
이미지 후처리에는 FP32를 반환한다. denoiser·VAE 정밀도는 바꾸지 않는다.
원본 upstream과 체크포인트는 수정하지 않았다. 디코더 가중치 자체는
7,610,380 → 3,805,190 bytes이며, 큰 절감 효과는 중간 텐서에서 발생한다.
최종 디코더 타일링은 추가하지 않았다.

```bash
bash scripts/run_mobile_vton_in_vllm.sh --config experiments/2026-10-23-mobile_vton/config_decoder_fp32.json
bash scripts/run_mobile_vton_in_vllm.sh --config experiments/2026-10-23-mobile_vton/config_decoder_fp16.json
bash scripts/run_mobile_vton_in_vllm.sh --config experiments/2026-10-23-mobile_vton/config_decoder_fp16_vae_tiled512.json
bash scripts/run_mobile_vton_in_vllm.sh --config experiments/2026-10-23-mobile_vton/config_decoder_fp16_vae_tiled512_lowload.json
.venv/bin/python scripts/compare_mobile_vton_memory.py --suite decoder --experiment experiments/2026-10-23-mobile_vton --output experiments/2026-10-23-mobile_vton/outputs_decoder_comparison
.venv/bin/python -m pytest -q tests/
```

동일 CUDA Python 환경에서 `scripts/replay_mobile_vton_decoder.py`에
`--experiment experiments/2026-10-23-mobile_vton`과
`--output experiments/2026-10-23-mobile_vton/outputs_decoder_comparison/precision_replay`를
전달하면 저장된 latent의 재검증을 실행한다. 환경·소스 revision·최종 스크립트
해시는 [환경 기록](../experiments/2026-10-23-mobile_vton/outputs_decoder_comparison/environment.json)에 있다.

추가한 정밀도 경계·입력 보존·NaN 검출 테스트를 포함해 `tests/`의 **9개
테스트가 통과**했고, 변경 파일 Ruff 및 컴파일 검사를 통과했다.

## 배포 판단과 한계

품질 변화가 작은 다음 후보는 FP16 최종 디코더이며, 최대 절감이 필요한
실험 설정은 FP16 + VAE 타일링 + 개선된 로더다. 두 설정을 config 파일로
분리해 저장했다. 타일링의 품질 검증은 계속 필요하다.

최저 메모리 설정의 CUDA reserved는 2.219GiB, 호스트 peak RSS는 2.228GiB다.
CUDA allocated 1.428GiB와 호스트 RSS는 다른 지표이며, CPU에서 BF16 모델을
구성하는 로딩 과정도 남아 있으므로 이를 안드로이드 총 RAM으로 환산할 수
없다. 드라이버·컨텍스트 메모리는 CUDA allocated에 포함되지 않는다.
**안드로이드 실행 파일 변환과 실기기 RAM·속도 검증은 아직 수행하지 않았다.**
