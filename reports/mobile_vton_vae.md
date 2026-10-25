# Mobile-VTON VAE 경량화 검증 — 2026-10-23

## 결과

원본 VAE의 사용하지 않는 디코더를 제거하면 출력 변화 없이 가중치 약
99.1MB를 제외할 수 있었다. 512 타일링은 VAE 인코딩 구간의 GPU 고점을
44.3% 낮췄다. 그러나 **전체 최대 메모리는 1.906 → 1.899GiB로 약 0.4%만
줄었다**. 최종 경량 디코더의 FP32 실행이 새로운 최대 메모리 지점이다.

| 설정 | VAE 가중치 | VAE 인코딩까지의 GPU 고점 | 전체 CUDA peak allocated | 평균 생성 시간 |
| --- | ---: | ---: | ---: | ---: |
| 단계별 해제 + 원본 VAE | 167.64MB | 1.906GiB | 1.906GiB | 45.483초 |
| 인코더만 로드 | 68.55MB | 1.811GiB | 1.899GiB | 45.376초 |
| 인코더만 + 512 타일 | 68.55MB | **1.062GiB** | **1.899GiB** | 44.404초 |

MB는 10진수, GiB는 2진수다. VAE 구간 수치는 첫 측정 생성에서 인코딩이
끝날 때까지의 누적 CUDA allocated 고점이며, denoiser 상주 가중치도 포함한다.
측정 도중 peak 통계를 초기화하지 않았다. 전체 peak에는 인코딩부터 최종
디코딩까지 모두 포함한다. 원본 대비 두 변형의 전체 절감은 각각 약 6.85MiB,
7.50MiB로 작다.

## 구현과 검증

- `scripts/mobile_vton_vae.py`는 원본과 같은 Encoder를 구성하고 safetensors에서
  `encoder.*`와 필요한 `quant_conv.*`만 읽는다. 원본 디코더를 생성하거나
  CPU/GPU에 가중치를 복사하지 않는다. 별도의 최종 경량 디코더는 유지한다.
- 가중치 49,545,475개가 제외되며, VAE 인코더 34,274,208개는 그대로다.
- Diffusers 0.34.0의 기존 타일 인코딩 연산을 사용한다. 타일 512, 겹침 25%,
  이동 간격 384로 설정하고, 사람·의류 입력마다 실제 **6회**의 인코더 실행을
  확인했다. 타일마다 최대 크기는 512×512다.
- 입력은 768×1024, latent mean은 모든 설정에서 `[1,16,128,96]`이다.
  두 텍스트 인코더는 동일 임베딩 캐시로 대체했고, 기존 CUDA INT4 denoiser
  체크포인트를 재로딩했다. CFG는 기존 배치 방식을 유지했다.
- 원본 체크포인트와 인코더 추출의 posterior가 정확히 같은지, 타일링이 실제로
  동작하며 Diffusers 원본과 같은 출력을 내는지 확인하는 테스트를 추가했다.
  저장소의 `tests/` 테스트 **7개 통과**, 변경 파일 Ruff 및 컴파일 검사 통과.

각 변형은 같은 seed 42, guidance 2.5, scheduler shift 3.0, BF16 활성값에서
1스텝 워밍업 후 28스텝을 두 번 실행했다. vLLM CUDA 이미지 안에서 PyTorch를
사용했으며 GPU는 RTX 3080 Laptop 16GB다. 컨테이너·패키지·소스 revision과
스크립트 해시는 [환경 기록](../experiments/2026-10-23-mobile_vton/outputs_vae_comparison/environment.json)에 있다.

## 출력과 판단

인코더만 로드한 출력은 기준 실행 및 이전 INT4 결과와 PNG SHA256이 완전히
같았다. 모든 설정에서 두 번의 반복 출력도 각각 같았다.

타일링 출력은 기존 INT4 기준 MAE 0.005523, PSNR 42.50dB였다. 전체 형태는
비슷하지만 색감·질감에 차이가 있으며 출력이 동일하지 않다. 기존 INT4의
의류 프린트 손실도 해결하지 않는다. 이 지표는 정답 이미지에 대한 품질
평가가 아니며 한 샘플로 품질을 통과시키지 않는다.

[전체 비교](../experiments/2026-10-23-mobile_vton/outputs_vae_comparison/comparison.jpg)
· [원본 크기](../experiments/2026-10-23-mobile_vton/outputs_vae_comparison/comparison_native.png)
· [의류 확대](../experiments/2026-10-23-mobile_vton/outputs_vae_comparison/garment_detail.jpg)
· [수치 JSON](../experiments/2026-10-23-mobile_vton/outputs_vae_comparison/comparison.json)

인코더만 로드하는 방식은 출력이 유지되는 개선으로 사용할 수 있다. 현재
파이프라인에서 타일링은 전체 고점 개선 효과가 작으므로 선택 옵션으로 둔다.
다음 최적화 대상은 FP32 최종 디코더의 중간 메모리이며, 낮은 정밀도 또는
타일 디코딩을 적용하려면 별도의 출력 검증이 필요하다.

후속 [최종 디코더 FP16 검증](mobile_vton_decoder.md)에서 같은 latent의 출력
차이를 확인했고, VAE 타일링·로딩 개선과 결합해 전체 고점을 1.428GiB까지
낮췄다. 위 표는 FP32 최종 디코더를 유지한 당시의 결과다.

## 재현

기존 환경·가중치·INT4 파일이 준비된 상태에서 다음을 순차 실행한다.

```bash
bash scripts/run_mobile_vton_in_vllm.sh --config experiments/2026-10-23-mobile_vton/config_vae_baseline.json
bash scripts/run_mobile_vton_in_vllm.sh --config experiments/2026-10-23-mobile_vton/config_vae_encoder_only.json
bash scripts/run_mobile_vton_in_vllm.sh --config experiments/2026-10-23-mobile_vton/config_vae_encoder_tiled512.json
.venv/bin/python scripts/compare_mobile_vton_memory.py --suite vae --experiment experiments/2026-10-23-mobile_vton --output experiments/2026-10-23-mobile_vton/outputs_vae_comparison
.venv/bin/python -m pytest -q tests/
```

GPU는 공유 환경이므로 소수 실행의 시간 차이를 성능 향상으로 일반화하지
않는다. 호스트 peak RSS는 각각 2.387 / 2.077 / 2.117GiB이며 CUDA allocated와
다른 지표다. CUDA reserved는 약 2.75GiB이고 드라이버·컨텍스트는 포함하지
않는다. **안드로이드 RAM·속도는 아직 측정하지 않았다.**
