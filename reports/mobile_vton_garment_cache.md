# Mobile-VTON 고정 의상 캐시 및 GPU 상주 테스트

## 결과

고정 의상의 VAE 확률분포와 DINO 이미지 임베딩을 GPU에서 사전 계산했다.
온라인에서는 DINO를 로드하지 않고, 필요한 네 모델과 선택한 의상 캐시를
계속 GPU에 상주시켰다. **CPU 모델 offload와 단계별 재로딩 없이** 테스트했다.

| 측정값 | 의상 캐시 없음·GPU 상주 | 의상 캐시 사용·GPU 상주 |
| --- | ---: | ---: |
| 로딩·워밍업 포함 CUDA 최대 allocated | 1.235GiB | **1.071GiB** |
| 로딩·워밍업 포함 CUDA 최대 reserved | 1.359GiB | **1.199GiB** |
| GPU 모델 가중치·버퍼 저장량 | 870.21MB | 697.05MB |
| CPU 프로세스 최대 RSS | 2.012GiB | **1.928GiB** |
| 생성 3회 평균 | 37.076초 | **42.389초** |
| 생성 시간 범위 | 34.980~38.761초 | **40.606~43.579초** |
| 모델 구성·GPU 로딩 | 3.160초 | 2.946초 |
| 1스텝 워밍업 | 2.384초 | 2.068초 |

동일 GPU 상주 조건에서 allocated가 **13.3% 감소**했다. 이번 공유 GPU
측정에서는 캐시 사용이 더 빨라지지 않았다. 두 설정을 차례로 실행했으므로
전체 시간 차이를 캐시 자체의 영향으로 확정할 수 없다. 28스텝 denoising은
그대로 남는다. 속도 향상은 이번 결과로 입증하지 않았다.

기존 최저값 **1.052GiB**는 VAE·디코더를 단계별로 해제하는 설정이었다.
이번 **1.071GiB는 모든 필요한 모델을 GPU에 계속 상주시킨 값**이다.
따라서 종전 최저값보다 메모리가 더 낮아진 실험은 아니다.

## 조건과 실제 실행 위치

768×1024, 28스텝, seed 42, guidance 2.5, scheduler shift 3.0, 배치 1을 유지했다.
두 denoiser의 Linear 가중치는 INT4, 활성값과 VAE는 BF16, 최종 디코더는
FP16이다. 기존 텍스트 임베딩 캐시, VAE 512px 타일링, 디코더 합성곱 32행
분할, 순차 CFG, 임시 텐서 해제, expandable segments도 유지했다.

| 구성 | 캐시 적용 후 온라인 실행 |
| --- | --- |
| 사람 VAE 인코더 | GPU 상주·매 생성마다 실행 |
| Tryon denoiser | INT4·GPU 상주·28스텝 실행 |
| Garment denoiser | INT4·GPU 상주·28스텝 실행 |
| 최종 디코더 | FP16·GPU 상주 |
| 의상 VAE 인코딩 | 캐시로 대체 |
| 의상 DINO 및 이미지 투영 계산 | 캐시로 대체·DINO 모델 미로드 |
| 텍스트 인코딩 | 기존 캐시 사용 |

모든 모델 파라미터와 버퍼가 시작·종료 시 모두 `cuda:0`임을 검사했다.
캐시 텐서도 GPU에 유지하며 phase load/release 이벤트는 0개다. VAE 호출은
생성당 두 번에서 사람 인코딩 한 번으로 줄었다. 생성 중 CPU로 latent를
복사·저장하던 디버그 감사도 끄고 검증했다.

신경망 계산에는 CPU를 사용하지 않는다. 파일 로딩, 입력 이미지 준비,
최종 이미지의 CPU 전송과 PNG 저장에는 CPU를 사용한다. Python·PyTorch·CUDA
런타임의 호스트 메모리도 남으므로 CPU RAM 사용량이 0이 되는 것은 아니다.

## 의상 사전 계산과 캐시 유효성

새 의상 캐시 파일은 **1,051,669 bytes = 1.052MB**다. 실제 텐서는 총
1,048,576 bytes이며, 기존 약 8.20MB 텍스트 캐시와 별도 파일이다.

- VAE Gaussian parameters: `[1, 32, 128, 96]`, BF16, 786,432 bytes.
- CFG를 포함한 투영 이미지 임베딩: `[2, 16, 4096]`, BF16, 262,144 bytes.

완성된 단일 sampled latent 대신 VAE 분포 파라미터를 저장했다. 생성 시
기존 위치에서 GPU로 샘플링하므로 사람 인코딩 이후의 난수 흐름과 seed
동작을 유지한다. DINO는 의상 이미지와 영 이미지의 조건을 계산한 뒤
이미지 투영 결과까지 저장했다. GarmentNet의 스텝별 특징은 캐시하지 않았다.

GPU 모델 준비 후 **의상 인코딩·캐시 저장은 0.550초**였다. 이 값은 사전
모델 로딩 3.281초, 이미지·텍스트 준비 0.201초, 캐시 identity 검증 준비
0.664초를 제외한다. 온라인 캐시 로드·GPU 설치는 **0.048초**였다.
이 비용들은 생성 42.389초에 포함되지 않는다.

의상 SHA256, 설명, 해상도, VAE 타일 크기, 정밀도, CFG 설정, 텍스트 캐시
해시, 소스·모델 revision, VAE·DINO·processor·INT4 projector 체크포인트
해시를 캐시에 기록한다. 불일치하거나 nonfinite 값이면 사용을 거부한다.
임베딩 검증을 위해 모델 파일을 해시하는 시작 비용은 별도로 기록했다.

선택한 의상 하나의 캐시만 GPU에 올렸다. 다른 고정 의상은 개별 캐시로
보관해 선택 시 로드할 수 있으며, 여러 의상 동시 상주 비용은 측정하지 않았다.

## 출력 검증과 시간 범위

캐시 유무 각각 세 번의 결과가 모두 같은 PNG SHA256을 가졌다. 이전
단계별 모델 해제 설정의 결과와도 같았다.

[입력 인물·의상·캐시 적용 결과](../experiments/2026-10-23-mobile_vton/outputs_garment_resident_cached/comparison_01.jpg)
· [캐시 적용 전후 원본 픽셀 비교](../experiments/2026-10-23-mobile_vton/outputs_garment_comparison/garment_cache_before_after.png)

```text
1f421d9d90b576db84dd19a7033dc403a650ba682d69a553c28f6763ad6b9b99
```

생성 시간에는 사람 VAE 인코딩, denoising, 최종 디코딩, 파이프라인 내부
이미지 준비·후처리와 최종 CPU 전송이 포함된다. 초기 모델 로딩, 처음의
입력 준비·캐시 설치·해시 검증, 워밍업, PNG 파일 저장은 제외한다.
각 프로세스에서 1스텝 워밍업 후 세 번 전체 생성을 측정했다.

GPU는 RTX 3080 Laptop 16GB이며 기존 vLLM CUDA 이미지에서 PyTorch를
실행했다. vLLM 서버는 사용하지 않았다. CUDA allocated/reserved는
드라이버·컨텍스트를 포함하지 않는다. CPU RSS 고점과 GPU 고점은 시점이
다를 수 있어 합산하지 않았다. 기존 INT4의 의류 프린트 손실과 VAE 타일링
품질 한계는 남아 있다. **안드로이드 실기기 추론 결과는 아니다.**

## 재현과 저장 위치

```bash
# 의상당 한 번 생성한다. 기존 캐시를 덮어쓰지 않는다.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash scripts/run_mobile_vton_in_vllm.sh --config experiments/2026-10-23-mobile_vton/config_garment_precompute.json
# 같은 상주 조건에서 각각 별도 프로세스로 실행한다.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash scripts/run_mobile_vton_in_vllm.sh --config experiments/2026-10-23-mobile_vton/config_garment_resident_uncached.json
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash scripts/run_mobile_vton_in_vllm.sh --config experiments/2026-10-23-mobile_vton/config_garment_resident_cached.json
.venv/bin/python scripts/compare_mobile_vton_garment_cache.py --experiment experiments/2026-10-23-mobile_vton --output experiments/2026-10-23-mobile_vton/outputs_garment_comparison
.venv/bin/python -m pytest -q tests/
```

[온라인 설정](../experiments/2026-10-23-mobile_vton/config_garment_resident_cached.json)
· [비교 원본 JSON](../experiments/2026-10-23-mobile_vton/outputs_garment_comparison/comparison.json)
· [환경 및 코드 해시](../experiments/2026-10-23-mobile_vton/outputs_garment_comparison/environment.json)

캐시는 `outputs_garment_precompute/garment_cache.pt`, 생성 PNG와 개별 측정은
`outputs_garment_resident_{uncached,cached}/`에 보관했다. 원본 upstream과
가중치는 수정하지 않았다. 캐시 불일치 거부, 샘플링·난수 흐름 보존,
인코더 호출 생략, latent CPU 감사 비활성화 테스트를 포함해 **27개 테스트가
통과**했다. Ruff·Black·컴파일 검사와 기존 compute·lifetimes 비교도 통과했다.
