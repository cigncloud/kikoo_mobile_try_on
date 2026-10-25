# Mobile-VTON 메모리 절감 검증

2026-10-23, vLLM CUDA 컨테이너 안의 PyTorch에서 같은 INT4 체크포인트와
768×1024, 28스텝, seed 42, guidance 2.5 조건을 비교했다. GPU는 RTX 3080
Laptop 16GB이며 안드로이드 실기기 측정은 아니다.

| 방식 | 평균 생성 시간 | CUDA peak allocated | INT4 기준 출력 |
| --- | ---: | ---: | --- |
| 기존 배치 CFG + 상주 전처리 모델 | 34.104초 | 2.229GiB | 기준 |
| CFG 두 분기 순차 실행 | 34.047초 | 2.231GiB | 미세 차이 |
| 전처리 모델 단계 로드·해제 | 36.081초 | **1.906GiB** | 동일 |
| 두 방법 동시 적용 | 42.986초 | **1.906GiB** | CFG와 동일한 미세 차이 |

전처리 모델 단계 해제는 VAE와 DINO 이미지 인코더를 인코딩 뒤 해제하고,
최종 디코더를 디코딩 시점에만 로드했다. 기존 2.229GiB에서 1.906GiB로
**14.5% 감소**했다. 출력 PNG 해시는 기존 INT4 실행과 완전히 같았고, 모델
가중치의 상주 저장량은 0.906GiB에서 0.582GiB로 줄었다. 대신 매 생성마다
모델을 다시 읽어 평균 시간이 약 5.8% 늘고 호스트 RSS가 증가했다.

CFG 순차 실행은 두 분기를 따로 실행했지만, 이 모델의 큰 가중치와 garment
conditioning이 계속 남아 있어 peak가 줄지 않았다. PyTorch 커널 실행 순서가
달라져 기존 출력과 MAE 0.000723, PSNR 52.83dB의 작은 차이가 생겼다. 따라서
현재 기본 경로에는 적용하지 않는다.

검증 코드는 `scripts/mobile_vton_memory.py`와
`scripts/compare_mobile_vton_memory.py`에 있으며, 실행 모드는
`--memory-mode phase_release` 또는 `combined`로 선택한다. 생성된 네 모드의
metrics와 비교 이미지는 `experiments/2026-10-23-mobile_vton/outputs_memory_*`에
있다. 이 결과는 안드로이드 RAM·GPU 메모리·속도를 보장하지 않으며, 실제 배포
전에는 Android 런타임과 대상 기기에서 다시 측정해야 한다.
