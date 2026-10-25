# Mobile-VTON 임시 텐서 해제 및 CUDA 메모리 풀 실험

## 결과

기존 최저 메모리 설정에서 **CUDA 최대 할당량을 1.428 → 1.263GiB로
11.5% 더 줄였다.** 계산이 끝난 임시 텐서를 바로 해제하고 CUDA allocator의
`expandable_segments:True`를 적용했다. 생성 구간의 최대 reserved도
**2.219 → 1.469GiB로 33.8% 감소**했다.

| 설정 | 로딩·워밍업 포함 최대 allocated | 생성 구간 최대 reserved | 평균 생성 시간 |
| --- | ---: | ---: | ---: |
| 이전 최저 설정 | 1.428GiB | 2.219GiB | 35.354초 |
| 임시 텐서 조기 해제 | 1.270GiB | 2.219GiB | 37.488초 |
| 위 설정 + expandable segments | **1.263GiB** | **1.469GiB** | 39.492초 |

조건은 768×1024, 28스텝, seed 42, guidance 2.5, scheduler shift 3.0,
텍스트 임베딩 캐시, 동일 CUDA INT4 denoiser 가중치, BF16 활성값이다.
인코더 전용 VAE의 512px 타일링, FP16 최종 디코더, 단계별 모델 해제,
INT4 전송 순서 개선도 유지했다. 각 설정은 별도 프로세스에서 1스텝
워밍업과 두 번의 전체 생성을 수행했다. 기준값은 이전 완료 실험을
재사용했고 입력·설정·임베딩 해시를 비교 스크립트에서 대조했다.
RTX 3080 Laptop 공유 GPU에서 측정했으므로 시간 차이로 속도 향상이나
저하를 확정하지 않는다. 속도 개선은 이번 실험에서 입증하지 않았다.

## 변경과 출력 검증

기존 파이프라인은 이전 스텝의 의류 특징, denoiser 입력, guidance 출력의
뷰 등을 다음 스텝이나 최종 디코딩까지 보관했다. 스케줄러 계산 직후
이 참조들을 삭제하고, 디코딩 직전에는 사용이 끝난 조건 텐서도 삭제했다.
매 스텝 약 153.6MiB, 디코딩 직전 약 7.3MiB의 live allocation이 해제됐다.
각 새 설정에서 워밍업·전체 생성을 합한 60쌍의 해제 전후 기록을 검증했다.

`scripts/mobile_vton_lifetimes.py`는 고정한 upstream 함수의 두 위치를 AST로
확인한 뒤 참조 삭제만 삽입한다. 예상 위치 수가 다르면 실행을 중단하며,
원본 upstream 파일과 학습 가중치는 수정하지 않는다. 조기 해제는
`release_working_tensors` 옵션으로 켠다. allocator 설정은 별도 환경변수다.

세 설정의 두 번 생성 결과가 모두 같은 PNG SHA256을 가졌다.

```text
1964494b4220f18fc871fc86b77255d7fc4001dd8b17df999ed51bd7e4a8c558
```

최종 디코더 입력 latent도 워밍업을 포함해 설정 간에 같았고 NaN·Inf는
없었다. 이번 변경에 따른 추가 이미지 차이는 이 샘플에서 0이다.
기존 INT4의 프린트 손실과 VAE 타일링의 질감 차이는 남아 있으며,
단일 샘플의 일치가 전체 상품 품질을 보장하지는 않는다.

## 재현과 기록

```bash
bash scripts/run_mobile_vton_in_vllm.sh --config experiments/2026-10-23-mobile_vton/config_lifetimes_release.json
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash scripts/run_mobile_vton_in_vllm.sh --config experiments/2026-10-23-mobile_vton/config_lifetimes_release_expandable.json
.venv/bin/python scripts/compare_mobile_vton_memory.py --suite lifetimes --experiment experiments/2026-10-23-mobile_vton --output experiments/2026-10-23-mobile_vton/outputs_lifetimes_comparison
.venv/bin/python -m pytest -q tests/
```

[비교 JSON](../experiments/2026-10-23-mobile_vton/outputs_lifetimes_comparison/comparison.json),
[비교 이미지](../experiments/2026-10-23-mobile_vton/outputs_lifetimes_comparison/comparison.jpg),
[원본 크기](../experiments/2026-10-23-mobile_vton/outputs_lifetimes_comparison/comparison_native.png),
[환경 및 코드 해시](../experiments/2026-10-23-mobile_vton/outputs_lifetimes_comparison/environment.json)에
측정값과 실행 조건을 저장했다. guidance 사용 여부별 출력·참조 해제 및
upstream 구조 변경 거부 테스트를 추가해 **12개 테스트가 통과**했다.
Ruff, Black, Python 컴파일, 셸 문법 검사와 기존 memory·VAE·decoder 비교
스크립트의 회귀 검증도 통과했다.

## 모바일 적용 한계와 다음 병목

allocated는 텐서 등의 실제 할당량, reserved는 PyTorch가 확보한 메모리 풀이다.
reserved는 생성 구간의 최고값이며 초기 로딩 전체의 최고값은 별도로 기록하지
않았다. 둘 다 드라이버·CUDA 컨텍스트를 포함하지 않는다. 최저 설정의 호스트
peak RSS는 2.220GiB로, CPU 로딩 비용은 여전히 남아 있다. 이 수치들을
안드로이드 총 RAM 요구량으로 환산할 수 없다. allocator 변경은 CUDA 런타임
최적화이며 안드로이드용 최적화가 아니다. **휴대폰 변환·실기기 실행은 미검증**이다.

해제 후에도 denoising 누적 고점은 약 1.255GiB, 최종 디코딩을 거친 고점은
1.263GiB다. 더 큰 절감에는 두 단계의 활성값 메모리를 함께 줄이는 작업이
필요하다. 최종 디코더에는 GroupNorm이 있어 단순 타일링은 통계를 바꿀 수
있으므로, 적용한다면 동일 latent에서 화질을 별도로 검증해야 한다.

## 추가 최적화 후보 — 아직 미측정

후속 [중간 계산 메모리 실험](mobile_vton_compute.md)에서 FFN·K/V 후보를
실측했다. 두 방법만으로 전체 고점은 줄지 않았고, 실제 병목인 합성곱의
행 분할과 순차 CFG를 결합해 1.052GiB를 측정했다. 아래는 당시의 후보
목록이며 최신 채택 판단과 품질 차이는 후속 보고서를 따른다.

다음 후보는 현재 고정한 코드와 설치된 라이브러리를 확인해 정리했다.
추가 절감량이나 이미지 일치는 아직 검증하지 않았다. 우선 레이어별 고점을
측정해 실제 최대값을 만드는 연산에 적용해야 한다.

1. **FFN 분할 계산:** upstream `models/attention.py`의 TryOn·Garment 블록은
   `set_chunk_feed_forward()`를 지원하지만 기본 `_chunk_size`는 `None`이다.
   토큰을 나눠 계산하면 FFN 확장 중간값을 줄일 수 있다. 분할 크기는 토큰
   길이를 나눌 수 있어야 하며 실행 시간과 INT4 연산의 수치 차이를 검증한다.
2. **Attention K/V 복제 제거:** 두 denoiser는 KV head가 1인데 현재 processor는
   `repeat_interleave`로 K/V를 query head 수만큼 복제한다. 설치된 PyTorch의
   native GQA를 사용하는 후보를 비교한다. 실제 절감은 사용 커널에 달려
   있으며 math 경로로 돌아가면 내부 복제가 남을 수 있다. 이 CUDA 변경을
   안드로이드 런타임 지원으로 해석하지 않는다.
3. **CFG 조건 생성부터 분리:** 앞선 순차 CFG 실험은 denoiser 호출만 나누고
   입력·의류 특징의 배치 복제는 유지해 전체 최대값을 줄이지 못했다. 다음
   후보는 각 분기에 필요한 조건을 그때 생성해 복제 자체를 줄이는 방식이다.
   기존 guidance 의미와 영 텐서 조건을 유지하며 추가 실행 비용을 측정한다.
4. **최종 디코더 분할 및 가중치 수명 조정:** 이미 적용한 VAE 인코더 타일링과
   별개다. 디코더의 큰 중간값을 영역별로 계산하되 GroupNorm의 전체 영역
   통계를 유지하는 구현을 검토한다. denoising 완료 후 가중치를 실제로
   해제하고 다음 요청에서 재로드하는 방법도 비교할 수 있다. CPU로 옮기는
   것만으로 휴대폰 전체 RAM이 줄었다고 판단하지 않는다. 디코딩만 개선해도
   denoising 고점 약 1.255GiB는 남는다.

원본 해상도를 유지하는 우선순위는 **레이어별 측정 → FFN/K/V 중간값 개선 →
CFG 조건 생성 개선과 디코더 메모리 개선**이다. 더 큰 구조 변경 후보는
활성값 INT8과 채널 수를 줄이는 증류이며, 별도의 보정·학습 및 의류 디테일
검증이 필요하다. 1GiB 미만 달성 여부는 이 후보들을 측정하기 전에는 확정할
수 없다.
