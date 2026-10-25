# DM-VTON Pixel 4 실기기 검증

> The original raw-input run below is retained for historical comparison. The
> official-input and automatic-preprocessing validation supersedes its quality
> conclusion; see [dm_vton_preprocessing.md](dm_vton_preprocessing.md).

## 판정

DM-VTON PF student를 ONNX→MNN으로 변환해 Pixel 4에서 **warp와 generator를
연속 실행했다.** 양자화 없는 FP32 모델, 192×256 출력에서 반복 추론은 약
**1.70초/장**, 커널 캐시가 있는 새 프로세스의 첫 결과는 **1.90초**, 최대
프로세스 RSS는 **347MiB**였다. 정지 사진용 지연 시간은 크게 개선됐지만,
약 0.59fps이므로 실시간 카메라 합성에는 부족하다. 이 샘플의 화질은 미합격이다.

추가 [공식 사용법 감사](dm_vton_usage_audit.md)에서 웹 데모의 사람 배경 제거·포즈
crop·의상 분할 전처리와 현재 구현의 차이를 확인했다. 아래 수치는 기존 임의 입력의
측정값이다. 공식 VITON-Clean 입력의 품질 재현과 전처리 수정 효과는 아직 미검증이다.

이전 Mobile-VTON은 768×1024에서 34분 41초·1.603GiB였다. 해상도와 생성 구조가
다르므로 두 결과를 동일 품질 조건의 가속 배율로 해석하지 않는다.

## 측정 결과

| 설정 | 반복 평균, 3회 | 프로세스 시작→첫 결과 | 최대 RSS | 출력 비교 |
| --- | ---: | ---: | ---: | --- |
| Pixel 4 OpenCL Low, FP16 연산 | 1.356초 | 41.17초 | 239.0MiB | 상대 RMSE 0.17418 |
| Pixel 4 OpenCL High, FP32 연산 | 1.690초 | 48.68초 | 341.4MiB | 상대 RMSE 0.00002198 |
| 상반신 입력, FP32, 캐시 생성 | 1.709초 | 48.46초 | 339.8MiB | CUDA 원본 대비 0.003213 |
| 상반신 입력, FP32, 캐시 재사용 | **1.695초** | **1.897초** | **347.0MiB** | 위와 동일 |
| 호스트 MNN CPU FP32 | 0.451초 | 0.519초 | 150.1MiB | 상대 RMSE 0.00002333 |
| RTX 3080 Laptop, 공식 CUDA/CuPy | 0.01655초 | 별도 미측정 | Torch 할당 160.5MiB | 변환 전 기준선 |

휴대폰은 Android 13, Snapdragon 855/Adreno 640, RAM 5.34GiB다. APK가 아닌
Android arm64 CLI로 측정했다. 두 세션을 동시에 유지하며 매번 입력을 복원했다.
반복 시간에는 입력 업로드, warp, 중간 출력 다운로드와 CPU를 거친 generator
입력 복사, generator, 최종 출력 다운로드가 포함된다. JPEG 전처리·PNG 인코딩,
ADB 전송, 파일 저장은 제외한다. cold 시간은 입력 파일 읽기와 모델 로딩·세션
준비를 포함한다. 캐시 저장은 반복 종료 후 수행했다.

최종 캐시 재사용 실행의 warp 평균은 1.663초, generator는 0.0324초였다.
커널 캐시는 1,510,232바이트이며 기기·드라이버·정밀도·모델별로 관리해야 한다.
캐시 없는 최초 실행의 약 49초는 여전히 남는다. 반복 3회뿐이며 지속 실행의
발열·배터리·프레임 안정성은 측정하지 않았다.

## 크기와 정밀도

두 FP32 MNN 파일은 합계 **37,720,932바이트(35.97MiB)**이고 파라미터는
9,350,124개다. INT4/INT8 모델이 아니다. Low/High는 실행 정밀도 설정이다.
모델 파일 크기와 실행 중 메모리는 다르다. CUDA에서도 모델 논리 저장량은
37.57MB지만 Torch 최대 할당은 168.30MB, 예약은 201.33MB였다.

휴대폰 RSS는 프로세스에 매핑된 메모리이며 전용 GPU VRAM 전체 수치가 아니다.
첫 forward 출력 backend는 warp 1,291개, generator 54개 모두 OpenCL(3)이었다.
이는 개별 커널이 모두 GPU에서 실행됐다는 추적 증거는 아니며 호스트 복사는
명시적으로 포함돼 있다.

CuPy 전용 7×7 correlation을 동일 순서의 ATen 연산으로 바꿔 ONNX로 내보냈다.
채널 순서·제로 패딩·stride는 독립적인 픽셀 계산 테스트로 검증했다. 원본 CUDA와
ATen CPU 최종 출력의 상대 RMSE는 0.006018이었다. 그 원인을 분리하지 않았으므로
완전한 동일 출력이라고 주장하지 않는다. FP32 MNN/Pixel 출력은 ATen CPU 기준과
거의 일치했다. FP16 후보는 수치 오차가 커서 기본 후보에서 제외했다.

## 화질과 입력 한계

한 사람·한 카디건을 전신 리사이즈와 수동 상반신 crop으로 비교했다.
의상 마스크는 흰 배경 임계값으로 만들었으며 공식 VITON edge가 아니다.
상반신 crop 좌표는 원본 768×1024에서 `(144, 0, 624, 640)`이다.

전신 입력은 의상이 과도하게 늘어나고 원래 보라색 티셔츠가 남았다. 상반신은
위치가 일부 개선됐으나 어깨·목·팔의 경계가 부자연스럽다. **공식 CUDA 원본에서도
같은 문제가 재현돼 모바일 변환 오류만으로 설명할 수 없다.** 여러 모델에 대한
동일한 상반신 데이터·마스크 평가를 거쳐야 최종 선정할 수 있다.

- [전신 비교](../experiments/2026-10-25-dm_vton/comparison.png)
- [상반신 비교](../experiments/2026-10-25-dm_vton/comparison_upper_body.png)
- [Pixel 4 결과](../experiments/2026-10-25-dm_vton/pixel4_upper_body_cached/result.png)
- [측정 JSON](../experiments/2026-10-25-dm_vton/summary.json)
- [재현 절차·아티팩트](../experiments/2026-10-25-dm_vton/README.md)

## 소스와 사용 조건

[공식 DM-VTON](https://github.com/KiseKloset/DM-VTON) revision
`5dcd04ee3f5b1fb392df1184c887c1a7e910c72e`, release `1.0.0` PF warp/gen을 사용했다.
공개 코드의 CC BY-NC-SA 4.0·학술용 조건은 상용 최종 선정 시 별도로 해결해야 한다.
현재 결과는 연구 후보의 기능·지연·메모리 검증이며 제품 품질 합격이 아니다.
