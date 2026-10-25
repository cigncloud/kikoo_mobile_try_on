# DM-VTON 공식 사용법 및 앱 입력 처리 감사

확인일: 2026-10-24. 공식 README, Colab, 추론 코드, 연결된 웹 데모 및 이슈를 확인했다.
이번 작업은 소스 감사이며 새로운 추론·화질 실험은 실행하지 않았다.

## 결론

현재 앱은 RGB 정규화와 warp→generator 흐름은 따르지만, **공식 웹 데모의 사람 배경 제거·포즈 기반 정렬·의상 분할 전처리를 재현하지 않았다.** 기존 임의 샘플의 화질 실패는 유효하나, 모델 자체의 품질 한계나 전처리 수정의 효과를 확정할 수 없다. 공식 VITON-Clean 입력으로 기준선을 먼저 재현해야 한다.

공식 데이터셋 추론은 준비된 사람·의상·마스크를 읽는다. 임의 사진을 받는 공식 웹 데모에는 별도의 전처리 모델이 있다. 추론 네트워크에 사람 parsing/pose 텐서를 넣지 않는다는 사실이 카메라 사진의 전처리를 생략해도 된다는 뜻은 아니다.

## 확인한 버전과 소스

| 소스 | 확인 내용 |
| --- | --- |
| [DM-VTON README][readme] | `5dcd04ee3f5b1fb392df1184c887c1a7e910c72e`; 현재 main과 로컬 revision 일치 |
| [공식 실행 명령][test-script]·[데이터 로더][dataset]·[파이프라인][pipeline] | 디렉터리, 텐서 범위, 정렬, 합성 순서 |
| [공식 Colab][colab] | notebook을 내려받아 데이터·가중치 다운로드 및 추론 셀 확인 |
| [README에서 연결한 KiseKloset 웹 데모][web-service] | `118e0da5037550912cf33b1b7bdd46ecf36141e7`; 임의 사진 전처리·배경 복원 |
| [release 1.0.0][release] | student `dmvton_pf_warp.pt` + `dmvton_pf_gen.pt`; teacher FS-VTON과 구별 |

Colab 사본 SHA256: `185cd6969283296bea435ed9e3ded327254628dd824d822cb723fc92acbb5d73`.
Google Drive notebook은 바뀔 수 있으므로 재현 시 버전도 기록한다.

## 공식 데이터셋으로 추론하는 방법

README의 VITON-Clean을 내려받아 다음 구조로 준비한다. 공식 test split은 416쌍이다.

```text
VITON_test/
├── test_pairs.txt       # 각 줄: person_filename garment_filename
├── test_img/           # 사람 RGB
├── test_color/         # 의상 RGB
└── test_edge/          # 의상 영역 마스크; test_color와 같은 파일명
```

공식 예제 크기는 **가로 192 × 세로 256**이다. 데이터 로더의 `size=(256,192)`는 자동 정렬 기능이 아니다. 고정 resize 문장은 주석 처리돼 있고, 실제 transform은 각 변을 가까운 16의 배수로 맞춘다. 재현 시험에서는 세 입력을 모두 미리 같은 192×256 크기로 준비한다.

- 사람·의상: RGB, NCHW `[1,3,256,192]`, 범위 `[-1,1]`.
- 의상 `edge`: **의상 전체 영역이 흰색인 foreground mask**다. 윤곽선만 그린 그림이 아니다. 배경 0, 의상 1, `[1,1,256,192]`; nearest resize, 범위 `[0,1]`. 내부에서 `edge > 0.5`로 이진화한다.
- 사람 parsing map, DensePose, OpenPose 파일은 데이터 로더의 학습 분기에만 있다. 테스트 입력으로 요구되지 않는다.
- PF student warp/gen 두 파일을 함께 사용하고 `eval()`, gradient 비활성화, `align_corners=True`를 유지한다.

공식 소스 디렉터리에서 데이터 경로를 실제 위치로 바꾸어 실행한다.

```bash
python test.py --project runs/test --name DM-VTON_demo \
  --device 0 --align_corners --batch_size 1 --workers 16 \
  --dataroot /path/to/VITON-Clean/VITON_test \
  --pf_warp_checkpoint /tmp/dmvton_pf_warp.pt \
  --pf_gen_checkpoint /tmp/dmvton_pf_gen.1.pt
```

`/tmp` 가중치는 기존 시험에서 받은 공식 release 파일이다. generator의 `.1`은 로컬 다운로드 파일명 차이다. 출력은 `runs/test/DM-VTON_demo*/results/{tryon,visualize}/`에 저장된다. `visualize`는 사람→의상→warped 의상→최종 결과 순서다.

이 명령은 추론 후 FID/LPIPS 평가도 수행한다. 소수 샘플의 육안 시험을 전체 416쌍의 공식 FID 재현으로 보고하지 않는다. README의 검증 환경은 Python 3.10, PyTorch 1.13.1, torchvision 0.14.1이며, requirements는 CuPy 11.0.0, 설치 스크립트는 CUDA toolkit 11.7을 지정한다. 기존 시험의 최신 환경·CuPy API 호환 패치는 환경까지 동일한 재현은 아니다.

## 임의 사진을 받는 공식 웹 데모

[TryonService][web-service]는 다음을 수행한다.

1. **배경 제거:** MediaPipe selfie segmentation을 0.5로 이진화해 사람 밖을 흰색으로 바꾸고 원본·마스크를 보관한다.
2. **포즈 crop:** YOLOv7 pose의 눈 2점·골반 2점 bbox에 위아래 각각 높이의 20%, 좌우 각각 폭의 100% 여백을 추가한다. 이미지 경계로 제한하고 포즈 검출 실패 시 추론하지 않는다.
3. **정렬:** 해당 영역을 중앙에서 3:4 비율로 자른 뒤 192×256으로 resize한다.
4. **의상 분할:** 192×256 의상에서 U²-Net의 작은 모델 `u2netp`로 foreground를 예측·정규화한다. 고정 의상은 마스크를 사전 계산할 수 있다.
5. **추론·복원:** warp→generator 결과를 원래 crop 위치에 붙이고 사람 마스크로 원래 배경을 복원한다.

이는 웹 데모의 구현 사실이며 임의 자세·의상의 품질 보장은 아니다. 웹 데모의 `mobile_warp.pt/mobile_gen.pt`와 main release 가중치의 내용 동일성은 이번 감사에서 확인하지 않았다. 재현할 때 버전을 섞지 않는다.

웹 데모도 검토 후 적용해야 한다. `pil_edge`를 전달하는 분기에서 실제 마스크 대신 `pil_clothes`를 grayscale 변환한다(51–53행). 사전 계산 마스크를 연결할 때 이 분기를 그대로 복사하면 안 된다. 이 문제를 현재 앱 실패 원인으로 확인한 것은 아니다.

## 현재 구현과의 차이

| 항목 | 현재 구현 | 확인 결과 |
| --- | --- | --- |
| 사람 배경 | 사진 배경 그대로 | 웹 데모의 사람 분할·흰 배경 처리 누락 |
| 사람 위치·크기 | 전신 resize 또는 수동 3:4 crop | 눈·골반 기준 crop과 다름 |
| 의상 | 다른 모델의 샘플·teaser crop | 공식 VITON-Clean 재현 입력이 아님 |
| 의상 마스크 | 초기 흰색 임계값; 앱은 밝은 배경 연결 영역 제거 | annotation/U²-Net 결과와 비교·검수하지 않음 |
| 정규화·합성 | RGB `[-1,1]`, edge 임계값, align true, warp→generator | 핵심 흐름은 공식 파이프라인과 일치 |
| 검증 | 임의 소수 샘플, 모바일 수치 비교 | 공식 샘플 품질 재현은 미실시 |
| 결과 복원 | 192×256 crop만 표시 | 웹 데모의 원본 배경 복원 누락 |

대조한 로컬 파일: `scripts/run_dm_vton_sample.py`, `scripts/export_dm_vton_mnn.py`, `scripts/prepare_android_assets.py`, Android의 `TryOnEngine.java`, `PhotoCropView.java`. 기존 수치·메모리 측정은 [Pixel 4 보고서](dm_vton_pixel4.md)에 있다.

## 다음 검증 순서

1. VITON-Clean 원본 사람·의상·edge와 release PF 가중치로 CUDA/CuPy 기준선을 재현한다. 먼저 여러 쌍의 네 칸 비교 이미지를 보관하고, 정량 재현에는 전체 test split을 사용한다.
2. 같은 입력 텐서를 MNN·Pixel 4에 넣어 warped 의상과 최종 결과를 각각 비교한다.
3. 사용자 사진에 웹 데모의 사람 분할·포즈 crop을 적용한다. 고정 의상 마스크를 사전 계산·육안 검수하고, 같은 사진의 전처리 전후 결과를 비교한다.
4. 품질 개선을 확인한 처리를 앱에 적용한다. 검출 실패를 처리하고, 전처리·추론·복원을 포함한 지연과 최대 RSS를 다시 측정한다.

확정된 것은 입력 처리의 차이다. 어느 전처리가 얼마나 기여하는지, 수정 후 제품에 쓸 수 있는지는 미검증이다.

## 이슈 확인

- [#14][issue14]: 저자는 custom images에 공식 Colab 절차와 `data_root` 교체를 안내했다.
- [#25][issue25]: 사람·배경 분리 실패 보고가 있다. 밝은 배경 제안은 다른 사용자의 댓글이며 저자가 보장한 해결책이 아니다.
- [#26][issue26]: 입력 크기 불일치 오류가 보고됐으며 완결된 공식 해결 절차는 확인되지 않았다.

[readme]: https://github.com/KiseKloset/DM-VTON/blob/5dcd04ee3f5b1fb392df1184c887c1a7e910c72e/README.md
[test-script]: https://github.com/KiseKloset/DM-VTON/blob/5dcd04ee3f5b1fb392df1184c887c1a7e910c72e/scripts/test.sh
[dataset]: https://github.com/KiseKloset/DM-VTON/blob/5dcd04ee3f5b1fb392df1184c887c1a7e910c72e/dataloader/viton_dataset.py#L39-L55
[pipeline]: https://github.com/KiseKloset/DM-VTON/blob/5dcd04ee3f5b1fb392df1184c887c1a7e910c72e/pipelines/dmvton_pipeline.py#L32-L59
[colab]: https://colab.research.google.com/drive/1oLg0qe0nqLuIeaklzwbkk3IOKmMb0clk
[web-service]: https://github.com/KiseKloset/KiseKloset/blob/118e0da5037550912cf33b1b7bdd46ecf36141e7/src/api/tryon/service/__init__.py#L33-L156
[release]: https://github.com/KiseKloset/DM-VTON/releases/tag/1.0.0
[issue14]: https://github.com/KiseKloset/DM-VTON/issues/14#issuecomment-1881380132
[issue25]: https://github.com/KiseKloset/DM-VTON/issues/25
[issue26]: https://github.com/KiseKloset/DM-VTON/issues/26
