# DM-VTON 모바일 전처리 검증

2026-10-24 Pixel 4(Android 13, Snapdragon 855/Adreno 640)에서 공식
VITON-Clean 입력과 실제 Android 전처리 경로를 검증했다. DM-VTON은 192×256
입력으로 유지하고, 사진은 MediaPipe Pose Landmarker Lite로 눈·골반 기준 3:4
crop을 계산한다. 골반이 프레임 밖인 셀피는 얼굴·어깨 랜드마크로 상반신 crop을
계산한다. Pixel 4에서는 Pose segmentation의 GPU delegate가 비정상
float 값을 반환해 전처리만 CPU delegate로 고정했다. warp와 generator는 계속
OpenCL FP32로 실행한다.

## 결과

| 항목 | 측정값 |
| --- | ---: |
| 첫 모델 초기화(커널 생성 포함) | 45.72초 |
| 자동 인물 전처리 | 0.207초 |
| 모델 추론(의상별) | 1.67~1.68초 |
| 추론+원본 해상도 복원(의상별) | 1.77~1.79초 |
| 전처리부터 3개 의상까지 전체 계측 | 5.59초 |
| 계측 중 최대 프로세스 RSS | 507,650,048B (약 484MiB) |
| 공식 CUDA 입력 대비 Pixel 4 상대 RMSE | 3.41e-4 |

반복 추론 출력은 서로 동일했고, 잘못된 tensor 크기와 사람 없는 사진은
거부했다. 카메라 임시 URI의 full-size write/decode, 취소 시 삭제, path
traversal 거부도 통과했다. 모델 초기화 후 캐시가 있으면 초기화 비용은 별도이며,
측정은 사진 디코드·PNG 저장·UI 렌더링을 제외한다.

## 품질 확인 범위

공식 CUDA 기준선의 8개 비교 이미지는
[official_comparison.jpg](../experiments/2026-10-24-dm_vton-preprocessing/official_comparison.jpg)에
있다. 이 검증은 전처리와 수치 일치 확인이며 FID/LPIPS나 임의 사용자 사진의
제품 품질 판정이 아니다. Pixel 4 결과는 계측 APK의 private files에 다음 이름으로
저장된다: `official_pixel4.png`, `person_aligned.png`,
`preprocessed_white_tee.png`, `preprocessed_black_tee.png`,
`preprocessed_navy_hoodie.png`.

## 재현

```bash
cd android
bash ./gradlew --offline assembleDebug assembleDebugAndroidTest lintDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb shell am instrument -w com.mobiletryon.demo.test/com.mobiletryon.demo.TryOnInstrumentation
adb shell run-as com.mobiletryon.demo cat files/instrumentation.json
```

의상은 공식 VITON-Clean `test_color`와 `test_edge`를 준비 스크립트에서
사전 변환한다. 흰색 원단과 인쇄 무늬를 보존하며, 런타임에 의상 분할을 반복하지
않는다. 연구용 데이터와 DM-VTON CC BY-NC-SA 조건은 상용 배포 전에 별도 검토가
필요하다.
