package com.mobiletryon.demo;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.util.Log;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.ByteBufferExtractor;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.List;

/** Mobile adaptation of KiseKloset's foreground whitening and eye/hip crop.
 * Own, call and close on one worker thread, like the GPU try-on sessions.
 */
public final class PersonPreprocessor implements AutoCloseable {
    private final PoseLandmarker landmarker;

    public PersonPreprocessor(Context context) {
        this(context, Delegate.CPU);
    }

    PersonPreprocessor(Context context, Delegate delegate) {
        landmarker = PoseLandmarker.createFromOptions(context,
            PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath("preprocess/pose_landmarker_lite.task").setDelegate(delegate).build())
                .setRunningMode(RunningMode.IMAGE).setNumPoses(2)
                .setMinPoseDetectionConfidence(0.5f).setMinPosePresenceConfidence(0.5f)
                .setOutputSegmentationMasks(true).build());
    }

    public Result prepare(Bitmap original) {
        long started = System.nanoTime();
        // Closing a BitmapImageBuilder MPImage recycles its bitmap: own a copy.
        float scale = Math.min(1f, 960f / Math.max(original.getWidth(), original.getHeight()));
        Bitmap resized = Bitmap.createScaledBitmap(original, Math.max(1, Math.round(original.getWidth() * scale)), Math.max(1, Math.round(original.getHeight() * scale)), true);
        Bitmap detection = resized.copy(Bitmap.Config.ARGB_8888, false);
        if (resized != original) resized.recycle();
        try (MPImage image = new BitmapImageBuilder(detection).build()) {
        PoseLandmarkerResult result = landmarker.detect(image);
        if (result.landmarks().size() != 1) throw new InputException("얼굴이 가려지지 않은 한 사람의 정면 사진을 선택해주세요.");
        List<NormalizedLandmark> pose = result.landmarks().get(0);
        Rect crop = upperBodyCrop(original.getWidth(), original.getHeight(), pose);
        Rect face = faceProtection(original.getWidth(), original.getHeight(), pose);
        List<MPImage> masks = result.segmentationMasks().orElse(java.util.Collections.emptyList());
        try {
            if (masks.size() != 1) throw new IllegalStateException("Missing person segmentation mask");
            MPImage mask = masks.get(0);
            FloatBuffer probabilities = ByteBufferExtractor.extract(mask).order(ByteOrder.nativeOrder()).asFloatBuffer();
            int maskWidth = mask.getWidth(), maskHeight = mask.getHeight();
            if (probabilities.remaining() != maskWidth * maskHeight) throw new IllegalStateException("Invalid segmentation dimensions");
            float[] smallMask = new float[maskWidth * maskHeight]; probabilities.get(smallMask);
            float maskMin = Float.POSITIVE_INFINITY, maskMax = Float.NEGATIVE_INFINITY;
            int maskSelected = 0;
            for (float value : smallMask) {
                if (!Float.isFinite(value) || value < -0.001f || value > 1.001f) throw new IllegalStateException("Invalid person mask probability: " + value);
                maskMin = Math.min(maskMin, value); maskMax = Math.max(maskMax, value);
                if (value > 0.5f) maskSelected++;
            }
            Log.i("TryOnPreprocess", "Mask " + maskWidth + "x" + maskHeight + " min=" + maskMin + " max=" + maskMax + " selected=" + maskSelected + "/" + smallMask.length);
            int width = original.getWidth(), height = original.getHeight();
            byte[] foreground = new byte[width * height];
            int[] pixels = new int[foreground.length]; original.getPixels(pixels, 0, width, 0, 0, width, height);
            int selected = 0;
            for (int y = 0; y < height; y++) {
                int maskY = Math.min(maskHeight - 1, (int) ((long) y * maskHeight / height));
                for (int x = 0; x < width; x++) {
                    int index = y * width + x, maskX = Math.min(maskWidth - 1, (int) ((long) x * maskWidth / width));
                    float probability = smallMask[maskY * maskWidth + maskX];
                    if (probability > 0.5f) { foreground[index] = 1; selected++; }
                    else pixels[index] = Color.WHITE;
                }
            }
            if (selected < pixels.length / 50) throw new InputException("사람이 너무 작게 보여요. 얼굴부터 골반까지 더 크게 촬영해주세요.");
            Bitmap white = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888);
            Bitmap aligned = Bitmap.createBitmap(TryOnEngine.WIDTH, TryOnEngine.HEIGHT, Bitmap.Config.ARGB_8888);
            new Canvas(aligned).drawBitmap(white, crop, new Rect(0, 0, TryOnEngine.WIDTH, TryOnEngine.HEIGHT), new Paint(Paint.FILTER_BITMAP_FLAG));
            white.recycle();
            return new Result(original, aligned, foreground, crop, face, (System.nanoTime() - started) / 1e9);
        } finally { for (MPImage mask : masks) mask.close(); }
        }
    }

    static Rect upperBodyCrop(int width, int height, List<NormalizedLandmark> points) {
        if (points.size() < 25) throw new InputException("얼굴이 가려지지 않은 사진이 필요해요.");
        NormalizedLandmark leftEye = points.get(2), rightEye = points.get(5);
        if (!visible(leftEye, 0.25f) || !visible(rightEye, 0.25f))
            throw new InputException("얼굴이 가려지지 않은 정면 셀피를 사용해주세요.");
        boolean hipsVisible = visible(points.get(23), 0.25f) && visible(points.get(24), 0.25f);
        boolean shouldersVisible = visible(points.get(11), 0.25f) && visible(points.get(12), 0.25f);
        if (!hipsVisible && !shouldersVisible)
            throw new InputException("얼굴과 어깨가 가려지지 않게 정면으로 촬영해주세요.");
        float left = 1, top = 1, right = 0, bottom = 0;
        int[] anchors = hipsVisible ? new int[]{2, 5, 23, 24} : new int[]{2, 5, 11, 12};
        for (int index : anchors) {
            NormalizedLandmark point = points.get(index);
            left = Math.min(left, point.x()); right = Math.max(right, point.x());
            top = Math.min(top, point.y()); bottom = Math.max(bottom, point.y());
        }
        if (!hipsVisible) {
            float shoulderWidth = Math.abs(points.get(11).x() - points.get(12).x());
            top = Math.max(0f, Math.min(points.get(2).y(), points.get(5).y()) - 0.08f);
            bottom = Math.min(1f, Math.max(bottom + shoulderWidth * 1.7f, top + 0.55f));
        }
        if (bottom - top < 0.2f || right - left < 0.05f) throw new InputException("얼굴부터 골반까지 보이는 정면 사진을 선택해주세요.");
        int t = (int) (top * height), b = (int) (bottom * height), l = (int) (left * width), r = (int) (right * width);
        int bodyHeight = b - t, bodyWidth = r - l;
        t = Math.max(0, t - bodyHeight / 5); b = Math.min(height, b + bodyHeight / 5);
        l = Math.max(0, l - bodyWidth); r = Math.min(width, r + bodyWidth);
        int h = b - t, w = r - l;
        if (h < 32 || w < 24) throw new InputException("상반신이 더 크게 보이는 사진을 선택해주세요.");
        int targetWidth = h * 3 < w * 4 ? h * 3 / 4 : w;
        int targetHeight = h * 3 < w * 4 ? h : w * 4 / 3;
        l += (w - targetWidth) / 2; t += (h - targetHeight) / 2;
        return new Rect(l, t, l + targetWidth, t + targetHeight);
    }

    private static boolean visible(NormalizedLandmark point, float threshold) {
        return Float.isFinite(point.x()) && Float.isFinite(point.y())
            && point.visibility().orElse(0f) >= threshold
            && point.presence().orElse(1f) >= threshold;
    }

    private static Rect faceProtection(int width, int height, List<NormalizedLandmark> points) {
        float leftEye = Math.min(points.get(2).x(), points.get(5).x());
        float rightEye = Math.max(points.get(2).x(), points.get(5).x());
        float eyeSpan = Math.max(rightEye - leftEye, 0.04f);
        float top = Math.min(points.get(2).y(), points.get(5).y()) - eyeSpan * 0.7f;
        float bottom = Math.max(Math.max(points.get(0).y(), points.get(9).y()), Math.max(points.get(2).y(), points.get(5).y())) + eyeSpan * 1.5f;
        int l = Math.max(0, (int) ((leftEye - eyeSpan * 1.2f) * width));
        int t = Math.max(0, (int) (top * height));
        int r = Math.min(width, (int) ((rightEye + eyeSpan * 1.2f) * width));
        int b = Math.min(height, (int) (bottom * height));
        return new Rect(l, t, Math.max(l + 1, r), Math.max(t + 1, b));
    }

    public static final class Result {
        public final Bitmap original, aligned;
        public final Rect crop;
        public final double seconds;
        private final byte[] foreground;
        private final Rect faceProtection;
        Result(Bitmap original, Bitmap aligned, byte[] foreground, Rect crop, Rect faceProtection, double seconds) {
            this.original = original; this.aligned = aligned; this.foreground = foreground; this.crop = crop; this.faceProtection = faceProtection; this.seconds = seconds;
        }
        public Bitmap restore(Bitmap prediction) {
            int width = original.getWidth(), height = original.getHeight();
            Bitmap output = original.copy(Bitmap.Config.ARGB_8888, true);
            new Canvas(output).drawBitmap(prediction, null, crop, new Paint(Paint.FILTER_BITMAP_FLAG));
            int[] pixels = new int[width * height], source = new int[pixels.length];
            output.getPixels(pixels, 0, width, 0, 0, width, height); original.getPixels(source, 0, width, 0, 0, width, height);
            for (int y = crop.top; y < crop.bottom; y++) for (int x = crop.left; x < crop.right; x++) {
                int index = y * width + x;
                if (foreground[index] == 0 || faceProtection.contains(x, y)) pixels[index] = source[index];
            }
            output.setPixels(pixels, 0, width, 0, 0, width, height);
            return output;
        }
    }

    public static final class InputException extends IllegalArgumentException {
        InputException(String message) { super(message); }
    }

    @Override public void close() { landmarker.close(); }
}
