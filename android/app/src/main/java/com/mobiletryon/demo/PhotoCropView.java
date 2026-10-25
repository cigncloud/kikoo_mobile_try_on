package com.mobiletryon.demo;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.View;

/** A fixed 3:4 crop with drag/pinch, constrained to stay inside the source. */
public final class PhotoCropView extends View {
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
    private final RectF crop = new RectF();
    private final ScaleGestureDetector zoom;
    private Bitmap image;
    private float scale, minimumScale, x, y, lastX, lastY;

    public PhotoCropView(Context context) {
        super(context);
        setContentDescription("상반신 맞추기. 두 손가락으로 확대하고 드래그해서 사진을 이동하세요.");
        zoom = new ScaleGestureDetector(context, new ScaleGestureDetector.SimpleOnScaleGestureListener() {
            @Override public boolean onScale(ScaleGestureDetector detector) {
                float next = Math.max(minimumScale, Math.min(minimumScale * 6, scale * detector.getScaleFactor()));
                float factor = next / scale;
                x = detector.getFocusX() - (detector.getFocusX() - x) * factor;
                y = detector.getFocusY() - (detector.getFocusY() - y) * factor;
                scale = next;
                constrain(); invalidate(); return true;
            }
        });
    }

    public void setImage(Bitmap bitmap) { image = bitmap; reset(); invalidate(); }

    private void reset() {
        if (image == null || crop.width() <= 0) return;
        minimumScale = Math.max(crop.width() / image.getWidth(), crop.height() / image.getHeight());
        scale = minimumScale;
        x = crop.centerX() - image.getWidth() * scale / 2;
        y = crop.centerY() - image.getHeight() * scale / 2;
        constrain();
    }

    private void constrain() {
        if (image == null) return;
        x = Math.max(crop.right - image.getWidth() * scale, Math.min(crop.left, x));
        y = Math.max(crop.bottom - image.getHeight() * scale, Math.min(crop.top, y));
    }

    @Override protected void onSizeChanged(int width, int height, int oldWidth, int oldHeight) {
        float cropWidth = Math.min(width - 24f, (height - 16f) * 0.75f);
        float cropHeight = cropWidth / 0.75f;
        crop.set((width - cropWidth) / 2, (height - cropHeight) / 2, (width + cropWidth) / 2, (height + cropHeight) / 2);
        reset();
    }

    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawColor(Color.rgb(233, 232, 225));
        if (image != null) {
            canvas.save(); canvas.clipRect(crop);
            Matrix matrix = new Matrix(); matrix.setScale(scale, scale); matrix.postTranslate(x, y);
            canvas.drawBitmap(image, matrix, paint); canvas.restore();
        } else {
            paint.setColor(Color.rgb(80, 94, 86)); paint.setTextAlign(Paint.Align.CENTER);
            paint.setTextSize(44 * getResources().getDisplayMetrics().scaledDensity);
            canvas.drawText("+", crop.centerX(), crop.centerY() - 8, paint);
            paint.setTextSize(15 * getResources().getDisplayMetrics().scaledDensity);
            canvas.drawText("내 사진으로 시작하기", crop.centerX(), crop.centerY() + 34, paint);
        }
        paint.setStyle(Paint.Style.STROKE); paint.setStrokeWidth(3); paint.setColor(Color.WHITE);
        canvas.drawRoundRect(crop, 8, 8, paint); paint.setStyle(Paint.Style.FILL);
    }

    @Override public boolean onTouchEvent(MotionEvent event) {
        if (!isEnabled() || image == null) return super.onTouchEvent(event);
        getParent().requestDisallowInterceptTouchEvent(true);
        zoom.onTouchEvent(event);
        if (event.getActionMasked() == MotionEvent.ACTION_DOWN) { lastX = event.getX(); lastY = event.getY(); return true; }
        if (event.getActionMasked() == MotionEvent.ACTION_MOVE) {
            if (!zoom.isInProgress() && event.getPointerCount() == 1) {
                x += event.getX() - lastX; y += event.getY() - lastY; constrain(); invalidate();
            }
            lastX = event.getX(); lastY = event.getY();
        }
        if (event.getActionMasked() == MotionEvent.ACTION_UP || event.getActionMasked() == MotionEvent.ACTION_CANCEL) {
            getParent().requestDisallowInterceptTouchEvent(false);
            if (event.getActionMasked() == MotionEvent.ACTION_UP) performClick();
        }
        return true;
    }

    @Override public boolean performClick() { super.performClick(); return true; }

    public Bitmap renderInput() {
        if (image == null || crop.width() <= 0) throw new IllegalStateException("사진을 먼저 선택해주세요.");
        float ratio = TryOnEngine.WIDTH / crop.width();
        Matrix matrix = new Matrix();
        matrix.setScale(scale * ratio, scale * ratio);
        matrix.postTranslate((x - crop.left) * ratio, (y - crop.top) * ratio);
        Bitmap bitmap = Bitmap.createBitmap(TryOnEngine.WIDTH, TryOnEngine.HEIGHT, Bitmap.Config.ARGB_8888);
        new Canvas(bitmap).drawBitmap(image, matrix, paint);
        return bitmap;
    }
}
