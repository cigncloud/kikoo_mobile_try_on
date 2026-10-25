package com.mobiletryon.demo;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ContentValues;
import android.content.ClipData;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.ImageDecoder;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.HorizontalScrollView;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private static final int PICK_PHOTO = 10, TAKE_PHOTO = 11;
    private static final int INK = Color.rgb(32, 55, 49), MUTED = Color.rgb(110, 117, 111), PAPER = Color.rgb(245, 244, 240);
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final ArrayList<LinearLayout> garmentCards = new ArrayList<>();
    private JSONArray catalog;
    private int selected = 0;
    private boolean busy, destroyed, hasPhoto, comparing;
    private Uri photoUri;
    private Uri captureUri;
    private TryOnEngine engine;
    private PersonPreprocessor preprocessor;
    private PersonPreprocessor.Result prepared;
    private ImageView sourceView;
    private Bitmap sourcePhoto;
    private ImageView resultView;
    private Bitmap result, before;
    private Button photoButton, cameraButton, generateButton, compareButton, saveButton, adjustButton;
    private TextView status;
    private ProgressBar progress;
    private LinearLayout resultActions;
    private ScrollView page;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        try {
            catalog = new JSONArray(TryOnEngine.readText(getAssets(), "catalog.json"));
            if (state != null) selected = Math.max(0, Math.min(catalog.length() - 1, state.getInt("garment", 0)));
            if (state != null && state.getString("capture") != null) captureUri = Uri.parse(state.getString("capture"));
            buildUi();
            if (state != null && state.getString("photo") != null) loadPhoto(Uri.parse(state.getString("photo")));
        } catch (Exception error) { showError("앱을 준비하지 못했어요.", error); }
    }

    private void buildUi() throws Exception {
        ScrollView scroll = new ScrollView(this); page = scroll; scroll.setFillViewport(true); scroll.setBackgroundColor(PAPER);
        LinearLayout content = new LinearLayout(this); content.setOrientation(LinearLayout.VERTICAL); content.setPadding(dp(22), dp(18), dp(22), dp(28));
        scroll.addView(content); setContentView(scroll);
        TextView eyebrow = text("MY FITTING ROOM", 11, MUTED); eyebrow.setLetterSpacing(0.18f); content.addView(eyebrow);
        LinearLayout heading = row();
        TextView title = text("입어보기", 30, INK); title.setTypeface(null, Typeface.BOLD);
        heading.addView(title, new LinearLayout.LayoutParams(0, dp(54), 1));
        Button about = button("앱 정보", false); about.setTextSize(12); about.setOnClickListener(v -> showAbout());
        heading.addView(about, new LinearLayout.LayoutParams(dp(86), dp(42))); content.addView(heading);
        TextView intro = text("내 사진에, 오늘의 옷을.", 15, MUTED); add(content, intro, 0, 12);
        FrameLayout photoFrame = new FrameLayout(this); photoFrame.setBackground(shape(Color.rgb(233, 232, 225), 0)); photoFrame.setClipToOutline(true);
        sourceView = new ImageView(this); sourceView.setScaleType(ImageView.ScaleType.FIT_CENTER); sourceView.setContentDescription("원본 사진. 얼굴과 어깨가 보이는 정면 셀피를 선택해주세요.");
        sourceView.setOnClickListener(v -> { if (!hasPhoto && !busy) pickPhoto(); });
        resultView = new ImageView(this); resultView.setScaleType(ImageView.ScaleType.FIT_CENTER); resultView.setVisibility(View.GONE);
        photoFrame.addView(sourceView, new FrameLayout.LayoutParams(-1, -1)); photoFrame.addView(resultView, new FrameLayout.LayoutParams(-1, -1));
        content.addView(photoFrame, new LinearLayout.LayoutParams(-1, dp(330)));
        LinearLayout photoActions = row();
        cameraButton = button("카메라로 촬영", false); cameraButton.setOnClickListener(v -> takePhoto());
        photoButton = button("사진 선택", false); photoButton.setOnClickListener(v -> pickPhoto());
        LinearLayout.LayoutParams cameraParams = new LinearLayout.LayoutParams(0, dp(48), 1); cameraParams.setMargins(0, 0, dp(8), 0);
        photoActions.addView(cameraButton, cameraParams); photoActions.addView(photoButton, new LinearLayout.LayoutParams(0, dp(48), 1)); add(content, photoActions, 10, 0);
        TextView guide = text("한 사람의 얼굴과 어깨가 보이는 정면 셀피를 골라주세요.\n골반까지 보이면 더 정확하게 맞춰드려요.", 12, MUTED); add(content, guide, 9, 17);
        TextView clothesTitle = text("오늘 입어볼 옷", 18, INK); clothesTitle.setTypeface(null, Typeface.BOLD); add(content, clothesTitle, 0, 10);
        HorizontalScrollView clothesScroll = new HorizontalScrollView(this); clothesScroll.setHorizontalScrollBarEnabled(false);
        LinearLayout clothes = row(); clothesScroll.addView(clothes);
        for (int i = 0; i < catalog.length(); i++) {
            final int index = i; JSONObject item = catalog.getJSONObject(i);
            LinearLayout card = new LinearLayout(this); card.setOrientation(LinearLayout.VERTICAL); card.setPadding(dp(8), dp(8), dp(8), dp(9)); card.setGravity(Gravity.CENTER);
            card.setContentDescription(item.getString("name")); card.setClickable(true);
            ImageView image = new ImageView(this); image.setScaleType(ImageView.ScaleType.FIT_CENTER);
            try (InputStream in = getAssets().open(item.getString("image"))) { image.setImageBitmap(BitmapFactory.decodeStream(in)); }
            card.addView(image, new LinearLayout.LayoutParams(dp(92), dp(112)));
            TextView name = text(item.getString("name"), 12, INK); name.setTypeface(null, Typeface.BOLD); name.setGravity(Gravity.CENTER); card.addView(name);
            TextView description = text(item.getString("description"), 10, MUTED); description.setGravity(Gravity.CENTER); card.addView(description);
            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(dp(112), -2); params.setMargins(0, 0, dp(9), 0); clothes.addView(card, params);
            card.setOnClickListener(v -> { if (!busy) { selected = index; updateSelection(); showCrop(); } }); garmentCards.add(card);
        }
        content.addView(clothesScroll); updateSelection();
        generateButton = button("이 옷 입어보기", true); generateButton.setOnClickListener(v -> generate()); add(content, generateButton, 17, 0);
        progress = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal); progress.setIndeterminate(true); progress.setVisibility(View.GONE); add(content, progress, 8, 0);
        status = text("사진을 선택하면 시작할 수 있어요.", 12, MUTED); status.setGravity(Gravity.CENTER); status.setAccessibilityLiveRegion(View.ACCESSIBILITY_LIVE_REGION_POLITE); add(content, status, 8, 0);
        resultActions = new LinearLayout(this); resultActions.setOrientation(LinearLayout.VERTICAL); resultActions.setVisibility(View.GONE);
        LinearLayout actions = row();
        compareButton = button("원본과 비교", false); compareButton.setOnClickListener(v -> toggleComparison());
        saveButton = button("사진에 저장", false); saveButton.setOnClickListener(v -> saveResult());
        LinearLayout.LayoutParams half = new LinearLayout.LayoutParams(0, dp(48), 1); half.setMargins(0, 0, dp(6), 0); actions.addView(compareButton, half);
        actions.addView(saveButton, new LinearLayout.LayoutParams(0, dp(48), 1)); resultActions.addView(actions);
        adjustButton = button("다른 옷 입어보기", false); adjustButton.setOnClickListener(v -> showCrop()); add(resultActions, adjustButton, 7, 0); add(content, resultActions, 12, 0);
        content.removeView(resultActions); content.addView(resultActions, content.indexOfChild(photoActions));
        TextView footer = text("사진은 폰 안에서만 처리해요.\n테스트용 미리보기로, 목·어깨 표현이 부자연스러울 수 있어요.", 11, MUTED); footer.setGravity(Gravity.CENTER); add(content, footer, 17, 0);
        setBusy(false);
    }

    private void pickPhoto() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT); intent.setType("image/*"); intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
        startActivityForResult(intent, PICK_PHOTO);
    }

    private void takePhoto() {
        if (busy) return;
        Intent camera = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (camera.resolveActivity(getPackageManager()) == null) {
            showError("사용할 수 있는 카메라 앱이 없어요.", new IllegalStateException("No camera activity")); return;
        }
        try {
            captureUri = CaptureProvider.create(this);
            camera.putExtra(MediaStore.EXTRA_OUTPUT, captureUri);
            camera.setClipData(ClipData.newRawUri("촬영 사진", captureUri));
            camera.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
            startActivityForResult(camera, TAKE_PHOTO);
        } catch (Exception error) {
            if (captureUri != null) CaptureProvider.discard(this, captureUri);
            captureUri = null; showError("카메라를 열지 못했어요.", error);
        }
    }

    @Override protected void onActivityResult(int request, int resultCode, Intent data) {
        super.onActivityResult(request, resultCode, data);
        if (request == PICK_PHOTO && resultCode == RESULT_OK && data != null && data.getData() != null) {
            Uri uri = data.getData();
            try { getContentResolver().takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION); }
            catch (SecurityException ignored) { /* Immediate use is still permitted. */ }
            loadPhoto(uri);
        }
        if (request == TAKE_PHOTO && captureUri != null) {
            Uri captured = captureUri; captureUri = null;
            if (resultCode == RESULT_OK) {
                try {
                    if (CaptureProvider.file(this, captured).length() == 0) throw new java.io.IOException("Empty camera output");
                    loadPhoto(captured);
                } catch (Exception error) { CaptureProvider.discard(this, captured); showError("촬영한 사진을 불러오지 못했어요. 다시 촬영해주세요.", error); }
            } else CaptureProvider.discard(this, captured);
        }
    }

    private void loadPhoto(Uri uri) {
        setBusy(true); status.setText("사진을 불러오는 중이에요…");
        worker.execute(() -> {
            try {
                Bitmap bitmap = ImageDecoder.decodeBitmap(ImageDecoder.createSource(getContentResolver(), uri), (decoder, info, source) -> {
                    int width = info.getSize().getWidth(), height = info.getSize().getHeight();
                    float ratio = Math.min(1f, 1600f / Math.max(width, height));
                    decoder.setTargetSize(Math.max(1, Math.round(width * ratio)), Math.max(1, Math.round(height * ratio)));
                    decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
                });
                if (prepared != null) { prepared.aligned.recycle(); prepared = null; }
                post(() -> { photoUri = uri; hasPhoto = true; sourcePhoto = bitmap; sourceView.setImageBitmap(bitmap); before = null; result = null; showCrop(); photoButton.setText("다른 사진 선택"); setBusy(false); status.setText("옷을 고르면 상반신을 자동으로 맞춰드려요."); });
            } catch (Exception error) { post(() -> { setBusy(false); showError("사진을 열지 못했어요. 다른 사진을 선택해주세요.", error); }); }
        });
    }

    private void generate() {
        if (busy || !hasPhoto) return;
        Bitmap input = sourcePhoto;
        JSONObject garment = catalog.optJSONObject(selected);
        setBusy(true); status.setText(engine == null ? "첫 사용을 준비하고 있어요. 처음 한 번은 약 1분 걸릴 수 있어요." : "옷을 입혀보고 있어요…");
        worker.execute(() -> {
            long start = System.nanoTime();
            try {
                if (preprocessor == null) preprocessor = new PersonPreprocessor(getApplicationContext());
                if (prepared == null) prepared = preprocessor.prepare(input);
                post(() -> status.setText(engine == null ? "첫 사용을 준비하고 있어요. 처음 한 번은 약 1분 걸릴 수 있어요." : "옷을 입혀보고 있어요…"));
                if (engine == null) engine = new TryOnEngine(getApplicationContext());
                Bitmap prediction = engine.run(prepared.aligned, getAssets(), garment);
                Bitmap output = prepared.restore(prediction); prediction.recycle();
                double wall = (System.nanoTime() - start) / 1e9;
                double[] stats = engine.stats();
                Log.i("MobileTryOn", "completed total_seconds=" + wall + " preprocess=" + prepared.seconds + " crop=" + prepared.crop + " warp=" + stats[0] + " generator=" + stats[1] + " peak_rss_bytes=" + stats[3]);
                post(() -> { before = input; result = output; comparing = false; resultView.setImageBitmap(output); resultView.setContentDescription("선택한 옷을 입힌 결과"); resultView.setVisibility(View.VISIBLE); sourceView.setVisibility(View.GONE); resultActions.setVisibility(View.VISIBLE); compareButton.setText("원본과 비교"); setBusy(false); status.setText(String.format(Locale.KOREA, "미리보기 완성 · %.1f초", wall)); page.post(() -> page.smoothScrollTo(0, 0)); });
            } catch (PersonPreprocessor.InputException error) { post(() -> { setBusy(false); showError(error.getMessage(), error); }); }
            catch (Exception | LinkageError error) { post(() -> { setBusy(false); showError("옷을 입히지 못했어요. 다시 시도해주세요.", error); }); }
        });
    }

    private void showCrop() { if (sourceView != null) { sourceView.setVisibility(View.VISIBLE); resultView.setVisibility(View.GONE); resultActions.setVisibility(View.GONE); comparing = false; } }

    private void toggleComparison() {
        if (busy || result == null) return;
        comparing = !comparing; resultView.setImageBitmap(comparing ? before : result);
        compareButton.setText(comparing ? "착용 결과 보기" : "원본과 비교");
        resultView.setContentDescription(comparing ? "입히기 전 원본 사진" : "선택한 옷을 입힌 결과");
    }

    private void saveResult() {
        if (busy || result == null) return;
        Bitmap output = result; setBusy(true); status.setText("사진에 저장하고 있어요…");
        worker.execute(() -> {
            Uri destination = null;
            try {
                ContentValues metadata = new ContentValues(); metadata.put(MediaStore.Images.Media.DISPLAY_NAME, "TryOn_" + System.currentTimeMillis() + ".png");
                metadata.put(MediaStore.Images.Media.MIME_TYPE, "image/png"); metadata.put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/MobileTryOn"); metadata.put(MediaStore.Images.Media.IS_PENDING, 1);
                destination = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, metadata);
                if (destination == null) throw new java.io.IOException("Cannot create gallery entry");
                try (OutputStream stream = getContentResolver().openOutputStream(destination)) {
                    if (stream == null || !output.compress(Bitmap.CompressFormat.PNG, 100, stream)) throw new java.io.IOException("Cannot save PNG");
                }
                ContentValues ready = new ContentValues(); ready.put(MediaStore.Images.Media.IS_PENDING, 0); getContentResolver().update(destination, ready, null, null);
                post(() -> { setBusy(false); status.setText("사진 > MobileTryOn 폴더에 저장했어요."); Toast.makeText(this, "저장했어요", Toast.LENGTH_SHORT).show(); });
            } catch (Exception error) {
                if (destination != null) { try { getContentResolver().delete(destination, null, null); } catch (Exception cleanup) { Log.w("MobileTryOn", "Cannot remove incomplete image", cleanup); } }
                post(() -> { setBusy(false); showError("사진을 저장하지 못했어요.", error); });
            }
        });
    }

    private void setBusy(boolean value) {
        busy = value;
        if (photoButton == null) return;
        photoButton.setEnabled(!value); cameraButton.setEnabled(!value); generateButton.setEnabled(!value && hasPhoto); sourceView.setEnabled(!value);
        progress.setVisibility(value ? View.VISIBLE : View.GONE);
        for (LinearLayout card : garmentCards) { card.setEnabled(!value); card.setAlpha(value ? 0.5f : 1f); }
        if (saveButton != null) { saveButton.setEnabled(!value); compareButton.setEnabled(!value); adjustButton.setEnabled(!value); }
    }

    private void updateSelection() { for (int i = 0; i < garmentCards.size(); i++) { garmentCards.get(i).setBackground(shape(Color.WHITE, i == selected ? INK : Color.rgb(222, 223, 215))); garmentCards.get(i).setSelected(i == selected); } }
    private void post(Runnable action) { ui.post(() -> { if (!destroyed && !isFinishing()) action.run(); }); }
    private void showError(String message, Throwable error) { Log.e("MobileTryOn", message, error); if (status != null) status.setText(message); new AlertDialog.Builder(this).setTitle("잠시만요").setMessage(message).setPositiveButton("확인", null).show(); }
    private void showAbout() { new AlertDialog.Builder(this).setTitle("입어보기 · 테스트 앱").setMessage("사진을 서버로 전송하지 않고 폰에서 처리합니다.\n\nDM-VTON 연구 모델 · CC BY-NC-SA 4.0\nMNN 3.6.1 · Apache 2.0\nMediaPipe Pose Lite · Apache 2.0\n의상·마스크: 공식 VITON-Clean 연구 샘플\n\n상업용 배포가 아닌 기능 검증용 앱입니다.").setPositiveButton("확인", null).show(); }
    @Override protected void onSaveInstanceState(Bundle state) { super.onSaveInstanceState(state); state.putInt("garment", selected); if (photoUri != null) state.putString("photo", photoUri.toString()); if (captureUri != null) state.putString("capture", captureUri.toString()); }
    @Override protected void onDestroy() { destroyed = true; worker.execute(() -> { if (engine != null) { engine.close(); engine = null; } if (preprocessor != null) { preprocessor.close(); preprocessor = null; } if (prepared != null) { prepared.aligned.recycle(); prepared = null; } }); worker.shutdown(); super.onDestroy(); }
    private int dp(int value) { return Math.round(value * getResources().getDisplayMetrics().density); }
    private TextView text(String value, int size, int color) { TextView view = new TextView(this); view.setText(value); view.setTextSize(size); view.setTextColor(color); view.setLineSpacing(dp(2), 1); return view; }
    private LinearLayout row() { LinearLayout view = new LinearLayout(this); view.setOrientation(LinearLayout.HORIZONTAL); view.setGravity(Gravity.CENTER_VERTICAL); return view; }
    private GradientDrawable shape(int fill, int stroke) { GradientDrawable drawable = new GradientDrawable(); drawable.setColor(fill); drawable.setCornerRadius(dp(12)); if (stroke != 0) drawable.setStroke(dp(2), stroke); return drawable; }
    private Button button(String title, boolean primary) { Button view = new Button(this); view.setText(title); view.setTextSize(15); view.setAllCaps(false); view.setTextColor(primary ? Color.WHITE : INK); view.setBackground(shape(primary ? INK : Color.WHITE, primary ? 0 : Color.rgb(221, 224, 215))); view.setMinHeight(dp(48)); view.setPadding(dp(12), dp(8), dp(12), dp(8)); return view; }
    private void add(LinearLayout parent, View child, int top, int bottom) { LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT); params.setMargins(0, dp(top), 0, dp(bottom)); parent.addView(child, params); }
}
