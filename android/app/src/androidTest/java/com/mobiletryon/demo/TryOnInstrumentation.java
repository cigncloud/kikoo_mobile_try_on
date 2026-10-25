package com.mobiletryon.demo;

import android.app.Activity;
import android.app.Instrumentation;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.ImageDecoder;
import android.graphics.BitmapFactory;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

/** Runs the real app's native pipeline against saved FP32 reference tensors. */
public final class TryOnInstrumentation extends Instrumentation {
    private boolean preprocessingOnly;
    @Override public void onCreate(Bundle arguments) { super.onCreate(arguments); preprocessingOnly = "true".equals(arguments.getString("preprocess_only")); start(); }
    @Override public void onStart() {
        new Thread(() -> {
            Bundle result = new Bundle();
            try {
                Context target = getTargetContext();
                AssetManager fixtures = getContext().getAssets();
                if (preprocessingOnly) {
                    Bitmap photo;
                    try (java.io.InputStream in = fixtures.open("official/person.png")) { photo = BitmapFactory.decodeStream(in); }
                    try (PersonPreprocessor preprocessor = new PersonPreprocessor(target, com.google.mediapipe.tasks.core.Delegate.CPU)) {
                        PersonPreprocessor.Result prepared = preprocessor.prepare(photo);
                        saveBitmap(target, "person_aligned.png", prepared.aligned);
                        result.putString("stream", "Preprocessing passed: " + prepared.seconds + " s, " + prepared.crop + "\n");
                    }
                    finish(Activity.RESULT_OK, result); return;
                }
                float[] person = TryOnEngine.readFloats(fixtures, "person.f32", TryOnEngine.PIXELS * 3);
                float[] clothes = TryOnEngine.readFloats(fixtures, "clothes.f32", TryOnEngine.PIXELS * 3);
                float[] edge = TryOnEngine.readFloats(fixtures, "edge.f32", TryOnEngine.PIXELS);
                float[] reference = TryOnEngine.readFloats(fixtures, "reference.f32", TryOnEngine.PIXELS * 3);
                JSONObject report = new JSONObject();
                Uri capture = CaptureProvider.create(target);
                try {
                    Bitmap photo = Bitmap.createBitmap(480, 640, Bitmap.Config.ARGB_8888); photo.eraseColor(Color.RED);
                    try (OutputStream out = target.getContentResolver().openOutputStream(capture)) {
                        if (out == null || !photo.compress(Bitmap.CompressFormat.JPEG, 90, out)) throw new AssertionError("Cannot write camera output");
                    }
                    Bitmap decoded = ImageDecoder.decodeBitmap(ImageDecoder.createSource(target.getContentResolver(), capture));
                    if (decoded.getWidth() != 480 || decoded.getHeight() != 640) throw new AssertionError("Camera output was downscaled to a thumbnail");
                    report.put("full_size_camera_uri_round_trip", true);
                    try {
                        CaptureProvider.file(target, Uri.parse("content://" + target.getPackageName() + ".capture/../models.json"));
                        throw new AssertionError("Capture provider accepted path traversal");
                    } catch (java.io.FileNotFoundException expected) { report.put("capture_path_traversal_rejected", true); }
                } finally { CaptureProvider.discard(target, capture); }
                if (target.getContentResolver().query(capture, null, null, null, null) != null) throw new AssertionError("Capture cleanup failed");
                report.put("camera_cancel_cleanup", true);
                try (TryOnEngine engine = new TryOnEngine(target)) {
                    report.put("initialization_seconds", engine.initializationSeconds);
                    float[] actual = engine.runRaw(person, clothes, edge);
                    double relative = relativeRmse(actual, reference);
                    if (relative > 0.001) throw new AssertionError("FP32 reference mismatch: " + relative);
                    report.put("reference_relative_rmse", relative);
                    float[] repeated = engine.runRaw(person, clothes, edge);
                    if (relativeRmse(repeated, actual) > 1e-6) throw new AssertionError("Refreshed-input repeat mismatch");
                    report.put("repeat_relative_rmse", relativeRmse(repeated, actual));
                    try { engine.runRaw(new float[1], clothes, edge); throw new AssertionError("Invalid shape accepted"); }
                    catch (IllegalArgumentException expected) { report.put("invalid_shape_rejected", true); }
                    JSONArray catalog = new JSONArray(TryOnEngine.readText(target.getAssets(), "catalog.json"));
                    if (catalog.length() < 3) throw new AssertionError("Missing garments");
                    JSONArray runs = new JSONArray();
                    float[] previous = null;
                    for (int i = 0; i < catalog.length(); i++) {
                        JSONObject garment = catalog.getJSONObject(i);
                        float[] pixels = engine.runRaw(person,
                            TryOnEngine.readFloats(target.getAssets(), garment.getString("tensor"), 3 * TryOnEngine.PIXELS),
                            TryOnEngine.readFloats(target.getAssets(), garment.getString("edge"), TryOnEngine.PIXELS));
                        if (previous != null && relativeRmse(pixels, previous) < 0.01) throw new AssertionError("Garments produced indistinguishable outputs");
                        previous = pixels;
                        double[] stats = engine.stats();
                        JSONObject item = new JSONObject(); item.put("id", garment.getString("id")); item.put("inference_seconds", stats[2]); item.put("process_peak_rss_bytes", stats[3]); runs.put(item);
                    }
                    report.put("garments", runs);

                    float[] officialPerson = TryOnEngine.readFloats(fixtures, "official/person.f32", TryOnEngine.PIXELS * 3);
                    float[] officialClothes = TryOnEngine.readFloats(fixtures, "official/clothes.f32", TryOnEngine.PIXELS * 3);
                    float[] officialEdge = TryOnEngine.readFloats(fixtures, "official/edge.f32", TryOnEngine.PIXELS);
                    float[] originalCuda = TryOnEngine.readFloats(fixtures, "official/cuda_reference.f32", TryOnEngine.PIXELS * 3);
                    float[] officialOutput = engine.runRaw(officialPerson, officialClothes, officialEdge);
                    double officialDifference = relativeRmse(officialOutput, originalCuda);
                    if (officialDifference > 0.01) throw new AssertionError("Official CUDA input control mismatch: " + officialDifference);
                    report.put("official_cuda_relative_rmse", officialDifference);
                    Bitmap photo;
                    try (java.io.InputStream in = fixtures.open("official/person.png")) { photo = BitmapFactory.decodeStream(in); }
                    if (photo == null) throw new AssertionError("Missing official photo fixture");
                    saveBitmap(target, "official_pixel4.png", engine.run(photo, target.getAssets(), catalog.getJSONObject(0)));
                    try (PersonPreprocessor preprocessor = new PersonPreprocessor(target)) {
                        long fullStart = System.nanoTime();
                        PersonPreprocessor.Result prepared = preprocessor.prepare(photo);
                        report.put("person_preprocess_seconds", prepared.seconds);
                        report.put("person_crop", prepared.crop.toShortString());
                        if (prepared.aligned.getWidth() != 192 || prepared.aligned.getHeight() != 256 || photo.isRecycled())
                            throw new AssertionError("Preprocessing changed source ownership or output size");
                        Rect bounds = prepared.crop;
                        if (bounds.left < 0 || bounds.top < 0 || bounds.right > photo.getWidth() || bounds.bottom > photo.getHeight())
                            throw new AssertionError("Automatic crop escaped image bounds");
                        saveBitmap(target, "person_aligned.png", prepared.aligned);
                        Bitmap marker = Bitmap.createBitmap(192, 256, Bitmap.Config.ARGB_8888); marker.eraseColor(Color.MAGENTA);
                        Bitmap restored = prepared.restore(marker);
                        if (restored.getWidth() != photo.getWidth() || restored.getHeight() != photo.getHeight())
                            throw new AssertionError("Restoration lost original dimensions");
                        if (restored.getPixel(0, 0) != photo.getPixel(0, 0)) throw new AssertionError("Restoration overwrote background");
                        restored.recycle(); marker.recycle();
                        JSONArray fullRuns = new JSONArray();
                        for (int i = 0; i < catalog.length(); i++) {
                            JSONObject garment = catalog.getJSONObject(i);
                            long start = System.nanoTime();
                            Bitmap prediction = engine.run(prepared.aligned, target.getAssets(), garment);
                            Bitmap output = prepared.restore(prediction);
                            saveBitmap(target, "preprocessed_" + garment.getString("id") + ".png", output);
                            JSONObject item = new JSONObject(); item.put("id", garment.getString("id")); item.put("inference_restore_seconds", (System.nanoTime() - start) / 1e9);
                            item.put("process_peak_rss_bytes", engine.stats()[3]); fullRuns.put(item);
                            output.recycle(); prediction.recycle();
                        }
                        report.put("preprocessed_garments", fullRuns);
                        report.put("validation_wall_seconds", (System.nanoTime() - fullStart) / 1e9);
                        prepared.aligned.recycle();
                        Bitmap empty = Bitmap.createBitmap(480, 640, Bitmap.Config.ARGB_8888); empty.eraseColor(Color.LTGRAY);
                        try { preprocessor.prepare(empty); throw new AssertionError("No-person image was accepted"); }
                        catch (PersonPreprocessor.InputException expected) { report.put("no_person_rejected", true); }
                        finally { empty.recycle(); }
                    }
                }
                report.put("status", "passed"); report.put("backend", "OpenCL FP32 required; no CPU fallback");
                File file = new File(target.getFilesDir(), "instrumentation.json");
                try (FileOutputStream out = new FileOutputStream(file)) { out.write(report.toString(2).getBytes(StandardCharsets.UTF_8)); }
                result.putString("stream", report.toString(2) + "\n");
                finish(Activity.RESULT_OK, result);
            } catch (Throwable error) {
                Log.e("TryOnTest", "FAILED", error); result.putString("stream", "FAILED: " + Log.getStackTraceString(error)); finish(Activity.RESULT_CANCELED, result);
            }
        }, "tryon-instrumentation").start();
    }
    private static void saveBitmap(Context target, String name, Bitmap image) throws Exception {
        try (FileOutputStream out = new FileOutputStream(new File(target.getFilesDir(), name))) {
            if (!image.compress(Bitmap.CompressFormat.PNG, 100, out)) throw new AssertionError("Cannot save validation image");
        }
    }
    private static double relativeRmse(float[] actual, float[] expected) {
        if (actual.length != expected.length) throw new AssertionError("Output length mismatch");
        double difference = 0, energy = 0;
        for (int i = 0; i < actual.length; i++) {
            if (!Float.isFinite(actual[i])) throw new AssertionError("Nonfinite output");
            difference += Math.pow(actual[i] - expected[i], 2); energy += Math.pow(expected[i], 2);
        }
        return Math.sqrt(difference / Math.max(energy, 1e-30));
    }
}
