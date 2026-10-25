package com.mobiletryon.demo;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

/** Owns GPU sessions. Create, use and close on the same background thread. */
public final class TryOnEngine implements AutoCloseable {
    public static final int WIDTH = 192, HEIGHT = 256, PIXELS = WIDTH * HEIGHT;
    static { System.loadLibrary("tryon_jni"); }
    private long handle;
    public final double initializationSeconds;

    public TryOnEngine(Context context) throws Exception {
        long start = System.nanoTime();
        JSONArray manifest = new JSONArray(readText(context.getAssets(), "models.json"));
        File models = new File(context.getNoBackupFilesDir(), "models");
        if (!models.isDirectory() && !models.mkdirs()) throw new IOException("Cannot create model directory");
        StringBuilder version = new StringBuilder("mnn361_fp32_");
        for (int i = 0; i < manifest.length(); i++) {
            JSONObject item = manifest.getJSONObject(i);
            String name = item.getString("file"), sha = item.getString("sha256");
            version.append(sha, 0, 12);
            File destination = new File(models, name);
            if (!destination.isFile() || destination.length() != item.getLong("bytes") || !sha.equals(digest(destination))) {
                File pending = new File(models, name + ".pending");
                try (InputStream in = context.getAssets().open("models/" + name); FileOutputStream out = new FileOutputStream(pending)) {
                    byte[] buffer = new byte[65536];
                    int count;
                    while ((count = in.read(buffer)) != -1) out.write(buffer, 0, count);
                }
                if (!sha.equals(digest(pending))) throw new IOException("Model checksum mismatch");
                if (!pending.renameTo(destination)) throw new IOException("Cannot install model");
            }
        }
        File cache = new File(context.getCodeCacheDir(), version.toString());
        if (!cache.isDirectory() && !cache.mkdirs()) throw new IOException("Cannot create kernel cache");
        handle = nativeCreate(models.getAbsolutePath(), cache.getAbsolutePath());
        if (handle == 0) throw new IllegalStateException("Cannot initialize model");
        initializationSeconds = (System.nanoTime() - start) / 1e9;
    }

    public float[] runRaw(float[] person, float[] clothes, float[] edge) {
        if (handle == 0) throw new IllegalStateException("Engine is closed");
        if (person.length != 3 * PIXELS || clothes.length != 3 * PIXELS || edge.length != PIXELS)
            throw new IllegalArgumentException("Input dimensions must be 192 x 256");
        return nativeRun(handle, person, clothes, edge);
    }

    public Bitmap run(Bitmap input, AssetManager assets, JSONObject garment) throws Exception {
        if (input.getWidth() != WIDTH || input.getHeight() != HEIGHT) throw new IllegalArgumentException("Unexpected crop size");
        int[] pixels = new int[PIXELS];
        input.getPixels(pixels, 0, WIDTH, 0, 0, WIDTH, HEIGHT);
        float[] person = new float[3 * PIXELS];
        for (int i = 0; i < PIXELS; i++) {
            person[i] = ((pixels[i] >> 16) & 255) / 127.5f - 1;
            person[PIXELS + i] = ((pixels[i] >> 8) & 255) / 127.5f - 1;
            person[2 * PIXELS + i] = (pixels[i] & 255) / 127.5f - 1;
        }
        float[] output = runRaw(person, readFloats(assets, garment.getString("tensor"), 3 * PIXELS), readFloats(assets, garment.getString("edge"), PIXELS));
        for (int i = 0; i < PIXELS; i++) {
            int red = channel(output[i]), green = channel(output[PIXELS + i]), blue = channel(output[2 * PIXELS + i]);
            pixels[i] = 0xff000000 | red << 16 | green << 8 | blue;
        }
        return Bitmap.createBitmap(pixels, WIDTH, HEIGHT, Bitmap.Config.ARGB_8888);
    }

    private static int channel(float value) {
        if (!Float.isFinite(value)) throw new IllegalStateException("Nonfinite image");
        return Math.max(0, Math.min(255, Math.round((value + 1) * 127.5f)));
    }

    public double[] stats() {
        if (handle == 0) throw new IllegalStateException("Engine is closed");
        return nativeStats(handle);
    }

    @Override public void close() {
        if (handle != 0) { nativeDestroy(handle); handle = 0; }
    }

    public static String readText(AssetManager assets, String path) throws IOException {
        return new String(readBytes(assets, path), StandardCharsets.UTF_8);
    }

    public static float[] readFloats(AssetManager assets, String path, int count) throws IOException {
        byte[] data = readBytes(assets, path);
        if (data.length != count * 4) throw new IOException("Tensor size mismatch: " + path);
        float[] values = new float[count];
        ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(values);
        return values;
    }

    private static byte[] readBytes(AssetManager assets, String path) throws IOException {
        try (InputStream in = assets.open(path); ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[16384];
            int count;
            while ((count = in.read(buffer)) != -1) out.write(buffer, 0, count);
            return out.toByteArray();
        }
    }

    private static String digest(File file) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream in = new FileInputStream(file)) {
            byte[] buffer = new byte[65536];
            int count;
            while ((count = in.read(buffer)) != -1) digest.update(buffer, 0, count);
        }
        StringBuilder hex = new StringBuilder();
        for (byte b : digest.digest()) hex.append(String.format(java.util.Locale.ROOT, "%02x", b & 255));
        return hex.toString();
    }

    private static native long nativeCreate(String modelDirectory, String cacheDirectory);
    private static native float[] nativeRun(long handle, float[] person, float[] clothes, float[] edge);
    private static native double[] nativeStats(long handle);
    private static native void nativeDestroy(long handle);
}
