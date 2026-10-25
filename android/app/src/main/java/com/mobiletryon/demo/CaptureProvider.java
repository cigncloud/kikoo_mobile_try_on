package com.mobiletryon.demo;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.MatrixCursor;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.provider.OpenableColumns;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

/** Grants the selected camera access to one private temporary JPEG. */
public final class CaptureProvider extends ContentProvider {
    public static Uri create(Context context) throws IOException {
        File directory = new File(context.getCacheDir(), "captures");
        if (!directory.isDirectory() && !directory.mkdirs()) throw new IOException("Cannot create capture directory");
        File file = File.createTempFile("capture_", ".jpg", directory);
        return new Uri.Builder().scheme("content").authority(context.getPackageName() + ".capture").appendPath(file.getName()).build();
    }

    static File file(Context context, Uri uri) throws FileNotFoundException {
        if (uri == null || !"content".equals(uri.getScheme()) || !(context.getPackageName() + ".capture").equals(uri.getAuthority())
            || uri.getPathSegments().size() != 1 || uri.getQuery() != null) throw new FileNotFoundException("Invalid capture URI");
        String name = uri.getLastPathSegment();
        if (name == null || !name.matches("capture_[0-9]+\\.jpg")) throw new FileNotFoundException("Invalid capture name");
        File result = new File(new File(context.getCacheDir(), "captures"), name);
        if (!result.isFile()) throw new FileNotFoundException("Capture does not exist");
        return result;
    }

    public static void discard(Context context, Uri uri) {
        try { file(context, uri).delete(); } catch (FileNotFoundException ignored) { }
    }

    @Override public boolean onCreate() { return true; }
    @Override public String getType(Uri uri) { return "image/jpeg"; }
    @Override public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
        return ParcelFileDescriptor.open(file(getContext(), uri), ParcelFileDescriptor.parseMode(mode));
    }
    @Override public Cursor query(Uri uri, String[] projection, String selection, String[] args, String order) {
        try {
            File target = file(getContext(), uri);
            String[] columns = projection != null ? projection : new String[] {OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE};
            MatrixCursor result = new MatrixCursor(columns);
            MatrixCursor.RowBuilder row = result.newRow();
            for (String column : columns) {
                if (OpenableColumns.DISPLAY_NAME.equals(column)) row.add(target.getName());
                else if (OpenableColumns.SIZE.equals(column)) row.add(target.length());
                else row.add(null);
            }
            return result;
        } catch (FileNotFoundException error) { return null; }
    }
    @Override public Uri insert(Uri uri, ContentValues values) { throw new UnsupportedOperationException(); }
    @Override public int update(Uri uri, ContentValues values, String selection, String[] args) { throw new UnsupportedOperationException(); }
    @Override public int delete(Uri uri, String selection, String[] args) { throw new UnsupportedOperationException(); }
}
