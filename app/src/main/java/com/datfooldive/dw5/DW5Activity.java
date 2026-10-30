package com.datfooldive.dw5;

import android.Manifest;
import android.app.AlertDialog;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.Settings;
import android.view.InputDevice;
import android.view.InputEvent;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.view.WindowManager;
import android.widget.Toast;

import java.io.File;

import org.libsdl.app.SDLActivity;
import org.libsdl.app.SDLControllerManager;

// SDL's activity running libmain.so. The game goes in a DW5 folder on the phone's storage
// (/storage/emulated/0/DW5) or at the top of an SD card. Reading it needs all files access, a special access Android
// grants on a settings page of its own, never among the app's ordinary permissions. This asks for it, and says where
// the game goes, each time the app comes back without them; the game itself waits and starts as soon as it finds the
// files.
public class DW5Activity extends SDLActivity {
    static final String SHARED = "DW5";

    private static DW5Activity current;  // for the toasts the game's threads ask for
    private AlertDialog dialog;
    private SettingsDialog settings;
    private TouchControls controls;
    private boolean notNow;  // the player turned the access down this run

    @Override
    protected String[] getLibraries() {
        return new String[] {"SDL3", "main"};
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        current = this;
        fillDisplay();
        // the controls over the game's surface, taking every touch on it
        if (mLayout != null) {
            controls = new TouchControls(this);
            mLayout.addView(controls, new ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT));
        }
        File own = getExternalFilesDir(null);
        if (own != null) new File(own, "assets").mkdirs();
    }

    @Override
    protected void onDestroy() {
        if (current == this) current = null;
        if (editor != null) editor.hide();
        super.onDestroy();
    }

    // native/host.cpp's ShowError: a toast, from the game's thread
    public static void showToast(String text) {
        DW5Activity a = current;
        if (a == null || text == null) return;
        a.runOnUiThread(() -> Toast.makeText(a, text, Toast.LENGTH_LONG).show());
    }

    // A controller some phones' game services pass on as a virtual device (MIUI's took a DualShock 4 on a Redmi
    // Note 8 Pro): SDL sees no controller in it and drops its input, so the game never learns it is there and the
    // on-screen controls never give way to it. Its buttons and sticks go to the game as port 0's here
    // (native/loose.cpp); a controller SDL has goes its own way as before.
    private static native void nativeLooseKey(int keyCode, boolean down);
    private static native void nativeLooseAxes(float lx, float ly, float rx, float ry, float lt, float rt, float hx,
                                               float hy);
    private static native void nativeLooseRelease();

    // native/settings_bridge.cpp: the panel's values, a change to one of them, and the layout editor's way in
    static native int[] nativeSettings();
    static native void nativeSettingsSet(int key, int value, boolean commit);
    static native void nativeSettingsDefaults();
    static native boolean[] nativeCheats();
    static native void nativeCheatSet(int index, boolean on);

    private LayoutEditor editor;

    // the panel's MOVE CONTROLS: it closes, the controls stay drawn where they are and a finger moves them; the bar
    // this puts at the bottom of the screen is the editor's Done and Reset
    public void beginLayoutEdit() {
        if (controls == null) return;
        if (editor == null) editor = new LayoutEditor(this, controls);
        editor.show();
        controls.beginEdit(editor);
        hideSystemBars();
    }

    // the gear (TouchControls): the settings panel over the game, which keeps running behind it
    void showSettings() {
        if (settings != null && settings.isShowing()) return;
        settings = new SettingsDialog(this);
        settings.setOnDismissListener(d -> {
            settings = null;
            hideSystemBars();
        });
        settings.show();
        hideSystemBars();
    }

    // a value in the panel changed: the controls read their opacity and right stick from it, the layout with DEFAULTS
    void settingsChanged() {
        if (controls == null) return;
        controls.reloadSettings();
        controls.reloadLayout();
    }

    // native/touch.cpp, from the game's thread: the settings file is loaded (the controls were made before it was, and
    // read the layout and the panel's values again now), or a controller came or went
    public void controlsSync(boolean pad) {
        runOnUiThread(() -> {
            if (controls == null) return;
            controls.reloadLayout();
            controls.reloadSettings();
            controls.setPad(pad);
        });
    }

    private static boolean loose(InputEvent e, int sources) {
        return (e.getSource() & sources) != 0 && !SDLControllerManager.isDeviceSDLJoystick(e.getDeviceId());
    }

    @Override
    public boolean dispatchKeyEvent(KeyEvent e) {
        boolean pad = KeyEvent.isGamepadButton(e.getKeyCode())
                || (e.getKeyCode() >= KeyEvent.KEYCODE_DPAD_UP && e.getKeyCode() <= KeyEvent.KEYCODE_DPAD_RIGHT);
        if (pad && loose(e, InputDevice.SOURCE_GAMEPAD | InputDevice.SOURCE_JOYSTICK)) {
            if (e.getRepeatCount() == 0) nativeLooseKey(e.getKeyCode(), e.getAction() == KeyEvent.ACTION_DOWN);
            return true;
        }
        return super.dispatchKeyEvent(e);
    }

    @Override
    public boolean dispatchGenericMotionEvent(MotionEvent e) {
        if (loose(e, InputDevice.SOURCE_JOYSTICK) && e.getActionMasked() == MotionEvent.ACTION_MOVE) {
            // the right stick on Z and RZ, the triggers on LTRIGGER and RTRIGGER or BRAKE and GAS, as Android has them
            nativeLooseAxes(e.getAxisValue(MotionEvent.AXIS_X), e.getAxisValue(MotionEvent.AXIS_Y),
                    e.getAxisValue(MotionEvent.AXIS_Z), e.getAxisValue(MotionEvent.AXIS_RZ),
                    Math.max(e.getAxisValue(MotionEvent.AXIS_LTRIGGER), e.getAxisValue(MotionEvent.AXIS_BRAKE)),
                    Math.max(e.getAxisValue(MotionEvent.AXIS_RTRIGGER), e.getAxisValue(MotionEvent.AXIS_GAS)),
                    e.getAxisValue(MotionEvent.AXIS_HAT_X), e.getAxisValue(MotionEvent.AXIS_HAT_Y));
            return true;
        }
        return super.dispatchGenericMotionEvent(e);
    }

    @Override
    protected void onResume() {
        super.onResume();
        hideSystemBars();
        askForGame();
    }

    @Override
    protected void onPause() {
        nativeLooseRelease();  // its ups go elsewhere while the game is not in front
        if (controls != null) controls.releaseAll();
        super.onPause();
    }

    // the game fills the display, cutout and all: without this a phone with a notch or a camera keeps a black band
    // on the side of it, and the status bar's space stays black above the picture
    @SuppressWarnings("deprecation")
    private void fillDisplay() {
        Window window = getWindow();
        WindowManager.LayoutParams lp = window.getAttributes();
        if (Build.VERSION.SDK_INT >= 30) {
            window.setDecorFitsSystemWindows(false);
            lp.layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS;
        } else {
            lp.layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES;
        }
        window.setAttributes(lp);
        // A swipe brings the bars back for a moment; they are hidden again as soon as the system has laid the window
        // out with them, so the game never keeps a strip of status bar or navigation bar over its picture.
        if (Build.VERSION.SDK_INT >= 30) {
            window.getDecorView().setOnApplyWindowInsetsListener((v, insets) -> {
                if (insets.isVisible(WindowInsets.Type.statusBars())
                        || insets.isVisible(WindowInsets.Type.navigationBars())) {
                    v.removeCallbacks(hideBars);
                    v.post(hideBars);
                }
                return insets;
            });
        }
    }

    private final Runnable hideBars = this::hideSystemBars;

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) hideSystemBars();
    }

    @SuppressWarnings("deprecation")
    private void hideSystemBars() {
        // SDL clears this when the game leaves fullscreen; left on, the system keeps room for the status bar
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_FORCE_NOT_FULLSCREEN);
        if (Build.VERSION.SDK_INT < 30) {
            getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                    | View.SYSTEM_UI_FLAG_FULLSCREEN | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                    | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                    | View.SYSTEM_UI_FLAG_LAYOUT_STABLE);
            return;
        }
        getWindow().setDecorFitsSystemWindows(false);
        WindowInsetsController c = getWindow().getInsetsController();
        if (c == null) return;
        c.hide(WindowInsets.Type.systemBars());
        c.setSystemBarsBehavior(WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);
    }

    private static final boolean ALL_FILES_ACCESS = Build.VERSION.SDK_INT >= 30;
    private static final int STORAGE_REQUEST = 0x4457;
    private boolean storageAsked;

    private boolean hasStorageAccess() {
        if (ALL_FILES_ACCESS) return Environment.isExternalStorageManager();
        return checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
                && checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
    }

    private void askForGame() {
        if (dialog != null && dialog.isShowing()) return;
        File own = getExternalFilesDir(null);
        if (own != null && hasGame(new File(own, "assets"))) return;
        if (!hasStorageAccess()) {
            if (!notNow) askAllFilesAccess();
            return;
        }
        File shared = new File(Environment.getExternalStorageDirectory(), SHARED);
        shared.mkdirs();
        if (hasGame(shared) || hasGame(new File(shared, "assets"))) return;
        tellWhereTheGameGoes();
    }

    // default.xbe or a disc image in the folder, as native/game_files.cpp looks for them
    private static boolean hasGame(File dir) {
        if (new File(dir, "default.xbe").exists()) return true;
        File[] files = dir.listFiles();
        if (files == null) return false;
        for (File f : files)
            if (f.getName().toLowerCase().endsWith(".iso")) return true;
        return false;
    }

    private void askAllFilesAccess() {
        dialog = new AlertDialog.Builder(this)
                .setTitle("Storage access")
                .setMessage("DW5 Android reads the game from a DW5 folder on your phone's storage:\n\n"
                        + "/storage/emulated/0/" + SHARED + "\n\n"
                        + (ALL_FILES_ACCESS
                                ? "On the next screen turn on \"Allow access to manage all files\" for DW5 Android, then come back."
                                : "Allow DW5 Android to access your storage on the next screen."))
                .setCancelable(false)
                .setPositiveButton("Allow access", (d, w) -> openAccessSettings())
                .setNegativeButton("Not now", (d, w) -> notNow = true)
                .show();
    }

    private void tellWhereTheGameGoes() {
        dialog = new AlertDialog.Builder(this)
                .setTitle("Game files")
                .setMessage("Copy the game into a DW5 folder on your phone's storage:\n\n"
                        + "/storage/emulated/0/" + SHARED + "\n\n"
                        + "or into a DW5 folder at the top of your SD card.\n\n"
                        + "Its disc image (.iso), or default.xbe with its data files beside it. The game starts as soon "
                        + "as they are there.")
                .setPositiveButton("OK", null)
                .show();
    }

    private void openAccessSettings() {
        if (!ALL_FILES_ACCESS) {
            boolean ask = !storageAsked || shouldShowRequestPermissionRationale(Manifest.permission.READ_EXTERNAL_STORAGE);
            if (ask) {
                storageAsked = true;
                requestPermissions(new String[] {Manifest.permission.READ_EXTERNAL_STORAGE,
                        Manifest.permission.WRITE_EXTERNAL_STORAGE}, STORAGE_REQUEST);
            } else {
                startActivity(new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                        Uri.parse("package:" + getPackageName())));
            }
            return;
        }
        try {
            startActivity(new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                    Uri.parse("package:" + getPackageName())));
        } catch (ActivityNotFoundException e) {
            try {
                startActivity(new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION));
            } catch (ActivityNotFoundException e2) {
                startActivity(new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                        Uri.parse("package:" + getPackageName())));
            }
        }
    }
}
