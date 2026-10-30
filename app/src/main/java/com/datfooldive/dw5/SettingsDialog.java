package com.datfooldive.dw5;

import android.app.Dialog;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.drawable.ColorDrawable;
import android.graphics.drawable.GradientDrawable;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.util.Locale;

// The settings panel, drawn by Android rather than by the game: the activity's own dialog over the running picture,
// in Dear ImGui's default dark look (square corners, one pixel borders, its blue frames and grabs). The values live in native/settings_bridge.cpp, which this reads once when it
// opens and writes back one at a time; a slider only writes the file when it is let go, not on every step of a drag.
class SettingsDialog extends Dialog {

    // the numbers native/settings_bridge.cpp reads and writes, in its order
    static final int RESOLUTION = 0, ANTI_ALIAS = 1, ANISO = 2, VSYNC = 3, FRAME_SKIP = 4, ASPECT = 5, FPS = 6,
            HDR = 7, HDR_AVAILABLE = 8, SCALE = 9, OPACITY = 10, OUT_W = 11, OUT_H = 12, COUNT = 13;

    // ImGui's StyleColorsDark, the alpha ones over the panel's background
    private static final int BG = 0xF00F0F0F;          // WindowBg
    private static final int BORDER = 0xFF4A4A55;      // Border
    private static final int TITLE = 0xFF294A7A;       // TitleBgActive
    private static final int TEXT = 0xFFFFFFFF;        // Text
    private static final int DIM = 0xFF808080;         // TextDisabled
    private static final int FRAME = 0xFF1D2F48;       // FrameBg
    private static final int BUTTON = 0xFF1F3B5E;      // Button
    private static final int BUTTON_ON = 0xFF3369AD;   // TabActive
    private static final int GRAB = 0xFF3D85E0;        // SliderGrab
    private static final int GRAB_ON = 0xFF4296FA;     // SliderGrabActive, CheckMark

    private static final String[] kAntiAlias = {"OFF", "SMAA", "MSAA 2X", "MSAA 4X"};
    private static final String[] kAniso = {"GAME DEFAULT", "2X", "4X", "8X", "16X"};
    private static final int[] kAnisoValues = {0, 2, 4, 8, 16};
    private static final String[] kAspect = {"ORIGINAL", "16:9", "STRETCH", "FILL", "ULTRAWIDE"};

    private final DW5Activity activity;  // the dialog's own context is a wrapper around it, not it
    private final int[] v = new int[COUNT];
    private final Slider[] bars = new Slider[COUNT];
    private boolean loading;  // a reload filling the widgets is not a change to write back

    private final TextView[] tabs = new TextView[5];
    private final LinearLayout[] pages = new LinearLayout[5];

    private TextView antiAliasText, anisoText, aspectText;
    private TextView vsync, frameSkip, fps, hdr;

    SettingsDialog(DW5Activity activity) {
        super(activity);
        this.activity = activity;
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        setCanceledOnTouchOutside(true);

        LinearLayout frame = new LinearLayout(activity);
        frame.setOrientation(LinearLayout.VERTICAL);
        frame.setBackground(box(BG, true));
        frame.setPadding(1, 1, 1, 1);

        // the title bar, as an ImGui window's
        TextView title = text("Settings", 14, TEXT);
        title.setBackgroundColor(TITLE);
        title.setPadding(dp(8), dp(4), dp(8), dp(4));
        frame.addView(title);

        LinearLayout root = new LinearLayout(activity);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(10), dp(6), dp(10), dp(8));
        frame.addView(root);

        LinearLayout bar = new LinearLayout(activity);
        bar.setOrientation(LinearLayout.HORIZONTAL);
        bar.setPadding(0, dp(2), 0, dp(4));
        String[] names = {"GRAPHICS", "PERFORMANCE", "DISPLAY", "CONTROLS", "CHEATS"};
        for (int i = 0; i < names.length; i++) {
            final int at = i;
            tabs[i] = chip(names[i], 12);
            tabs[i].setOnClickListener(x -> select(at));
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f);
            lp.setMargins(i == 0 ? 0 : dp(2), 0, 0, 0);
            bar.addView(tabs[i], lp);
        }
        root.addView(bar);
        root.addView(line());

        for (int i = 0; i < pages.length; i++) {
            pages[i] = new LinearLayout(activity);
            pages[i].setOrientation(LinearLayout.VERTICAL);
            pages[i].setVisibility(View.GONE);
        }
        graphics(pages[0]);
        performance(pages[1]);
        display(pages[2]);
        controls(pages[3]);
        cheats(pages[4]);

        LinearLayout holder = new LinearLayout(activity);
        holder.setOrientation(LinearLayout.VERTICAL);
        for (LinearLayout p : pages) holder.addView(p);
        // the rows held to what the screen leaves under the title, the tabs and the footer; a longer page scrolls
        ScrollView scroll = new ScrollView(activity) {
            @Override
            protected void onMeasure(int widthSpec, int heightSpec) {
                int max = (int) (getResources().getDisplayMetrics().heightPixels * 0.55f);
                super.onMeasure(widthSpec, MeasureSpec.makeMeasureSpec(max, MeasureSpec.AT_MOST));
            }
        };
        scroll.addView(holder);
        root.addView(scroll);

        root.addView(line());
        root.addView(footer());

        setContentView(frame);
        select(0);
        reload();
    }

    // the panel a fixed share of the screen wide, as tall as its rows ask for
    @Override
    protected void onStart() {
        super.onStart();
        Window window = getWindow();
        if (window == null) return;
        window.setBackgroundDrawable(new ColorDrawable(0));
        window.addFlags(WindowManager.LayoutParams.FLAG_DIM_BEHIND);
        WindowManager.LayoutParams lp = window.getAttributes();
        lp.dimAmount = 0.65f;
        lp.gravity = Gravity.CENTER;
        int screenW = getContext().getResources().getDisplayMetrics().widthPixels;
        lp.width = Math.min((int) (screenW * 0.62f), dp(660));
        lp.height = WindowManager.LayoutParams.WRAP_CONTENT;
        window.setAttributes(lp);
    }

    // ---------------------------------------------------------------- the rows

    private void graphics(LinearLayout box) {
        slider(box, "RENDER RESOLUTION", RESOLUTION, 20, 200, 5);
        antiAliasText = stepper(box, "ANTI-ALIASING", kAntiAlias, ANTI_ALIAS);
        anisoText = stepper(box, "ANISOTROPIC FILTERING", kAniso, ANISO);
    }

    private void performance(LinearLayout box) {
        vsync = toggle(box, "VERTICAL SYNC", VSYNC);
        frameSkip = toggle(box, "FRAME SKIP", FRAME_SKIP);
    }

    private void display(LinearLayout box) {
        aspectText = stepper(box, "ASPECT RATIO", kAspect, ASPECT);
        fps = toggle(box, "FPS COUNTER", FPS);
        hdr = toggle(box, "HDR", HDR);
    }

    private void controls(LinearLayout box) {
        slider(box, "BUTTON SIZE", SCALE, 50, 150, 1);
        slider(box, "BUTTON OPACITY", OPACITY, 20, 100, 1);

        TextView edit = chip("MOVE CONTROLS", 13);
        edit.setPadding(0, dp(7), 0, dp(7));
        edit.setOnClickListener(x -> {
            activity.beginLayoutEdit();
            dismiss();
        });
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        lp.topMargin = dp(10);
        box.addView(edit, lp);
    }

    // the game has no codes of its own: these are kept up by the port (src/common/cheats.cpp), and saved with the settings
    private void cheats(LinearLayout box) {
        box.addView(text("Kept until switched off. Health and musou take effect in battle; the unlocks stay once the game saves.", 11, DIM));
        String[] names = {"INFINITE HEALTH", "INFINITE MUSOU", "UNLOCK ALL", "ALL WEAPONS", "MAX OFFICERS",
                "MAX BODYGUARDS", "INFINITE RAGE", "FREEZE TIMER", "MAX SPEED"};
        boolean[] on = DW5Activity.nativeCheats();
        for (int i = 0; i < names.length; i++) {
            final int at = i;
            LinearLayout row = new LinearLayout(getContext());
            row.setOrientation(LinearLayout.HORIZONTAL);
            row.setGravity(Gravity.CENTER_VERTICAL);
            row.setPadding(0, dp(3), 0, dp(3));
            row.addView(text(names[i], 13, TEXT), new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
            TextView sw = text("", 14, GRAB_ON);
            sw.setGravity(Gravity.CENTER);
            sw.setBackgroundColor(FRAME);
            sw.setClickable(true);
            sw.setOnClickListener(x -> {
                boolean next = !Boolean.TRUE.equals(sw.getTag());
                paint(sw, next);
                DW5Activity.nativeCheatSet(at, next);
            });
            paint(sw, at < on.length && on[at]);
            row.addView(sw, new LinearLayout.LayoutParams(dp(26), dp(26)));
            box.addView(row);
        }
    }

    // a row whose value is dragged: the label, and an ImGui slider with the value inside it
    private void slider(LinearLayout box, String label, int key, int min, int max, int step) {
        LinearLayout row = new LinearLayout(getContext());
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(0, dp(3), 0, dp(3));
        row.addView(text(label, 13, TEXT), new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        Slider bar = new Slider(getContext(), (max - min) / step);
        bar.listener = new Slider.Listener() {
            @Override
            public void changed(int progress) {
                if (loading) return;
                DW5Activity.nativeSettingsSet(key, min + progress * step, false);
                show(key, min, step);
            }

            @Override
            public void released(int progress) {
                DW5Activity.nativeSettingsSet(key, min + progress * step, true);
            }
        };
        row.addView(bar, new LinearLayout.LayoutParams(0, dp(26), 1.1f));
        box.addView(row);
        bars[key] = bar;
    }

    // a slider's value inside it: the percentage, and for the resolution the pixels it makes
    private void show(int key, int min, int step) {
        int value = min + bars[key].progress * step;
        if (key == RESOLUTION)
            bars[key].setText(String.format(Locale.US, "%d%% (%dx%d)", value, v[OUT_W] * value / 100,
                    v[OUT_H] * value / 100));
        else bars[key].setText(String.format(Locale.US, "%d%%", value));
    }

    private TextView stepper(LinearLayout box, String label, String[] names, int key) {
        LinearLayout row = new LinearLayout(getContext());
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(0, dp(3), 0, dp(3));
        row.addView(text(label, 13, TEXT), new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        // an ImGui combo's look: the value in a frame, the arrows as buttons beside it
        LinearLayout control = new LinearLayout(getContext());
        control.setOrientation(LinearLayout.HORIZONTAL);
        TextView value = text(names[0], 13, TEXT);
        value.setGravity(Gravity.CENTER);
        value.setBackgroundColor(FRAME);

        TextView down = chip("<", 13);
        TextView up = chip(">", 13);
        down.setOnClickListener(x -> step(key, -1, names.length, value, names));
        up.setOnClickListener(x -> step(key, 1, names.length, value, names));
        LinearLayout.LayoutParams gap = new LinearLayout.LayoutParams(dp(26), ViewGroup.LayoutParams.MATCH_PARENT);
        gap.setMargins(dp(2), 0, 0, 0);
        control.addView(value, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f));
        control.addView(down, gap);
        control.addView(up, new LinearLayout.LayoutParams(gap));
        row.addView(control, new LinearLayout.LayoutParams(0, dp(26), 1.1f));
        box.addView(row);
        return value;
    }

    private void step(int key, int delta, int count, TextView value, String[] names) {
        int i = (index(key, count) + delta) % count;
        if (i < 0) i += count;
        int written = key == ANISO ? kAnisoValues[i] : i;
        v[key] = written;
        DW5Activity.nativeSettingsSet(key, written, true);
        value.setText(names[i]);
    }

    // the index the setting's value sits at in its table
    private int index(int key, int count) {
        if (key == ANISO) {
            for (int i = 0; i < kAnisoValues.length; i++)
                if (kAnisoValues[i] == v[ANISO]) return i;
            return 0;
        }
        return Math.max(0, Math.min(v[key], count - 1));
    }

    // A checkbox of this panel's own, as ImGui draws one: android.widget.Switch and CheckBox draw nothing at all under
    // the game's bare theme (Theme.NoTitleBar.Fullscreen has no style for them, so they come out empty)
    private TextView toggle(LinearLayout box, String label, int key) {
        LinearLayout row = new LinearLayout(getContext());
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(0, dp(3), 0, dp(3));
        row.addView(text(label, 13, TEXT), new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        TextView sw = text("", 14, GRAB_ON);
        sw.setGravity(Gravity.CENTER);
        sw.setBackgroundColor(FRAME);
        sw.setClickable(true);
        sw.setFocusable(true);
        sw.setOnClickListener(x -> {
            if (loading) return;
            boolean on = !Boolean.TRUE.equals(sw.getTag());
            paint(sw, on);
            DW5Activity.nativeSettingsSet(key, on ? 1 : 0, true);
        });
        paint(sw, false);
        row.addView(sw, new LinearLayout.LayoutParams(dp(26), dp(26)));
        box.addView(row);
        return sw;
    }

    // the switch as it reads for a state, whether that came from a tap or from the file
    private void paint(TextView sw, boolean on) {
        sw.setTag(on);
        sw.setText(on ? "\u2713" : "");
    }

    // ---------------------------------------------------------------- the panel

    private void select(int at) {
        for (int i = 0; i < pages.length; i++) {
            pages[i].setVisibility(i == at ? View.VISIBLE : View.GONE);
            tabs[i].setBackground(box(i == at ? BUTTON_ON : BUTTON, false));
        }
    }

    private LinearLayout footer() {
        LinearLayout row = new LinearLayout(getContext());
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(0, dp(6), 0, 0);
        row.addView(text("Changes are saved as you make them", 11, DIM),
                new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        TextView back = chip("BACK", 13);
        back.setOnClickListener(x -> dismiss());
        TextView defaults = chip("DEFAULTS", 13);
        defaults.setOnClickListener(x -> {
            DW5Activity.nativeSettingsDefaults();
            reload();
        });
        LinearLayout.LayoutParams one = new LinearLayout.LayoutParams(dp(90), dp(28));
        one.setMargins(dp(4), 0, 0, 0);
        LinearLayout.LayoutParams two = new LinearLayout.LayoutParams(dp(90), dp(28));
        two.setMargins(dp(4), 0, 0, 0);
        row.addView(back, one);
        row.addView(defaults, two);
        return row;
    }

    // the values as the file has them now, into the widgets
    private void reload() {
        System.arraycopy(DW5Activity.nativeSettings(), 0, v, 0, COUNT);
        loading = true;
        bars[RESOLUTION].setProgress((v[RESOLUTION] - 20) / 5);
        show(RESOLUTION, 20, 5);
        antiAliasText.setText(kAntiAlias[Math.min(v[ANTI_ALIAS], 3)]);
        anisoText.setText(kAniso[index(ANISO, kAniso.length)]);
        paint(vsync, v[VSYNC] != 0);
        paint(frameSkip, v[FRAME_SKIP] != 0);
        aspectText.setText(kAspect[Math.min(v[ASPECT], kAspect.length - 1)]);
        paint(fps, v[FPS] != 0);
        paint(hdr, v[HDR] != 0);
        hdr.setEnabled(v[HDR_AVAILABLE] != 0);
        hdr.setAlpha(v[HDR_AVAILABLE] != 0 ? 1f : 0.4f);
        bars[SCALE].setProgress(v[SCALE] - 50);
        show(SCALE, 50, 1);
        bars[OPACITY].setProgress(v[OPACITY] - 20);
        show(OPACITY, 20, 1);
        loading = false;
    }

    // ---------------------------------------------------------------- the pieces

    private TextView text(String s, float sp, int color) {
        TextView t = new TextView(getContext());
        t.setText(s);
        t.setTextSize(TypedValue.COMPLEX_UNIT_SP, sp);
        t.setTextColor(color);
        return t;
    }

    // an ImGui button
    private TextView chip(String s, float sp) {
        TextView t = text(s, sp, TEXT);
        t.setGravity(Gravity.CENTER);
        t.setBackground(box(BUTTON, false));
        t.setClickable(true);
        t.setFocusable(true);
        t.setPadding(dp(6), dp(4), dp(6), dp(4));
        return t;
    }

    // square, filled, a one pixel border when asked
    private static GradientDrawable box(int color, boolean border) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(color);
        if (border) d.setStroke(1, BORDER);
        return d;
    }

    // ImGui's separator
    private View line() {
        View rule = new View(getContext());
        rule.setBackgroundColor(BORDER);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 1);
        lp.setMargins(0, dp(3), 0, dp(3));
        rule.setLayoutParams(lp);
        return rule;
    }

    // ImGui's SliderInt: a flat frame, a narrow grab at the value, the value written across the middle. A finger
    // anywhere on it takes the grab there, and the value is written once more when it lets go
    private static class Slider extends View {
        interface Listener {
            void changed(int progress);
            void released(int progress);
        }

        final int max;
        int progress;
        Listener listener;
        private String text = "";
        private boolean held;
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final float grabW;

        Slider(Context c, int max) {
            super(c);
            this.max = Math.max(max, 1);
            float density = c.getResources().getDisplayMetrics().density;
            grabW = 10 * density;
            paint.setTextSize(13 * c.getResources().getDisplayMetrics().scaledDensity);
            paint.setTextAlign(Paint.Align.CENTER);
        }

        void setProgress(int p) {
            progress = Math.max(0, Math.min(p, max));
            invalidate();
        }

        void setText(String s) {
            text = s;
            invalidate();
        }

        @Override
        protected void onDraw(Canvas canvas) {
            float w = getWidth(), h = getHeight(), pad = 2;
            paint.setColor(FRAME);
            canvas.drawRect(0, 0, w, h, paint);
            float x = pad + (w - 2 * pad - grabW) * progress / max;
            paint.setColor(held ? GRAB_ON : GRAB);
            canvas.drawRect(x, pad, x + grabW, h - pad, paint);
            paint.setColor(TEXT);
            canvas.drawText(text, w / 2, h / 2 - (paint.descent() + paint.ascent()) / 2, paint);
        }

        @Override
        public boolean onTouchEvent(MotionEvent e) {
            switch (e.getActionMasked()) {
                case MotionEvent.ACTION_DOWN:
                    held = true;
                    getParent().requestDisallowInterceptTouchEvent(true);
                    // fall through: the press moves the grab as a drag does
                case MotionEvent.ACTION_MOVE: {
                    float span = Math.max(getWidth() - grabW, 1);
                    int p = Math.round((e.getX() - grabW / 2) / span * max);
                    p = Math.max(0, Math.min(p, max));
                    if (p != progress) {
                        progress = p;
                        if (listener != null) listener.changed(p);
                    }
                    invalidate();
                    return true;
                }
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_CANCEL:
                    held = false;
                    invalidate();
                    if (listener != null) listener.released(progress);
                    return true;
                default:
                    return true;
            }
        }
    }

    private int dp(float value) {
        return (int) (value * getContext().getResources().getDisplayMetrics().density + 0.5f);
    }
}
