package com.datfooldive.dw5;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.drawable.GradientDrawable;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;

// Dear ImGui's default dark look, shared by the in-game panel (SettingsDialog) and the layout editor's bar
// (LayoutEditor) so the two are the same app: square corners, one pixel borders, its blue frames and grabs.
final class Theme {

    // ImGui's StyleColorsDark, the alpha one over a panel's background
    static final int BG = 0xF00F0F0F;         // WindowBg
    static final int BORDER = 0xFF4A4A55;     // Border
    static final int TITLE = 0xFF294A7A;      // TitleBgActive
    static final int TEXT = 0xFFFFFFFF;       // Text
    static final int DIM = 0xFF808080;        // TextDisabled
    static final int FRAME = 0xFF1D2F48;      // FrameBg
    static final int BUTTON = 0xFF1F3B5E;     // Button
    static final int BUTTON_ON = 0xFF3369AD;  // TabActive
    static final int GRAB = 0xFF3D85E0;       // SliderGrab
    static final int GRAB_ON = 0xFF4296FA;    // SliderGrabActive, CheckMark

    private Theme() {}

    static int dp(Context c, float value) {
        return (int) (value * c.getResources().getDisplayMetrics().density + 0.5f);
    }

    static TextView text(Context c, String s, float sp, int color) {
        TextView t = new TextView(c);
        t.setText(s);
        t.setTextSize(TypedValue.COMPLEX_UNIT_SP, sp);
        t.setTextColor(color);
        return t;
    }

    // an ImGui button
    static TextView chip(Context c, String s, float sp) {
        TextView t = text(c, s, sp, TEXT);
        t.setGravity(Gravity.CENTER);
        t.setBackground(box(BUTTON, false));
        t.setClickable(true);
        t.setFocusable(true);
        t.setPadding(dp(c, 6), dp(c, 4), dp(c, 6), dp(c, 4));
        return t;
    }

    // square, filled, a one pixel border when asked
    static GradientDrawable box(int color, boolean border) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(color);
        if (border) d.setStroke(1, BORDER);
        return d;
    }

    // ImGui's separator
    static View line(Context c) {
        View rule = new View(c);
        rule.setBackgroundColor(BORDER);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 1);
        lp.setMargins(0, dp(c, 3), 0, dp(c, 3));
        rule.setLayoutParams(lp);
        return rule;
    }

    // ImGui's SliderInt: a flat frame, a narrow grab at the value, the value written across the middle. A finger
    // anywhere on it takes the grab there, and the value is written once more when it lets go.
    static class Slider extends View {
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
}
