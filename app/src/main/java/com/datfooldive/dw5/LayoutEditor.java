package com.datfooldive.dw5;

import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;

// The layout editor's own bar: the screen stays the controls' (TouchControls), so a finger anywhere else still drags a
// control, and only the strip itself takes a touch. A control a finger takes is the one its size bar sizes, 50 to 150%
// of its kind's. The layout is written out as each drag or size ends; Reset puts the pad's own back, sizes and all.
class LayoutEditor {

    private static final int kMin = 50, kMax = 150, kStep = 5;

    private final DW5Activity activity;
    private final TouchControls controls;
    private View bar;
    private TextView hint;
    private Theme.Slider size;

    LayoutEditor(DW5Activity activity, TouchControls controls) {
        this.activity = activity;
        this.controls = controls;
    }

    void show() {
        if (bar != null) return;
        LinearLayout row = new LinearLayout(activity);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(dp(10), dp(6), dp(10), dp(6));
        row.setClickable(true);  // a tap on the bar itself is not a tap on the game behind it
        // the settings panel's look (Theme): its window, square, a one pixel border
        row.setBackground(Theme.box(Theme.BG, true));

        hint = Theme.text(activity, "TAP A CONTROL TO MOVE OR SIZE IT", 13, Theme.DIM);
        LinearLayout.LayoutParams hlp = new LinearLayout.LayoutParams(dp(150), ViewGroup.LayoutParams.WRAP_CONTENT);
        row.addView(hint, hlp);

        // the selected control's size, there once a control is taken
        size = new Theme.Slider(activity, (kMax - kMin) / kStep);
        size.listener = new Theme.Slider.Listener() {
            @Override
            public void changed(int progress) {
                controls.setSelectedScale(kMin + progress * kStep, false);
                size.setText((kMin + progress * kStep) + "%");
            }

            @Override
            public void released(int progress) {
                controls.setSelectedScale(kMin + progress * kStep, true);
            }
        };
        size.setVisibility(View.INVISIBLE);
        LinearLayout.LayoutParams slp = new LinearLayout.LayoutParams(dp(200), dp(26));
        slp.setMargins(dp(8), 0, 0, 0);
        row.addView(size, slp);

        TextView reset = Theme.chip(activity, "RESET", 13);
        reset.setOnClickListener(x -> controls.resetLayout());
        TextView done = Theme.chip(activity, "DONE", 13);
        done.setBackground(Theme.box(Theme.BUTTON_ON, false));
        done.setOnClickListener(x -> {
            controls.endEdit();
            hide();
        });
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(dp(90), dp(28));
        lp.setMargins(dp(8), 0, 0, 0);
        row.addView(reset, lp);
        LinearLayout.LayoutParams lp2 = new LinearLayout.LayoutParams(dp(90), dp(28));
        lp2.setMargins(dp(4), 0, 0, 0);
        row.addView(done, lp2);

        FrameLayout.LayoutParams at = new FrameLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM | Gravity.CENTER_HORIZONTAL);
        at.bottomMargin = dp(26);
        ((ViewGroup) activity.getWindow().getDecorView()).addView(row, at);
        bar = row;
    }

    // a control taken: the bar names it and sizes it
    void select(int unit, int percent) {
        if (bar == null || unit < 0 || unit >= TouchControls.NAMES.length) return;
        hint.setText(TouchControls.NAMES[unit] + " SIZE");
        hint.setTextColor(Theme.TEXT);
        int p = Math.max(kMin, Math.min(percent, kMax));
        size.setProgress((p - kMin) / kStep);
        size.setText(p + "%");
        size.setVisibility(View.VISIBLE);
    }

    void hide() {
        if (bar == null) return;
        ((ViewGroup) bar.getParent()).removeView(bar);
        bar = null;
    }

    private int dp(float value) {
        return (int) (value * activity.getResources().getDisplayMetrics().density + 0.5f);
    }
}
