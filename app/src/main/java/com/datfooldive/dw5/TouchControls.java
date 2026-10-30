package com.datfooldive.dw5;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.MotionEvent;
import android.view.View;

// The on-screen controls, laid out as the original Xbox pad's: the left stick walks, the four face buttons and the
// rest are the pad's, each drawn as what it does in the game with the pad's letter on a badge, and a finger on the
// right is the right stick, which turns the camera. A view over the game: each control is an image (res/drawable-nodpi, made by
// tools/controls_art.py), drawn again only when something changes, and what the fingers hold goes to the game over
// JNI (native/touch.cpp), so the game's thread draws nothing for them. The layout, each control's place and size, is
// native's, in the settings file; in the layout editor a finger moves a control and the editor's bar sizes it.
@SuppressLint("ViewConstructor")
class TouchControls extends View {

    // native/touch.cpp: what the fingers hold, and the layout
    private static native void nativeTouchPad(int buttons, int analog, float lx, float ly, float rx, float ry);
    private static native void nativeTouchSwipe(float dx, float dy);
    private static native float[] nativeControls();
    private static native void nativeControlSet(int unit, float x, float y, float scale, boolean commit);
    private static native void nativeControlsReset();

    // the layout's controls, in native/touch.cpp's Unit order
    static final int STICK = 0, A = 1, B = 2, X = 3, Y = 4, DPAD = 5, START = 6, BACK = 7, LT = 8, RT = 9, L3 = 10,
            R3 = 11, GEAR = 12, MAP = 13, ORDERS = 14, RSTICK = 15, UNITS = 16;
    static final String[] NAMES = {"STICK", "A", "B", "X", "Y", "D-PAD", "START", "BACK", "LT", "RT", "L3", "R3",
            "SETTINGS", "MAP", "ORDERS", "RIGHT STICK"};

    // the pad's buttons as input::Pad has them: the digital ones' bits, and the analog ones' places
    private static final int UP = 1, DOWN = 1 << 1, LEFT = 1 << 2, RIGHT = 1 << 3, BIT_START = 1 << 4,
            BIT_BACK = 1 << 5, LEFT_THUMB = 1 << 6, RIGHT_THUMB = 1 << 7;
    private static final int AN_A = 0, AN_B = 1, AN_X = 2, AN_Y = 3, AN_BLACK = 4, AN_WHITE = 5, AN_LT = 6, AN_RT = 7;

    // the round buttons other than the face buttons, and what each presses: a digital bit, or an analog place (-1 none)
    private static final int[] ROUND = {START, BACK, LT, RT, L3, R3, MAP, ORDERS};
    private static final int[] ROUND_BIT = {BIT_START, BIT_BACK, 0, 0, LEFT_THUMB, RIGHT_THUMB, 0, 0};
    private static final int[] ROUND_ANALOG = {-1, -1, AN_LT, AN_RT, -1, -1, AN_BLACK, AN_WHITE};
    private static final int[] FACE_ANALOG = {AN_A, AN_B, AN_X, AN_Y};
    private static final int[] ARROW_BIT = {UP, DOWN, LEFT, RIGHT};
    private static final float[][] DIRS = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};

    // what a finger is doing
    private static final int NONE = -1, ROLE_STICK = 0, ROLE_CAMERA = 1, ROLE_GEAR = 2, ROLE_FACE = 10,
            ROLE_ROUND = 20, ROLE_ARROW = 30;

    // the panel's RIGHT STICK
    private static final int FLOATING = 0, FIXED = 1, SWIPE = 2;

    // an image is its control's radius 100 in a 260 box: drawn this many radii out from the centre
    private static final float ART = 1.3f;

    private final DW5Activity activity;
    private final float[] ux = new float[UNITS], uy = new float[UNITS], scale = new float[UNITS];
    private final Bitmap[][] art = new Bitmap[UNITS][2];  // each control normal and held
    private final Bitmap[][] arrowArt = new Bitmap[4][2];
    private Bitmap stickArt;
    private final Bitmap[] knobArt = new Bitmap[2];
    private final Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG);
    private final Paint ring = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF dst = new RectF();

    // the panel's settings the controls read
    private int opacity = 100, cameraMode = FLOATING;
    private float sensitivity = 3;
    private boolean pad;  // a controller is plugged in: only the gear is left

    // the fingers
    private static final int MAX = 10;
    private final int[] ids = new int[MAX], roles = new int[MAX];
    private final float[] fx = new float[MAX], fy = new float[MAX], ox = new float[MAX], oy = new float[MAX];
    private int fingers;
    private int buttons, analog;
    private float lx, ly, rx, ry;
    private boolean camHeld;
    private float camX, camY;  // the floating right stick's middle while a finger holds it

    // the layout editor
    private boolean editing;
    private int selected = -1, drag = -1, dragId = -1;
    private float grabX, grabY;
    private LayoutEditor editor;

    TouchControls(DW5Activity activity) {
        super(activity);
        this.activity = activity;
        ring.setStyle(Paint.Style.STROKE);
        load();
        reloadLayout();
        reloadSettings();
    }

    // ---------------------------------------------------------------- what native and the panel tell it

    void reloadLayout() {
        float[] v = nativeControls();
        if (v == null) return;
        for (int i = 0; i < UNITS && i * 3 + 2 < v.length; i++) {
            ux[i] = v[i * 3];
            uy[i] = v[i * 3 + 1];
            scale[i] = v[i * 3 + 2];
        }
        invalidate();
    }

    void reloadSettings() {
        int[] v = DW5Activity.nativeSettings();
        if (v == null) return;
        opacity = Math.max(20, Math.min(v[SettingsDialog.OPACITY], 100));
        cameraMode = Math.max(FLOATING, Math.min(v[SettingsDialog.CAMERA_MODE], SWIPE));
        sensitivity = Math.max(0.5f, Math.min(v[SettingsDialog.CAMERA] / 100f, 3f));
        invalidate();
    }

    void setPad(boolean connected) {
        if (pad == connected) return;
        pad = connected;
        if (pad) releaseAll();
        invalidate();
    }

    // what the fingers held lets go: the panel opened, the game went behind, or a controller took over
    void releaseAll() {
        fingers = 0;
        camHeld = false;
        buttons = analog = 0;
        lx = ly = rx = ry = 0;
        send();
        invalidate();
    }

    // ---------------------------------------------------------------- the layout editor

    void beginEdit(LayoutEditor editor) {
        releaseAll();
        this.editor = editor;
        editing = true;
        selected = drag = -1;
        invalidate();
    }

    void endEdit() {
        editing = false;
        selected = drag = -1;
        editor = null;
        invalidate();
    }

    // the editor's size bar: the selected control's size, written to the file when it is let go
    void setSelectedScale(int percent, boolean commit) {
        if (selected < 0) return;
        scale[selected] = percent / 100f;
        nativeControlSet(selected, ux[selected], uy[selected], scale[selected], commit);
        invalidate();
    }

    void resetLayout() {
        nativeControlsReset();
        reloadLayout();
        if (editor != null && selected >= 0) editor.select(selected, Math.round(scale[selected] * 100));
    }

    // ---------------------------------------------------------------- the layout: each control's centre and radius

    private float side() {
        return Math.min(getWidth(), getHeight());
    }

    private float cx(int u) {
        return ux[u] * getWidth();
    }

    private float cy(int u) {
        return uy[u] * getHeight();
    }

    // a control's radius, its kind's times its own size: native/touch.cpp's sizes before it moved here
    private float radius(int u) {
        float k = side() * Math.max(0.5f, Math.min(scale[u], 1.5f));
        switch (u) {
            case STICK: case RSTICK: return k * 0.11f;
            case A: case B: case X: case Y: return k * 0.075f;
            case LT: case RT: return k * 0.06f;
            case DPAD: return k * 0.057f;  // an arrow's
            case GEAR: return k * 0.0322f * 1.3f;
            default: return k * 0.052f;
        }
    }

    // an arrow's centre: around the d-pad's middle
    private float arrowX(int a) {
        return cx(DPAD) + DIRS[a][0] * side() * scale[DPAD] * 0.100f;
    }

    private float arrowY(int a) {
        return cy(DPAD) + DIRS[a][1] * side() * scale[DPAD] * 0.100f;
    }

    // the right stick's reach: the more sensitive, the shorter the drag to a full push
    private float camReach() {
        return side() * scale[RSTICK] * 0.10f / sensitivity;
    }

    private static boolean in(float x, float y, float cx, float cy, float r) {
        return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r;
    }

    // ---------------------------------------------------------------- drawing

    private void load() {
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inScaled = false;
        int[][] ids = {
            {STICK, 0, 0}, {A, R.drawable.ctl_a, R.drawable.ctl_a_on}, {B, R.drawable.ctl_b, R.drawable.ctl_b_on},
            {X, R.drawable.ctl_x, R.drawable.ctl_x_on}, {Y, R.drawable.ctl_y, R.drawable.ctl_y_on},
            {START, R.drawable.ctl_start, R.drawable.ctl_start_on}, {BACK, R.drawable.ctl_back, R.drawable.ctl_back_on},
            {LT, R.drawable.ctl_lt, R.drawable.ctl_lt_on}, {RT, R.drawable.ctl_rt, R.drawable.ctl_rt_on},
            {L3, R.drawable.ctl_l3, R.drawable.ctl_l3_on}, {R3, R.drawable.ctl_r3, R.drawable.ctl_r3_on},
            {GEAR, R.drawable.ctl_gear, R.drawable.ctl_gear_on}, {MAP, R.drawable.ctl_map, R.drawable.ctl_map_on},
            {ORDERS, R.drawable.ctl_orders, R.drawable.ctl_orders_on},
        };
        for (int[] e : ids) {
            if (e[1] == 0) continue;
            art[e[0]][0] = BitmapFactory.decodeResource(getResources(), e[1], o);
            art[e[0]][1] = BitmapFactory.decodeResource(getResources(), e[2], o);
        }
        int[][] arrows = {{R.drawable.ctl_arrow_up, R.drawable.ctl_arrow_up_on},
            {R.drawable.ctl_arrow_down, R.drawable.ctl_arrow_down_on},
            {R.drawable.ctl_arrow_left, R.drawable.ctl_arrow_left_on},
            {R.drawable.ctl_arrow_right, R.drawable.ctl_arrow_right_on}};
        for (int a = 0; a < 4; a++) {
            arrowArt[a][0] = BitmapFactory.decodeResource(getResources(), arrows[a][0], o);
            arrowArt[a][1] = BitmapFactory.decodeResource(getResources(), arrows[a][1], o);
        }
        stickArt = BitmapFactory.decodeResource(getResources(), R.drawable.ctl_stick, o);
        knobArt[0] = BitmapFactory.decodeResource(getResources(), R.drawable.ctl_knob, o);
        knobArt[1] = BitmapFactory.decodeResource(getResources(), R.drawable.ctl_knob_on, o);
    }

    private void image(Canvas c, Bitmap b, float x, float y, float r) {
        if (b == null) return;
        float h = r * ART;
        dst.set(x - h, y - h, x + h, y + h);
        c.drawBitmap(b, null, dst, paint);
    }

    private void stick(Canvas c, float x, float y, float r, float px, float py) {
        image(c, stickArt, x, y, r);
        image(c, knobArt[px != 0 || py != 0 ? 1 : 0], x + px * r * 0.55f, y - py * r * 0.55f, r * 0.40f);
    }

    @Override
    protected void onDraw(Canvas c) {
        paint.setAlpha(Math.round(255 * opacity / 100f));
        image(c, art[GEAR][0], cx(GEAR), cy(GEAR), radius(GEAR));
        // a controller takes the pad's place: its controls are not drawn while one is plugged in, the gear staying
        if (pad) return;

        stick(c, cx(STICK), cy(STICK), radius(STICK), lx, ly);
        if (cameraMode == FIXED) {
            stick(c, cx(RSTICK), cy(RSTICK), radius(RSTICK), rx, ry);
        } else if (camHeld && cameraMode == FLOATING) {
            float r = camReach();
            image(c, knobArt[0], camX, camY, r);
            image(c, knobArt[1], camX + rx * r, camY - ry * r, r * 0.36f);
        }
        for (int i = 0; i < 4; i++) {
            boolean on = (analog & (1 << FACE_ANALOG[i])) != 0;
            image(c, art[A + i][on ? 1 : 0], cx(A + i), cy(A + i), radius(A + i));
        }
        for (int a = 0; a < 4; a++)
            image(c, arrowArt[a][(buttons & ARROW_BIT[a]) != 0 ? 1 : 0], arrowX(a), arrowY(a), radius(DPAD));
        for (int i = 0; i < ROUND.length; i++) {
            int u = ROUND[i];
            boolean on = ROUND_BIT[i] != 0 ? (buttons & ROUND_BIT[i]) != 0 : (analog & (1 << ROUND_ANALOG[i])) != 0;
            image(c, art[u][on ? 1 : 0], cx(u), cy(u), radius(u));
        }
        if (editing) drawEditor(c);
    }

    // the editor: every control ringed in the panel's blue, the one the size bar sizes solid and thicker
    private void drawEditor(Canvas c) {
        float density = getResources().getDisplayMetrics().density;
        for (int u = 0; u < UNITS; u++) {
            if (u == RSTICK && cameraMode != FIXED) continue;
            boolean on = u == selected;
            ring.setColor(on ? Theme.GRAB_ON : (Theme.GRAB & 0x00FFFFFF) | 0xC0000000);
            ring.setStrokeWidth((on ? 3f : 1.5f) * density);
            c.drawCircle(cx(u), cy(u), u == DPAD ? radius(DPAD) * 2.6f : radius(u), ring);
        }
    }

    // ---------------------------------------------------------------- the fingers

    // the control under a finger, for the editor: -1 when none is
    private int hitUnit(float x, float y) {
        if (in(x, y, cx(GEAR), cy(GEAR), radius(GEAR) * 1.1f)) return GEAR;
        for (int u : ROUND)
            if (in(x, y, cx(u), cy(u), radius(u) * 1.15f)) return u;
        for (int u = A; u <= Y; u++)
            if (in(x, y, cx(u), cy(u), radius(u))) return u;
        if (in(x, y, cx(DPAD), cy(DPAD), radius(DPAD) * 2.6f)) return DPAD;
        if (in(x, y, cx(STICK), cy(STICK), radius(STICK))) return STICK;
        if (cameraMode == FIXED && in(x, y, cx(RSTICK), cy(RSTICK), radius(RSTICK))) return RSTICK;
        return -1;
    }

    private int hitRole(float x, float y) {
        if (in(x, y, cx(GEAR), cy(GEAR), radius(GEAR) * 1.1f)) return ROLE_GEAR;
        if (pad) return NONE;
        for (int i = 0; i < 4; i++)
            if (in(x, y, cx(A + i), cy(A + i), radius(A + i) * 1.15f)) return ROLE_FACE + i;
        for (int i = 0; i < ROUND.length; i++)
            if (in(x, y, cx(ROUND[i]), cy(ROUND[i]), radius(ROUND[i]) * 1.15f)) return ROLE_ROUND + i;
        for (int a = 0; a < 4; a++)
            if (in(x, y, arrowX(a), arrowY(a), radius(DPAD) * 1.1f)) return ROLE_ARROW + a;
        float w = getWidth();
        if (x < w * 0.45f && in(x, y, cx(STICK), cy(STICK), radius(STICK) * 1.7f)) return ROLE_STICK;
        if (cameraMode == FIXED) return in(x, y, cx(RSTICK), cy(RSTICK), radius(RSTICK) * 1.7f) ? ROLE_CAMERA : NONE;
        return x > w * 0.45f ? ROLE_CAMERA : NONE;
    }

    @SuppressLint("ClickableViewAccessibility")
    @Override
    public boolean onTouchEvent(MotionEvent e) {
        if (editing) return editTouch(e);
        int action = e.getActionMasked();
        switch (action) {
            case MotionEvent.ACTION_DOWN:
            case MotionEvent.ACTION_POINTER_DOWN: {
                int i = e.getActionIndex();
                down(e.getPointerId(i), e.getX(i), e.getY(i));
                break;
            }
            case MotionEvent.ACTION_MOVE:
                for (int i = 0; i < e.getPointerCount(); i++) move(e.getPointerId(i), e.getX(i), e.getY(i));
                break;
            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_POINTER_UP: {
                int i = e.getActionIndex();
                up(e.getPointerId(i));
                break;
            }
            case MotionEvent.ACTION_CANCEL:
                releaseAll();
                return true;
            default:
                return true;
        }
        send();
        invalidate();
        return true;
    }

    private void down(int id, float x, float y) {
        int role = hitRole(x, y);
        if (role == NONE || fingers >= MAX) return;
        if (role == ROLE_GEAR) {
            // what the fingers held lets go: the panel opens over the game, which reads no touch until it closes
            releaseAll();
            activity.showSettings();
            return;
        }
        int f = fingers++;
        ids[f] = id;
        roles[f] = role;
        fx[f] = ox[f] = x;
        fy[f] = oy[f] = y;
        if (role == ROLE_STICK) stickTo(x, y);
        else if (role == ROLE_CAMERA && cameraMode == FIXED) rightStickTo(x, y, cx(RSTICK), cy(RSTICK));
        else if (role == ROLE_CAMERA && cameraMode == FLOATING) {
            camX = x;
            camY = y;
            camHeld = true;
        } else if (role >= ROLE_ARROW) buttons |= ARROW_BIT[role - ROLE_ARROW];
        else if (role >= ROLE_ROUND) press(role - ROLE_ROUND, true);
        else if (role >= ROLE_FACE) analog |= 1 << FACE_ANALOG[role - ROLE_FACE];
    }

    private void move(int id, float x, float y) {
        for (int f = 0; f < fingers; f++) {
            if (ids[f] != id) continue;
            float dx = x - fx[f], dy = y - fy[f];
            fx[f] = x;
            fy[f] = y;
            if (roles[f] == ROLE_STICK) {
                stickTo(x, y);
            } else if (roles[f] == ROLE_CAMERA && cameraMode == FIXED) {
                rightStickTo(x, y, cx(RSTICK), cy(RSTICK));
            } else if (roles[f] == ROLE_CAMERA && cameraMode == SWIPE) {
                // a drag turns the view: each pixel adds to the stick's push, which native eases back, so the turn
                // follows the swipe's speed; at 300% sensitivity a twenty-fifth of the short side is a full push
                float full = side() * 0.12f / sensitivity;
                if (dx != 0 || dy != 0) nativeTouchSwipe(dx / full, -dy / full);
            } else if (roles[f] == ROLE_CAMERA) {
                // the right stick, its middle where the finger went down: held over as long as the finger stays, as
                // the command menu's choice needs; past its reach the middle follows the finger
                float reach = camReach();
                float sx = (x - ox[f]) / reach, sy = (y - oy[f]) / reach;
                float len = (float) Math.hypot(sx, sy);
                if (len > 1) {
                    sx /= len;
                    sy /= len;
                    ox[f] = x - sx * reach;
                    oy[f] = y - sy * reach;
                    camX = ox[f];
                    camY = oy[f];
                }
                rx = sx;
                ry = -sy;
            }
            // an arrow stays the one first touched, wherever the thumb slides: no direction changes by itself
            return;
        }
    }

    private void up(int id) {
        for (int f = 0; f < fingers; f++) {
            if (ids[f] != id) continue;
            int role = roles[f];
            if (role == ROLE_STICK) {
                lx = ly = 0;
            } else if (role == ROLE_CAMERA) {
                // a swipe's turn eases out in native, as the flick it was; a stick held lets go at once
                rx = ry = 0;
                camHeld = false;
            } else if (role >= ROLE_ARROW) {
                buttons &= ~ARROW_BIT[role - ROLE_ARROW];
            } else if (role >= ROLE_ROUND) {
                press(role - ROLE_ROUND, false);
            } else if (role >= ROLE_FACE) {
                analog &= ~(1 << FACE_ANALOG[role - ROLE_FACE]);
            }
            fingers--;
            ids[f] = ids[fingers];
            roles[f] = roles[fingers];
            fx[f] = fx[fingers];
            fy[f] = fy[fingers];
            ox[f] = ox[fingers];
            oy[f] = oy[fingers];
            return;
        }
    }

    private void press(int round, boolean down) {
        if (ROUND_BIT[round] != 0) buttons = down ? buttons | ROUND_BIT[round] : buttons & ~ROUND_BIT[round];
        else {
            int bit = 1 << ROUND_ANALOG[round];
            analog = down ? analog | bit : analog & ~bit;
        }
    }

    private void stickTo(float x, float y) {
        float r = radius(STICK);
        float dx = (x - cx(STICK)) / r, dy = (y - cy(STICK)) / r;
        float len = (float) Math.hypot(dx, dy);
        if (len > 1) {
            dx /= len;
            dy /= len;
        }
        lx = dx;
        ly = -dy;  // screen y down, the stick's up
    }

    // the fixed right stick: its push from its middle, a full one at the sensitivity's reach
    private void rightStickTo(float x, float y, float cx, float cy) {
        float reach = camReach();
        float sx = (x - cx) / reach, sy = (y - cy) / reach;
        float len = (float) Math.hypot(sx, sy);
        if (len > 1) {
            sx /= len;
            sy /= len;
        }
        rx = sx;
        ry = -sy;
    }

    private void send() {
        nativeTouchPad(buttons, analog, lx, ly, rx, ry);
    }

    // the editor: a finger takes the control under it, moves it, and selects it for the size bar
    private boolean editTouch(MotionEvent e) {
        int i = e.getActionIndex();
        switch (e.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
            case MotionEvent.ACTION_POINTER_DOWN: {
                if (drag >= 0) break;
                float x = e.getX(i), y = e.getY(i);
                int u = hitUnit(x, y);
                if (u < 0) break;
                drag = u;
                dragId = e.getPointerId(i);
                grabX = x - cx(u);
                grabY = y - cy(u);
                if (selected != u) {
                    selected = u;
                    if (editor != null) editor.select(u, Math.round(scale[u] * 100));
                }
                invalidate();
                break;
            }
            case MotionEvent.ACTION_MOVE: {
                if (drag < 0) break;
                int p = e.findPointerIndex(dragId);
                if (p < 0) break;
                ux[drag] = Math.max(0, Math.min((e.getX(p) - grabX) / getWidth(), 1));
                uy[drag] = Math.max(0, Math.min((e.getY(p) - grabY) / getHeight(), 1));
                nativeControlSet(drag, ux[drag], uy[drag], scale[drag], false);
                invalidate();
                break;
            }
            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_POINTER_UP:
            case MotionEvent.ACTION_CANCEL:
                if (drag >= 0 && (e.getActionMasked() == MotionEvent.ACTION_CANCEL || e.getPointerId(i) == dragId)) {
                    nativeControlSet(drag, ux[drag], uy[drag], scale[drag], true);
                    drag = -1;
                }
                break;
            default:
                break;
        }
        return true;
    }
}
