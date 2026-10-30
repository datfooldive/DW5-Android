// The on-screen controls' native side. The controls themselves, laid out as the original Xbox pad's (the left stick
// walks, the four face buttons and the rest are the pad's, a drag on the right turns the camera), are an Android view
// over the game, TouchControls.java: it draws them from images and reads the fingers, so the game's thread draws
// nothing for them. What the fingers hold comes here over JNI (native/settings_bridge.cpp) and is overlaid on the pad
// the keyboard left (src/common/keyboard.cpp), so the game reads one report as on the console. The layout, each
// control's place and size, lives in the settings file and is kept here.
#include <toml++/toml.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>

#include "app/app.h"
#include "common/options.h"
#include "core/input.h"
#include "core/log.h"
#include "host.h"
#include "loose.h"

namespace touch {

namespace {

std::atomic<float> g_lx{0}, g_ly{0};  // the stick, -1 to 1, y up
std::atomic<float> g_rx{0}, g_ry{0};  // the right stick held, -1 to 1, y up
std::atomic<float> g_sx{0}, g_sy{0};  // the right stick swiped: each drag adds to it, Apply eases it back
std::atomic<uint32_t> g_analog{0};    // a bit per analog button, in input::Pad::analog order: held is full
std::atomic<uint32_t> g_buttons{0};   // input::Pad::buttons bits
std::atomic<int> g_pad{-1};           // a controller is plugged in, as the overlay was last told: -1 not yet told

// where each control sits: its centre as a fraction of the screen, in this order (TouchControls.java has the same).
// These are the pad's own layout, what a fresh install gets and what the editor's RESET puts back.
enum Unit : int {
    kUnitStick, kUnitA, kUnitB, kUnitX, kUnitY, kUnitDpad, kUnitStart, kUnitBack, kUnitLT, kUnitRT, kUnitL3, kUnitR3,
    kUnitGear, kUnitMap, kUnitOrders, kUnitRStick, kUnitCount  // the later ones last, so an older saved layout keeps its places
};
struct Point {
    float x, y;
};
constexpr Point kDefaultUnits[kUnitCount] = {
    {0.130f, 0.760f},  // the stick
    {0.855f, 0.835f},  // A, down: the diamond far enough apart that a button's badge clears the next button over
    {0.917f, 0.700f},  // B, right
    {0.793f, 0.700f},  // X, left
    {0.855f, 0.565f},  // Y, up
    {0.130f, 0.393f},  // the arrows' middle
    {0.580f, 0.090f},  // START
    {0.420f, 0.090f},  // BACK
    {0.090f, 0.090f},  // LT
    {0.910f, 0.090f},  // RT
    {0.235f, 0.920f},  // L3
    {0.700f, 0.920f},  // R3
    {0.500f, 0.090f},  // the gear
    {0.812f, 0.430f},  // MAP, the pad's Black: over the diamond's top corners
    {0.898f, 0.430f},  // ORDERS, the pad's White
    {0.640f, 0.700f},  // the fixed right stick, between the stick clicks and the diamond
};
// the places a fresh install had while the controls were drawn as pills: a saved layout with a unit still on its old
// place never moved it, and takes the new one
constexpr struct {
    int unit;
    Point at;
} kFormerUnits[] = {{kUnitA, {0.855f, 0.820f}}, {kUnitB, {0.909f, 0.700f}}, {kUnitX, {0.801f, 0.700f}},
                    {kUnitY, {0.855f, 0.580f}}, {kUnitMap, {0.255f, 0.090f}}, {kUnitOrders, {0.745f, 0.090f}}};
std::array<Point, kUnitCount> g_units = [] {
    std::array<Point, kUnitCount> a{};
    std::copy(std::begin(kDefaultUnits), std::end(kDefaultUnits), a.begin());
    return a;
}();
// each control's size, a fraction of its kind's: the layout editor's size bar, 0.5 to 1.5
constexpr float kScaleMin = 0.5f, kScaleMax = 1.5f;
std::array<float, kUnitCount> g_scale = [] {
    std::array<float, kUnitCount> a{};
    a.fill(1.0f);
    return a;
}();

// the panel's RIGHT STICK swipe: a drag turns the view, as a stick pushed for a moment
bool Swiping() { return game::Opt().camera_mode == 2; }

}  // namespace

void Init() { LOGI("touch: on-screen controls, drawn by the activity's overlay"); }

void Set(uint32_t buttons, uint32_t analog, float lx, float ly, float rx, float ry) {
    g_buttons = buttons;
    g_analog = analog;
    g_lx = std::clamp(lx, -1.0f, 1.0f);
    g_ly = std::clamp(ly, -1.0f, 1.0f);
    g_rx = std::clamp(rx, -1.0f, 1.0f);
    g_ry = std::clamp(ry, -1.0f, 1.0f);
}

void Swipe(float dx, float dy) {
    g_sx = std::clamp(g_sx.load() + dx, -1.0f, 1.0f);
    g_sy = std::clamp(g_sy.load() + dy, -1.0f, 1.0f);
}

void Apply(input::Pad& pad) {
    uint32_t analog = g_analog.load();
    uint32_t buttons = g_buttons.load();
    float lx = g_lx.load(), ly = g_ly.load();
    float sx = Swiping() ? g_sx.load() : 0.0f, sy = Swiping() ? g_sy.load() : 0.0f;
    float rx = std::clamp(g_rx.load() + sx, -1.0f, 1.0f), ry = std::clamp(g_ry.load() + sy, -1.0f, 1.0f);
    if (analog || buttons || lx || ly || rx || ry) pad.connected = true;
    for (int i = 0; i < 8; i++)
        if (analog & (1u << i)) pad.analog[i] = 0xFF;
    pad.buttons |= uint16_t(buttons);
    if (lx || ly) {
        pad.lx = int16_t(std::lround(lx * 32767.0f));
        pad.ly = int16_t(std::lround(ly * 32767.0f));
    }
    if (rx || ry) {
        pad.rx = int16_t(std::lround(rx * 32767.0f));
        pad.ry = int16_t(std::lround(ry * 32767.0f));
    }
    // a swipe's turn eases out over the next few polls, as a stick held for a moment would: a slow ease keeps the
    // deflection up between the finger's motion events, where a fast one dropped it under the deadzone
    if (sx || sy) {
        g_sx = std::fabs(sx) < 0.02f ? 0.0f : sx * 0.8f;
        g_sy = std::fabs(sy) < 0.02f ? 0.0f : sy * 0.8f;
    }
    // a controller the phone took over comes in over the top of both
    loose::Merge(pad);
}

void Frame() {
    // The overlay is made with the activity, before this thread has read the settings file: told at the first frame,
    // it reads the layout and the panel's values then, as the file has them. After that a controller plugged in takes
    // the pad's place: the overlay hides its controls, the gear staying for the panel.
    const int pad = input::Connected() || loose::Connected() ? 1 : 0;
    if (g_pad.exchange(pad) == pad) return;
    if (pad) {
        Set(0, 0, 0, 0, 0, 0);
        g_sx = g_sy = 0;
    }
    host::ControlsSync(pad != 0);
}

int Units() { return kUnitCount; }

void GetUnit(int unit, float& x, float& y, float& scale) {
    if (unit < 0 || unit >= kUnitCount) return;
    x = g_units[size_t(unit)].x;
    y = g_units[size_t(unit)].y;
    scale = g_scale[size_t(unit)];
}

void SetUnit(int unit, float x, float y, float scale, bool commit) {
    if (unit < 0 || unit >= kUnitCount) return;
    g_units[size_t(unit)] = {std::clamp(x, 0.0f, 1.0f), std::clamp(y, 0.0f, 1.0f)};
    g_scale[size_t(unit)] = std::clamp(scale, kScaleMin, kScaleMax);
    if (commit) app::Save();
}

void ResetUnits() {
    std::copy(std::begin(kDefaultUnits), std::end(kDefaultUnits), g_units.begin());
    g_scale.fill(1.0f);
    app::Save();
}

// the layout as [[controls]] in the settings file, one entry per unit in the order above: its place and its size
void LoadLayout(const void* table) {
    const toml::table& t = *static_cast<const toml::table*>(table);
    // a file from before the sizes were each control's has the panel's one BUTTON SIZE for all of them
    const double former = std::clamp(t["controls_scale"].value_or(100.0), 50.0, 150.0) / 100.0;
    g_scale.fill(float(former));
    const toml::array* a = t["controls"].as_array();
    if (!a) return;
    for (size_t i = 0; i < a->size() && i < size_t(kUnitCount); i++) {
        const toml::table* e = (*a)[i].as_table();
        if (!e) continue;
        g_units[i].x = float((*e)["x"].value_or(double(kDefaultUnits[i].x)));
        g_units[i].y = float((*e)["y"].value_or(double(kDefaultUnits[i].y)));
        g_scale[i] = std::clamp(float((*e)["scale"].value_or(former)), kScaleMin, kScaleMax);
    }
    for (const auto& f : kFormerUnits)
        if (g_units[size_t(f.unit)].x == f.at.x && g_units[size_t(f.unit)].y == f.at.y)
            g_units[size_t(f.unit)] = kDefaultUnits[f.unit];
}

void SaveLayout(void* table) {
    toml::table& t = *static_cast<toml::table*>(table);
    toml::array a;
    for (int i = 0; i < kUnitCount; i++)
        a.push_back(toml::table{{"x", double(g_units[size_t(i)].x)},
                                {"y", double(g_units[size_t(i)].y)},
                                {"scale", double(g_scale[size_t(i)])}});
    t.insert_or_assign("controls", std::move(a));
}

}  // namespace touch
