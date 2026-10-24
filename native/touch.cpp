// The on-screen controls, laid out as the original Xbox pad's: the left stick walks, the four face buttons are the
// pad's, and a drag on the right turns the camera. Their art is the pad's own colours, from native/ui. Touches come
// from SDL's finger events in window coordinates; what they leave is overlaid on the pad the keyboard left
// (src/common/keyboard.cpp), so the game reads one report as on the console.
#include <SDL3/SDL.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <mutex>
#include <vector>

#include <imgui.h>

#include "core/input.h"
#include "core/log.h"
#include "host.h"
#include "menu.h"
#include "ui_icons.h"
#include "vk_ui.h"

namespace touch {

namespace {

// what a finger is doing; the gear is past the pad's buttons
enum Role : int { kNone = -1, kStick = 0, kCamera = 1, kButton = 2 };
// the pad's analog buttons, in input::Pad::analog order
enum Analog : int { kA = 0, kB = 1, kX = 2, kY = 3, kWhite = 4, kBlack = 5, kLeftTrigger = 6, kRightTrigger = 7 };
enum Bit : uint16_t { kUp = 1 << 0, kDown = 1 << 1, kLeft = 1 << 2, kRight = 1 << 3, kStart = 1 << 4, kBack = 1 << 5 };
// the on-screen buttons, in the order the layout and the art have them
enum Button : int { kBtnA = 0, kBtnB = 1, kBtnX = 2, kBtnY = 3, kBtnStart = 4, kBtnBack = 5, kBtnDpad = 6,
                     kBtnLT = 7, kBtnRT = 8, kBtnCount = 9 };

std::atomic<float> g_lx{0}, g_ly{0};  // the stick, -1 to 1, y up
std::atomic<float> g_rx{0}, g_ry{0};  // the camera's pending turn, -1 to 1
std::atomic<uint64_t> g_analog{0};    // one byte per analog button
std::atomic<uint32_t> g_buttons{0};   // Bit

struct Touch {
    SDL_FingerID id;
    int role;
    float x, y;  // window pixels
};
std::mutex g_lock;
std::vector<Touch> g_touches;

struct Layout {
    float w, h, s;
    float stick[2], stick_r;
    float btn[4][2], btn_r;  // A B X Y, a diamond as the pad has them: Y up, X left, B right, A down
    float dpad[2];           // the arrows' middle
    float arrow[4][2], arrow_r;  // up, down, left, right: apart, each a button of its own
    float pill[kBtnCount][2];  // START BACK LT RT, by their button
    float pill_w, pill_h;
    float gear[2], gear_r;  // the settings panel's button, between BACK and START
};
constexpr int kGear = kButton + kBtnCount;
constexpr int kArrow = kGear + 1;  // the four arrows after it, up down left right
constexpr uint32_t kArrowBits[4] = {kUp, kDown, kLeft, kRight};

Layout Make(float w, float h) {
    Layout l{};
    l.w = w;
    l.h = h;
    l.s = std::min(w, h);
    l.stick_r = l.s * 0.15f;
    l.stick[0] = w * 0.15f;
    l.stick[1] = h * 0.70f;
    l.btn_r = l.s * 0.052f;
    float d = l.s * 0.088f;
    float cx = w * 0.855f, cy = h * 0.70f;
    l.btn[kBtnA][0] = cx;      l.btn[kBtnA][1] = cy + d;
    l.btn[kBtnB][0] = cx + d;  l.btn[kBtnB][1] = cy;
    l.btn[kBtnX][0] = cx - d;  l.btn[kBtnX][1] = cy;
    l.btn[kBtnY][0] = cx;      l.btn[kBtnY][1] = cy - d;
    // the arrows far enough apart that a thumb on one does not touch the next
    l.arrow_r = l.s * 0.062f;
    l.dpad[0] = w * 0.33f;
    l.dpad[1] = h * 0.80f;
    float off = l.s * 0.105f;
    const float dirs[4][2] = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
    for (int a = 0; a < 4; a++) {
        l.arrow[a][0] = l.dpad[0] + dirs[a][0] * off;
        l.arrow[a][1] = l.dpad[1] + dirs[a][1] * off;
    }
    l.pill_w = l.s * 0.115f;
    l.pill_h = l.s * 0.052f;
    l.pill[kBtnBack][0] = w * 0.42f;  l.pill[kBtnBack][1] = h * 0.09f;
    l.pill[kBtnStart][0] = w * 0.58f; l.pill[kBtnStart][1] = h * 0.09f;
    l.pill[kBtnLT][0] = w * 0.09f;    l.pill[kBtnLT][1] = h * 0.09f;
    l.pill[kBtnRT][0] = w * 0.91f;    l.pill[kBtnRT][1] = h * 0.09f;
    l.gear[0] = w * 0.5f;
    l.gear[1] = h * 0.09f;
    l.gear_r = l.pill_h * 0.62f;
    return l;
}

bool InCircle(float x, float y, const float c[2], float r) {
    return (x - c[0]) * (x - c[0]) + (y - c[1]) * (y - c[1]) <= r * r;
}

bool InPill(const Layout& l, float x, float y, const float c[2]) {
    return std::fabs(x - c[0]) <= l.pill_w / 2 && std::fabs(y - c[1]) <= l.pill_h / 2;
}

int HitTest(const Layout& l, float x, float y) {
    if (InCircle(x, y, l.gear, l.gear_r * 1.4f)) return kGear;
    // the panel takes every other touch while it is open
    if (menu::Open()) return kNone;
    for (int b = 0; b < 4; b++)
        if (InCircle(x, y, l.btn[b], l.btn_r * 1.15f)) return kButton + b;
    for (int b : {kBtnStart, kBtnBack, kBtnLT, kBtnRT})
        if (InPill(l, x, y, l.pill[b])) return kButton + b;
    for (int a = 0; a < 4; a++)
        if (InCircle(x, y, l.arrow[a], l.arrow_r * 1.1f)) return kArrow + a;
    if (x < l.w * 0.45f && InCircle(x, y, l.stick, l.stick_r * 1.7f)) return kStick;
    if (x > l.w * 0.45f) return kCamera;
    return kNone;
}

void PressButton(int index, bool down) {
    static const int kAnalogOf[4] = {kA, kB, kX, kY};
    if (index < 4) {
        uint64_t v = g_analog.load();
        uint32_t shift = uint32_t(kAnalogOf[index]) * 8;
        v = down ? (v | (0xFFull << shift)) : (v & ~(0xFFull << shift));
        g_analog = v;
    } else if (index == kBtnStart || index == kBtnBack) {
        uint16_t bit = index == kBtnStart ? kStart : kBack;
        uint32_t v = g_buttons.load();
        v = down ? (v | bit) : (v & ~uint32_t(bit));
        g_buttons = v;
    } else if (index == kBtnLT || index == kBtnRT) {
        uint64_t v = g_analog.load();
        uint32_t shift = uint32_t(index == kBtnLT ? kLeftTrigger : kRightTrigger) * 8;
        v = down ? (v | (0xFFull << shift)) : (v & ~(0xFFull << shift));
        g_analog = v;
    }
}

// an arrow held or let go: its bit alone, the others as they are
void PressArrow(int arrow, bool down) {
    uint32_t v = g_buttons.load();
    v = down ? (v | kArrowBits[arrow]) : (v & ~kArrowBits[arrow]);
    g_buttons = v;
}

void StickTo(float x, float y, const Layout& l) {
    float dx = (x - l.stick[0]) / l.stick_r;
    float dy = (y - l.stick[1]) / l.stick_r;
    float len = std::hypot(dx, dy);
    if (len > 1) {
        dx /= len;
        dy /= len;
    }
    g_lx = std::clamp(dx, -1.0f, 1.0f);
    g_ly = std::clamp(-dy, -1.0f, 1.0f);  // screen y down, the stick's up
}

// the buttons' art, made into textures the first frame they are drawn
ImTextureID g_tex[4];
const char* const kArt[4] = {ui_icons::kBtnA, ui_icons::kBtnB, ui_icons::kBtnX, ui_icons::kBtnY};

void EnsureArt() {
    if (g_tex[0]) return;
    for (int b = 0; b < 4; b++)
        g_tex[b] = vkui::Texture(reinterpret_cast<const uint8_t*>(kArt[b]), ui_icons::kBtnSize, ui_icons::kBtnSize);
}

}  // namespace

void Init() { LOGI("touch: on-screen controls"); }

void Event(const SDL_Event& e) {
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(host::Window(), &w, &h);
    Layout lay = Make(float(std::max(w, 1)), float(std::max(h, 1)));
    switch (e.type) {
        case SDL_EVENT_FINGER_DOWN: {
            float x = e.tfinger.x * lay.w, y = e.tfinger.y * lay.h;
            int role = HitTest(lay, x, y);
            if (role == kNone) return;
            std::lock_guard lock(g_lock);
            if (role == kGear) {
                // what the fingers held lets go: the panel opens over the game, which reads no touch until it closes
                menu::Toggle();
                g_touches.clear();
                g_analog = 0;
                g_buttons = 0;
                g_lx = g_ly = g_rx = g_ry = 0;
                return;
            }
            g_touches.push_back({e.tfinger.fingerID, role, x, y});
            if (role == kStick) StickTo(x, y, lay);
            else if (role >= kArrow) PressArrow(role - kArrow, true);
            else if (role >= kButton && role < kButton + kBtnCount) PressButton(role - kButton, true);
            break;
        }
        case SDL_EVENT_FINGER_MOTION: {
            std::lock_guard lock(g_lock);
            for (Touch& t : g_touches) {
                if (t.id != e.tfinger.fingerID) continue;
                float x = e.tfinger.x * lay.w, y = e.tfinger.y * lay.h;
                float dx = x - t.x, dy = y - t.y;
                t.x = x;
                t.y = y;
                if (t.role == kStick) {
                    StickTo(x, y, lay);
                } else if (t.role == kCamera) {
                    // a drag turns the view: the deflection the stick would give, eased back in Apply
                    g_rx = std::clamp(g_rx.load() + dx / (lay.s * 0.12f), -1.0f, 1.0f);
                    g_ry = std::clamp(g_ry.load() - dy / (lay.s * 0.12f), -1.0f, 1.0f);
                }
                // an arrow stays the one first touched, wherever the thumb slides: no direction changes by itself
                break;
            }
            break;
        }
        case SDL_EVENT_FINGER_UP:
        case SDL_EVENT_FINGER_CANCELED: {
            std::lock_guard lock(g_lock);
            for (size_t i = 0; i < g_touches.size(); i++) {
                Touch& t = g_touches[i];
                if (t.id != e.tfinger.fingerID) continue;
                if (t.role == kStick) {
                    g_lx = 0;
                    g_ly = 0;
                } else if (t.role >= kArrow) {
                    PressArrow(t.role - kArrow, false);
                } else if (t.role >= kButton && t.role < kButton + kBtnCount) {
                    PressButton(t.role - kButton, false);
                }
                g_touches.erase(g_touches.begin() + long(i));
                break;
            }
            break;
        }
        default:
            break;
    }
}

void Apply(input::Pad& pad) {
    uint64_t analog = g_analog.load();
    uint32_t buttons = g_buttons.load();
    float lx = g_lx.load(), ly = g_ly.load();
    float rx = g_rx.load(), ry = g_ry.load();
    if (analog || buttons || lx || ly || rx || ry) pad.connected = true;
    for (int i = 0; i < 8; i++) {
        uint8_t v = uint8_t(analog >> (i * 8));
        if (v) pad.analog[i] = v;
    }
    pad.buttons |= uint16_t(buttons);
    if (lx || ly) {
        pad.lx = int16_t(std::lround(std::clamp(lx, -1.0f, 1.0f) * 32767.0f));
        pad.ly = int16_t(std::lround(std::clamp(ly, -1.0f, 1.0f) * 32767.0f));
    }
    if (rx || ry) {
        pad.rx = int16_t(std::lround(std::clamp(rx, -1.0f, 1.0f) * 32767.0f));
        pad.ry = int16_t(std::lround(std::clamp(ry, -1.0f, 1.0f) * 32767.0f));
        // the drag's turn eases out over the next few polls, as a stick held for a moment would
        g_rx = std::fabs(rx) < 0.02f ? 0.0f : rx * 0.55f;
        g_ry = std::fabs(ry) < 0.02f ? 0.0f : ry * 0.55f;
    }
}

void Draw() {
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    if (!dl) return;
    EnsureArt();
    ImGuiIO& io = ImGui::GetIO();
    Layout l = Make(io.DisplaySize.x, io.DisplaySize.y);
    const ImU32 fill = IM_COL32(255, 255, 255, 30), line = IM_COL32(255, 255, 255, 80);
    const ImU32 label = IM_COL32(255, 255, 255, 210);
    ImFont* font = ImGui::GetFont();  // io.FontDefault stays null unless set: the frame's font, the atlas's first

    // the gear, drawn as the pills are: a dark disc with a thin rim, the gear's outline in it, gold while the panel
    // is open
    {
        ImVec2 c(l.gear[0], l.gear[1]);
        float r = l.gear_r;
        ImU32 stroke = menu::Open() ? IM_COL32(235, 190, 95, 235) : label;
        dl->AddCircleFilled(c, r * 1.3f, IM_COL32(20, 20, 20, 150), 40);
        dl->AddCircle(c, r * 1.3f, line, 40, 1.5f);
        // eight teeth: each a flat top at the outer radius and a flat valley at the inner one, slanted sides between
        constexpr int kTeeth = 8;
        constexpr float kStep = 6.2831853f / kTeeth;
        float outer = r * 0.86f, inner = r * 0.64f;
        ImVec2 pts[kTeeth * 4];
        for (int i = 0; i < kTeeth; i++) {
            float a = float(i) * kStep;
            const float corners[4][2] = {{-0.30f, inner}, {-0.17f, outer}, {0.17f, outer}, {0.30f, inner}};
            for (int k = 0; k < 4; k++) {
                float ak = a + corners[k][0] * kStep;
                pts[i * 4 + k] = ImVec2(c.x + std::cos(ak) * corners[k][1], c.y + std::sin(ak) * corners[k][1]);
            }
        }
        float width = std::max(r * 0.11f, 1.5f);
        dl->AddPolyline(pts, kTeeth * 4, stroke, ImDrawFlags_Closed, width);
        dl->AddCircle(c, r * 0.27f, stroke, 24, width);
    }
    // the panel is over the game: the pad's controls are not drawn and take no touch until it closes
    if (menu::Open()) return;

    // the stick
    dl->AddCircleFilled(ImVec2(l.stick[0], l.stick[1]), l.stick_r, fill, 48);
    dl->AddCircle(ImVec2(l.stick[0], l.stick[1]), l.stick_r, line, 48, 2.0f);
    dl->AddCircleFilled(ImVec2(l.stick[0] + g_lx * l.stick_r * 0.72f, l.stick[1] - g_ly * l.stick_r * 0.72f),
                        l.stick_r * 0.36f, IM_COL32(255, 255, 255, 70), 32);

    // the four face buttons, the pad's own art
    for (int b = 0; b < 4; b++) {
        ImVec2 c(l.btn[b][0], l.btn[b][1]);
        ImVec2 a(c.x - l.btn_r, c.y - l.btn_r), z(c.x + l.btn_r, c.y + l.btn_r);
        if (g_tex[b]) dl->AddImage(g_tex[b], a, z);
        else {
            dl->AddCircleFilled(c, l.btn_r, fill, 32);
            dl->AddCircle(c, l.btn_r, line, 32, 2.0f);
        }
    }

    // the arrows: four round buttons apart, each with its arrow, lit while held
    {
        uint32_t held = g_buttons.load();
        const float dirs[4][2] = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
        for (int a = 0; a < 4; a++) {
            ImVec2 c(l.arrow[a][0], l.arrow[a][1]);
            float r = l.arrow_r;
            bool on = held & kArrowBits[a];
            dl->AddCircleFilled(c, r, on ? IM_COL32(255, 255, 255, 90) : fill, 32);
            dl->AddCircle(c, r, line, 32, 2.0f);
            // the arrow head pointing away from the middle
            float dx = dirs[a][0], dy = dirs[a][1], t = r * 0.42f;
            ImVec2 tip(c.x + dx * t, c.y + dy * t);
            ImVec2 b1(c.x - dx * t * 0.55f - dy * t, c.y - dy * t * 0.55f + dx * t);
            ImVec2 b2(c.x - dx * t * 0.55f + dy * t, c.y - dy * t * 0.55f - dx * t);
            dl->AddTriangleFilled(tip, b1, b2, label);
        }
    }

    // START, BACK and the two triggers
    static const struct {
        int button;
        const char* text;
    } kPills[] = {{kBtnBack, "BACK"}, {kBtnStart, "START"}, {kBtnLT, "LT"}, {kBtnRT, "RT"}};
    for (const auto& p : kPills) {
        ImVec2 c(l.pill[p.button][0], l.pill[p.button][1]);
        dl->AddRectFilled(ImVec2(c.x - l.pill_w / 2, c.y - l.pill_h / 2), ImVec2(c.x + l.pill_w / 2, c.y + l.pill_h / 2),
                          IM_COL32(20, 20, 20, 150), l.pill_h / 2);
        dl->AddRect(ImVec2(c.x - l.pill_w / 2, c.y - l.pill_h / 2), ImVec2(c.x + l.pill_w / 2, c.y + l.pill_h / 2),
                    line, l.pill_h / 2, 0, 1.5f);
        float size = l.pill_h * 0.46f;
        ImVec2 t = font->CalcTextSizeA(size, FLT_MAX, 0, p.text);
        dl->AddText(font, size, ImVec2(c.x - t.x / 2, c.y - t.y / 2), label, p.text);
    }
}

}  // namespace touch
