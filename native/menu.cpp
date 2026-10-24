// The settings panel the gear opens. Each setting is a row of choices, the one in use lit; a tap on another takes it
// at once and saves it (SDL turns touches into the mouse ImGui reads, a pad's D-pad and A move through them). It
// holds what changes the game on a phone: how sharp and how far it draws against how fast it runs.
#include "menu.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iterator>

#include <imgui.h>

#include "app/app.h"
#include "common/options.h"

namespace menu {

namespace {

std::atomic<bool> g_open{false};

constexpr ImVec4 kPanelBg{0.06f, 0.05f, 0.05f, 0.95f};
constexpr ImVec4 kAccent{0.72f, 0.14f, 0.10f, 1.0f};  // the lit choice
constexpr ImVec4 kAccentHovered{0.82f, 0.22f, 0.14f, 1.0f};
constexpr ImVec4 kGold{0.92f, 0.74f, 0.36f, 1.0f};
constexpr ImVec4 kChip{0.16f, 0.14f, 0.13f, 1.0f};
constexpr ImVec4 kChipHovered{0.26f, 0.22f, 0.20f, 1.0f};
constexpr ImVec4 kText{0.93f, 0.91f, 0.88f, 1.0f};

const char* const kOffOn[] = {"OFF", "ON"};
const char* const kSkipNames[] = {"OFF", "AUTO"};

// the choices of a row as buttons under its label, the one at chosen lit; the index tapped, -1 for none
int Choices(const char* label, const char* const* names, int count, int chosen) {
    float em = ImGui::GetFontSize();
    ImGui::PushID(label);
    ImGui::Dummy(ImVec2(0, em * 0.35f));
    ImGui::TextUnformatted(label);
    // the choices share the row's width
    float gap = ImGui::GetStyle().ItemSpacing.x;
    float width = (ImGui::GetContentRegionAvail().x - gap * float(count - 1)) / float(count);
    ImVec2 size(width, ImGui::GetFrameHeight() * 1.5f);
    int tapped = -1;
    for (int i = 0; i < count; i++) {
        if (i) ImGui::SameLine();
        bool lit = i == chosen;
        ImGui::PushStyleColor(ImGuiCol_Button, lit ? kAccent : kChip);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, lit ? kAccentHovered : kChipHovered);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, kAccentHovered);
        ImGui::PushStyleColor(ImGuiCol_Text, lit ? ImVec4(1, 1, 1, 1) : kText);
        if (ImGui::Button(names[i], size) && !lit) tapped = i;
        ImGui::PopStyleColor(4);
    }
    ImGui::PopID();
    return tapped;
}

int OffOn(const char* label, bool on) { return Choices(label, kOffOn, 2, on ? 1 : 0); }

// the index of v in values, the nearest when it is not one
template <size_t N>
int IndexOf(const uint32_t (&values)[N], uint32_t v) {
    auto far = [&](int i) { return values[i] > v ? values[i] - v : v - values[i]; };
    int at = 0;
    for (int i = 1; i < int(N); i++)
        if (far(i) < far(at)) at = i;
    return at;
}

constexpr uint32_t kResolutions[] = {33, 50, 67, 75, 100};
const char* const kResolutionNames[] = {"33%", "50%", "67%", "75%", "100%"};
// SMAA filters the finished frame, MSAA samples the scene: one or the other
const char* const kAntiAliasNames[] = {"OFF", "SMAA", "MSAA 2X", "MSAA 4X"};
constexpr uint32_t kDistances[] = {1, 2, 3, 4};
const char* const kDistanceNames[] = {"1X", "2X", "3X", "4X"};

int AntiAliasIndex(const game::Options& o) {
    if (o.msaa >= 4) return 3;
    if (o.msaa >= 2) return 2;
    return o.smaa ? 1 : 0;
}

void Heading(const char* text) {
    float em = ImGui::GetFontSize();
    ImGui::Dummy(ImVec2(0, em * 0.5f));
    ImGui::PushStyleColor(ImGuiCol_Text, kGold);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::GetWindowDrawList()->AddLine(p, ImVec2(p.x + ImGui::GetContentRegionAvail().x, p.y),
                                        ImGui::GetColorU32(ImVec4(kGold.x, kGold.y, kGold.z, 0.4f)), 1.0f);
    ImGui::Dummy(ImVec2(0, em * 0.2f));
}

void DrawRows() {
    app::Settings& s = app::Get();
    game::Options& o = game::Opt();
    bool changed = false;
    int i;

    Heading("GRAPHICS");
    if ((i = Choices("Resolution", kResolutionNames, int(std::size(kResolutionNames)),
                     IndexOf(kResolutions, s.render_percent))) >= 0) {
        s.render_percent = kResolutions[i];
        changed = true;
    }
    if ((i = Choices("Anti-aliasing", kAntiAliasNames, int(std::size(kAntiAliasNames)), AntiAliasIndex(o))) >= 0) {
        o.smaa = i == 1;
        o.msaa = i == 2 ? 2 : i == 3 ? 4 : 0;
        changed = true;
    }
    if ((i = OffOn("Glow", o.bloom)) >= 0) {
        o.bloom = i == 1;
        changed = true;
    }
    if ((i = Choices("Draw distance", kDistanceNames, int(std::size(kDistanceNames)),
                     IndexOf(kDistances, o.draw_distance))) >= 0) {
        o.draw_distance = kDistances[i];
        game::ApplyDrawDistance();
        changed = true;
    }
    if ((i = Choices("Frame skip", kSkipNames, 2, o.frame_skip ? 1 : 0)) >= 0) {
        o.frame_skip = i == 1;
        changed = true;
    }
    if ((i = OffOn("VSync", s.vsync)) >= 0) {
        s.vsync = i == 1;
        changed = true;
    }

    Heading("DISPLAY");
    if ((i = OffOn("FPS counter", o.fps_counter)) >= 0) {
        o.fps_counter = i == 1;
        changed = true;
    }

    if (changed) app::Save();
}

}  // namespace

bool Open() { return g_open.load(std::memory_order_relaxed); }

void Toggle() { g_open.store(!Open(), std::memory_order_relaxed); }

void Draw() {
    if (!Open()) return;
    ImGuiIO& io = ImGui::GetIO();
    // finger sized: the font twice ImGui's 13 pixels on a 1080 line screen
    io.FontGlobalScale = std::max(1.0f, io.DisplaySize.y / 540.0f);
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    float em = ImGui::GetFontSize();
    // under the gear (native/touch.cpp, at the top's middle), which closes it as it opened it
    ImGui::SetNextWindowPos(ImVec2(vp->GetCenter().x, vp->Size.y * 0.17f), ImGuiCond_Always, ImVec2(0.5f, 0.0f));
    ImGui::SetNextWindowSize(ImVec2(std::min(em * 36, vp->Size.x * 0.9f), vp->Size.y * 0.8f), ImGuiCond_Always);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, kPanelBg);
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(kAccent.x, kAccent.y, kAccent.z, 0.8f));
    ImGui::PushStyleColor(ImGuiCol_Text, kText);
    ImGui::PushStyleColor(ImGuiCol_ScrollbarGrab, kChipHovered);
    ImGui::PushStyleColor(ImGuiCol_NavHighlight, kGold);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.5f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, em * 0.6f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, em * 0.45f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(em * 1.2f, em * 0.9f));
    ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, em * 0.8f);
    constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                                        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoTitleBar |
                                        ImGuiWindowFlags_NoMove;
    if (ImGui::Begin("##settings", nullptr, kFlags)) {
        float footer = ImGui::GetFrameHeight() * 1.5f + ImGui::GetStyle().ItemSpacing.y * 2;
        ImGui::BeginChild("##rows", ImVec2(0, -footer), ImGuiChildFlags_NavFlattened);
        DrawRows();
        ImGui::EndChild();
        ImGui::PushStyleColor(ImGuiCol_Button, kChip);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, kChipHovered);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, kAccentHovered);
        if (ImGui::Button("BACK TO THE GAME", ImVec2(-1, ImGui::GetFrameHeight() * 1.5f))) Toggle();
        ImGui::PopStyleColor(3);
    }
    ImGui::End();
    ImGui::PopStyleVar(5);
    ImGui::PopStyleColor(5);
}

}  // namespace menu
