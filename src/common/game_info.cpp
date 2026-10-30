#include "app/app.h"

#include <imgui.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>

#include "common/options.h"
#include "host.h"

namespace game {
void InstallCaptureSignal();
void PadHook(int port, input::Pad& p);

void Init() { InstallCaptureSignal(); }
// the frame rate in the top left corner above LT; the on-screen controls are the activity's overlay, told here when a
// controller comes or goes
void Ui(double fps) {
    touch::Frame();
    if (!game::Opt().fps_counter) return;
    ImGuiIO& io = ImGui::GetIO();
    float size = std::max(io.DisplaySize.y * 0.03f, 12.0f);
    char text[16];
    std::snprintf(text, sizeof(text), "%.0f FPS", fps);
    ImVec2 at(size * 0.5f, size * 0.25f);
    ImVec2 extent = ImGui::GetFont()->CalcTextSizeA(size, FLT_MAX, 0, text);
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    dl->AddRectFilled(ImVec2(at.x - size * 0.25f, at.y), ImVec2(at.x + extent.x + size * 0.25f, at.y + extent.y),
                      IM_COL32(0, 0, 0, 120), size * 0.2f);
    // green near the cap (60 when unlimited), amber under it: the title slows down with its frame rate
    uint32_t cap = app::Get().fps_cap ? app::Get().fps_cap : 60;
    bool full = fps >= cap * 0.93;
    dl->AddText(ImGui::GetFont(), size, at, full ? IM_COL32(120, 230, 120, 230) : IM_COL32(240, 200, 90, 230), text);
}
}  // namespace game

namespace {
// the controls' layout lives in the settings file's table: options hold flags and single values only
void LoadControls(const void* table) { touch::LoadLayout(table); }
void SaveControls(void* table) { touch::SaveLayout(table); }
}  // namespace

// what the runtime asks of the game; strings must be set, hooks may stay null
const app::GameInfo app::kGame = {
    .title = L"Dynasty Warriors 5",         // window and splash
    .short_title = L"Dynasty Warriors 5",   // message boxes
    .name = L"DW5",                         // window classes, crash dumps, handover event
    .log = "dw5",                           // log file prefix
    .settings = "settings.toml",            // settings file beside the exe
    .splash = L"splash",                    // optional splash.png or splash.bmp beside the exe
    .missing = L"Put the game in the assets folder: its disc image (.iso), or default.xbe with its folders beside it.",
    .images = {{"default.xbe", "DW5"}},     // each recompiled xbe and its exe, for title launches
    .options = {{"msaa", nullptr, &game::Opt().msaa},  // game settings kept in the settings file
                {"smaa", &game::Opt().smaa, nullptr},
                {"fps_counter", &game::Opt().fps_counter, nullptr},
                {"frame_skip", &game::Opt().frame_skip, nullptr},
                {"aspect", nullptr, &game::Opt().aspect},
                {"controls_opacity", nullptr, &game::Opt().controls_opacity},
                {"camera_sensitivity", nullptr, &game::Opt().camera_sensitivity},
                {"camera_mode", nullptr, &game::Opt().camera_mode},
                {"hdr_output", &game::Opt().hdr_output, nullptr},
                {"cheat_infinite_health", &game::Opt().cheats[0], nullptr},
                {"cheat_infinite_musou", &game::Opt().cheats[1], nullptr},
                {"cheat_unlock_all", &game::Opt().cheats[2], nullptr},
                {"cheat_all_weapons", &game::Opt().cheats[3], nullptr},
                {"cheat_max_officers", &game::Opt().cheats[4], nullptr},
                {"cheat_max_bodyguards", &game::Opt().cheats[5], nullptr},
                {"cheat_infinite_rage", &game::Opt().cheats[6], nullptr},
                {"cheat_freeze_timer", &game::Opt().cheats[7], nullptr},
                {"cheat_max_speed", &game::Opt().cheats[8], nullptr}},
    .load = LoadControls,                   // reads more from the settings file's toml::table
    .save = SaveControls,                   // writes more into it
    .init = game::Init,                     // after the settings load, before the window and boot
    .loaded = nullptr,                      // image in memory, before its entry point runs
    .window = nullptr,                      // the game window is up
    .message = nullptr,                     // window messages first; true takes one
    .ui = game::Ui,                         // ImGui drawing each frame
    .pad = game::PadHook,                   // changes a polled pad
    .rumble = nullptr,                      // changes a rumble before it is set
};
