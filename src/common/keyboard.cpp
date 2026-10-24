// the keyboard as the controller in port 1, on top of a real pad there if one is plugged in
#include <SDL3/SDL.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "core/input.h"
#include "host.h"

namespace game {

namespace {

enum Button : uint16_t {
    kUp = 1 << 0,
    kDown = 1 << 1,
    kLeft = 1 << 2,
    kRight = 1 << 3,
    kStart = 1 << 4,
    kBack = 1 << 5,
    kLeftThumb = 1 << 6,
    kRightThumb = 1 << 7,
};
enum Analog { kA, kB, kX, kY, kBlack, kWhite, kLeftTrigger, kRightTrigger };

constexpr struct {
    SDL_Scancode key;
    uint16_t button;
} kButtons[] = {
    {SDL_SCANCODE_UP, kUp},         {SDL_SCANCODE_DOWN, kDown},        {SDL_SCANCODE_LEFT, kLeft},
    {SDL_SCANCODE_RIGHT, kRight},   {SDL_SCANCODE_RETURN, kStart},     {SDL_SCANCODE_BACKSPACE, kBack},
    {SDL_SCANCODE_ESCAPE, kBack},   {SDL_SCANCODE_LSHIFT, kLeftThumb}, {SDL_SCANCODE_RSHIFT, kRightThumb},
};

constexpr struct {
    SDL_Scancode key;
    Analog analog;
} kAnalogs[] = {
    {SDL_SCANCODE_SPACE, kA}, {SDL_SCANCODE_J, kA},     {SDL_SCANCODE_K, kB},
    {SDL_SCANCODE_U, kX},     {SDL_SCANCODE_I, kY},     {SDL_SCANCODE_O, kBlack},
    {SDL_SCANCODE_L, kWhite}, {SDL_SCANCODE_Q, kLeftTrigger}, {SDL_SCANCODE_E, kRightTrigger},
};

// a pair of keys as one stick axis, the pad's own deflection kept when no key is down
int16_t Axis(const bool* keys, SDL_Scancode neg, SDL_Scancode pos, int16_t pad) {
    int v = (keys[pos] ? 32767 : 0) - (keys[neg] ? 32767 : 0);
    return v ? int16_t(v) : pad;
}

// XBE_PAD_SCRIPT: a file of "seconds buttons [seconds held]" lines, buttons joined by + (UP DOWN LEFT RIGHT START
// BACK A B X Y), held a tenth of a second by default and timed from the first poll; to reach a scene unattended
struct Press {
    double at, len;
    uint16_t buttons;
    uint8_t analogs;  // bit per Analog
};

std::vector<Press> LoadScript() {
    std::vector<Press> v;
    const char* path = std::getenv("XBE_PAD_SCRIPT");
    FILE* f = path ? std::fopen(path, "r") : nullptr;
    if (!f) return v;
    char line[160];
    while (std::fgets(line, sizeof(line), f)) {
        char names[96];
        Press p{0, 0.1, 0, 0};
        if (line[0] == '#' || std::sscanf(line, "%lf %95s %lf", &p.at, names, &p.len) < 2) continue;
        for (char* n = std::strtok(names, "+"); n; n = std::strtok(nullptr, "+")) {
            constexpr struct {
                const char* name;
                uint16_t button;
                int analog;
            } kNames[] = {{"UP", kUp, -1},       {"DOWN", kDown, -1}, {"LEFT", kLeft, -1}, {"RIGHT", kRight, -1},
                          {"START", kStart, -1}, {"BACK", kBack, -1}, {"A", 0, kA},        {"B", 0, kB},
                          {"X", 0, kX},          {"Y", 0, kY}};
            for (auto& k : kNames)
                if (!std::strcmp(n, k.name)) {
                    p.buttons |= k.button;
                    if (k.analog >= 0) p.analogs |= uint8_t(1 << k.analog);
                }
        }
        v.push_back(p);
    }
    std::fclose(f);
    return v;
}

void Script(input::Pad& p) {
    static const std::vector<Press> presses = LoadScript();
    if (presses.empty()) return;
    static const auto start = std::chrono::steady_clock::now();
    double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    p.connected = true;
    for (const Press& s : presses) {
        if (now < s.at || now >= s.at + s.len) continue;
        p.buttons |= s.buttons;
        for (int a = 0; a < 8; a++)
            if (s.analogs >> a & 1) p.analog[a] = 0xFF;
    }
}

}  // namespace

// the keyboard over a pad: keys held add to it, the sticks keep the pad's deflection while no key is down
void Keyboard(input::Pad& p) {
    const bool* keys = SDL_GetKeyboardState(nullptr);
    if (!keys) return;
    p.connected = true;
    for (auto [key, button] : kButtons)
        if (keys[key]) p.buttons |= button;
    for (auto [key, analog] : kAnalogs)
        if (keys[key]) p.analog[analog] = 0xFF;
    p.lx = Axis(keys, SDL_SCANCODE_A, SDL_SCANCODE_D, p.lx);
    p.ly = Axis(keys, SDL_SCANCODE_S, SDL_SCANCODE_W, p.ly);  // y up, as the console's stick
    p.rx = Axis(keys, SDL_SCANCODE_KP_4, SDL_SCANCODE_KP_6, p.rx);
    p.ry = Axis(keys, SDL_SCANCODE_KP_2, SDL_SCANCODE_KP_8, p.ry);
}

// GameInfo::pad: the keyboard and the on-screen controls on port 1
void PadHook(int port, input::Pad& p) {
    if (port != 0) return;
    Keyboard(p);
    touch::Apply(p);
    Script(p);
}

}  // namespace game
