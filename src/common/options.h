#pragma once
#include <atomic>
#include <cstdint>

// The renderer's own settings. The title has no options screen here, so these are the
// defaults a fresh install would use; they are read every frame by d3d/vulkan_linux.cpp.
namespace game {

struct Options {
    bool bloom = true;        // the glow added over bright parts of the frame
    bool fog = true;
    uint32_t msaa = 0;        // samples a pixel of the scene, 0 off
    bool smaa = false;        // over the finished frame
    bool hdr_output = false;  // HDR10 on a display that offers it
    bool fps_counter = true;     // the frame rate in the top left corner (Android)
    bool frame_skip = true;      // frames left undrawn while the drawing is behind, the game at its speed (Android)
    uint32_t aspect = 4;         // the frame's shape on the screen: 0 the TV's, 1 16:9, 2 the whole screen stretched,
                                 // 3 the whole screen covered, what overflows cut, 4 ultrawide: the camera sees as
                                 // wide as the screen and the 2D layer keeps its 16:9 shape in the middle (Android)
    uint32_t controls_opacity = 100;  // the on-screen controls' opacity, 20 to 100% (Android)
    uint32_t controls_scale = 100;    // their size, 50 to 150% (Android); the layout editor sets where they sit
    bool cheats[9] = {};              // the CHEATS tab's switches, in common/cheats.cpp's order (Android)
};

inline Options& Opt() {
    static Options options;
    return options;
}

inline std::atomic<bool> g_hdr_display{false};  // the window's display has HDR on, from the renderer

// ultrawide: the screen's width over 16:9's at the same height, 1 when off; the renderer sets it each frame and the
// camera hook (game/hooks.cpp) widens the projection by it
inline std::atomic<float> g_wide{1.0f};

}  // namespace game
