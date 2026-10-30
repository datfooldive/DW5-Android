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
    uint32_t aspect = 1;         // the frame's shape on the screen: 0 the TV's, else widescreen: the camera sees as
                                 // wide as the screen and the 2D layer keeps its 16:9 shape in the middle (Android);
                                 // a file from before keeps 1 to 4, its other shapes, which all read as widescreen
    uint32_t controls_opacity = 100;  // the on-screen controls' opacity, 20 to 100% (Android)
    uint32_t camera_sensitivity = 300;  // the on-screen right stick, 50 to 300%: at 100 a full push is a tenth of the
                                        // screen's short side from where the finger went down (Android)
    uint32_t camera_mode = 0;  // the on-screen right stick: 0 floating, centred where the finger went down; 1 fixed,
                               // a stick of its own the layout editor places; 2 swipe, a drag turns the view (Android)
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
