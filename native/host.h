#pragma once
#include <filesystem>

#include <SDL3/SDL_events.h>

#include "core/input.h"

struct SDL_Window;

// the Android host, shared between native/main.cpp, the Vulkan backend and the touch controls
namespace host {

SDL_Window* Window();
bool Foreground();                 // the activity is in front: the renderer waits while it is not
uint32_t SurfaceEpoch();           // counts returns to the front: a new window, its surface made again
std::filesystem::path FilesDir();  // internal, writable: logs, settings, hdd, shader cache
std::filesystem::path GameDir();   // where the game's files were found, empty until FindGame finds them
void Toast(const char* text);      // the activity's toast, any thread
// once the settings file is loaded, and as a controller comes or goes; any thread
void ControlsSync(bool pad);

}  // namespace host

namespace touch {

// The controls are TouchControls.java's, an overlay the activity draws; what they hold and where they sit come here
// (native/settings_bridge.cpp).
void Init();
// what the fingers hold: input::Pad::buttons bits, a bit per analog button held, the sticks -1 to 1 with y up
void Set(uint32_t buttons, uint32_t analog, float lx, float ly, float rx, float ry);
void Swipe(float dx, float dy);  // the right stick swiped by a drag, as a share of a full push; eases out in Apply
// the layout, each control's centre as a fraction of the screen and its size as a fraction of its kind's: read from
// the settings file's table before the first frame, written back into it
void LoadLayout(const void* table);
void SaveLayout(void* table);
int Units();
void GetUnit(int unit, float& x, float& y, float& scale);
void SetUnit(int unit, float x, float y, float scale, bool commit);  // commit writes the file
void ResetUnits();  // the pad's own layout back, every control at its kind's size
// gameplay: the sticks and buttons of the on-screen controls, over the pad the keyboard left
void Apply(input::Pad& pad);
void Frame();  // once a frame: tells the overlay when a controller comes or goes

}  // namespace touch
