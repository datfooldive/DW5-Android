#pragma once
#include <filesystem>

#include <SDL3/SDL_events.h>

#include "core/input.h"

struct SDL_Window;

// the Android host, shared between native/main.cpp, the Vulkan backend and the touch controls
namespace host {

SDL_Window* Window();
bool Foreground();                 // the activity is in front: the renderer waits while it is not
std::filesystem::path FilesDir();  // internal, writable: logs, settings, hdd, shader cache
std::filesystem::path GameDir();   // where the game's files were found, empty until FindGame finds them
void Toast(const char* text);      // the activity's toast, any thread

}  // namespace host

namespace touch {

void Init();
void Event(const SDL_Event& e);  // a finger event, window thread
// gameplay: the sticks and buttons of the on-screen controls, over the pad the keyboard left
void Apply(input::Pad& pad);
void Draw();  // inside the ImGui frame, UiLock held

}  // namespace touch
