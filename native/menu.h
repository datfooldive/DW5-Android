#pragma once

// the settings panel the on-screen gear opens: the renderer's settings and the game's options, saved as they change
namespace menu {

bool Open();
void Toggle();
void Draw();  // inside the ImGui frame, UiLock held

}  // namespace menu
