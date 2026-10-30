// The Android host: SDL's window and event loop, the app's lifecycle, the settings and folders; the game is described
// by app::kGame (src/common/game_info.cpp), as on Linux, whose host (sdk/runtime/app/main_linux.cpp) this follows
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_system.h>
#include <jni.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

#include <imgui.h>

#include "app/app.h"
#include "core/input.h"
#include "core/log.h"
#include "host.h"
#include "image_symbols.h"
#include "kernel/kernel.h"

namespace app {

std::filesystem::path g_game_dir;  // native/game_files.cpp: where the game's files were found

namespace {

SDL_Window* g_window = nullptr;
std::vector<std::string> g_args;
// SDL_RegisterEvents' range: what other threads ask of the window thread
uint32_t g_event_base = 0;
enum : uint32_t { kApplyDisplay, kOpenWindow, kRun, kEventCount };
std::atomic<bool> g_running{true};
std::atomic<bool> g_foreground{true};
// each return to the front: Android hands a new window, whose surface the renderer makes again (host::SurfaceEpoch)
std::atomic<uint32_t> g_surface_epoch{0};

std::atomic<uint32_t> g_pixel_w{1280};
std::atomic<uint32_t> g_pixel_h{720};
std::atomic<float> g_density{1};

std::mutex g_open_lock;
std::condition_variable g_open_cv;
uint64_t g_open_asked = 0;
uint64_t g_open_done = 0;

bool HasArg(const char* arg) { return std::find(g_args.begin(), g_args.end(), arg) != g_args.end(); }

void PushEvent(uint32_t kind, int32_t code = 0, void* data = nullptr) {
    SDL_Event e{};
    e.type = g_event_base + kind;
    e.user.code = code;
    e.user.data1 = data;
    SDL_PushEvent(&e);
}

// window thread: what other threads read of the window
void UpdateSizes() {
    if (!g_window) return;
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(g_window, &w, &h);
    g_pixel_w = uint32_t(std::max(w, 1));
    g_pixel_h = uint32_t(std::max(h, 1));
    float density = SDL_GetWindowPixelDensity(g_window);
    g_density = density > 0 ? density : 1.0f;
}

ImGuiKey KeyOf(SDL_Keycode k) {
    if (k >= SDLK_A && k <= SDLK_Z) return ImGuiKey(ImGuiKey_A + (k - SDLK_A));
    if (k >= SDLK_0 && k <= SDLK_9) return ImGuiKey(ImGuiKey_0 + (k - SDLK_0));
    switch (k) {
        case SDLK_TAB: return ImGuiKey_Tab;
        case SDLK_LEFT: return ImGuiKey_LeftArrow;
        case SDLK_RIGHT: return ImGuiKey_RightArrow;
        case SDLK_UP: return ImGuiKey_UpArrow;
        case SDLK_DOWN: return ImGuiKey_DownArrow;
        case SDLK_BACKSPACE: return ImGuiKey_Backspace;
        case SDLK_SPACE: return ImGuiKey_Space;
        case SDLK_RETURN: return ImGuiKey_Enter;
        case SDLK_ESCAPE: return ImGuiKey_Escape;
        default: return ImGuiKey_None;
    }
}

int MouseButtonOf(uint8_t b) {
    switch (b) {
        case SDL_BUTTON_LEFT: return 0;
        case SDL_BUTTON_RIGHT: return 1;
        case SDL_BUTTON_MIDDLE: return 2;
        default: return -1;
    }
}

// the event as ImGui's input, in window pixels as the renderer draws it; UiLock held. SDL turns touches into mouse
// events on Android, which is what the panel takes
void FeedImGui(const SDL_Event& e) {
    if (!ImGui::GetCurrentContext()) return;
    ImGuiIO& io = ImGui::GetIO();
    float density = g_density;
    switch (e.type) {
        case SDL_EVENT_MOUSE_MOTION:
            io.AddMouseSourceEvent(ImGuiMouseSource_Mouse);
            io.AddMousePosEvent(e.motion.x * density, e.motion.y * density);
            break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
        case SDL_EVENT_MOUSE_BUTTON_UP:
            if (int b = MouseButtonOf(e.button.button); b >= 0) {
                io.AddMouseSourceEvent(ImGuiMouseSource_Mouse);
                io.AddMouseButtonEvent(b, e.type == SDL_EVENT_MOUSE_BUTTON_DOWN);
            }
            break;
        case SDL_EVENT_TEXT_INPUT:
            io.AddInputCharactersUTF8(e.text.text);
            break;
        case SDL_EVENT_KEY_DOWN:
        case SDL_EVENT_KEY_UP:
            if (ImGuiKey key = KeyOf(e.key.key); key != ImGuiKey_None) io.AddKeyEvent(key, e.type == SDL_EVENT_KEY_DOWN);
            break;
        case SDL_EVENT_WINDOW_FOCUS_GAINED:
        case SDL_EVENT_WINDOW_FOCUS_LOST:
            io.AddFocusEvent(e.type == SDL_EVENT_WINDOW_FOCUS_GAINED);
            break;
        default:
            break;
    }
}

void Handle(const SDL_Event& e) {
    {
        std::lock_guard lock(UiLock());
        FeedImGui(e);
    }
    if (e.type >= g_event_base && e.type < g_event_base + kEventCount) {
        switch (e.type - g_event_base) {
            case kApplyDisplay:
                break;  // Android shows the window full screen and does not resize it
            case kOpenWindow: {
                SDL_ShowWindow(g_window);
                SDL_RaiseWindow(g_window);
                std::lock_guard lock(g_open_lock);
                g_open_done++;
                g_open_cv.notify_all();
                break;
            }
            case kRun: {
                std::unique_ptr<std::function<void()>> fn(static_cast<std::function<void()>*>(e.user.data1));
                (*fn)();
                break;
            }
        }
        return;
    }
    switch (e.type) {
        case SDL_EVENT_QUIT:
        case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            LOGR("window: quit event 0x%x", unsigned(e.type));
            g_running = false;
            break;
        case SDL_EVENT_WILL_ENTER_BACKGROUND:
            g_foreground = false;
            break;
        case SDL_EVENT_DID_ENTER_FOREGROUND:
            if (!g_foreground) g_surface_epoch++;
            g_foreground = true;
            UpdateSizes();
            break;
        case SDL_EVENT_TERMINATING:
            g_running = false;
            break;
        case SDL_EVENT_LOW_MEMORY:
            LOGW("app: low memory");
            break;
        case SDL_EVENT_GAMEPAD_ADDED:
        case SDL_EVENT_GAMEPAD_REMOVED:
            input::OnDevice(e.type == SDL_EVENT_GAMEPAD_ADDED, e.gdevice.which);
            break;
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
        case SDL_EVENT_WINDOW_DISPLAY_CHANGED:
        case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
            UpdateSizes();
            break;
        case SDL_EVENT_WINDOW_FOCUS_LOST:
            break;
        default:
            break;
    }
}

// UTF-8 from a wide string: wchar_t is UTF-32 here
std::string Narrow(const std::wstring& w) {
    std::string s;
    for (wchar_t c : w) {
        auto u = uint32_t(c);
        if (u < 0x80) {
            s += char(u);
        } else if (u < 0x800) {
            s += char(0xC0 | (u >> 6));
            s += char(0x80 | (u & 0x3F));
        } else if (u < 0x10000) {
            s += char(0xE0 | (u >> 12));
            s += char(0x80 | ((u >> 6) & 0x3F));
            s += char(0x80 | (u & 0x3F));
        } else {
            s += char(0xF0 | (u >> 18));
            s += char(0x80 | ((u >> 12) & 0x3F));
            s += char(0x80 | ((u >> 6) & 0x3F));
            s += char(0x80 | (u & 0x3F));
        }
    }
    return s;
}

// the settings file the port starts with: the screen's own size, two thirds of its height rendered, 60 a second as
// the title's timing counts on (at 30 it runs at half speed)
void WriteDefaultSettings(const std::filesystem::path& path) {
    std::ofstream f(path);
    if (!f) return;
    f << "display_mode = \"borderless\"\n"
         "width = 0\n"
         "height = 0\n"
         "render_percent = 67\n"
         "fps_cap = 60\n"
         "vsync = true\n"
         "anisotropic_filtering = 0\n";
}

}  // namespace

std::string Utf8(const std::wstring& text) { return Narrow(text); }

// the saves were kept in the app's own files folder, which an uninstall wipes: the first run that finds the game
// takes them into the game's folder, so the progress already made is not left behind. True when the game's folder is
// the one to use, false when the old saves could not be moved and stay where they are
bool TakeSaves(const std::filesystem::path& to) {
    std::error_code ec;
    const std::filesystem::path from = ExeDir() / "hdd";
    if (std::filesystem::exists(to, ec)) return true;           // already there, the app's own left alone
    if (!std::filesystem::is_directory(from, ec)) return true;  // a fresh install: made there
    std::filesystem::rename(from, to, ec);
    if (ec) {
        ec.clear();
        std::filesystem::copy(from, to, std::filesystem::copy_options::recursive, ec);
        if (ec) {
            LOGW("saves: %s did not move to %s: %s", from.c_str(), to.c_str(), ec.message().c_str());
            std::error_code ignored;
            std::filesystem::remove_all(to, ignored);  // what was half copied
            return false;
        }
        std::filesystem::remove_all(from, ec);
    }
    LOGI("saves: moved from %s to %s", from.c_str(), to.c_str());
    return true;
}

std::filesystem::path ExeDir() {
    static const std::filesystem::path dir = [] {
        const char* internal = SDL_GetAndroidInternalStoragePath();
        if (internal && *internal) return std::filesystem::path(internal);
        return std::filesystem::path(SDL_GetBasePath() ? SDL_GetBasePath() : ".");
    }();
    return dir;
}

int Instance() { return 0; }

void* Window() { return g_window; }

void Quit() {
    SDL_Event e{};
    e.type = SDL_EVENT_QUIT;
    SDL_PushEvent(&e);
}

void Post(std::function<void()> fn) { PushEvent(kRun, 0, new std::function<void()>(std::move(fn))); }

void ShowError(const std::wstring& text) {
    std::string utf8 = Narrow(text);
    LOGE("error: %s", utf8.c_str());
    host::Toast(utf8.c_str());
    // the toast outlives the process only while it is up: give it a moment, then let the activity close
    std::this_thread::sleep_for(std::chrono::seconds(3));
}

bool AskYesNo(const std::wstring& text) {
    ShowError(text);
    return false;
}

std::recursive_mutex& UiLock() {
    static std::recursive_mutex m;
    return m;
}

void ApplyDisplay() {}
void SetDisplay(uint32_t) {}

void OpenGameWindow(bool) {
    std::unique_lock lock(g_open_lock);
    uint64_t ticket = ++g_open_asked;
    PushEvent(kOpenWindow);
    g_open_cv.wait(lock, [&] { return g_open_done >= ticket || !g_running; });
}

Mode Desktop() { return {g_pixel_w.load(), g_pixel_h.load()}; }
Mode WindowPixels() { return {g_pixel_w.load(), g_pixel_h.load()}; }

void UiFrame() {
    using clock = std::chrono::steady_clock;
    static clock::time_point last = clock::now();
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(float(g_pixel_w), float(g_pixel_h));
    io.DisplayFramebufferScale = ImVec2(1, 1);
    auto now = clock::now();
    io.DeltaTime = std::max(std::chrono::duration<float>(now - last).count(), 1e-4f);
    last = now;
    // the on-screen D-pad and A move through the panel: port 1's poll takes the touch controls with the keyboard
    input::Pad p;
    for (int port = 0; port < 4 && !p.connected; port++) p = port == 0 ? input::Poll(0) : input::Read(port);
    if (p.connected) io.BackendFlags |= ImGuiBackendFlags_HasGamepad;
    else io.BackendFlags &= ~ImGuiBackendFlags_HasGamepad;
    if (!p.connected) return;
    constexpr uint8_t kPressed = 0x40;
    auto key = [&](ImGuiKey k, bool down) { io.AddKeyEvent(k, down); };
    key(ImGuiKey_GamepadDpadUp, p.buttons & 0x01);
    key(ImGuiKey_GamepadDpadDown, p.buttons & 0x02);
    key(ImGuiKey_GamepadDpadLeft, p.buttons & 0x04);
    key(ImGuiKey_GamepadDpadRight, p.buttons & 0x08);
    key(ImGuiKey_GamepadStart, p.buttons & 0x10);
    key(ImGuiKey_GamepadBack, p.buttons & 0x20);
    key(ImGuiKey_GamepadFaceDown, p.analog[0] > kPressed);
    key(ImGuiKey_GamepadFaceRight, p.analog[1] > kPressed);
    key(ImGuiKey_GamepadFaceLeft, p.analog[2] > kPressed);
    key(ImGuiKey_GamepadFaceUp, p.analog[3] > kPressed);
    key(ImGuiKey_GamepadR1, p.analog[4] > kPressed);
    key(ImGuiKey_GamepadL1, p.analog[5] > kPressed);
    key(ImGuiKey_GamepadL2, p.analog[6] > kPressed);
    key(ImGuiKey_GamepadR2, p.analog[7] > kPressed);
    constexpr float kDead = 8000.0f;
    auto stick = [&](ImGuiKey k, float v) {
        float a = std::clamp((v - kDead) / (32767.0f - kDead), 0.0f, 1.0f);
        io.AddKeyAnalogEvent(k, a > 0.1f, a);
    };
    stick(ImGuiKey_GamepadLStickLeft, -float(p.lx));
    stick(ImGuiKey_GamepadLStickRight, float(p.lx));
    stick(ImGuiKey_GamepadLStickUp, float(p.ly));
    stick(ImGuiKey_GamepadLStickDown, -float(p.ly));
}

void Pace() {
    using clock = std::chrono::steady_clock;
    static clock::time_point next = clock::now();
    uint32_t cap = Get().fps_cap;
    auto now = clock::now();
    if (!cap) {
        next = now;
        return;
    }
    auto period = std::chrono::nanoseconds(1000000000ull / cap);
    next += period;
    if (next < now - period) next = now;
    while (clock::now() + std::chrono::milliseconds(1) < next) std::this_thread::sleep_for(std::chrono::microseconds(500));
    while (clock::now() < next) std::this_thread::yield();
}

// the port has no splash picture and one instance only
bool OpenSplash() { return false; }
void SplashFirstFrame() {}
void CloseSplash() {}
bool StartHandover() { return false; }
bool HandingOver() { return false; }
void FinishHandover() {}
bool PrepareGameCode(const GameFiles&) { return true; }

}  // namespace app

namespace host {

SDL_Window* Window() { return app::g_window; }
bool Foreground() { return app::g_foreground; }
uint32_t SurfaceEpoch() { return app::g_surface_epoch; }
std::filesystem::path FilesDir() { return app::ExeDir(); }
std::filesystem::path GameDir() { return app::g_game_dir; }

void Toast(const char* text) {
    JNIEnv* env = static_cast<JNIEnv*>(SDL_GetAndroidJNIEnv());
    if (!env) return;
    jclass cls = env->FindClass("com/datfooldive/dw5/DW5Activity");
    if (!cls) return;
    jmethodID mid = env->GetStaticMethodID(cls, "showToast", "(Ljava/lang/String;)V");
    if (mid) {
        jstring s = env->NewStringUTF(text);
        env->CallStaticVoidMethod(cls, mid, s);
        env->DeleteLocalRef(s);
    }
    env->DeleteLocalRef(cls);
}

}  // namespace host

int main(int argc, char** argv) {
    // SDL sets the activity's orientation from the window it is asked for, over the manifest's sensorLandscape
    SDL_SetHint(SDL_HINT_ORIENTATIONS, "LandscapeLeft LandscapeRight");
    app::g_args.assign(argv + 1, argv + argc);
    // no log file: the release keeps nothing on the device

    std::string title = app::Narrow(app::kGame.title);
    SDL_SetAppMetadata(title.c_str(), nullptr, nullptr);
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        LOGE("SDL video failed: %s", SDL_GetError());
        app::ShowError(L"No display to open the game window on, see the log in the logs folder.");
        return 1;
    }
    app::g_event_base = SDL_RegisterEvents(app::kEventCount);
    app::UpdateSizes();

    std::error_code ec;
    if (!std::filesystem::exists(app::ExeDir() / app::kGame.settings, ec))
        app::WriteDefaultSettings(app::ExeDir() / app::kGame.settings);
    app::Load();
    // 30 a second was the first default, which the title plays in slow motion at; the port has no settings screen,
    // so a 30 can only be that default
    if (app::Get().fps_cap == 30) {
        app::Get().fps_cap = 60;
        app::Save();
    }
    if (app::kGame.init) app::kGame.init();
    app::LogSystem();

    app::GameFiles game;
    if (!app::FindGame(game)) return 1;
    if (!app::PrepareGameCode(game)) return 1;
    // the saves go in the game's own folder, beside its files: an uninstall takes the app's folder, not this one.
    // An install that still has them in the app's own files takes them along, and keeps them there if they cannot
    // be moved, so no progress is lost
    std::filesystem::path hdd = game.dir / "hdd";
    if (!app::TakeSaves(hdd)) hdd = app::ExeDir() / "hdd";

    app::g_window = SDL_CreateWindow(title.c_str(), 1280, 720, SDL_WINDOW_VULKAN | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!app::g_window) {
        LOGE("SDL window failed: %s", SDL_GetError());
        app::ShowError(L"The game window could not be opened, see the log in the logs folder.");
        return 1;
    }
    app::UpdateSizes();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.BackendPlatformName = "sdl3 (android)";
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;

    input::Init();
    touch::Init();
    if (app::kGame.window) app::kGame.window(app::g_window);
    if (!kernel::Boot(std::move(game.xbe), IMAGE_XBE, game.dir.string(), hdd.string())) {
        app::ShowError(L"The game failed to start, see the log in the logs folder.");
        return 1;
    }

    // Polled, then a short sleep: SDL_WaitEventTimeout on Android pumps its events again each time anything is
    // queued, its own bookkeeping included, and kept this thread at 85% of a big core while the game ran. A touch
    // waits 4 ms at most.
    constexpr auto kIdle = std::chrono::milliseconds(4);
    while (app::g_running) {
        SDL_Event e;
        bool any = false;
        while (app::g_running && SDL_PollEvent(&e)) {
            app::Handle(e);
            any = true;
        }
        if (!any) std::this_thread::sleep_for(kIdle);
    }
    {
        std::lock_guard lock(app::g_open_lock);
        app::g_open_cv.notify_all();
    }
    logx::Flush();
    kernel::Shutdown(0);
    return 0;
}
