// the host on Linux: an SDL window, its event loop, ImGui's input and crash logs; the game is described by app::kGame
#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <SDL3/SDL.h>
#include <imgui.h>

#include "app/app.h"
#include "core/input.h"
#include "core/log.h"
#include "gpu/backend.h"
#include "image_symbols.h"
#include "kernel/kernel.h"

namespace app {

namespace {

SDL_Window* g_window = nullptr;
std::vector<std::string> g_args;
// SDL_RegisterEvents' range: what other threads ask of the window thread
uint32_t g_event_base = 0;
enum : uint32_t { kApplyDisplay, kOpenWindow, kRun, kEventCount };
bool g_running = true;
bool g_mode_changed = false;  // a fullscreen mode other than the desktop's is set

// kept by the window thread for the others: the window in pixels, pixels per SDL window unit, the desktop mode
std::atomic<uint32_t> g_pixel_w{1280};
std::atomic<uint32_t> g_pixel_h{720};
std::atomic<float> g_density{1};
std::atomic<uint32_t> g_desktop_w{0};
std::atomic<uint32_t> g_desktop_h{0};
std::atomic<uint32_t> g_display{0};  // SetDisplay's, 0 for the window's own

// OpenGameWindow waits for the window thread to have opened it
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

// the crash's guest thread, its registers and return addresses to the log; then the default action, a core dump
void OnCrash(int sig, siginfo_t* info, void*) {
    if (kernel::Thread* t = kernel::Current(); t && t->ctx) {
        LOGE("signal %d (%s) at %p on guest thread %X", sig, strsignal(sig), info ? info->si_addr : nullptr, t->id);
        if (info && (sig == SIGSEGV || sig == SIGBUS)) {
            uint64_t va = uint64_t(reinterpret_cast<uintptr_t>(info->si_addr));
            uint64_t g = uint64_t(reinterpret_cast<uintptr_t>(kernel::g_base));
            LOGE("  access %llX, guest %08llX", (unsigned long long)va, (unsigned long long)(va - g));
        }
        kernel::LogGuestStack(*t->ctx, 256);
    } else {
        LOGE("signal %d (%s) at %p", sig, strsignal(sig), info ? info->si_addr : nullptr);
    }
    logx::Flush();
    signal(sig, SIG_DFL);
    raise(sig);
}

void InstallCrashHandler() {
    struct sigaction sa{};
    sa.sa_sigaction = OnCrash;
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
    sigemptyset(&sa.sa_mask);
    for (int sig : {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT}) sigaction(sig, &sa, nullptr);
}

void DumpThreads() {
    gpu::g_capture = true;
    for (auto& t : kernel::Threads())
        if (t->ctx) {
            LOGI("thread %X", t->id);
            kernel::LogGuestStack(*t->ctx, 32);
        }
}

SDL_DisplayID WindowDisplay() {
    if (uint32_t d = g_display) return d;
    SDL_DisplayID d = g_window ? SDL_GetDisplayForWindow(g_window) : 0;
    return d ? d : SDL_GetPrimaryDisplay();
}

// window thread: what other threads read of the window and its display
void UpdateSizes() {
    if (g_window) {
        int w = 0;
        int h = 0;
        SDL_GetWindowSizeInPixels(g_window, &w, &h);
        g_pixel_w = uint32_t(std::max(w, 1));
        g_pixel_h = uint32_t(std::max(h, 1));
        float density = SDL_GetWindowPixelDensity(g_window);
        g_density = density > 0 ? density : 1.0f;
    }
    if (const SDL_DisplayMode* m = SDL_GetDesktopDisplayMode(WindowDisplay())) {
        float scale = m->pixel_density > 0 ? m->pixel_density : 1.0f;
        g_desktop_w = uint32_t(std::lround(m->w * scale));
        g_desktop_h = uint32_t(std::lround(m->h * scale));
    }
}

// window thread only
void ApplyNow() {
    const Settings& s = Get();
    SDL_DisplayID display = WindowDisplay();
    if (s.display_mode == DisplayMode::Windowed) {
        SDL_SetWindowFullscreen(g_window, false);
        SDL_SetWindowBordered(g_window, true);
        SDL_SetWindowResizable(g_window, true);
        g_mode_changed = false;
        Mode out = Output();
        float density = g_density;
        SDL_SetWindowSize(g_window, int(std::lround(out.width / density)), int(std::lround(out.height / density)));
        SDL_SetWindowPosition(g_window, SDL_WINDOWPOS_CENTERED_DISPLAY(display), SDL_WINDOWPOS_CENTERED_DISPLAY(display));
    } else {
        // borderless is the desktop's own mode on that display; the desktop's size is no mode change either
        const SDL_DisplayMode* desktop = SDL_GetDesktopDisplayMode(display);
        SDL_DisplayMode mode{};
        bool found = false;
        if (s.display_mode == DisplayMode::Fullscreen && s.width && s.height) {
            float scale = desktop && desktop->pixel_density > 0 ? desktop->pixel_density : 1.0f;
            found = SDL_GetClosestFullscreenDisplayMode(display, int(s.width / scale), int(s.height / scale), 0.0f, true,
                                                        &mode);
        }
        if (!found && desktop) {
            mode = *desktop;
            found = true;
        }
        g_mode_changed = found && desktop && (mode.w != desktop->w || mode.h != desktop->h);
        SDL_SetWindowFullscreenMode(g_window, found ? &mode : nullptr);
        SDL_SetWindowFullscreen(g_window, true);
    }
    UpdateSizes();
}

void OpenNow(bool apply) {
    CloseSplash();
    if (apply) ApplyNow();
    SDL_ShowWindow(g_window);
    SDL_RaiseWindow(g_window);
}

ImGuiKey KeyOf(SDL_Keycode k) {
    if (k >= SDLK_A && k <= SDLK_Z) return ImGuiKey(ImGuiKey_A + (k - SDLK_A));
    if (k >= SDLK_0 && k <= SDLK_9) return ImGuiKey(ImGuiKey_0 + (k - SDLK_0));
    if (k >= SDLK_F1 && k <= SDLK_F12) return ImGuiKey(ImGuiKey_F1 + (k - SDLK_F1));
    if (k >= SDLK_KP_1 && k <= SDLK_KP_9) return ImGuiKey(ImGuiKey_Keypad1 + (k - SDLK_KP_1));
    switch (k) {
        case SDLK_TAB: return ImGuiKey_Tab;
        case SDLK_LEFT: return ImGuiKey_LeftArrow;
        case SDLK_RIGHT: return ImGuiKey_RightArrow;
        case SDLK_UP: return ImGuiKey_UpArrow;
        case SDLK_DOWN: return ImGuiKey_DownArrow;
        case SDLK_PAGEUP: return ImGuiKey_PageUp;
        case SDLK_PAGEDOWN: return ImGuiKey_PageDown;
        case SDLK_HOME: return ImGuiKey_Home;
        case SDLK_END: return ImGuiKey_End;
        case SDLK_INSERT: return ImGuiKey_Insert;
        case SDLK_DELETE: return ImGuiKey_Delete;
        case SDLK_BACKSPACE: return ImGuiKey_Backspace;
        case SDLK_SPACE: return ImGuiKey_Space;
        case SDLK_RETURN: return ImGuiKey_Enter;
        case SDLK_ESCAPE: return ImGuiKey_Escape;
        case SDLK_APOSTROPHE: return ImGuiKey_Apostrophe;
        case SDLK_COMMA: return ImGuiKey_Comma;
        case SDLK_MINUS: return ImGuiKey_Minus;
        case SDLK_PERIOD: return ImGuiKey_Period;
        case SDLK_SLASH: return ImGuiKey_Slash;
        case SDLK_SEMICOLON: return ImGuiKey_Semicolon;
        case SDLK_EQUALS: return ImGuiKey_Equal;
        case SDLK_LEFTBRACKET: return ImGuiKey_LeftBracket;
        case SDLK_BACKSLASH: return ImGuiKey_Backslash;
        case SDLK_RIGHTBRACKET: return ImGuiKey_RightBracket;
        case SDLK_GRAVE: return ImGuiKey_GraveAccent;
        case SDLK_CAPSLOCK: return ImGuiKey_CapsLock;
        case SDLK_KP_0: return ImGuiKey_Keypad0;
        case SDLK_KP_PERIOD: return ImGuiKey_KeypadDecimal;
        case SDLK_KP_DIVIDE: return ImGuiKey_KeypadDivide;
        case SDLK_KP_MULTIPLY: return ImGuiKey_KeypadMultiply;
        case SDLK_KP_MINUS: return ImGuiKey_KeypadSubtract;
        case SDLK_KP_PLUS: return ImGuiKey_KeypadAdd;
        case SDLK_KP_ENTER: return ImGuiKey_KeypadEnter;
        case SDLK_LCTRL: return ImGuiKey_LeftCtrl;
        case SDLK_LSHIFT: return ImGuiKey_LeftShift;
        case SDLK_LALT: return ImGuiKey_LeftAlt;
        case SDLK_LGUI: return ImGuiKey_LeftSuper;
        case SDLK_RCTRL: return ImGuiKey_RightCtrl;
        case SDLK_RSHIFT: return ImGuiKey_RightShift;
        case SDLK_RALT: return ImGuiKey_RightAlt;
        case SDLK_RGUI: return ImGuiKey_RightSuper;
        default: return ImGuiKey_None;
    }
}

int MouseButtonOf(uint8_t b) {
    switch (b) {
        case SDL_BUTTON_LEFT: return 0;
        case SDL_BUTTON_RIGHT: return 1;
        case SDL_BUTTON_MIDDLE: return 2;
        case SDL_BUTTON_X1: return 3;
        case SDL_BUTTON_X2: return 4;
        default: return -1;
    }
}

// the event as ImGui's input, in window pixels as the renderer draws it; UiLock held
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
        case SDL_EVENT_MOUSE_WHEEL:
            io.AddMouseWheelEvent(-e.wheel.x, e.wheel.y);
            break;
        case SDL_EVENT_WINDOW_MOUSE_LEAVE:
            io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);
            break;
        case SDL_EVENT_TEXT_INPUT:
            io.AddInputCharactersUTF8(e.text.text);
            break;
        case SDL_EVENT_KEY_DOWN:
        case SDL_EVENT_KEY_UP: {
            bool down = e.type == SDL_EVENT_KEY_DOWN;
            SDL_Keymod m = e.key.mod;
            io.AddKeyEvent(ImGuiMod_Ctrl, (m & SDL_KMOD_CTRL) != 0);
            io.AddKeyEvent(ImGuiMod_Shift, (m & SDL_KMOD_SHIFT) != 0);
            io.AddKeyEvent(ImGuiMod_Alt, (m & SDL_KMOD_ALT) != 0);
            io.AddKeyEvent(ImGuiMod_Super, (m & SDL_KMOD_GUI) != 0);
            if (ImGuiKey key = KeyOf(e.key.key); key != ImGuiKey_None) io.AddKeyEvent(key, down);
            break;
        }
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
    if (kGame.event && kGame.event(&e)) return;
    if (e.type >= g_event_base && e.type < g_event_base + kEventCount) {
        switch (e.type - g_event_base) {
            case kApplyDisplay:
                ApplyNow();
                break;
            case kOpenWindow: {
                OpenNow(e.user.code != 0);
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
        case SDL_EVENT_KEY_DOWN:
            // F3 captures the next frame when XBE_CAPTURE_DIR is set, as on Windows, and dumps the threads
            if (e.key.key == SDLK_F3 && !e.key.repeat) {
                gpu::g_capture = true;
                DumpThreads();
            }
            break;
        case SDL_EVENT_QUIT:
        case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            LOGR("window: quit event 0x%x", unsigned(e.type));
            g_running = false;
            break;
        case SDL_EVENT_GAMEPAD_ADDED:
        case SDL_EVENT_GAMEPAD_REMOVED:
            input::OnDevice(e.type == SDL_EVENT_GAMEPAD_ADDED, e.gdevice.which);
            break;
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
        case SDL_EVENT_WINDOW_DISPLAY_CHANGED:
        case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
        case SDL_EVENT_DISPLAY_CURRENT_MODE_CHANGED:
        case SDL_EVENT_DISPLAY_DESKTOP_MODE_CHANGED:
            UpdateSizes();
            break;
        case SDL_EVENT_WINDOW_FOCUS_LOST:
            // a changed display mode steps aside for the desktop, minimized, and comes back on return
            if (g_mode_changed && !kernel::g_relaunching) {
                SDL_SetWindowFullscreenMode(g_window, nullptr);
                g_mode_changed = false;
                SDL_MinimizeWindow(g_window);
            }
            break;
        case SDL_EVENT_WINDOW_FOCUS_GAINED:
            if (Get().display_mode == DisplayMode::Fullscreen && Get().width && Get().height && !g_mode_changed &&
                !(SDL_GetWindowFlags(g_window) & SDL_WINDOW_HIDDEN)) {
                PushEvent(kApplyDisplay);
            }
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

}  // namespace

std::string Utf8(const std::wstring& text) { return Narrow(text); }

std::filesystem::path ExeDir() {
    static const std::filesystem::path dir = [] {
        std::error_code ec;
        auto self = std::filesystem::read_symlink("/proc/self/exe", ec);
        return ec ? std::filesystem::current_path() : self.parent_path();
    }();
    return dir;
}

int Instance() {
    static const int n = [] {
        auto it = std::find(g_args.begin(), g_args.end(), "--instance");
        return it != g_args.end() && it + 1 != g_args.end() ? std::clamp(std::atoi((it + 1)->c_str()), 0, 8) : 0;
    }();
    return n;
}

void* Window() { return g_window; }

void Quit() {
    SDL_Event e{};
    e.type = SDL_EVENT_QUIT;
    SDL_PushEvent(&e);
}

void Post(std::function<void()> fn) { PushEvent(kRun, 0, new std::function<void()>(std::move(fn))); }

void ShowError(const std::wstring& text) {
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, Narrow(kGame.short_title).c_str(), Narrow(text).c_str(), g_window);
}

bool AskYesNo(const std::wstring& text) {
    const SDL_MessageBoxButtonData buttons[] = {
        {SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT | SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT, 0, "No"},
        {0, 1, "Yes"},
    };
    std::string title = Narrow(kGame.short_title);
    std::string message = Narrow(text);
    SDL_MessageBoxData box{SDL_MESSAGEBOX_WARNING, g_window, title.c_str(), message.c_str(), 2, buttons, nullptr};
    int answer = 0;
    return SDL_ShowMessageBox(&box, &answer) && answer == 1;
}

std::recursive_mutex& UiLock() {
    static std::recursive_mutex m;
    return m;
}

void ApplyDisplay() { PushEvent(kApplyDisplay); }

void SetDisplay(uint32_t display) { g_display = display; }

void OpenGameWindow(bool apply) {
    std::unique_lock lock(g_open_lock);
    uint64_t ticket = ++g_open_asked;
    PushEvent(kOpenWindow, apply ? 1 : 0);
    g_open_cv.wait(lock, [&] { return g_open_done >= ticket || !g_running; });
}

Mode Desktop() {
    uint32_t w = g_desktop_w;
    uint32_t h = g_desktop_h;
    return w && h ? Mode{w, h} : Mode{1920, 1080};
}

Mode WindowPixels() { return {g_pixel_w, g_pixel_h}; }

void UiFrame() {
    using clock = std::chrono::steady_clock;
    static clock::time_point last = clock::now();
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(float(g_pixel_w), float(g_pixel_h));
    io.DisplayFramebufferScale = ImVec2(1, 1);
    auto now = clock::now();
    io.DeltaTime = std::max(std::chrono::duration<float>(now - last).count(), 1e-4f);
    last = now;
    // the first controller plugged in moves through the panel
    input::Pad p;
    for (int port = 0; port < 4 && !p.connected; port++) p = input::Read(port);
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
    key(ImGuiKey_GamepadL3, p.buttons & 0x40);
    key(ImGuiKey_GamepadR3, p.buttons & 0x80);
    key(ImGuiKey_GamepadFaceDown, p.analog[0] > kPressed);
    key(ImGuiKey_GamepadFaceRight, p.analog[1] > kPressed);
    key(ImGuiKey_GamepadFaceLeft, p.analog[2] > kPressed);
    key(ImGuiKey_GamepadFaceUp, p.analog[3] > kPressed);
    key(ImGuiKey_GamepadR1, p.analog[4] > kPressed);
    key(ImGuiKey_GamepadL1, p.analog[5] > kPressed);
    key(ImGuiKey_GamepadL2, p.analog[6] > kPressed);
    key(ImGuiKey_GamepadR2, p.analog[7] > kPressed);
    // the left stick past a dead zone, as the D-pad's analog
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

}  // namespace app

int main(int argc, char** argv) {
    app::g_args.assign(argv + 1, argv + argc);
    app::InstallCrashHandler();
    // one log a session: a relaunch into another image carries on its predecessor's
    bool relaunched = app::HasArg("--relaunched");
    std::string log = app::kGame.log;
    if (app::Instance() > 1) log += "-" + std::to_string(app::Instance());
    logx::Init((app::ExeDir() / "logs").string(), log, relaunched);

    std::string title = app::Narrow(app::kGame.title);
    SDL_SetAppMetadata(title.c_str(), nullptr, nullptr);
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        LOGE("SDL video failed: %s", SDL_GetError());
        app::ShowError(L"No display to open the game window on, see the log in the logs folder.");
        return 1;
    }
    app::g_event_base = SDL_RegisterEvents(app::kEventCount);
    app::UpdateSizes();

    app::Load();
    if (app::kGame.init) app::kGame.init();
    if (relaunched && !logx::kVerbose) LOGR("Relaunched: %s", IMAGE_XBE);
    else app::LogSystem();

    app::GameFiles game;
    if (!app::FindGame(game)) return 1;
    if (!app::PrepareGameCode(game)) return 1;

    bool splash = !relaunched && app::OpenSplash();
    bool handover = app::StartHandover();

    app::g_window = SDL_CreateWindow(title.c_str(), 1280, 720,
                                     SDL_WINDOW_VULKAN | SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_HIDDEN |
                                         SDL_WINDOW_RESIZABLE);
    if (!app::g_window) {
        LOGE("SDL window failed: %s", SDL_GetError());
        app::ShowError(L"The game window could not be opened, see the log in the logs folder.");
        return 1;
    }
    app::UpdateSizes();
    // behind a splash the window stays hidden until the game's first frame; handed over, it is set up hidden
    if (handover) {
        app::ApplyNow();
    } else if (!splash) {
        app::OpenNow(true);
    }
    SDL_StartTextInput(app::g_window);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.BackendPlatformName = "sdl3 (host)";
    // the game sets the cursor, the panel included
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad |
                      ImGuiConfigFlags_NoMouseCursorChange;

    input::Init();
    if (app::kGame.window) app::kGame.window(app::g_window);
    std::string hdd = app::Instance() > 1 ? "hdd" + std::to_string(app::Instance()) : "hdd";
    if (app::Instance()) SDL_SetWindowTitle(app::g_window, (title + " [" + std::to_string(app::Instance()) + "]").c_str());
    if (!kernel::Boot(std::move(game.xbe), IMAGE_XBE, game.dir.string(), (app::ExeDir() / hdd).string())) {
        app::ShowError(L"The game failed to start, see the log in the logs folder.");
        return 1;
    }

    while (app::g_running) {
        SDL_Event e;
        if (!SDL_WaitEventTimeout(&e, 100)) continue;
        app::Handle(e);
        while (app::g_running && SDL_PollEvent(&e)) app::Handle(e);
    }
    {
        // wakes a thread still waiting for the window to open
        std::lock_guard lock(app::g_open_lock);
        app::g_open_cv.notify_all();
    }
    if (app::g_mode_changed) SDL_SetWindowFullscreenMode(app::g_window, nullptr);
    logx::Flush();
    kernel::Shutdown(0);
}
