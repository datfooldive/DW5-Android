// The boot splash: GameInfo::splash .bmp or .png beside the exe, in a borderless window. The game boots behind it with
// its window hidden and waits at its first frame until the splash has been up kMinimum, since this boot takes a
// fraction of a second. No file, or a relaunch: no splash.
#include <atomic>
#include <chrono>
#include <thread>

#include <SDL3/SDL.h>

#include "app/app.h"
#include "core/log.h"

namespace app {

namespace {

constexpr auto kMinimum = std::chrono::seconds(1);

SDL_Window* g_splash = nullptr;
std::atomic<bool> g_waiting{false};
std::chrono::steady_clock::time_point g_until;

// the picture premultiplied, as a transparent window's surface takes it; a picture without alpha comes out opaque
SDL_Surface* Read(const std::filesystem::path& path) {
    SDL_Surface* loaded = SDL_LoadSurface(path.c_str());
    if (!loaded) {
        LOGW("splash: %s could not be read (%s)", path.c_str(), SDL_GetError());
        return nullptr;
    }
    SDL_Surface* argb = SDL_ConvertSurface(loaded, SDL_PIXELFORMAT_ARGB8888);
    SDL_DestroySurface(loaded);
    if (argb && !SDL_PremultiplySurfaceAlpha(argb, false)) LOGW("splash: no premultiplied alpha (%s)", SDL_GetError());
    return argb;
}

}  // namespace

bool OpenSplash() {
    std::filesystem::path path;
    SDL_Surface* image = nullptr;
    for (const wchar_t* ext : {L".bmp", L".png"}) {
        std::error_code ec;
        path = ExeDir() / (Utf8(kGame.splash) + Utf8(ext));
        if (std::filesystem::exists(path, ec) && (image = Read(path))) break;
    }
    if (!image) return false;

    g_splash = SDL_CreateWindow(Utf8(kGame.title).c_str(), image->w, image->h,
                                SDL_WINDOW_BORDERLESS | SDL_WINDOW_TRANSPARENT | SDL_WINDOW_UTILITY |
                                    SDL_WINDOW_HIDDEN);
    if (!g_splash) {
        SDL_DestroySurface(image);
        return false;
    }
    SDL_SetWindowPosition(g_splash, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    SDL_ShowWindow(g_splash);
    if (SDL_Surface* surface = SDL_GetWindowSurface(g_splash)) {
        SDL_SetSurfaceBlendMode(image, SDL_BLENDMODE_NONE);
        SDL_BlitSurface(image, nullptr, surface, nullptr);
        SDL_UpdateWindowSurface(g_splash);
    }
    LOGI("splash: %s, %dx%d", path.c_str(), image->w, image->h);
    SDL_DestroySurface(image);
    g_until = std::chrono::steady_clock::now() + kMinimum;
    g_waiting = true;
    return true;
}

void SplashFirstFrame() {
    if (!g_waiting.exchange(false)) return;
    std::this_thread::sleep_until(g_until);
    OpenGameWindow();
}

void CloseSplash() {
    if (g_splash) SDL_DestroyWindow(g_splash);
    g_splash = nullptr;
}

}  // namespace app
