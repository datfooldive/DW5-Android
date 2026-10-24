// the mixer's stereo output through SDL's audio stream
#include <SDL3/SDL.h>

#include <vector>

#include "audio/audio.h"
#include "core/log.h"

namespace audio {

namespace {

constexpr uint32_t kOutChannels = 2;
constexpr uint32_t kFrames = kRate / 100;  // the mixer renders 10 ms at a time

struct Output {
    SDL_AudioStream* stream = nullptr;
    std::function<void(float*, uint32_t)> render;
    std::vector<float> mix = std::vector<float>(kFrames * kOutChannels);
};
Output g_out;

// SDL asks for more: the mixer renders it 10 ms at a time
void SDLCALL Feed(void* user, SDL_AudioStream* stream, int additional, int) {
    Output& o = *static_cast<Output*>(user);
    while (additional > 0) {
        o.render(o.mix.data(), kFrames);
        int bytes = int(kFrames * kOutChannels * sizeof(float));
        SDL_PutAudioStreamData(stream, o.mix.data(), bytes);
        additional -= bytes;
    }
}

}  // namespace

void StartOutput(std::function<void(float* out, uint32_t frames)> render) {
    g_out.render = std::move(render);
    if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
        LOGE("audio: SDL audio unavailable, no sound: %s", SDL_GetError());
        return;
    }
    SDL_AudioSpec spec{SDL_AUDIO_F32, int(kOutChannels), int(kRate)};
    g_out.stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, Feed, &g_out);
    if (!g_out.stream) {
        LOGE("audio: no audio device, no sound: %s", SDL_GetError());
        return;
    }
    SDL_ResumeAudioStreamDevice(g_out.stream);
    LOGR("audio: SDL3 %u Hz stereo", kRate);
}

}  // namespace audio
