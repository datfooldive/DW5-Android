// SIGUSR1 asks for a frame capture as F3 does, for a capture taken from a script
#include <csignal>

#include "gpu/backend.h"

namespace game {

namespace {
void OnCaptureSignal(int) { gpu::g_capture = true; }
}  // namespace

void InstallCaptureSignal() { std::signal(SIGUSR1, OnCaptureSignal); }

}  // namespace game
