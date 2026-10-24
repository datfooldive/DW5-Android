#pragma once
#include <cstdint>

// Ambient occlusion reaches the renderer as a clear with a flag no D3DCLEAR value uses, so the render thread keeps it
// in order with the draws. Its four rects carry the sixteen floats Occlusion sends
namespace gpu {
constexpr uint32_t kClearOcclusion = 0x100;
constexpr uint32_t kOcclusionRects = 4;
}  // namespace gpu

namespace d3d {
// m: the camera to screen matrix, row vectors, as SetTransform takes it; gtao: GTAO, else SSAO
void Occlusion(const float* m, bool gtao);
}  // namespace d3d
