// Hooks for default.xbe. A strong definition replaces the recompiled function at that address, for example:
//
// #include "xdk/hle.h"
//
// X86_EXTERN_FUNC(__imp__sub_00012345);
// X86_FUNC(sub_00012345) {
//     __imp__sub_00012345(ctx, base);  // the original
// }
#include <cmath>

#include "common/options.h"
#include "core/log.h"
#include "kernel/memory.h"
#include "xdk/hle.h"

using namespace kernel;

// Ultrawide. The game projects for a 4:3 frame (width over height 640 / 480) whatever the TV, which a 16:9 one
// stretches. With ultrawide on, the frame is stretched game::g_wide times wider than on a 16:9 TV, to fill the screen,
// so the cameras' aspect is raised as much over 16:9's: the field of view grows across, nothing is stretched, and the
// culling is widened to match.
namespace {

int g_widen_depth;  // inside a camera whose projections are widened

// the screen's width over height divided by the 4:3 the game projects for, 1 off
float Widen() {
    float wide = game::g_wide;
    return wide > 1 ? 16.0f / 9.0f * wide * 0.75f : 1.0f;
}

}  // namespace

// D3DXMatrixPerspectiveFovLH(out ecx, fovy, aspect, near, far), stdcall: a widened camera's projections take the
// screen's aspect
X86_EXTERN_FUNC(__imp__sub_001DF9FC);
X86_FUNC(sub_001DF9FC) {
    float m = Widen();
    if (g_widen_depth && m > 1) G<float>(ctx.esp + 8) *= m;
    __imp__sub_001DF9FC(ctx, base);
}

// The battle's camera (0x18D8A0, its index in eax, 4 the event one): the state at 0x47FCB0 + index * 0x100, the
// horizontal field of view in degrees at +0x98 plus +0x9C, the focal length in pixels for a 640 wide frame at +0xBC,
// and the half angles the culling keeps, across at +0xB4 and up at +0xB8. Its three projections (0x5AAA70, 0x5AA770
// and 0x5AA5F0, 0x40 each per index) are widened, then the culling's angle across.
X86_EXTERN_FUNC(__imp__sub_0018D8A0);
X86_FUNC(sub_0018D8A0) {
    uint32_t state = 0x47FCB0 + (ctx.eax << 8);
    float m = Widen();
    g_widen_depth++;
    __imp__sub_0018D8A0(ctx, base);
    g_widen_depth--;
    if (m > 1) {
        float& across = G<float>(state + 0xB4);
        across = std::atan(std::tan(across) * m);
    }
}

// A camera's matrices (0x198110, the camera in eax): eye at +0x0, target at +0x10, the vertical field of view at +0x24,
// near +0x28, far +0x2C, width over height +0x30, +0x34 set for an orthographic one (its bounds from +0x38). It builds
// the projection at +0x80 with D3DXMatrixPerspectiveFovLH (0x1DF9FC), view times projection at +0xC0.
X86_EXTERN_FUNC(__imp__sub_00198110);
X86_FUNC(sub_00198110) {
    g_widen_depth++;
    __imp__sub_00198110(ctx, base);
    g_widen_depth--;
}

// The library's kick and wait for idle (0x243EE7, called from the texture upload at 0x243C3F): it kicks the
// pushbuffer through 0x244620 and spins until the NV2A reports the channel idle at miniport + 0x100 bit 24,
// which no host code raises because the drawing is the host's. Nothing is left to wait for, and the value it
// hands back is the completion the caller asked for, zero.
X86_FUNC(sub_00243EE7) {
    uint32_t out = R32(ctx.esp + 4);
    if (out) W32(out, 0);
}
