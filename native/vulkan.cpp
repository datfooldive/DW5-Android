// The title's Direct3D 8 calls drawn with Vulkan: d3d12.cpp's renderer on Linux, the same passes, targets and
// read-backs, its HLSL compiled to SPIR-V at run time by shaderc
#include <algorithm>
#include <bit>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
// the loader's libvulkan.so on Android exports Vulkan 1.1: VMA is told so, and the later entry points below are looked
// up on the device at run time
#define VMA_VULKAN_VERSION 1001000
#include <vk_mem_alloc.h>

// entry points past Vulkan 1.1, which the Android loader's stub does not export (the NDK's libvulkan.so for API 29
// stops at 1.1): looked up on the device at run time, by their core names on a 1.3 device and by their KHR / EXT names
// on an older one that has the extensions
PFN_vkCmdBeginRenderingKHR p_vkCmdBeginRendering = nullptr;
PFN_vkCmdEndRenderingKHR p_vkCmdEndRendering = nullptr;
PFN_vkCmdSetPrimitiveTopologyEXT p_vkCmdSetPrimitiveTopology = nullptr;
PFN_vkCmdBindVertexBuffers2EXT p_vkCmdBindVertexBuffers2 = nullptr;
PFN_vkGetSemaphoreCounterValueKHR p_vkGetSemaphoreCounterValue = nullptr;
PFN_vkWaitSemaphoresKHR p_vkWaitSemaphores = nullptr;
#define vkCmdBeginRendering p_vkCmdBeginRendering
#define vkCmdEndRendering p_vkCmdEndRendering
#define vkCmdSetPrimitiveTopology p_vkCmdSetPrimitiveTopology
#define vkCmdBindVertexBuffers2 p_vkCmdBindVertexBuffers2
#define vkGetSemaphoreCounterValue p_vkGetSemaphoreCounterValue
#define vkWaitSemaphores p_vkWaitSemaphores

#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>

#include "app/app.h"
#include "common/options.h"
#include "core/diag.h"
#include "core/log.h"
#include "d3d/gpu_frame.h"
#include "d3d/occlusion.h"
#include "d3d/smaa_shader.h"
#include "d3d/smaa_textures.h"
#include "gpu/backend.h"
#include "gpu/dxgi_format.h"
#include "gpu/shaders.h"
#include "gpu/texture.h"
#include "gpu/vertex.h"
#include "host.h"
#include "kernel/kernel.h"
#include "vk_shader.h"
#include "vk_ui.h"

namespace gpu {

namespace {

constexpr int kFrames = 3;
// the vertex, index and constant data of a frame; a phone's frame is a few megabytes, and the ring is pinned memory
constexpr uint64_t kUploadSize = 32ull << 20;
// pipelines made before the cache is written to disk again
constexpr uint32_t kPipeCacheSave = 64;
// b0 and b1 are dynamic uniform buffers of this range, bound at each draw's offsets: the larger of the vertex
// constants and DrawConstants, every constant upload reserving all of it
constexpr uint64_t kUboRange = (std::max<uint64_t>(sizeof(gpu::State::vsc), sizeof(gpu::DrawConstants)) + 255) & ~255ull;
constexpr VkFormat kBgra = VK_FORMAT_B8G8R8A8_UNORM;
// HDR10 output: back buffers in PQ over BT.2020; SDR white and peak in nits, XBE_HDR_PAPER_NITS and
// XBE_HDR_PEAK_NITS when set, as Linux has no system setting to read them from
constexpr VkFormat kHdr10Format = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
constexpr VkFormat kOutFrameFormat = VK_FORMAT_R16G16B16A16_UNORM;  // the frame before PQ
constexpr int kOutFrame = 0;
constexpr int kOutPanel = 1;
constexpr int kOutTargets = 2;
constexpr float kDefaultPaperNits = 200.0f;
constexpr float kDefaultPeakNits = 1000.0f;
constexpr float kMaxPeakNits = 4000.0f;  // brighter is a display without calibration reporting PQ's 10000
// the descriptor set every pipeline shares: b0 and b1, t0 to t3, s0 to s3, as the HLSL registers number them
constexpr uint32_t kBindingCb = 0;
constexpr uint32_t kBindingTex = 2;
constexpr uint32_t kBindingSampler = 6;
constexpr uint32_t kBindings = 10;

// perf diagnostic: where a frame's time goes, logged once a second with diag::Perf
using PerfClock = std::chrono::steady_clock;
double PerfMs(PerfClock::time_point t) {
    return std::chrono::duration<double, std::milli>(PerfClock::now() - t).count();
}
struct Perf {
    // milliseconds
    double draw = 0;
    double assemble = 0;
    double bind = 0;
    double sig = 0;
    double texup = 0;
    double compile = 0;
    double clear = 0;
    double copy = 0;
    double present = 0;
    double wait = 0;
    double swap = 0;
    double sync = 0;
    // counts and bytes
    uint64_t draws = 0;
    uint64_t verts = 0;
    uint64_t upload = 0;
    uint64_t overflow = 0;
    uint64_t sigs = 0;
    uint64_t sig_bytes = 0;
    uint64_t texups = 0;
    uint64_t texup_bytes = 0;
    uint64_t newtex = 0;
    uint64_t targets = 0;
    uint64_t rtups = 0;
    uint64_t shrinks = 0;
    uint64_t syncs = 0;
    uint64_t psos = 0;
    uint64_t shaders = 0;
    uint64_t raw = 0;
    uint64_t lates = 0;
};
Perf g_perf;
struct PerfScope {
    double& acc;
    PerfClock::time_point t = PerfClock::now();
    ~PerfScope() { acc += PerfMs(t); }
};

// F3 captures only with diag::CaptureDir set
bool CaptureOn() {
    if (diag::CaptureDir().empty()) LOG_FIRST(1, "capture: set XBE_CAPTURE_DIR");
    return !diag::CaptureDir().empty();
}

const char* kBlitHlsl = R"(
cbuffer B : register(b1) { float4 uvrect; float4 opts; };
Texture2D img : register(t0);
Texture2D lut : register(t1);
SamplerState ls : register(s0);
SamplerState ps : register(s1);
struct O { float4 p : SV_Position; float2 uv : TEXCOORD0; };
O vs(uint id : SV_VertexID) {
    O o;
    float2 t = float2((id << 1) & 2, id & 2);
    o.p = float4(t * float2(2, -2) + float2(-1, 1), 0, 1);
    o.uv = uvrect.xy + t * uvrect.zw;
    return o;
}
float4 psmain(O i) : SV_Target {
    float4 c = 0;
    int n = max((int)opts.z, 1);
    float2 px = fwidth(i.uv);
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) c += img.SampleLevel(ls, i.uv + px * ((float2(x, y) + 0.5) / n - 0.5), 0);
    c /= n * n;
    c.rgb *= opts.w;
    if (opts.x > 0) c.rgb = float3(lut.Sample(ps, float2(c.r, 0.5)).r, lut.Sample(ps, float2(c.g, 0.5)).g, lut.Sample(ps, float2(c.b, 0.5)).b);
    return float4(c.rgb, opts.y > 0 ? c.a : 1);
}
)";

// a colour clear of some channels: the colour drawn with the others masked off
const char* kClearHlsl = R"(
cbuffer B : register(b1) { float4 colour; float4 unused; };
float4 vs(uint id : SV_VertexID) : SV_Position {
    float2 t = float2((id << 1) & 2, id & 2);
    return float4(t * float2(2, -2) + float2(-1, 1), 0, 1);
}
float4 psmain() : SV_Target { return colour; }
)";

// HDR10 output: the frame with the panel over it, its highlights stretched towards the display's peak
const char* kHdrHlsl = R"(
cbuffer P : register(b1) { float4 p0; float4 p1; };
Texture2D frame : register(t0);
Texture2D ui : register(t1);
struct O { float4 p : SV_Position; float2 uv : TEXCOORD0; };
O vs(uint id : SV_VertexID) {
    O o;
    float2 t = float2((id << 1) & 2, id & 2);
    o.p = float4(t * float2(2, -2) + float2(-1, 1), 0, 1);
    o.uv = t;
    return o;
}
static const float3 kLum = float3(0.2126, 0.7152, 0.0722);
static const float3x3 k709To2020 = {0.6274, 0.3293, 0.0433, 0.0691, 0.9195, 0.0114, 0.0164, 0.0880, 0.8956};
// gamma 2.2 as SDR monitors show it; sRGB's linear toe lifts the darks
float3 to_linear(float3 c) {
    return pow(c, 2.2);
}
// inverse Reinhard on luminance so hues keep: the brighter, the more it gains, white reaches most
float3 expand(float3 c, float most) {
    return c / (1 - dot(c, kLum) * (1 - 1 / most));
}
float3 pq(float3 nits) {
    float3 y = pow(saturate(nits / 10000), 0.1593017578125);
    return pow((0.8359375 + 18.8515625 * y) / (1 + 18.6875 * y), 78.84375);
}
// p0: SDR white nits, peak nits
float4 hdr10(O i) : SV_Target {
    int3 px = int3(i.p.xy, 0);
    float3 f = min(expand(to_linear(frame.Load(px).rgb), max(p0.y / p0.x, 1)) * p0.x, p0.y);
    float4 u = ui.Load(px);
    float3 panel = to_linear(u.a > 0 ? u.rgb / u.a : 0) * p0.x;
    return float4(pq(mul(k709To2020, lerp(f, panel, u.a))), 1);
}
)";

const char* kSmaaConfig = R"(
cbuffer P : register(b1) { float4 metrics; };
#define SMAA_RT_METRICS metrics
#define SMAA_CUSTOM_SL
#define SMAA_THRESHOLD 0.05
#define SMAA_MAX_SEARCH_STEPS 16
#define SMAA_MAX_SEARCH_STEPS_DIAG 8
#define SMAA_CORNER_ROUNDING 25
SamplerState LinearSampler : register(s0);
SamplerState PointSampler : register(s1);
#define SMAATexture2D(tex) Texture2D tex
#define SMAATexturePass2D(tex) tex
#define SMAASampleLevelZero(tex, coord) tex.SampleLevel(LinearSampler, coord, 0)
#define SMAASampleLevelZeroPoint(tex, coord) tex.SampleLevel(PointSampler, coord, 0)
#define SMAASampleLevelZeroOffset(tex, coord, offset) tex.SampleLevel(LinearSampler, coord, 0, offset)
#define SMAASample(tex, coord) tex.Sample(LinearSampler, coord)
#define SMAASamplePoint(tex, coord) tex.Sample(PointSampler, coord)
#define SMAASampleOffset(tex, coord, offset) tex.Sample(LinearSampler, coord, offset)
#define SMAAGather(tex, coord) tex.Gather(LinearSampler, coord, 0)
#define SMAA_FLATTEN [flatten]
#define SMAA_BRANCH [branch]
)";

// after kSmaaHlsl: edges from the frame's luma, blend weights from edges, area and search, then the frame blended
const char* kSmaaPasses = R"(
Texture2D tex0 : register(t0);
Texture2D tex1 : register(t1);
Texture2D tex2 : register(t2);
struct SmaaEdgeV { float4 p : SV_Position; float2 uv : TEXCOORD0; float4 o[3] : TEXCOORD1; };
struct SmaaWeightV { float4 p : SV_Position; float2 uv : TEXCOORD0; float2 pix : TEXCOORD1; float4 o[3] : TEXCOORD2; };
struct SmaaBlendV { float4 p : SV_Position; float2 uv : TEXCOORD0; float4 o : TEXCOORD1; };
float4 corner(uint id, out float2 uv) {
    uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
SmaaEdgeV vs_edges(uint id : SV_VertexID) {
    SmaaEdgeV o;
    o.p = corner(id, o.uv);
    SMAAEdgeDetectionVS(o.uv, o.o);
    return o;
}
SmaaWeightV vs_weights(uint id : SV_VertexID) {
    SmaaWeightV o;
    o.p = corner(id, o.uv);
    SMAABlendingWeightCalculationVS(o.uv, o.pix, o.o);
    return o;
}
SmaaBlendV vs_blend(uint id : SV_VertexID) {
    SmaaBlendV o;
    o.p = corner(id, o.uv);
    SMAANeighborhoodBlendingVS(o.uv, o.o);
    return o;
}
float4 smaa_edges(SmaaEdgeV i) : SV_Target { return float4(SMAALumaEdgeDetectionPS(i.uv, i.o, tex0), 0, 0); }
float4 smaa_weights(SmaaWeightV i) : SV_Target { return SMAABlendingWeightCalculationPS(i.uv, i.pix, i.o, tex0, tex1, tex2, 0); }
float4 smaa_blend(SmaaBlendV i) : SV_Target { return SMAANeighborhoodBlendingPS(i.uv, i.o, tex0, tex1); }
)";
constexpr int kSmaaTargets = 3;

const char* kDepthHlsl = R"(
#ifdef MS
Texture2DMS<float> img : register(t0);
#else
Texture2D<float> img : register(t0);
#endif
SamplerState ps : register(s1);
struct O { float4 p : SV_Position; float2 uv : TEXCOORD0; };
O vs(uint id : SV_VertexID) {
    O o;
    o.uv = float2((id << 1) & 2, id & 2);
    o.p = float4(o.uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return o;
}
float psmain(O i) : SV_Depth {
#ifdef MS
    uint w, h, n;
    img.GetDimensions(w, h, n);
    return img.Load(int2(i.uv * float2(w, h)), 0);
#else
    return img.SampleLevel(ps, i.uv, 0);
#endif
}
)";

// from the scene's depth: occlusion pass, then a blur across and one down that multiplies it into the colour.
// The sky is far: a pixel's samples there span less than one pixel and it stays lit
const char* kAoHlsl = R"(
cbuffer A : register(b1) { float4 proj; float4 zp; float4 vp; float4 opt; float4 fog; float4 fogcolor; };
// stencil bit 7: a blended or alpha tested surface wrote the depth last, it counts as the far plane like the sky
#ifdef MS
Texture2DMS<float> depth : register(t0);
Texture2DMS<uint2> stencil : register(t2);
float Raw(int2 p) { return stencil.Load(p, 0).g & 0x80 ? 1 : depth.Load(p, 0); }
#else
Texture2D<float> depth : register(t0);
Texture2D<uint2> stencil : register(t2);
float Raw(int2 p) { return stencil.Load(int3(p, 0)).g & 0x80 ? 1 : depth.Load(int3(p, 0)); }
#endif
Texture2D<float2> ao : register(t1);
float4 vs(uint id : SV_VertexID) : SV_Position {
    float2 t = float2((id << 1) & 2, id & 2);
    return float4(t * float2(2, -2) + float2(-1, 1), 0, 1);
}
// view space position of a pixel. proj: 1 / scale xy, centre xy; zp: z = x / (depth - y); vp: viewport in pixels
float3 View(float2 px) {
    int2 p = clamp(int2(px), int2(vp.xy), int2(vp.xy + vp.zw) - 1);
    float z = zp.x / (Raw(p) - zp.y);
    float2 ndc = (p + 0.5 - vp.xy) / vp.zw * float2(2, -2) + float2(-1, 1);
    return float3((ndc - proj.zw) * proj.xy * z, z);
}
// normal from the nearer neighbour each way: no bend over depth edges
float3 Normal(float2 pos, float3 c) {
    float3 l = View(pos - float2(1, 0));
    float3 r = View(pos + float2(1, 0));
    float3 u = View(pos - float2(0, 1));
    float3 d = View(pos + float2(0, 1));
    float3 dx = abs(r.z - c.z) < abs(c.z - l.z) ? r - c : c - l;
    float3 dy = abs(d.z - c.z) < abs(c.z - u.z) ? d - c : c - u;
    return normalize(cross(dx, dy));
}
static const float kDither[16] = {0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5};
// opt: world radius, most pixels it may span; out: occlusion, view z
float2 occlusion(float4 pos : SV_Position) : SV_Target {
    float3 c = View(pos.xy);
    float2 rad = opt.x * 0.5 * vp.zw / (proj.xy * c.z);
    float span = max(rad.x, rad.y);
    if (span < 1) return float2(1, c.z);
    float3 n = Normal(pos.xy, c);
    float k = min(1, opt.y / span);
    float R = opt.x * k;
    int2 cell = int2(pos.xy) & 3;
    float turn = kDither[cell.y * 4 + cell.x] * (6.2831853 / 16);
    float a = 0;
    for (int i = 0; i < 16; i++) {
        float f = (i + 0.5) / 16;
        float t = i * 2.3999632 + turn;
        float3 v = (View(pos.xy + float2(cos(t), sin(t)) * rad * (k * f)) - c) / R;
        float vv = dot(v, v);
        float w = max(1 - vv, 0);
        a += w * w * w * max((dot(v, n) - 0.05) / (vv + 0.01), 0);
    }
    return float2(max(1 - a * 0.25, 0), c.z);
}
static const float kHalfPi = 1.5707963;
// GTAO (Jimenez 2016) at XeGTAO's high preset: 3 slices, 3 steps a side, squared step spread, falloff over the outer
// 61.5% of the radius, power 2.2. Per slice the highest horizon each side, the visible arc between integrated against
// the normal projected into the slice
float2 gtao(float4 pos : SV_Position) : SV_Target {
    float3 c = View(pos.xy);
    float2 rad = opt.x * 0.5 * vp.zw / (proj.xy * c.z);
    float span = max(rad.x, rad.y);
    if (span < 1) return float2(1, c.z);
    float3 n = Normal(pos.xy, c);
    float3 v = normalize(-c);
    float k = min(1, opt.y / span);
    float R = opt.x * k;
    int2 cell = int2(pos.xy) & 3;
    float slice_noise = kDither[cell.y * 4 + cell.x] / 16;
    float step_noise = kDither[cell.x * 4 + cell.y] / 16;
    float vis = 0;
    for (int s = 0; s < 3; s++) {
        float phi = (s + slice_noise) * (kHalfPi * 2 / 3);
        float3 dir = float3(cos(phi), sin(phi), 0);
        // in pixels; screen y runs down, view y up
        float2 omega = float2(dir.x, -dir.y) * rad * k;
        float3 ortho = dir - dot(dir, v) * v;
        float3 axis = normalize(cross(ortho, v));
        float3 pn = n - axis * dot(n, axis);
        float pn_len = max(length(pn), 1e-4);
        float cos_n = saturate(dot(pn, v) / pn_len);
        float an = sign(dot(ortho, pn)) * acos(cos_n);
        // x: the +omega side, y: the -omega side; lowest is flat along the slice's tangent
        float2 low = float2(cos(an + kHalfPi), cos(an - kHalfPi));
        float2 hc = low;
        for (int j = 0; j < 3; j++) {
            float f = (j + step_noise) / 3;
            // at least 1.3 pixels out, off the centre pixel
            float2 off = omega * (f * f + 1.3 / length(omega));
            float3 d0 = View(pos.xy + off) - c;
            float3 d1 = View(pos.xy - off) - c;
            float2 len = max(float2(length(d0), length(d1)), 1e-4);
            float2 w = saturate((R - len) / (R * 0.615));
            float2 sc = float2(dot(d0, v), dot(d1, v)) / len;
            hc = max(hc, lerp(low, sc, w));
        }
        float h0 = an + clamp(-acos(hc.y) - an, -kHalfPi, kHalfPi);
        float h1 = an + clamp(acos(hc.x) - an, -kHalfPi, kHalfPi);
        float sin_n = sin(an);
        float arc0 = (cos_n + 2 * h0 * sin_n - cos(2 * h0 - an)) / 4;
        float arc1 = (cos_n + 2 * h1 * sin_n - cos(2 * h1 - an)) / 4;
        vis += pn_len * (arc0 + arc1);
    }
    return float2(pow(saturate(vis / 3), 2.2), c.z);
}
// opt.zw: step; 5 taps average the 4 pixel dither, depth edges left out
float Smooth(float2 pos, out float z) {
    int2 p = int2(pos);
    int2 lo = int2(vp.xy);
    int2 hi = int2(vp.xy + vp.zw) - 1;
    z = ao.Load(int3(p, 0)).y;
    float sum = 0;
    float n = 0;
    for (int i = -2; i <= 2; i++) {
        float2 s = ao.Load(int3(clamp(p + i * int2(opt.zw), lo, hi), 0));
        float w = (abs(i) == 2 ? 0.5 : 1) * saturate(2 - abs(s.y - z) / (z * 0.02));
        sum += s.x * w;
        n += w;
    }
    return sum / n;
}
float2 occlusion_blur(float4 pos : SV_Position) : SV_Target {
    float z;
    float a = Smooth(pos.xy, z);
    return float2(a, z);
}
// the fog is already drawn: only the scene share under it is occluded, blended as colour * a + rgb
float4 occlusion_apply(float4 pos : SV_Position) : SV_Target {
    float z;
    float a = Smooth(pos.xy, z);
    float fogged = min(saturate(fog.x * z + fog.y), fog.z);
    return float4(fogcolor.rgb * fogged * (1 - a), a);
}
)";
constexpr int kAoTargets = 2;
constexpr int kAoConstants = 24;  // kAoHlsl's b1
constexpr VkFormat kAoFormat = VK_FORMAT_R16G16_SFLOAT;
constexpr float kAoRadius = 0.8f;    // world units, the engine's are metres
constexpr float kAoMaxSpan = 0.15f;  // of the view's height, the most one pixel's samples may cover

// a rectangle in target pixels
struct Rect {
    int32_t left;
    int32_t top;
    int32_t right;
    int32_t bottom;
};

float Bits(uint32_t v) {
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}

void Color(uint32_t argb, float* o) {
    o[0] = ((argb >> 16) & 255) / 255.0f;
    o[1] = ((argb >> 8) & 255) / 255.0f;
    o[2] = (argb & 255) / 255.0f;
    o[3] = (argb >> 24) / 255.0f;
}

VkBlendFactor Blend(uint32_t v, bool alpha) {
    switch (v) {
        case 0: return VK_BLEND_FACTOR_ZERO;
        case 1: return VK_BLEND_FACTOR_ONE;
        case 0x300: return alpha ? VK_BLEND_FACTOR_SRC_ALPHA : VK_BLEND_FACTOR_SRC_COLOR;
        case 0x301: return alpha ? VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA : VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case 0x302: return VK_BLEND_FACTOR_SRC_ALPHA;
        case 0x303: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case 0x304: return VK_BLEND_FACTOR_DST_ALPHA;
        case 0x305: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case 0x306: return alpha ? VK_BLEND_FACTOR_DST_ALPHA : VK_BLEND_FACTOR_DST_COLOR;
        case 0x307: return alpha ? VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA : VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case 0x308: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        case 0x8001: case 0x8003: return VK_BLEND_FACTOR_CONSTANT_COLOR;
        case 0x8002: case 0x8004: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
        default: return VK_BLEND_FACTOR_ONE;
    }
}

VkBlendOp BlendOp(uint32_t v) {
    switch (v) {
        case 0x800A: return VK_BLEND_OP_SUBTRACT;
        case 0x800B: return VK_BLEND_OP_REVERSE_SUBTRACT;
        case 0x8007: return VK_BLEND_OP_MIN;
        case 0x8008: return VK_BLEND_OP_MAX;
        default: return VK_BLEND_OP_ADD;
    }
}

// D3D's comparison numbering, NEVER 1 to ALWAYS 8, which the shaders' constants take less one
uint32_t Cmp(uint32_t v) {
    uint32_t n = v >= 0x200 && v <= 0x207 ? v - 0x200 : 7;
    return n + 1;
}

// Vulkan's runs in the same order from 0
VkCompareOp CompareOp(uint32_t v) { return VkCompareOp(Cmp(v) - 1); }

VkStencilOp StencilOp(uint32_t v) {
    switch (v) {
        case 0: return VK_STENCIL_OP_ZERO;
        case 0x1E01: return VK_STENCIL_OP_REPLACE;
        case 0x1E02: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
        case 0x1E03: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
        case 0x150A: return VK_STENCIL_OP_INVERT;
        case 0x8507: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
        case 0x8508: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
        default: return VK_STENCIL_OP_KEEP;
    }
}

// The game's 2D layer (menus, text, HUD) leaves oPos.w at 0: screen space. Clamped to 2^-64 its clip space
// coordinates are near 1e-20 and AMD drops parts of those draws; w = 1 draws the same, as XYZRHW does
std::string VertexProgram(const VertexShader& vs, uint16_t regs, uint64_t decode) {
    const std::string clamp = "    p.w = magclamp(p.w);\n";
    const std::string screen_space = "    p.w = p.w == 0 ? 1 : magclamp(p.w);\n";
    std::string src = VertexProgramHlsl(vs.program, regs, decode);
    size_t at = src.find(clamp);
    if (at != std::string::npos) src.replace(at, clamp.size(), screen_space);
    return src;
}

// A small draw whose every float2 input lies in 0..1: sprites, HUD pieces, text. Upscaled, the far edge's last half
// texel is drawn too and wrap blends in the opposite edge (thin lines on HUD sprites); clamp takes the edge texel.
bool UnitTexcoords(const Assembled& a) {
    constexpr float kLo = -1e-4f;
    constexpr float kHi = 1.0001f;
    constexpr uint32_t kSmallDrawVertices = 4096;
    if (!a.vertices.empty() || !a.vertex_count || a.vertex_count > kSmallDrawVertices) return false;
    bool any = false;
    for (int r = 0; r < 16; r++) {
        const VertexElement& e = a.elements[size_t(r)];
        if (!(a.regs >> r & 1) || e.format != DXGI_FORMAT_R32G32_FLOAT) continue;
        const VertexBuffer& vb = a.buffers[e.buffer];
        if (!vb.src || !vb.stride || uint64_t(a.vertex_count) * vb.stride > vb.bytes) return false;
        for (uint32_t v = 0; v < a.vertex_count; v++) {
            float uv[2];
            std::memcpy(uv, vb.src + size_t(v) * vb.stride + e.offset, sizeof(uv));
            if (!(uv[0] >= kLo && uv[0] <= kHi && uv[1] >= kLo && uv[1] <= kHi)) return false;
        }
        any = true;
    }
    return any;
}

constexpr uint32_t kTaddressWrap = 1;
constexpr uint32_t kTaddressClamp = 3;

// UnitTexcoords only changes stages that wrap
bool AnyStageWraps(const State& s) {
    for (const auto& t : s.tss) {
        if (t[kTssAddressU] == kTaddressWrap || t[kTssAddressV] == kTaddressWrap) return true;
    }
    return false;
}

VkSamplerAddressMode Address(uint32_t v) {
    switch (v) {
        case 2: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case 3: case 5: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case 4: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        default: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }
}

// what a console format is uploaded as; a depth texture read with a comparison sampler needs a depth format
VkFormat HostFormat(Host h, bool depth) {
    switch (h) {
        case Host::BC1: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        case Host::BC2: return VK_FORMAT_BC2_UNORM_BLOCK;
        case Host::BC3: return VK_FORMAT_BC3_UNORM_BLOCK;
        case Host::R32F: return depth ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_R32_SFLOAT;
        default: return kBgra;
    }
}

// the input assembler's vertex formats, named as DXGI ones by vertex.cpp
VkFormat VertexFormat(uint32_t dxgi) {
    switch (dxgi) {
        case DXGI_FORMAT_R32G32B32A32_FLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
        case DXGI_FORMAT_R32G32B32_FLOAT: return VK_FORMAT_R32G32B32_SFLOAT;
        case DXGI_FORMAT_R16G16B16A16_SNORM: return VK_FORMAT_R16G16B16A16_SNORM;
        case DXGI_FORMAT_R16G16B16A16_SINT: return VK_FORMAT_R16G16B16A16_SINT;
        case DXGI_FORMAT_R32G32_FLOAT: return VK_FORMAT_R32G32_SFLOAT;
        case DXGI_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_R16G16_SNORM: return VK_FORMAT_R16G16_SNORM;
        case DXGI_FORMAT_R16G16_SINT: return VK_FORMAT_R16G16_SINT;
        case DXGI_FORMAT_R32_FLOAT: return VK_FORMAT_R32_SFLOAT;
        case DXGI_FORMAT_R32_UINT: return VK_FORMAT_R32_UINT;
        case DXGI_FORMAT_R8G8_UNORM: return VK_FORMAT_R8G8_UNORM;
        case DXGI_FORMAT_R16_SNORM: return VK_FORMAT_R16_SNORM;
        case DXGI_FORMAT_R16_SINT: return VK_FORMAT_R16_SINT;
        case DXGI_FORMAT_R8_UNORM: return VK_FORMAT_R8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
        default: return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
}

bool IsDepthFormat(VkFormat f) {
    return f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_D32_SFLOAT ||
           f == VK_FORMAT_D16_UNORM;
}

bool HasStencil(VkFormat f) { return f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT; }

// bytes of a 4x4 block of the BC formats, 0 for the others
uint32_t BlockBytes(VkFormat f) {
    if (f == VK_FORMAT_BC1_RGBA_UNORM_BLOCK) return 8;
    if (f == VK_FORMAT_BC2_UNORM_BLOCK || f == VK_FORMAT_BC3_UNORM_BLOCK) return 16;
    return 0;
}

// ---- DXT on the CPU, for GPUs without BC textures (most phones), as Black-Android does it

void Rgb565(uint16_t c, uint8_t* bgr) {
    bgr[2] = uint8_t((c >> 11) * 255 / 31);
    bgr[1] = uint8_t(((c >> 5) & 63) * 255 / 63);
    bgr[0] = uint8_t((c & 31) * 255 / 31);
}

// one 4x4 block's colours, BGRA, out 16 x 4 bytes; dxt1 alpha for the 3 colour mode
void ColourBlock(const uint8_t* b, bool dxt1, uint8_t* out) {
    uint16_t c0 = uint16_t(b[0] | b[1] << 8), c1 = uint16_t(b[2] | b[3] << 8);
    uint8_t pal[4][4] = {};
    Rgb565(c0, pal[0]);
    Rgb565(c1, pal[1]);
    for (int i = 0; i < 4; i++) pal[i][3] = 255;
    for (int k = 0; k < 3; k++) {
        if (c0 > c1 || !dxt1) {
            pal[2][k] = uint8_t((2 * pal[0][k] + pal[1][k]) / 3);
            pal[3][k] = uint8_t((pal[0][k] + 2 * pal[1][k]) / 3);
        } else {
            pal[2][k] = uint8_t((pal[0][k] + pal[1][k]) / 2);
            pal[3][k] = 0;
        }
    }
    if (dxt1 && c0 <= c1) pal[3][3] = 0;
    uint32_t bits = uint32_t(b[4] | b[5] << 8 | b[6] << 16 | uint32_t(b[7]) << 24);
    for (int i = 0; i < 16; i++) std::memcpy(out + i * 4, pal[(bits >> (i * 2)) & 3], 4);
}

// a level of DXT1, 3 or 5 blocks (row_pitch bytes a block row, z slices) into w x h BGRA texels a slice
void DecodeDxt(Host host, const std::vector<uint8_t>& in, uint32_t row_pitch, uint32_t w, uint32_t h, uint32_t z,
               std::vector<uint8_t>& out) {
    uint32_t block = host == Host::BC1 ? 8 : 16;
    uint32_t bw = std::max(1u, (w + 3) / 4), bh = std::max(1u, (h + 3) / 4);
    out.assign(size_t(w) * h * z * 4, 0);
    uint8_t px[64];
    for (uint32_t s = 0; s < z; s++)
        for (uint32_t by = 0; by < bh; by++)
            for (uint32_t bx = 0; bx < bw; bx++) {
                size_t at = (size_t(s) * bh + by) * row_pitch + size_t(bx) * block;
                if (at + block > in.size()) continue;
                const uint8_t* b = &in[at];
                ColourBlock(host == Host::BC1 ? b : b + 8, host == Host::BC1, px);
                if (host == Host::BC2) {
                    for (int i = 0; i < 16; i++) px[i * 4 + 3] = uint8_t(((b[i / 2] >> ((i & 1) * 4)) & 15) * 17);
                } else if (host == Host::BC3) {
                    uint8_t a[8] = {b[0], b[1]};
                    if (a[0] > a[1]) {
                        for (int i = 1; i < 7; i++) a[i + 1] = uint8_t(((7 - i) * a[0] + i * a[1]) / 7);
                    } else {
                        for (int i = 1; i < 5; i++) a[i + 1] = uint8_t(((5 - i) * a[0] + i * a[1]) / 5);
                        a[6] = 0;
                        a[7] = 255;
                    }
                    uint64_t bits = 0;
                    for (int i = 0; i < 6; i++) bits |= uint64_t(b[2 + i]) << (8 * i);
                    for (int i = 0; i < 16; i++) px[i * 4 + 3] = a[(bits >> (3 * i)) & 7];
                }
                for (uint32_t y = 0; y < 4 && by * 4 + y < h; y++)
                    for (uint32_t x = 0; x < 4 && bx * 4 + x < w; x++)
                        std::memcpy(&out[((size_t(s) * h + by * 4 + y) * w + bx * 4 + x) * 4], px + (y * 4 + x) * 4, 4);
            }
}

uint32_t TexelBytes(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8_UNORM: return 1;
        case VK_FORMAT_R8G8_UNORM: return 2;
        case VK_FORMAT_R16G16B16A16_UNORM: return 8;
        default: return 4;
    }
}

constexpr uint32_t kFmtD16 = 0x2C;     // D3DFMT_D16
constexpr uint32_t kFmtLinD16 = 0x30;  // D3DFMT_LIN_D16

// TexInfo::shadow of a depth texture
uint8_t ShadowDepthKind(uint32_t format) { return format == kFmtD16 || format == kFmtLinD16 ? 2 : 1; }

// bytes per row of a 32 bit surface in guest memory, rounded to 64
uint32_t GuestPitch(uint32_t width) { return (width * 4 + 63) & ~63u; }

// an image and the views draws, passes and attachments take of it; destroyed once the frame that last used it is
// done (the backend holds it in that frame's garbage)
struct Image {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;  // null for a swap chain image
    VkImageView view = VK_NULL_HANDLE;     // sampled: colour, or a depth format's depth
    VkImageView attach = VK_NULL_HANDLE;   // as an attachment: colour, or depth and stencil
    VkImageView stencil = VK_NULL_HANDLE;  // a depth format's stencil
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    uint32_t w = 1;
    uint32_t h = 1;
    uint32_t depth = 1;
    uint32_t levels = 1;
    uint32_t layers = 1;
    uint32_t samples = 1;
};
using ImageRef = std::shared_ptr<Image>;

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    uint8_t* map = nullptr;
    uint64_t size = 0;
};
using BufferRef = std::shared_ptr<Buffer>;

// a console surface the renderer draws into
struct Target {
    ImageRef res;
    bool depth = false;
    uint32_t gw = 0, gh = 0, w = 0, h = 0;
    uint32_t data = 0, bytes = 0;
    uint64_t epoch = 0, sig = 0;  // last lock seen, guest memory as last matched
    bool drawn = false;           // the GPU wrote it since the last readback
    std::vector<uint8_t> read;    // what Sync left in guest memory
    ImageRef fb;                  // copy read by draws that sample the target they render to
    VkFormat fmt = kBgra;
    float range = 1;  // colour held divided by it
};

// MemSig hashes a surface up to kSigWholeMax bytes whole, a bigger one as evenly spaced samples
constexpr uint32_t kSigWholeMax = 65536;
constexpr uint32_t kSigSamples = 64;
constexpr uint32_t kSigSampleBytes = 64;

uint64_t MemSig(uint32_t data, uint32_t total) {
    const uint8_t* p = static_cast<const uint8_t*>(kernel::HostPtr(data));
    uint64_t sig = Hash(&total, sizeof(total));
    g_perf.sigs++;
    g_perf.sig_bytes += total <= kSigWholeMax ? total : kSigSamples * kSigSampleBytes;
    if (total <= kSigWholeMax) return Hash(p, total, sig);
    for (uint32_t i = 0; i < kSigSamples; i++) {
        const uint8_t* sample = p + uint64_t(total - kSigSampleBytes) * i / (kSigSamples - 1);
        sig = Hash(sample, kSigSampleBytes, sig);
    }
    return sig;
}

// MemSig samples a big surface and misses small writes (XFONT text): after a Sync every byte is compared
bool Changed(const Target& t) {
    if (t.drawn || t.read.empty()) return false;
    uint32_t row = t.gw * 4;
    uint32_t pitch = GuestPitch(t.gw);
    const uint8_t* p = static_cast<const uint8_t*>(kernel::HostPtr(t.data));
    for (uint32_t y = 0; y < t.gh; y++) {
        if (std::memcmp(p + size_t(y) * pitch, &t.read[size_t(y) * row], row)) return true;
    }
    return false;
}

// a lock alone proves nothing, reads lock too: only changed bytes mean the CPU wrote it
bool LatchCpuWrite(Target& t) {
    uint64_t e = DirtyEpoch(t.data);
    if (e <= t.epoch) return false;
    t.epoch = e;
    uint64_t sig = MemSig(t.data, t.bytes);
    if (sig == t.sig && !Changed(t)) return false;
    t.sig = sig;
    return true;
}

uint32_t Ms(const Target& t) { return t.res->samples; }

// A guest edge as a host pixel: the low edge rounds down and the high edge up, so two rects that meet in guest space
// meet in host space too, and no pixel between them is left out. Truncating both edges loses the last row and column
// of every rect at a fractional render scale, which shows as a line or a cut where two rects join
int32_t HostEdge(double v, float sc, uint32_t limit, bool high) {
    double h = double(v) * double(sc);
    int32_t i = int32_t(high ? std::ceil(h) : std::floor(h));
    return std::clamp<int32_t>(i, 0, int32_t(limit));
}

// the viewport in target pixels
Rect ViewRect(const Viewport& v, const Target& t) {
    float sc = float(t.w) / float(t.gw);
    Rect r;
    r.left = HostEdge(int32_t(v.x), sc, t.w, false);
    r.top = HostEdge(int32_t(v.y), sc, t.h, false);
    r.right = HostEdge(int32_t(v.x + v.w), sc, t.w, true);
    r.bottom = HostEdge(int32_t(v.y + v.h), sc, t.h, true);
    return r;
}

// b1 of kAoHlsl: proj, zp with the viewport's depth range, vp, opt (its step written per pass), fog, fog colour
void OcclusionConstants(const Viewport& v, const Rect& r, const float p[4 * kOcclusionRects],
                        float c[kAoConstants]) {
    float zr = v.max_z - v.min_z;
    float w = float(r.right - r.left);
    float h = float(r.bottom - r.top);
    const float out[kAoConstants] = {p[0], p[1], p[2], p[3], p[4] * zr, v.min_z + p[5] * zr, 0, 0,
                                     float(r.left), float(r.top), w, h, kAoRadius, kAoMaxSpan * h, 0, 0,
                                     p[8], p[9], p[10], 0, p[12], p[13], p[14], 0};
    std::memcpy(c, out, sizeof(out));
}


struct Tex {
    ImageRef res;
    TextureDesc d;
    VkFormat fmt = VK_FORMAT_UNDEFINED;
    uint32_t levels = 1;
    uint64_t sig = 0;
    uint64_t frame = 0;
    uint64_t palette = 0;  // hash of the colours a P8 texture was decoded with
    uint32_t epoch = 0;    // g_texture_epoch at the last check
    bool gen_mips = false;  // the title gave one level, the rest are made from it (anisotropic filtering)
    Host decode = Host::None;  // DXT decoded on the CPU into a BGRA image: the GPU has no BC textures
};

struct PsoKey {
    uint64_t vs;
    uint64_t ps;
    uint32_t regs;
    uint32_t topo;
    uint32_t has_rt;
    uint32_t has_ds;
    uint32_t elements[16];  // per register: DXGI format, buffer << 8, offset << 16
    uint32_t blend;
    uint32_t src;
    uint32_t dst;
    uint32_t op;
    uint32_t wmask;
    uint32_t zen;
    uint32_t zwrite;
    uint32_t zfunc;
    uint32_t sten;
    uint32_t sfunc;
    uint32_t smask;
    uint32_t swmask;
    uint32_t sfail;
    uint32_t szfail;
    uint32_t spass;
    uint32_t cull;
    uint32_t ccw;
    uint32_t fill;
    uint32_t unclipped;  // no clipping at the near and far planes: NearDepthHlsl draws
    int32_t zbias;
    float zslope;
    uint32_t rtfmt;
    uint32_t msaa;     // samples of a multisampled target, 0 without
    uint32_t ao_mark;  // the stencil holds ambient occlusion's mark, not the game's state
    // without extended dynamic state: the strip and the vertex strides, which are then the pipeline's (0 with it)
    uint32_t strip;
    uint32_t strides[16];
};

// Blit's settings past the source and its rectangle
struct BlitOptions {
    bool gamma = false;       // through the gamma lookup
    bool alpha = false;       // the source's alpha, else 1
    uint32_t taps = 1;        // n x n box filter per pixel
    float mul = 1;            // colour scale
    VkFormat rtv = kBgra;     // the target's format
    bool over = false;        // premultiplied over the target, which keeps its alpha
    uint32_t samples = 1;     // the target's
};

// Shrink's settings past the source and the size
struct ShrinkOptions {
    uint32_t taps = 1;
    ImageRef into;  // a copy from an earlier call to draw over again
    float mul = 1;
};

// a draw's four texture stages: BindStageTexturesAndTargets fills them, the constants, views and samplers read them
struct DrawTextures {
    TexInfo info[4];
    Image* res[4] = {};
    uint32_t levels[4] = {1, 1, 1, 1};
    bool rt[4] = {};            // stage samples a render target
    float k[4] = {1, 1, 1, 1};  // texels per render scale texel, 1 without an extension
    float wide[4] = {};         // host texels per console texel of a target read into a smaller one, 0 otherwise
    bool blur[4] = {};          // wide for a blur between small targets
    uint32_t gw[4] = {};        // a render target's guest width, 0 for any other texture
    bool gen_mips[4] = {};      // levels the host made: filtered across them whatever the title's mip filter
};

// a draw's shaders: what they are made from, their cache keys, the hashes Compile returned and the modules
struct DrawShaders {
    bool prog = false;  // a vertex program, else the fixed function pipeline
    bool fixed_transform = false;  // fixed function on untransformed vertices: the shader reads matrices and lights
    FixedKey fk;
    uint64_t vkey = 0;
    uint64_t vh = 0;
    uint64_t pkey = 0;
    uint64_t ph = 0;
    bool near_depth = false;  // the pixel shader writes the depth: NearDepthHlsl
    VkShaderModule vs_code = VK_NULL_HANDLE;
    VkShaderModule ps_code = VK_NULL_HANDLE;
};

// the render target's console size in pixels, or the depth buffer's without one
struct GuestExtent {
    uint32_t w;
    uint32_t h;
};

// what a draw or pass reads: b0 and b1, t0 to t3, s0 to s3
struct Bindings {
    VkDescriptorBufferInfo cb[2] = {};
    VkImageView views[4] = {};
    VkImageView stencil = VK_NULL_HANDLE;  // t2 of the occlusion pass: a depth target's stencil
    VkSampler samplers[4] = {};
};

// a full screen pass's pipeline: one colour target or a depth one, how it blends
enum class PassBlend { None, Over, Factor, Occlusion };
struct PassPipe {
    VkShaderModule vs;
    VkShaderModule ps;
    const char* vs_entry = "vs";  // the HLSL entry points, for the log: vk_shader names each SPIR-V one "main"
    const char* ps_entry = "psmain";
    VkFormat color = VK_FORMAT_UNDEFINED;
    VkFormat depth = VK_FORMAT_UNDEFINED;  // DepthPass's, which writes depth alone
    uint32_t samples = 1;
    PassBlend blend = PassBlend::None;
    uint32_t wmask = 0xF;  // VkColorComponentFlags
};


#define VK_CHECK(call)                                                          \
    do {                                                                        \
        VkResult vk_result = (call);                                            \
        if (vk_result != VK_SUCCESS) LOGE("vulkan: %s failed (%d)", #call, int(vk_result)); \
    } while (0)

struct PipelineParts;

class VulkanBackend final : public Backend {
public:
    ~VulkanBackend() override;
    bool Init(uint32_t width, uint32_t height) override;
    void Clear(const State& s, uint32_t count, const uint8_t* rects, uint32_t flags, uint32_t color, float z,
               uint32_t stencil) override;
    void Draw(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices) override;
    void Copy(const Surface& src, const Surface& dst, uint32_t count, uint32_t src_rects, uint32_t dst_points) override;
    void Present(const Surface& front) override;
    void Sync(uint32_t data, uint32_t size) override;
    void SyncLate(uint32_t data) override;
    bool DeliverLate(uint32_t data) override;
    void RecordLate(uint32_t data) override;
    void Flush() override;
    void SetGamma(const uint8_t* ramp) override;
    void WaitGpuFrame(uint64_t n);

private:
    struct Frame {
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        BufferRef upload;
        uint64_t used = 0;
        uint64_t fence = 0;
        std::vector<ImageRef> garbage;
        std::vector<BufferRef> garbage_buffers;
        VkSemaphore acquired = VK_NULL_HANDLE;  // the swap chain image this frame presents is ready
        std::vector<VkDescriptorPool> pools;    // the draws' descriptor sets, reset with the frame
        size_t pool_at = 0;
        std::unordered_map<uint64_t, VkDescriptorSet> sets;  // by the buffers, views and samplers written in them
    };
    struct Alloc {
        VkBuffer buffer;
        uint64_t off;
        uint8_t* cpu;
    };

    // resources
    ImageRef CreateImage(VkFormat fmt, uint32_t w, uint32_t h, VkImageUsageFlags usage, uint32_t samples = 1,
                         uint32_t levels = 1, uint32_t depth = 1, bool cube = false);
    void DestroyImage(Image* img);
    BufferRef CreateBuffer(uint64_t size, VkBufferUsageFlags usage, bool readback);
    void DestroyBuffer(Buffer* b);
    void Release(ImageRef r) {
        if (r) CurrentFrame().garbage.push_back(std::move(r));
    }
    Alloc Upload(uint64_t size, uint64_t align = 256);
    Frame& CurrentFrame() { return frames_[frame_ % kFrames]; }
    // recording: layouts, the attachments drawn to, the pipeline and what it reads
    void Transition(Image& img, VkImageLayout want);
    // load: LOAD keeps what the target held, DONT_CARE for a pass that covers every pixel of it
    void SetAttachments(Image* color, Image* depth, VkAttachmentLoadOp load = VK_ATTACHMENT_LOAD_OP_LOAD);
    void EnsureRendering();
    void EndRendering();
    void HostReadBarrier();
    void SetViewport(float x, float y, float w, float h);
    void BindPipeline(VkPipeline pso);
    void SetBlendConstants(const float f[4]);
    void SetStencilReference(uint32_t ref);
    void SetTopology(VkPrimitiveTopology topology);
    void ForgetState();
    void SetScissor(const Rect& r);
    void SetFullViewport(uint32_t w, uint32_t h);
    void SetBlitTarget(Image* rtv, uint32_t w, uint32_t h);
    void PushBindings(const Bindings& b);
    VkDescriptorBufferInfo ConstantBuffer(const void* data, uint64_t bytes);
    void Begin();
    void Submit(bool wait);
    void WaitFence(uint64_t value);
    VkFence SubmitFence(uint64_t value);
    void PollFences();
    void ResetPools(Frame& f);
    VkDescriptorSet AllocateSet();
    VkRenderPass CompatiblePass(VkFormat color, VkFormat depth, uint32_t samples,
                                VkAttachmentLoadOp load = VK_ATTACHMENT_LOAD_OP_LOAD);
    VkFramebuffer Framebuffer(Image* color, Image* depth, uint32_t w, uint32_t h);
    void DropFramebuffers(VkImageView v);
    void CompletePipeline(PipelineParts& p);
    uint64_t CompletedFence();
    // a surface's place in targets_: its key, console size and entry, end when it has none
    struct TargetSlot {
        uint32_t data;
        uint64_t key;
        uint32_t gw;
        uint32_t gh;
        std::unordered_map<uint64_t, Target>::iterator it;
        bool sized;  // the entry is there at the surface's size
    };
    Target* FindTarget(const Surface& s, bool depth);
    Target* EnsureTarget(const Surface& s, bool depth);
    // EnsureTarget's steps, in call order
    TargetSlot FindTargetSlot(const Surface& s, bool depth);
    void ReleaseTarget(std::unordered_map<uint64_t, Target>::iterator it);
    bool CreateTargetResource(Target& t);
    void ClearNewTarget(Target& t);
    // UploadTarget, then its step
    void UploadTarget(Target& t, const Surface& s);
    ImageRef CreatePixelTexture(const std::vector<uint8_t>& px, uint32_t pitch, uint32_t w, uint32_t h);
public:
    // RGBA pixels as an ImGui texture, for the touch controls' buttons
    ImTextureID AddUiTexture(const uint8_t* rgba, uint32_t w, uint32_t h);

private:
    void Blit(Image* src, const float uvrect[4], const BlitOptions& o);
    VkSampler EnsureSampler(const VkSamplerCreateInfo& info, const VkSamplerCustomBorderColorCreateInfoEXT* border);
    void BlitSamplers(VkSampler (&out)[4]);
    void DrawFullScreen(VkPipeline pso, Bindings& b, bool samplers_set = false);
    Image* Feedback(Target& t);
    Image* Flat(Target& t);
    uint32_t Samples(uint32_t want);
    bool Pass(const char* hlsl, const char* vs_entry, const char* entry, VkFormat rtv, Image* rh, uint32_t w,
              uint32_t h, Image* const src[3], const float p[8], float blend);
    VkPipeline EnsurePassPipeline(const char* hlsl, const char* vs_entry, const char* entry, VkFormat rtv, float blend);
    VkPipeline CreatePassPipeline(const PassPipe& p);
    bool BindTargets(const State& s);
    bool DepthCopy(const State& s);
    bool DepthPass(Image* src, Image* dst, uint32_t w, uint32_t h);
    VkPipeline EnsureDepthCopyPipeline(bool ms, uint32_t samples);
    // one full screen pass of kAoHlsl
    struct AoPass {
        const char* entry;
        VkFormat out = kAoFormat;
        uint32_t samples = 1;  // of the target written
        bool blend = false;    // onto that target, OcclusionBlend
        Image* rtv = nullptr;
        Image* depth = nullptr;  // the scene's depth and stencil, read by the first pass
        bool ms_depth = false;
        Image* src = nullptr;  // the pass before
        float step_x = 0;
        float step_y = 0;
    };
    void Occlusion(const State& s, const float p[4 * kOcclusionRects]);
    bool OcclusionFits(const Target* rt, const Target* ds) const;
    bool EnsureOcclusionTargets(uint32_t w, uint32_t h);
    VkPipeline EnsureOcclusionPipeline(const AoPass& a);
    bool OcclusionPass(const AoPass& a, float c[kAoConstants], const Rect& rect, uint32_t w, uint32_t h);
    Target* draw_rt_ = nullptr;  // the draw's targets, the bound ones go when a texture's copy is drawn
    Target* draw_ds_ = nullptr;
    // Draw, in call order, each callee right after its caller
    bool AssembleDraw(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices,
                      Assembled& a);
    void EnsureVertexShader(const State& s, const Assembled& a, DrawShaders& sh);
    bool BindStageTexturesAndTargets(const State& s, DrawTextures& stages);
    Image* BindTexture(const State& s, int stage, DrawTextures& stages);
    Image* BindTargetTexture(const State& s, int stage, const TextureDesc& d, Target& t, bool box,
                             DrawTextures& stages);
    Image* BindGuestTexture(const State& s, int stage, const TextureDesc& d, const FormatInfo& fi,
                            DrawTextures& stages);
    bool CreateTexture(Tex& tx, const TextureDesc& d, const FormatInfo& fi, uint64_t key);
    void UploadTexture(Tex& tx, const TextureDesc& d, const uint8_t* p, const uint32_t* colours);
    void GenerateMips(Image& img);
    void UploadTextureLevel(Tex& tx, const TextureDesc& d, uint32_t f, uint32_t l, const std::vector<uint8_t>& px,
                            uint32_t pitch);
    void EnsurePixelShader(const PixelKey& pk, DrawShaders& sh);
    void BuildPipelineKey(const State& s, const Assembled& a, uint64_t vh, uint64_t ph, PsoKey& k);
    void KeyDepthStencil(const State& s, const Assembled& a, PsoKey& k);
    bool MarksOcclusion(const PsoKey& k) const;
    VkPipeline EnsurePipeline(const PsoKey& k, VkShaderModule vs, VkShaderModule ps);
    float TargetScale() const;  // host pixels per console pixel of the bound target
    bool WideDraw() const;      // ultrawide on and the bound target the frame's size: its 2D layer is pulled in
    float RenderTargetShift(const State& s, const DrawShaders& sh) const;
    void UploadConstants(const State& s, const DrawShaders& sh, const PsoKey& k, const DrawTextures& stages,
                         GuestExtent size, DrawConstants& c, Bindings& b);
    void FillTextureConstants(const State& s, const DrawShaders& sh, const DrawTextures& stages, DrawConstants& c);
    void FillTexelConstants(int i, const TextureDesc& d, const DrawTextures& stages, DrawConstants& c);
    void CaptureDraw(const State& s, uint32_t prim, uint32_t count, const PsoKey& k, const DrawConstants& c,
                     const DrawTextures& stages);
    void CaptureTextures(const State& s, const DrawTextures& stages);
    void EndDrawCapture(const State& s, const Assembled& a, uint32_t count, const DrawConstants& c);
    void CaptureInputs(const State& s, const Assembled& a, const DrawConstants& c);
    void CaptureShaders(const State& s, const Assembled& a, const DrawShaders& sh, const PixelKey& pk);
    void CaptureShader(const char* kind, uint64_t key, const std::string& src);
    void SetTextureViewsAndSamplers(const State& s, const Assembled& a, DrawTextures& stages, Bindings& b);
    void StageSamplers(const State& s, const DrawTextures& stages, bool clamp, VkSampler (&out)[4]);
    bool Linear(VkFormat f);
    std::unordered_map<VkFormat, bool> linear_;
    void SetViewportScissor(const State& s, uint32_t gw, bool pretransformed);
    void SetBlendFactorStencilRef(const State& s, const PsoKey& k);
    void DrawGeometry(const Assembled& a);
    // shaders by source hash: Compile returns the module and sets hash
    VkShaderModule Compile(const std::string& src, const char* entry, const char* profile, uint64_t& hash);
    VkPipeline EnsureBlitPipeline(VkFormat rtv, bool over, uint32_t samples);
    VkPipeline EnsureClearPipeline(VkFormat rtv, uint32_t samples, uint32_t wmask);
    void ClearChannels(Target& t, const float colour[4], uint32_t wmask, const std::vector<VkClearRect>& rs);
    // Init, in order, each callee right after its caller
    bool CreateInstance();
    bool PickDevice();
    bool CreateDevice();
    // the pipeline cache on disk, keyed to the driver: a phone without one recompiles every pipeline on each launch
    void LoadPipelineCache();
    void SavePipelineCache();
    bool CreateLayouts();
    void CreateFrames();
    void CreateDefaults();
    bool InitUi();
    void LoadSettings();
    // Present, in call order, each callee right after its caller
    void FinishCapture(const Surface& front);
    void StartCapture();
    void UpdateHdrDisplay();
    void ResizeSwap();
    bool EnsureSurface();
    bool CreateSurface();
    bool CreateSwapchain(uint32_t w, uint32_t h, bool hdr);
    void DestroySwapchainImages();
    bool Acquire(uint32_t& index);
    void UploadGammaRamp();
    void DrawFrame(Target* t, uint32_t bi);
    Image* PrepareShownFrame(Target* t);
    Image* Smaa(Image* src, VkFormat fmt, uint32_t w, uint32_t h);
    struct SmaaLookup {
        VkFormat fmt;
        uint32_t w;
        uint32_t h;
        uint32_t bpp;  // bytes per texel
        const uint8_t* data;
    };
    bool EnsureSmaaLookups();
    ImageRef CreateLookupTexture(const SmaaLookup& l);
    bool EnsureSmaaTargets(VkFormat fmt, uint32_t w, uint32_t h);
    bool DrawSmaaPasses(Image* src, VkFormat fmt, uint32_t w, uint32_t h, const char* hlsl);
    // the picture's place in the back buffer, and the share of the target's rows it shows
    struct FramePlace {
        float x = 0, y = 0, w = 0, h = 0;
        float rows = 1;
    };
    FramePlace PlaceFrame(const Target& t);
    // black outside the picture: the bars alone, so the whole surface is not written every frame
    void ClearBars(Image& bb, const FramePlace& p, const VkClearColorValue& colour);
    void DrawSdrFrame(Target* t, Image* r, Image* bb);
    bool EnsureOutTargets();
    void DrawHdrFrame(Target* t, Image* r, Image* bb);
    void BlitFrame(Target& t, Image* r, VkFormat rtv);
    void RenderUi();
    void SubmitPresent(bool acquired, uint32_t bi);
    void ReleaseOldTextures();
    void ResetFrame();
    void CountFps();
    void ApplySettings();
    ImageRef Shrink(Image* src, uint32_t w, uint32_t h, const ShrinkOptions& o = {});
    bool ReadBack(Image* res, uint32_t w, uint32_t h, std::vector<uint8_t>& out);
    void CopyImageToBuffer(Image* src, VkBuffer dst, uint32_t w, uint32_t h);
    // a resource CapSave shrinks to w x h and writes, colour times mul
    struct CapImage {
        Image* res;
        uint32_t w;
        uint32_t h;
        float mul = 1;
    };
    void CapSave(const CapImage& img, const char* name) { CapSave(img, cap_dir_ / (std::string(name) + ".bmp")); }
    void CapSave(const CapImage& img, const std::filesystem::path& file);
    BufferRef RecordSnapshot(uint32_t bi);
    void SaveSnapshot(const Buffer& b);
    void CapSave(Target& t, const char* name, float mul);
    void CapStep(const Surface& rt);
    // Clear's arguments past the state
    struct ClearArgs {
        uint32_t count;
        const uint8_t* rects;
        uint32_t flags;
        uint32_t color;
        float z;
        uint32_t stencil;
    };
    void ClearTarget(Target& t, const State& s, const ClearArgs& c, std::vector<VkClearRect>& rs);
    // Copy's guest lists: count src_rects, dst_points 0 or one per rect
    struct CopyList {
        uint32_t count;
        uint32_t src_rects;
        uint32_t dst_points;
    };
    void CopyByDraw(const Target& src_t, Image* src_res, Target& dst_t, const CopyList& l);
    void CopyByRegion(const Target& src_t, Image* src_res, Target& dst_t, const CopyList& l);
    // the target as a guest sized BGRA copy source, and the shrunk copy that holds it when it is not one already
    struct CopySource {
        Image* res;
        ImageRef down;
    };
    CopySource PrepareCopySource(Target& t, const ShrinkOptions& o);
    // RecordLate's steps, under late_lock_
    struct Late;
    void ResetLate(Late& l, const Target& t);
    Image* PrepareLateSource(Target& t, Late& l);
    void RecordLateCopy(Late& l, uint32_t slot, Image* src);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props_{};
    VkDevice dev_ = VK_NULL_HANDLE;
    VkPipelineCache pipe_cache_ = VK_NULL_HANDLE;
    std::filesystem::path pipe_cache_file_;
    uint32_t pipe_new_ = 0;  // pipelines made since the cache was last written
    uint32_t family_ = 0;
    VkQueue queue_ = VK_NULL_HANDLE;
    VmaAllocator vma_ = nullptr;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSwapchainKHR swap_ = VK_NULL_HANDLE;
    std::vector<ImageRef> bb_;
    std::vector<VkSemaphore> rendered_;  // per swap chain image: its frame's commands are done
    VkFormat swap_fmt_ = kBgra;
    VkFormat ui_fmt_ = kBgra;  // the SDR swap chain's, which the UI's pipeline is made for
    VkPresentModeKHR present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
    bool swap_dirty_ = false;  // out of date or suboptimal: made again before the next present
    uint32_t surface_epoch_ = 0;     // host::SurfaceEpoch the surface was made at
    bool surface_lost_ = false;      // the driver said the window is gone
    uint32_t surface_failures_ = 0;  // presents in a row without a surface, for the log
    bool hdr_ext_ = false;     // VK_EXT_hdr_metadata
    bool custom_border_ = false;
    bool maintenance5_ = false;
    bool bc_ = true;  // BC textures; most phones have none, and DXT is decoded on the CPU
    // what Vulkan 1.3 brings, each with its stand-in on an older driver
    bool dynamic_rendering_ = true;   // else render passes (Vulkan 1.1 drivers without VK_KHR_dynamic_rendering)
    bool khr_rendering_ = false;      // dynamic rendering from its extension, below 1.3
    bool timeline_ = true;            // else a fence per submit stands in for the timeline's values
    bool dynamic_state_ = true;       // else the topology and the vertex strides are part of the pipeline
    bool ext_dynamic_state_ = false;  // the dynamic state from VK_EXT_extended_dynamic_state, below 1.3
    struct Pending {
        uint64_t value;
        VkFence fence;
    };
    std::mutex fence_lock_;  // the game thread waits on frames too (WaitGpuFrame)
    std::vector<Pending> pending_;
    std::vector<VkFence> spare_fences_;
    std::atomic<uint64_t> fenced_{0};  // the highest value whose fence signalled
    std::unordered_map<uint64_t, VkRenderPass> passes_;
    struct Fb {
        VkImageView color, depth;
        VkFramebuffer fb;
    };
    std::unordered_map<uint64_t, Fb> framebuffers_;
    float max_aniso_ = 1;
    bool depth_clamp_ = false;  // without the feature depthClampEnable is invalid: some drivers then draw nothing
    VkFormat depth_fmt_ = VK_FORMAT_D24_UNORM_S8_UINT;
    VkSemaphore fence_ = VK_NULL_HANDLE;  // a timeline: each submit signals fence_value_
    uint64_t fence_value_ = 0;
    Frame frames_[kFrames];
    uint64_t frame_ = 0;
    VkCommandBuffer cl_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorPool ui_pool_ = VK_NULL_HANDLE;
    Image* att_color_ = nullptr;  // the attachments the next draw renders to
    VkAttachmentLoadOp att_load_ = VK_ATTACHMENT_LOAD_OP_LOAD;
    Image* att_depth_ = nullptr;
    bool rendering_ = false;      // a vkCmdBeginRendering with them is open
    // What the command buffer has bound and set, so a draw sets only what changed: Mali's driver works over the
    // state dirtied since the last draw on every one, thousands a frame. ForgetState drops it all when something
    // else may have changed it (a new command buffer, ImGui's drawing).
    struct BoundState {
        VkPipeline pso = VK_NULL_HANDLE;
        bool viewport_set = false, scissor_set = false, blend_set = false, ref_set = false, topology_set = false;
        bool vsc_set = false;
        VkViewport viewport{};
        VkRect2D scissor{};
        float blend[4] = {};
        uint32_t ref = 0;
        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkDescriptorSet set = VK_NULL_HANDLE;  // bound with these dynamic offsets
        uint32_t offsets[2] = {};
        std::array<std::array<float, 4>, 192> vsc{};  // the vertex constants last uploaded, and where
        VkDescriptorBufferInfo vsc_cb{};
    } bound_;
    std::unordered_map<uint64_t, VkShaderModule> blobs_;
    std::unordered_map<uint64_t, uint64_t> shader_src_;  // vertex and pixel program keys to source hash
    std::unordered_map<uint64_t, VkPipeline> psos_;
    PsoKey last_pso_key_{};  // the last pipeline EnsurePipeline found, by its whole key
    VkPipeline last_pso_ = VK_NULL_HANDLE;
    std::unordered_map<uint64_t, VkSampler> samplers_;
    std::unordered_map<uint64_t, Target> targets_;
    std::unordered_map<uint64_t, Tex> textures_;  // by data address, a P8 texture's with its palette's above
    std::unordered_map<uint64_t, VkPipeline> blit_psos_;
    ImageRef dummy_[3];  // what a stage without a texture reads: 2D, 3D, cube
    std::vector<ImageRef> ui_images_;  // the touch controls' art, made once and kept
    // one frame capture: every draw in a text file, each target as the frame leaves it, the textures it read
    FILE* cap_ = nullptr;
    std::filesystem::path cap_dir_;
    uint32_t shown_rows_ = 0;  // the presented surface's rows the title drew, 0 all
    uint32_t cap_i_ = 0;
    uint32_t cap_rt_ = 0;
    std::unordered_set<uint32_t> cap_seen_;
    std::unordered_set<uint64_t> cap_sh_;
    struct CapTex {
        ImageRef res;
        uint32_t data;
        uint32_t w;
        uint32_t h;
    };
    std::vector<CapTex> cap_tex_;
    ImageRef lut_;
    uint8_t ramp_[768] = {};
    bool ramp_dirty_ = false;
    bool gamma_ = false;
    Target* bound_rt_ = nullptr;
    Target* bound_ds_ = nullptr;
    uint32_t guest_w_ = 640;
    uint32_t guest_h_ = 480;
    uint32_t bb_w_ = 0;
    uint32_t bb_h_ = 0;
    float scale_ = 1.0f;
    float wide_ = 1.0f;      // ultrawide: the screen's width over 16:9's at its height, 1 off (game::g_wide)
    uint32_t msaa_ = 1;      // samples in use
    uint32_t msaa_set_ = 0;  // the setting they came from
    // ambient occlusion: the raw pass and its first blur, each with view z beside it
    ImageRef ao_[kAoTargets];
    uint32_t ao_w_ = 0;
    uint32_t ao_h_ = 0;
    uint64_t ao_frame_ = 0;  // presents_ plus one when the pass last ran, 0 before it ever did
    // surfaces locked without a wait: copies in flight, the newest finished one goes to guest memory
    struct Late {
        BufferRef rb[kFrames];
        ImageRef down;
        uint64_t fence[kFrames] = {};
        uint64_t frame[kFrames] = {};
        uint64_t delivered = 0;
        uint32_t gw = 0;
        uint32_t gh = 0;
        uint32_t next = 0;
    };
    std::unordered_map<uint32_t, Late> late_;
    std::mutex late_lock_;                      // DeliverLate runs on the title's thread
    std::atomic<uint64_t> presents_seen_{0};    // presents_ for DeliverLate
    std::atomic<uint64_t> present_fence_[kFrames + 1] = {};  // by presents_, for WaitGpuFrame
    Assembled asm_;  // kept between draws for its buffers
    // HDR10 output: the surface takes it, the output asked and in use, the frame and panel targets it draws first
    bool hdr_display_ = false;
    bool hdr_want_ = false;
    bool hdr_out_ = false;
    bool vsync_ = true;
    float paper_nits_ = kDefaultPaperNits;
    float peak_nits_ = kDefaultPeakNits;
    ImageRef out_[kOutTargets];
    uint32_t out_w_ = 0;
    uint32_t out_h_ = 0;
    // SMAA: edges, blend weights and the blended frame, at the frame's size; its area and search lookups
    ImageRef smaa_[kSmaaTargets];
    ImageRef smaa_lut_[2];
    uint32_t smaa_w_ = 0;
    uint32_t smaa_h_ = 0;
    VkFormat smaa_fmt_ = VK_FORMAT_UNDEFINED;
    uint64_t presents_ = 0;
    double fps_ = 0;
    std::chrono::steady_clock::time_point fps_t_ = std::chrono::steady_clock::now();
    uint64_t fps_n_ = 0;
};


VulkanBackend::~VulkanBackend() {
    if (!dev_) return;
    vkDeviceWaitIdle(dev_);
    SavePipelineCache();
    if (pipe_cache_) vkDestroyPipelineCache(dev_, pipe_cache_, nullptr);
}

bool VulkanBackend::Init(uint32_t width, uint32_t height) {
    guest_w_ = width;
    guest_h_ = height;
    // the game plays on with sound when this fails, so the step is named on screen rather than a black one
    auto fail = [&](const char* step) {
        char text[256];
        std::snprintf(text, sizeof(text), "Renderer failed: %s (%s).", step,
                      phys_ ? props_.deviceName : "no GPU");
        LOGE("%s", text);
        host::Toast(text);
        return false;
    };
    if (!CreateInstance()) return fail("no Vulkan instance");
    // a slow phone may not have handed the window over yet
    for (int i = 0; i < 250 && !CreateSurface(); i++) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (!surface_) return fail("no window surface");
    if (!PickDevice()) return fail("no usable GPU");
    if (!CreateDevice()) return fail("no Vulkan device");
    if (!CreateLayouts()) return fail("no pipeline layout");
    CreateFrames();
    Begin();
    CreateDefaults();
    UpdateHdrDisplay();
    app::Mode px = app::WindowPixels();
    vsync_ = app::Get().vsync;
    if (!CreateSwapchain(px.width, px.height, false)) return fail("no swap chain");
    if (!InitUi()) return fail("no UI");
    LoadSettings();
    LOGI("vulkan: %ux%u guest, %ux%u window, render scale %.3f", width, height, bb_w_, bb_h_, scale_);
    return true;
}

bool VulkanBackend::CreateInstance() {
    uint32_t count = 0;
    const char* const* sdl_ext = SDL_Vulkan_GetInstanceExtensions(&count);
    std::vector<const char*> ext(sdl_ext, sdl_ext + count);
    uint32_t n = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> have(n);
    vkEnumerateInstanceExtensionProperties(nullptr, &n, have.data());
    for (auto& e : have)
        if (!std::strcmp(e.extensionName, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME)) ext.push_back(e.extensionName);
    std::vector<const char*> layers;
    // XBE_VK_DEBUG: the validation layer, as XBE_D3D12_DEBUG has D3D12's
    if (std::getenv("XBE_VK_DEBUG")) layers.push_back("VK_LAYER_KHRONOS_validation");
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    std::string name = app::Utf8(app::kGame.title);
    ai.pApplicationName = name.c_str();
    ai.pEngineName = "XboxXBE-Recomp";
    uint32_t loader = VK_API_VERSION_1_1;
    vkEnumerateInstanceVersion(&loader);
    ai.apiVersion = std::min<uint32_t>(loader, VK_API_VERSION_1_3);
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &ai;
    ci.enabledExtensionCount = uint32_t(ext.size());
    ci.ppEnabledExtensionNames = ext.data();
    ci.enabledLayerCount = uint32_t(layers.size());
    ci.ppEnabledLayerNames = layers.data();
    VkResult r = vkCreateInstance(&ci, nullptr, &instance_);
    if (r != VK_SUCCESS && !layers.empty()) {
        LOGW("vulkan: no validation layer, going on without it");
        ci.enabledLayerCount = 0;
        r = vkCreateInstance(&ci, nullptr, &instance_);
    }
    if (r != VK_SUCCESS) {
        LOGE("vulkan: no instance (%d), a Vulkan 1.1 driver is needed", int(r));
        return false;
    }
    return true;
}

constexpr uint32_t kVendorArm = 0x13B5;

// a discrete GPU first on hybrid laptops; it needs Vulkan 1.1 and a queue that presents. What 1.2 and 1.3 add is taken
// from the core or the extensions where the driver has it, and stood in for where it does not
bool VulkanBackend::PickDevice() {
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance_, &n, nullptr);
    std::vector<VkPhysicalDevice> list(n);
    vkEnumeratePhysicalDevices(instance_, &n, list.data());
    int best = -1;
    for (VkPhysicalDevice pd : list) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pd, &p);
        if (p.apiVersion < VK_API_VERSION_1_1) continue;
        uint32_t en = 0;
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, nullptr);
        std::vector<VkExtensionProperties> ext(en);
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, ext.data());
        auto has = [&](const char* name) {
            return std::any_of(ext.begin(), ext.end(), [&](auto& e) { return !std::strcmp(e.extensionName, name); });
        };
        if (!has(VK_KHR_SWAPCHAIN_EXTENSION_NAME)) continue;
        bool v12 = p.apiVersion >= VK_API_VERSION_1_2, v13 = p.apiVersion >= VK_API_VERSION_1_3;
        bool rendering = v13 || (has(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME) &&
                                 (v12 || (has(VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME) &&
                                          has(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME))));
        bool dynamic_state = v13 || has(VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME);
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qs(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qs.data());
        for (uint32_t q = 0; q < qn; q++) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(pd, q, surface_, &present);
            if (!(qs[q].queueFlags & VK_QUEUE_GRAPHICS_BIT) || !present) continue;
            int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 : 1;
            if (score > best) {
                best = score;
                phys_ = pd;
                family_ = q;
                props_ = p;
                hdr_ext_ = has(VK_EXT_HDR_METADATA_EXTENSION_NAME);
                custom_border_ = has(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME);
                // VK_KHR_maintenance5 builds on dynamic rendering
                maintenance5_ = has(VK_KHR_MAINTENANCE_5_EXTENSION_NAME) && rendering;
                dynamic_rendering_ = rendering;
                khr_rendering_ = rendering && !v13;
                // the 1.1 drivers' timeline extension lost the device on the first submit (Mali-G76, Black-Android):
                // fences there
                timeline_ = v12;
                dynamic_state_ = dynamic_state;
                ext_dynamic_state_ = dynamic_state && !v13;

            }
            break;
        }
    }
    if (!phys_) {
        LOGE("vulkan: no GPU with Vulkan 1.1 and a queue that presents to the window");
        return false;
    }
    // Extended dynamic state, maintenance5 and custom border colours are taken on Mali only, the one GPU the port is
    // tried on; the others go Black-Android's way, which is played on Adreno, without them
    if (props_.vendorID != kVendorArm) {
        dynamic_state_ = ext_dynamic_state_ = maintenance5_ = custom_border_ = false;
    }
    LOGR("GPU: %s (Vulkan %u.%u.%u, driver %08X)", props_.deviceName, VK_API_VERSION_MAJOR(props_.apiVersion),
         VK_API_VERSION_MINOR(props_.apiVersion), VK_API_VERSION_PATCH(props_.apiVersion), props_.driverVersion);
    // reported, not warned: release logs drop warnings and these are what a bug report needs
    if (!dynamic_rendering_) LOGR("vulkan: no dynamic rendering, using render passes");
    if (!timeline_) LOGR("vulkan: below Vulkan 1.2, using fences for the timeline");
    if (!dynamic_state_) LOGR("vulkan: no extended dynamic state, topology and strides in the pipelines");
    // D24S8 as the console's depth, else D32S8 (AMD has no D24S8)
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(phys_, VK_FORMAT_D24_UNORM_S8_UINT, &fp);
    constexpr VkFormatFeatureFlags kNeed = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                           VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    depth_fmt_ = (fp.optimalTilingFeatures & kNeed) == kNeed ? VK_FORMAT_D24_UNORM_S8_UINT
                                                             : VK_FORMAT_D32_SFLOAT_S8_UINT;
    if (depth_fmt_ != VK_FORMAT_D24_UNORM_S8_UINT) LOGI("vulkan: no D24S8, depth is D32S8");
    return true;
}

bool VulkanBackend::CreateDevice() {
    VkPhysicalDeviceFeatures have{};
    vkGetPhysicalDeviceFeatures(phys_, &have);
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.features.samplerAnisotropy = have.samplerAnisotropy;
    f2.features.fillModeNonSolid = have.fillModeNonSolid;
    f2.features.depthClamp = have.depthClamp;
    depth_clamp_ = have.depthClamp;
    f2.features.shaderClipDistance = have.shaderClipDistance;
    f2.features.depthBiasClamp = have.depthBiasClamp;
    f2.features.textureCompressionBC = have.textureCompressionBC;
    bc_ = have.textureCompressionBC;
    if (!bc_) LOGR("vulkan: the GPU has no BC textures, DXT is decoded on the CPU");
    max_aniso_ = have.samplerAnisotropy ? props_.limits.maxSamplerAnisotropy : 1.0f;
    // the per-feature structs rather than VkPhysicalDeviceVulkan1xFeatures, which a 1.1 device does not know; the
    // chain holds only what is used, asked first, and what the driver lists but cannot do is stood in for
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    VkPhysicalDeviceDynamicRenderingFeatures rendering{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES};
    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT dynamic_state{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT};
    // links the structs in use after *tail, giving back the last pNext
    auto chain = [&](void** tail) {
        timeline.pNext = rendering.pNext = dynamic_state.pNext = nullptr;
        if (timeline_) *std::exchange(tail, &timeline.pNext) = &timeline;
        if (dynamic_rendering_) *std::exchange(tail, &rendering.pNext) = &rendering;
        if (ext_dynamic_state_) *std::exchange(tail, &dynamic_state.pNext) = &dynamic_state;
        return tail;
    };
    {
        VkPhysicalDeviceFeatures2 q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        chain(&q.pNext);
        vkGetPhysicalDeviceFeatures2(phys_, &q);
        if (timeline_ && !timeline.timelineSemaphore) {
            LOGR("vulkan: timeline semaphores not supported, using fences");
            timeline_ = false;
        }
        if (dynamic_rendering_ && !rendering.dynamicRendering) {
            LOGR("vulkan: dynamic rendering listed but not supported, using render passes");
            dynamic_rendering_ = khr_rendering_ = maintenance5_ = false;
        }
        if (ext_dynamic_state_ && !dynamic_state.extendedDynamicState) {
            LOGR("vulkan: extended dynamic state listed but not supported, topology and strides in the pipelines");
            dynamic_state_ = ext_dynamic_state_ = false;
        }
    }
    timeline.timelineSemaphore = VK_TRUE;
    rendering.dynamicRendering = VK_TRUE;
    dynamic_state.extendedDynamicState = VK_TRUE;
    VkPhysicalDeviceCustomBorderColorFeaturesEXT border{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
    border.customBorderColors = VK_TRUE;
    border.customBorderColorWithoutFormat = VK_TRUE;
    VkPhysicalDeviceMaintenance5FeaturesKHR m5{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR};
    m5.maintenance5 = VK_TRUE;
    void** tail = chain(&f2.pNext);
    std::vector<const char*> ext = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    if (khr_rendering_) {
        ext.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
        if (props_.apiVersion < VK_API_VERSION_1_2) {  // what dynamic rendering builds on is not core yet
            ext.push_back(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
            ext.push_back(VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME);
        }
    }
    if (ext_dynamic_state_) ext.push_back(VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME);
    if (hdr_ext_) ext.push_back(VK_EXT_HDR_METADATA_EXTENSION_NAME);
    if (custom_border_) {
        VkPhysicalDeviceCustomBorderColorFeaturesEXT q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
        VkPhysicalDeviceFeatures2 qf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &q};
        vkGetPhysicalDeviceFeatures2(phys_, &qf);
        custom_border_ = q.customBorderColors && q.customBorderColorWithoutFormat;
    }
    if (custom_border_) {
        ext.push_back(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME);
        *tail = &border;
        tail = &border.pNext;
    }
    // points take size 1 without the vertex shader writing one
    if (maintenance5_) {
        ext.push_back(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
        *tail = &m5;
        tail = &m5.pNext;
    }
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = family_;
    qi.queueCount = 1;
    qi.pQueuePriorities = &priority;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.pNext = &f2;
    di.queueCreateInfoCount = 1;
    di.pQueueCreateInfos = &qi;
    di.enabledExtensionCount = uint32_t(ext.size());
    di.ppEnabledExtensionNames = ext.data();
    VkResult r = vkCreateDevice(phys_, &di, nullptr, &dev_);
    if (r != VK_SUCCESS) {
        LOGE("vulkan: no device (%d)", int(r));
        return false;
    }
    vkGetDeviceQueue(dev_, family_, 0, &queue_);
    LoadPipelineCache();
    bool v13 = props_.apiVersion >= VK_API_VERSION_1_3;
    auto load = [&](const char* core, const char* ext) { return vkGetDeviceProcAddr(dev_, v13 ? core : ext); };
    if (dynamic_rendering_) {
        p_vkCmdBeginRendering =
            reinterpret_cast<PFN_vkCmdBeginRenderingKHR>(load("vkCmdBeginRendering", "vkCmdBeginRenderingKHR"));
        p_vkCmdEndRendering =
            reinterpret_cast<PFN_vkCmdEndRenderingKHR>(load("vkCmdEndRendering", "vkCmdEndRenderingKHR"));
    }
    if (dynamic_state_) {
        p_vkCmdSetPrimitiveTopology = reinterpret_cast<PFN_vkCmdSetPrimitiveTopologyEXT>(
            load("vkCmdSetPrimitiveTopology", "vkCmdSetPrimitiveTopologyEXT"));
        p_vkCmdBindVertexBuffers2 = reinterpret_cast<PFN_vkCmdBindVertexBuffers2EXT>(
            load("vkCmdBindVertexBuffers2", "vkCmdBindVertexBuffers2EXT"));
    }
    if (timeline_) {
        p_vkGetSemaphoreCounterValue =
            reinterpret_cast<PFN_vkGetSemaphoreCounterValueKHR>(vkGetDeviceProcAddr(dev_, "vkGetSemaphoreCounterValue"));
        p_vkWaitSemaphores = reinterpret_cast<PFN_vkWaitSemaphoresKHR>(vkGetDeviceProcAddr(dev_, "vkWaitSemaphores"));
    }
    if ((dynamic_rendering_ && (!p_vkCmdBeginRendering || !p_vkCmdEndRendering)) ||
        (dynamic_state_ && (!p_vkCmdSetPrimitiveTopology || !p_vkCmdBindVertexBuffers2)) ||
        (timeline_ && (!p_vkGetSemaphoreCounterValue || !p_vkWaitSemaphores))) {
        LOGE("vulkan: device functions missing");
        return false;
    }
    VmaAllocatorCreateInfo ac{};
    ac.physicalDevice = phys_;
    ac.device = dev_;
    ac.instance = instance_;
    ac.vulkanApiVersion = VK_API_VERSION_1_1;  // as VMA_VULKAN_VERSION: VMA uses nothing past 1.1
    if (vmaCreateAllocator(&ac, &vma_) != VK_SUCCESS) {
        LOGE("vulkan: no allocator");
        return false;
    }
    VkSemaphoreTypeCreateInfo ti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &ti};
    if (timeline_) VK_CHECK(vkCreateSemaphore(dev_, &si, nullptr, &fence_));
    // HLSL to SPIR-V at run time: the NDK has no shaderc, glslang's HLSL front end is the same compiler; no clip
    // distance, as on Mali and in Black-Android: the path the port is seen to draw right on
    vkshader::Init(host::FilesDir() / "shadercache", false);
    return true;
}

namespace {

// the cache file's header: a cache another driver wrote is not one this one can read
struct PipeCacheHeader {
    char tag[8];
    uint32_t version;
    uint32_t vendor;
    uint32_t device;
    uint32_t driver;
    uint32_t api;
    uint32_t uuid[VK_UUID_SIZE / 4];
};

PipeCacheHeader PipeHeader(const VkPhysicalDeviceProperties& p) {
    PipeCacheHeader h{};
    std::memcpy(h.tag, "DW5VK", 5);
    h.version = 1;
    h.vendor = p.vendorID;
    h.device = p.deviceID;
    h.driver = p.driverVersion;
    h.api = p.apiVersion;
    std::memcpy(h.uuid, p.pipelineCacheUUID, VK_UUID_SIZE);
    return h;
}

}  // namespace

void VulkanBackend::LoadPipelineCache() {
    PipeCacheHeader h = PipeHeader(props_);
    pipe_cache_file_ = host::FilesDir() / "pipelinecache" / "cache.bin";
    std::vector<uint8_t> data;
    if (FILE* f = std::fopen(pipe_cache_file_.c_str(), "rb")) {
        PipeCacheHeader file_h{};
        if (std::fread(&file_h, sizeof(file_h), 1, f) == 1 && std::memcmp(&file_h, &h, sizeof(h)) == 0 &&
            std::fseek(f, 0, SEEK_END) == 0) {
            long n = std::ftell(f);
            if (n > long(sizeof(h))) {
                data.resize(size_t(n) - sizeof(h));
                std::fseek(f, long(sizeof(h)), SEEK_SET);
                if (std::fread(data.data(), 1, data.size(), f) != data.size()) data.clear();
            }
        }
        std::fclose(f);
    }
    VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    ci.initialDataSize = data.size();
    ci.pInitialData = data.data();
    if (vkCreatePipelineCache(dev_, &ci, nullptr, &pipe_cache_) != VK_SUCCESS) {
        pipe_cache_ = VK_NULL_HANDLE;
        LOGW("vulkan: no pipeline cache");
        return;
    }
    LOGI("vulkan: pipeline cache %zu bytes from disk", data.size());
}

void VulkanBackend::SavePipelineCache() {
    if (!pipe_cache_) return;
    size_t n = 0;
    if (vkGetPipelineCacheData(dev_, pipe_cache_, &n, nullptr) != VK_SUCCESS || n == 0) return;
    std::vector<uint8_t> data(n);
    if (vkGetPipelineCacheData(dev_, pipe_cache_, &n, data.data()) != VK_SUCCESS || n == 0) return;
    std::error_code ec;
    std::filesystem::create_directories(pipe_cache_file_.parent_path(), ec);
    if (ec) return;
    PipeCacheHeader h = PipeHeader(props_);
    std::filesystem::path tmp = pipe_cache_file_;
    tmp += ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return;
    bool ok = std::fwrite(&h, sizeof(h), 1, f) == 1 && std::fwrite(data.data(), 1, n, f) == n;
    if (std::fclose(f) != 0) ok = false;
    if (ok) std::filesystem::rename(tmp, pipe_cache_file_, ec);
}

// one push descriptor set for every pipeline: b0 b1 as uniform buffers, t0 to t3 sampled images, s0 to s3 samplers
bool VulkanBackend::CreateLayouts() {
    VkDescriptorSetLayoutBinding b[kBindings] = {};
    for (uint32_t i = 0; i < kBindings; i++) {
        b[i].binding = i;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        b[i].descriptorType = i < kBindingTex       ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
                              : i < kBindingSampler ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                                                    : VK_DESCRIPTOR_TYPE_SAMPLER;
    }
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = kBindings;
    li.pBindings = b;
    if (vkCreateDescriptorSetLayout(dev_, &li, nullptr, &set_layout_) != VK_SUCCESS) return false;
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &set_layout_;
    return vkCreatePipelineLayout(dev_, &pl, nullptr, &layout_) == VK_SUCCESS;
}

void VulkanBackend::CreateFrames() {
    for (auto& f : frames_) {
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pi.queueFamilyIndex = family_;
        VK_CHECK(vkCreateCommandPool(dev_, &pi, nullptr, &f.pool));
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = f.pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(dev_, &ai, &f.cmd));
        f.upload = CreateBuffer(kUploadSize,
                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                false);
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(dev_, &si, nullptr, &f.acquired));
    }
    cl_ = CurrentFrame().cmd;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cl_, &bi);
}

// the gamma ramp as a 256 x 1 lookup, and the textures a stage without one reads
void VulkanBackend::CreateDefaults() {
    constexpr VkImageUsageFlags kSampled = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    lut_ = CreateImage(kBgra, 256, 1, kSampled);
    dummy_[0] = CreateImage(kBgra, 1, 1, kSampled);
    dummy_[1] = CreateImage(kBgra, 1, 1, kSampled, 1, 1, 2);
    dummy_[2] = CreateImage(kBgra, 1, 1, kSampled, 1, 1, 1, true);
    for (ImageRef* i : {&lut_, &dummy_[0], &dummy_[1], &dummy_[2]}) {
        Transition(**i, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkClearColorValue black{};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0,
                                      VK_REMAINING_ARRAY_LAYERS};
        vkCmdClearColorImage(cl_, (*i)->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
        Transition(**i, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
}

bool VulkanBackend::InitUi() {
    // one set per texture the UI draws: the font, the four face buttons and the eight pills, and again as many when the
    // button style is changed. Sixteen was the font's and a few more, and a style change took it past that: the new
    // textures then failed to make a set, and every frame tried again, each try a new image and a full barrier
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pi.maxSets = 64;
    pi.poolSizeCount = 1;
    pi.pPoolSizes = &size;
    if (vkCreateDescriptorPool(dev_, &pi, nullptr, &ui_pool_) != VK_SUCCESS) return false;
    ImGui_ImplVulkan_InitInfo ii{};
    ii.Instance = instance_;
    ii.PhysicalDevice = phys_;
    ii.Device = dev_;
    ii.QueueFamily = family_;
    ii.Queue = queue_;
    ii.DescriptorPool = ui_pool_;
    ii.MinImageCount = 2;
    ii.ImageCount = kFrames;
    ii.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    ii.UseDynamicRendering = dynamic_rendering_;
    // the panel draws over the SDR back buffer, or into the panel target of an HDR10 frame, made in the same format
    ui_fmt_ = swap_fmt_;
    if (!dynamic_rendering_) ii.RenderPass = CompatiblePass(ui_fmt_, VK_FORMAT_UNDEFINED, 1);
    ii.PipelineRenderingCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    ii.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    ii.PipelineRenderingCreateInfo.pColorAttachmentFormats = &ui_fmt_;  // kept: ImGui holds the pointer
    // the backend is built without prototypes (CMakeLists.txt): its functions come from the device, and the rendering
    // pair is the one looked up above, which a 1.3 device without the KHR extension gives out only by its core name
    auto loader = [](const char* name, void* self) -> PFN_vkVoidFunction {
        auto* vk = static_cast<VulkanBackend*>(self);
        if (!std::strcmp(name, "vkCmdBeginRenderingKHR")) return reinterpret_cast<PFN_vkVoidFunction>(p_vkCmdBeginRendering);
        if (!std::strcmp(name, "vkCmdEndRenderingKHR")) return reinterpret_cast<PFN_vkVoidFunction>(p_vkCmdEndRendering);
        PFN_vkVoidFunction f = vkGetDeviceProcAddr(vk->dev_, name);
        return f ? f : vkGetInstanceProcAddr(vk->instance_, name);
    };
    std::lock_guard lock(app::UiLock());
    if (!ImGui_ImplVulkan_LoadFunctions(loader, this)) {
        LOGE("vulkan: the UI's Vulkan functions could not be loaded");
        return false;
    }
    return ImGui_ImplVulkan_Init(&ii);
}

void VulkanBackend::LoadSettings() {
    scale_ = app::RenderScale(guest_h_);
    msaa_set_ = game::Opt().msaa;
    msaa_ = Samples(msaa_set_);
}

ImageRef VulkanBackend::CreateImage(VkFormat fmt, uint32_t w, uint32_t h, VkImageUsageFlags usage, uint32_t samples,
                                    uint32_t levels, uint32_t depth, bool cube) {
    ImageRef img(new Image, [this](Image* i) { DestroyImage(i); });
    img->format = fmt;
    img->w = w;
    img->h = h;
    img->depth = depth;
    img->levels = levels;
    img->layers = cube ? 6 : 1;
    img->samples = samples;
    bool ds = IsDepthFormat(fmt);
    img->aspect = !ds ? VK_IMAGE_ASPECT_COLOR_BIT
                      : HasStencil(fmt) ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
    ci.imageType = depth > 1 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    ci.format = fmt;
    ci.extent = {w, h, depth};
    ci.mipLevels = levels;
    ci.arrayLayers = img->layers;
    ci.samples = VkSampleCountFlagBits(samples);
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(vma_, &ci, &ai, &img->image, &img->alloc, nullptr) != VK_SUCCESS) {
        LOGE("vulkan: no %ux%u image, format %d, %u samples", w, h, int(fmt), samples);
        return nullptr;
    }
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = img->image;
    vi.viewType = cube ? VK_IMAGE_VIEW_TYPE_CUBE : depth > 1 ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
    vi.format = fmt;
    vi.subresourceRange = {ds ? VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT) : img->aspect, 0, levels, 0, img->layers};
    VK_CHECK(vkCreateImageView(dev_, &vi, nullptr, &img->view));
    img->attach = img->view;
    if (ds && (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
        vi.subresourceRange.aspectMask = img->aspect;
        VK_CHECK(vkCreateImageView(dev_, &vi, nullptr, &img->attach));
    }
    if (HasStencil(fmt) && (usage & VK_IMAGE_USAGE_SAMPLED_BIT)) {
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
        VK_CHECK(vkCreateImageView(dev_, &vi, nullptr, &img->stencil));
    }
    return img;
}

void VulkanBackend::DestroyImage(Image* img) {
    DropFramebuffers(img->attach);
    if (img->stencil) vkDestroyImageView(dev_, img->stencil, nullptr);
    if (img->attach && img->attach != img->view) vkDestroyImageView(dev_, img->attach, nullptr);
    if (img->view) vkDestroyImageView(dev_, img->view, nullptr);
    if (img->alloc) vmaDestroyImage(vma_, img->image, img->alloc);
    delete img;
}

BufferRef VulkanBackend::CreateBuffer(uint64_t size, VkBufferUsageFlags usage, bool readback) {
    BufferRef b(new Buffer, [this](Buffer* p) { DestroyBuffer(p); });
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | (readback ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                                                            : VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    // coherent, as Black-Android takes it: nothing here flushes what the CPU wrote; every device has such a type
    ai.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(vma_, &ci, &ai, &b->buffer, &b->alloc, &info) != VK_SUCCESS) {
        LOGE("vulkan: no %llu byte buffer", (unsigned long long)size);
        return nullptr;
    }
    b->map = static_cast<uint8_t*>(info.pMappedData);
    b->size = size;
    return b;
}

void VulkanBackend::DestroyBuffer(Buffer* b) {
    if (b->alloc) vmaDestroyBuffer(vma_, b->buffer, b->alloc);
    delete b;
}

VulkanBackend::Alloc VulkanBackend::Upload(uint64_t size, uint64_t align) {
    Frame& f = CurrentFrame();
    g_perf.upload += size;
    uint64_t o = (f.used + align - 1) & ~(align - 1);
    if (o + size <= kUploadSize) {
        f.used = o + size;
        return {f.upload->buffer, o, f.upload->map + o};
    }
    // too big for the ring: a buffer of its own, released with the frame
    g_perf.overflow++;
    BufferRef b = CreateBuffer(std::max<uint64_t>(size, 1),
                               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               false);
    f.garbage_buffers.push_back(b);
    return {b->buffer, 0, b->map};
}

// every layout change waits for all earlier work, as D3D12's transitions between states do
void VulkanBackend::Transition(Image& img, VkImageLayout want) {
    if (img.layout == want) return;
    EndRendering();
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.oldLayout = img.layout;
    b.newLayout = want;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img.image;
    b.subresourceRange = {img.aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    vkCmdPipelineBarrier(cl_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
    img.layout = want;
}

// copies into buffers the CPU reads once the fence is past them
void VulkanBackend::HostReadBarrier() {
    EndRendering();
    VkMemoryBarrier m{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    m.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    m.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cl_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &m, 0, nullptr, 0,
                         nullptr);
}

// the attachments of the next draws; rendering to them starts at the first that needs it
void VulkanBackend::SetAttachments(Image* color, Image* depth, VkAttachmentLoadOp load) {
    if (color == att_color_ && depth == att_depth_ && load == att_load_) return;
    EndRendering();
    att_load_ = load;
    att_color_ = color;
    att_depth_ = depth;
}

void VulkanBackend::EnsureRendering() {
    if (rendering_ || (!att_color_ && !att_depth_)) return;
    if (att_color_) Transition(*att_color_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    if (att_depth_) Transition(*att_depth_, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    // the renderings before wrote the same attachments in the same layout, which Transition leaves without a barrier:
    // their writes come first (Black-Android). Without it a tiler (Mali) loads this pass's tiles before the last pass
    // stored them, and blocks from the top left show what was there before: the sky's box, the menu's staircase
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    constexpr VkPipelineStageFlags kAttachmentStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                                       VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    vkCmdPipelineBarrier(cl_, kAttachmentStages, kAttachmentStages, 0, 1, &mb, 0, nullptr, 0, nullptr);
    VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    uint32_t w = UINT32_MAX;
    uint32_t h = UINT32_MAX;
    if (att_color_) {
        ca.imageView = att_color_->attach;
        ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        ca.loadOp = att_load_;
        ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &ca;
        w = att_color_->w;
        h = att_color_->h;
    }
    if (att_depth_) {
        da.imageView = att_depth_->attach;
        da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        da.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        ri.pDepthAttachment = &da;
        if (HasStencil(att_depth_->format)) ri.pStencilAttachment = &da;
        w = std::min(w, att_depth_->w);
        h = std::min(h, att_depth_->h);
    }
    ri.renderArea = {{0, 0}, {w, h}};
    ri.layerCount = 1;
    if (dynamic_rendering_) {
        vkCmdBeginRendering(cl_, &ri);
    } else {
        VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        bi.renderPass = CompatiblePass(att_color_ ? att_color_->format : VK_FORMAT_UNDEFINED,
                                       att_depth_ ? att_depth_->format : VK_FORMAT_UNDEFINED,
                                       (att_color_ ? att_color_ : att_depth_)->samples, att_load_);
        bi.framebuffer = Framebuffer(att_color_, att_depth_, w, h);
        bi.renderArea = ri.renderArea;
        vkCmdBeginRenderPass(cl_, &bi, VK_SUBPASS_CONTENTS_INLINE);
    }
    rendering_ = true;
}

void VulkanBackend::EndRendering() {
    if (!rendering_) return;
    if (dynamic_rendering_) vkCmdEndRendering(cl_);
    else vkCmdEndRenderPass(cl_);
    rendering_ = false;
}

// without dynamic rendering: the attachments loaded and stored and left in their layouts, as EnsureRendering's
// dynamic rendering does; a pipeline needs only a compatible pass, formats and samples
VkRenderPass VulkanBackend::CompatiblePass(VkFormat color, VkFormat depth, uint32_t samples,
                                           VkAttachmentLoadOp load) {
    uint64_t key = uint64_t(color) | uint64_t(depth) << 24 | uint64_t(samples) << 40 |
                   uint64_t(load == VK_ATTACHMENT_LOAD_OP_DONT_CARE) << 60;
    if (auto it = passes_.find(key); it != passes_.end()) return it->second;
    VkAttachmentDescription at[2] = {};
    VkAttachmentReference cr{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference dr{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sp{};
    sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    auto count = VkSampleCountFlagBits(std::max(samples, 1u));
    uint32_t n = 0;
    if (color != VK_FORMAT_UNDEFINED) {
        at[n] = {0, color, count, load, VK_ATTACHMENT_STORE_OP_STORE,
                 VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        cr.attachment = n++;
        sp.colorAttachmentCount = 1;
        sp.pColorAttachments = &cr;
    }
    if (depth != VK_FORMAT_UNDEFINED) {
        at[n] = {0, depth, count, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_LOAD_OP_LOAD,
                 VK_ATTACHMENT_STORE_OP_STORE, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        dr.attachment = n++;
        sp.pDepthStencilAttachment = &dr;
    }
    VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    ci.attachmentCount = n;
    ci.pAttachments = at;
    ci.subpassCount = 1;
    ci.pSubpasses = &sp;
    VkRenderPass rp = VK_NULL_HANDLE;
    if (vkCreateRenderPass(dev_, &ci, nullptr, &rp) != VK_SUCCESS) LOGE("vulkan: render pass failed");
    passes_[key] = rp;
    return rp;
}

VkFramebuffer VulkanBackend::Framebuffer(Image* color, Image* depth, uint32_t w, uint32_t h) {
    VkImageView cv = color ? color->attach : VK_NULL_HANDLE, dv = depth ? depth->attach : VK_NULL_HANDLE;
    const uint64_t k[4] = {uint64_t(cv), uint64_t(dv), w, h};
    uint64_t key = Hash(k, sizeof(k));
    if (auto it = framebuffers_.find(key); it != framebuffers_.end()) return it->second.fb;
    VkImageView views[2];
    uint32_t n = 0;
    if (cv) views[n++] = cv;
    if (dv) views[n++] = dv;
    VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    ci.renderPass = CompatiblePass(color ? color->format : VK_FORMAT_UNDEFINED,
                                   depth ? depth->format : VK_FORMAT_UNDEFINED, (color ? color : depth)->samples);
    ci.attachmentCount = n;
    ci.pAttachments = views;
    ci.width = w;
    ci.height = h;
    ci.layers = 1;
    VkFramebuffer fb = VK_NULL_HANDLE;
    if (vkCreateFramebuffer(dev_, &ci, nullptr, &fb) != VK_SUCCESS) LOGE("vulkan: framebuffer %ux%u failed", w, h);
    framebuffers_[key] = {cv, dv, fb};
    return fb;
}

// a view's framebuffers go with it; its image is destroyed only once the GPU is past the frames that used it
void VulkanBackend::DropFramebuffers(VkImageView v) {
    if (!v || framebuffers_.empty()) return;
    for (auto it = framebuffers_.begin(); it != framebuffers_.end();) {
        if (it->second.color == v || it->second.depth == v) {
            vkDestroyFramebuffer(dev_, it->second.fb, nullptr);
            it = framebuffers_.erase(it);
        } else {
            ++it;
        }
    }
}

// the HLSL's clip space has y up: the viewport is flipped, as D3D's maps it
void VulkanBackend::SetViewport(float x, float y, float w, float h) {
    VkViewport vp{x, y + h, w, -h, 0, 1};
    if (bound_.viewport_set && !std::memcmp(&vp, &bound_.viewport, sizeof(vp))) return;
    bound_.viewport = vp;
    bound_.viewport_set = true;
    vkCmdSetViewport(cl_, 0, 1, &vp);
}

void VulkanBackend::SetScissor(const Rect& r) {
    VkRect2D sc{{std::max(r.left, 0), std::max(r.top, 0)},
                {uint32_t(std::max(r.right - std::max(r.left, 0), 0)), uint32_t(std::max(r.bottom - std::max(r.top, 0), 0))}};
    if (bound_.scissor_set && !std::memcmp(&sc, &bound_.scissor, sizeof(sc))) return;
    bound_.scissor = sc;
    bound_.scissor_set = true;
    vkCmdSetScissor(cl_, 0, 1, &sc);
}

void VulkanBackend::BindPipeline(VkPipeline pso) {
    if (pso == bound_.pso) return;
    bound_.pso = pso;
    vkCmdBindPipeline(cl_, VK_PIPELINE_BIND_POINT_GRAPHICS, pso);
}

void VulkanBackend::SetBlendConstants(const float f[4]) {
    if (bound_.blend_set && !std::memcmp(f, bound_.blend, sizeof(bound_.blend))) return;
    std::memcpy(bound_.blend, f, sizeof(bound_.blend));
    bound_.blend_set = true;
    vkCmdSetBlendConstants(cl_, f);
}

void VulkanBackend::SetStencilReference(uint32_t ref) {
    if (bound_.ref_set && ref == bound_.ref) return;
    bound_.ref = ref;
    bound_.ref_set = true;
    vkCmdSetStencilReference(cl_, VK_STENCIL_FACE_FRONT_AND_BACK, ref);
}

void VulkanBackend::SetTopology(VkPrimitiveTopology topology) {
    if (!dynamic_state_ || (bound_.topology_set && topology == bound_.topology)) return;
    bound_.topology = topology;
    bound_.topology_set = true;
    vkCmdSetPrimitiveTopology(cl_, topology);
}

// the command buffer's state unknown again: everything is set on the next draw
void VulkanBackend::ForgetState() {
    std::array<std::array<float, 4>, 192> vsc = bound_.vsc;  // kept, it only means anything with vsc_set
    bound_ = BoundState{};
    bound_.vsc = vsc;
}

void VulkanBackend::SetFullViewport(uint32_t w, uint32_t h) {
    SetViewport(0, 0, float(w), float(h));
    SetScissor({0, 0, int32_t(w), int32_t(h)});
}

// one w x h colour target for a Blit, the draw's bound targets forgotten
void VulkanBackend::SetBlitTarget(Image* rtv, uint32_t w, uint32_t h) {
    SetAttachments(rtv, nullptr);
    bound_rt_ = bound_ds_ = nullptr;
    SetFullViewport(w, h);
}

// One descriptor set per frame for each set of buffers, views and samplers, written once, the constant buffers
// dynamic: a draw binds it with its constants' offsets. Mali's driver takes this far faster than ten descriptors
// pushed per draw (Black-Android binds its draws so), and a draw like the last binds nothing.
void VulkanBackend::PushBindings(const Bindings& b) {
    VkDescriptorBufferInfo cb[2];
    for (uint32_t i = 0; i < 2; i++)
        cb[i] = b.cb[i].buffer ? b.cb[i] : VkDescriptorBufferInfo{CurrentFrame().upload->buffer, 0, kUboRange};
    struct Key {
        VkBuffer buffers[2];
        VkImageView views[4], stencil;
        VkSampler samplers[4];
    } key{{cb[0].buffer, cb[1].buffer}, {b.views[0], b.views[1], b.views[2], b.views[3]}, b.stencil,
          {b.samplers[0], b.samplers[1], b.samplers[2], b.samplers[3]}};
    uint64_t h = Hash(&key, sizeof(key));
    uint32_t offsets[2] = {uint32_t(cb[0].offset), uint32_t(cb[1].offset)};
    Frame& f = CurrentFrame();
    VkDescriptorSet& set = f.sets[h];
    if (!set) {
        set = AllocateSet();
        VkDescriptorBufferInfo buffers[2] = {{cb[0].buffer, 0, kUboRange}, {cb[1].buffer, 0, kUboRange}};
        VkDescriptorImageInfo images[4];
        VkDescriptorImageInfo samplers[4];
        VkWriteDescriptorSet w[kBindings];
        uint32_t n = 0;
        for (uint32_t i = 0; i < 2; i++) {
            w[n] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w[n].dstSet = set;
            w[n].dstBinding = kBindingCb + i;
            w[n].descriptorCount = 1;
            w[n].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            w[n].pBufferInfo = &buffers[i];
            n++;
        }
        for (uint32_t i = 0; i < 4; i++) {
            images[i] = {VK_NULL_HANDLE, b.views[i] ? b.views[i] : dummy_[0]->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            w[n] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w[n].dstSet = set;
            w[n].dstBinding = kBindingTex + i;
            w[n].descriptorCount = 1;
            w[n].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            w[n].pImageInfo = &images[i];
            n++;
        }
        for (uint32_t i = 0; i < 4; i++) {
            if (!b.samplers[i]) continue;
            samplers[i] = {b.samplers[i], VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
            w[n] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w[n].dstSet = set;
            w[n].dstBinding = kBindingSampler + i;
            w[n].descriptorCount = 1;
            w[n].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            w[n].pImageInfo = &samplers[i];
            n++;
        }
        vkUpdateDescriptorSets(dev_, n, w, 0, nullptr);
    }
    // every pipeline has layout_: what was bound stays bound across them
    if (bound_.set == set && bound_.offsets[0] == offsets[0] && bound_.offsets[1] == offsets[1]) return;
    bound_.set = set;
    bound_.offsets[0] = offsets[0];
    bound_.offsets[1] = offsets[1];
    vkCmdBindDescriptorSets(cl_, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &set, 2, offsets);
}

// a set from the frame's pools, a new pool when they are full
VkDescriptorSet VulkanBackend::AllocateSet() {
    constexpr uint32_t kSetsPerPool = 1024;
    Frame& f = CurrentFrame();
    for (;;) {
        if (f.pool_at == f.pools.size()) {
            const VkDescriptorPoolSize sizes[3] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 2 * kSetsPerPool},
                                                   {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4 * kSetsPerPool},
                                                   {VK_DESCRIPTOR_TYPE_SAMPLER, 4 * kSetsPerPool}};
            VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            pi.maxSets = kSetsPerPool;
            pi.poolSizeCount = 3;
            pi.pPoolSizes = sizes;
            VkDescriptorPool pool = VK_NULL_HANDLE;
            VK_CHECK(vkCreateDescriptorPool(dev_, &pi, nullptr, &pool));
            f.pools.push_back(pool);
        }
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = f.pools[f.pool_at];
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &set_layout_;
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(dev_, &ai, &set) == VK_SUCCESS) return set;
        f.pool_at++;
    }
}

void VulkanBackend::ResetPools(Frame& f) {
    for (VkDescriptorPool p : f.pools) vkResetDescriptorPool(dev_, p, 0);
    f.pool_at = 0;
    f.sets.clear();
}

// constants in the upload ring, kUboRange reserved so the binding's fixed range stays inside what was allocated
VkDescriptorBufferInfo VulkanBackend::ConstantBuffer(const void* data, uint64_t bytes) {
    Alloc a = Upload(kUboRange, 256);
    std::memcpy(a.cpu, data, std::min(bytes, kUboRange));
    return {a.buffer, a.off, kUboRange};
}

void VulkanBackend::Begin() {
    // a new command buffer, and the upload ring the constants were in may be reused
    ForgetState();
    bound_rt_ = bound_ds_ = nullptr;
    att_color_ = att_depth_ = nullptr;
    rendering_ = false;
}

void VulkanBackend::Submit(bool wait) {
    EndRendering();
    vkEndCommandBuffer(cl_);
    uint64_t value = ++fence_value_;
    VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    ts.signalSemaphoreValueCount = 1;
    ts.pSignalSemaphoreValues = &value;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, timeline_ ? &ts : nullptr};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cl_;
    si.signalSemaphoreCount = timeline_ ? 1 : 0;
    si.pSignalSemaphores = &fence_;
    if (timeline_) {
        VK_CHECK(vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE));
    } else {
        std::lock_guard lock(fence_lock_);
        VK_CHECK(vkQueueSubmit(queue_, 1, &si, SubmitFence(value)));
    }
    Frame& f = CurrentFrame();
    f.fence = fence_value_;
    if (wait) {
        WaitFence(fence_value_);
        vkResetCommandPool(dev_, f.pool, 0);
        ResetPools(f);
        f.garbage.clear();
        f.garbage_buffers.clear();
        f.used = 0;
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cl_, &bi);
        Begin();
    }
}

void VulkanBackend::Flush() { Submit(true); }

void VulkanBackend::WaitFence(uint64_t value) {
    if (!timeline_) {
        if (!value || CompletedFence() >= value) return;
        // the lock is held through the wait so no fence is reset under it; the GPU needs no thread to get there
        std::lock_guard lock(fence_lock_);
        for (const Pending& p : pending_)
            if (p.value >= value) {
                vkWaitForFences(dev_, 1, &p.fence, VK_TRUE, UINT64_MAX);
                break;
            }
        PollFences();
        return;
    }
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &fence_;
    wi.pValues = &value;
    VK_CHECK(vkWaitSemaphores(dev_, &wi, UINT64_MAX));
}

uint64_t VulkanBackend::CompletedFence() {
    if (!timeline_) {
        std::lock_guard lock(fence_lock_);
        PollFences();
        return fenced_;
    }
    uint64_t v = 0;
    vkGetSemaphoreCounterValue(dev_, fence_, &v);
    return v;
}

// without the timeline: the fences that signalled, in submit order, give their values back; fence_lock_ held
void VulkanBackend::PollFences() {
    size_t done = 0;
    while (done < pending_.size() && vkGetFenceStatus(dev_, pending_[done].fence) == VK_SUCCESS) {
        fenced_ = pending_[done].value;
        vkResetFences(dev_, 1, &pending_[done].fence);
        spare_fences_.push_back(pending_[done].fence);
        done++;
    }
    pending_.erase(pending_.begin(), pending_.begin() + done);
}

// the fence the submit of this value signals; fence_lock_ held
VkFence VulkanBackend::SubmitFence(uint64_t value) {
    VkFence f = VK_NULL_HANDLE;
    if (!spare_fences_.empty()) {
        f = spare_fences_.back();
        spare_fences_.pop_back();
    } else {
        VkFenceCreateInfo ci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK_CHECK(vkCreateFence(dev_, &ci, nullptr, &f));
    }
    pending_.push_back({value, f});
    return f;
}


// targets_ key of a surface's colour or depth target
uint64_t TargetMapKey(uint32_t data, bool depth) { return uint64_t(data) << 1 | (depth ? 1 : 0); }

// a resized target gives back its resources once the frame is done
void VulkanBackend::ReleaseTarget(std::unordered_map<uint64_t, Target>::iterator it) {
    if (bound_rt_ == &it->second || bound_ds_ == &it->second) bound_rt_ = bound_ds_ = nullptr;
    if (att_color_ == it->second.res.get() || att_depth_ == it->second.res.get()) SetAttachments(nullptr, nullptr);
    Release(it->second.res);
    Release(it->second.fb);
    targets_.erase(it);
}

// the host image of a new target, at render scale; the scene and its depth multisampled
bool VulkanBackend::CreateTargetResource(Target& t) {
    t.w = std::max(1u, uint32_t(std::lround(t.gw * scale_)));
    t.h = std::max(1u, uint32_t(std::lround(t.gh * scale_)));
    // the game's small maps and blur chain stay as they are
    uint32_t samples = t.gw == guest_w_ && t.gh == guest_h_ ? msaa_ : 1;
    constexpr VkImageUsageFlags kCommon = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                          VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (t.depth) {
        t.res = CreateImage(depth_fmt_, t.w, t.h, kCommon | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, samples);
    } else {
        t.res = CreateImage(t.fmt, t.w, t.h, kCommon | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, samples);
    }
    if (!t.res) LOGE("vulkan: no %ux%u target with %u samples", t.w, t.h, samples);
    return t.res != nullptr;
}

// a new target starts cleared: colour transparent black, depth far
void VulkanBackend::ClearNewTarget(Target& t) {
    Image& img = *t.res;
    Transition(img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageSubresourceRange range{img.aspect, 0, 1, 0, 1};
    if (t.depth) {
        VkClearDepthStencilValue v{1.0f, 0};
        vkCmdClearDepthStencilImage(cl_, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &range);
    } else {
        VkClearColorValue v{};
        vkCmdClearColorImage(cl_, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &v, 1, &range);
    }
}

VulkanBackend::TargetSlot VulkanBackend::FindTargetSlot(const Surface& s, bool depth) {
    TargetSlot slot;
    slot.data = mem::VirtualFromPhysical(s.data);
    slot.key = TargetMapKey(slot.data, depth);
    SurfaceSize(s, slot.gw, slot.gh);
    slot.it = targets_.find(slot.key);
    slot.sized = slot.it != targets_.end() && slot.it->second.gw == slot.gw && slot.it->second.gh == slot.gh;
    return slot;
}

// the surface's target at its size, null when there is none
Target* VulkanBackend::FindTarget(const Surface& s, bool depth) {
    if (!s.header) return nullptr;
    TargetSlot slot = FindTargetSlot(s, depth);
    if (!slot.sized) return nullptr;
    return &slot.it->second;
}

// the surface's target, made again at its size when that changed
Target* VulkanBackend::EnsureTarget(const Surface& s, bool depth) {
    if (!s.header) return nullptr;
    TargetSlot slot = FindTargetSlot(s, depth);
    if (slot.sized) return &slot.it->second;
    if (slot.it != targets_.end()) ReleaseTarget(slot.it);
    Target t;
    t.depth = depth;
    t.gw = slot.gw;
    t.gh = slot.gh;
    if (!CreateTargetResource(t)) return nullptr;
    g_perf.targets++;
    ClearNewTarget(t);
    t.data = slot.data;
    t.bytes = GuestPitch(slot.gw) * slot.gh;
    if (depth) t.bytes = slot.gw * slot.gh * 4;
    t.epoch = DirtyEpoch(slot.data);
    t.sig = MemSig(slot.data, t.bytes);
    Target& ref = targets_[slot.key];
    ref = std::move(t);
    return &ref;
}

// the sample count the GPU has for the scene's formats, the one asked for or the next under it
uint32_t VulkanBackend::Samples(uint32_t want) {
    uint32_t n = std::clamp(want, 1u, 8u);
    auto has = [&](VkFormat f, VkImageUsageFlags usage) {
        VkImageFormatProperties p{};
        VkResult r = vkGetPhysicalDeviceImageFormatProperties(phys_, f, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                                                              usage | VK_IMAGE_USAGE_SAMPLED_BIT, 0, &p);
        return r == VK_SUCCESS && (p.sampleCounts & n);
    };
    while (n > 1 && !(has(kBgra, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) &&
                      has(depth_fmt_, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))) {
        n--;
    }
    // Vulkan counts in powers of two
    while (n & (n - 1)) n--;
    LOGI("vulkan: MSAA %ux (asked %u)", n, want);
    return n;
}

bool VulkanBackend::BindTargets(const State& s) {
    Target* rt = EnsureTarget(s.rt, false);
    Target* ds = s.rs[kRsZEnable] || s.rs[kRsStencilEnable] ? EnsureTarget(s.zs, true) : nullptr;
    if (rt && ds && (rt->w != ds->w || rt->h != ds->h || Ms(*rt) != Ms(*ds))) ds = nullptr;
    if (!rt && !ds) return false;
    if (rt) {
        if (LatchCpuWrite(*rt)) UploadTarget(*rt, s.rt);
        rt->drawn = true;
    }
    if (ds) ds->drawn = true;
    if (rt != bound_rt_ || ds != bound_ds_) {
        SetAttachments(rt ? rt->res.get() : nullptr, ds ? ds->res.get() : nullptr);
        bound_rt_ = rt;
        bound_ds_ = ds;
    }
    return true;
}

// a depth buffer drawn as colour into another depth buffer's memory: copied as depth
bool VulkanBackend::DepthCopy(const State& s) {
    TextureDesc d;
    if (!s.rt.header || !s.textures[0] || !Describe(s.textures[0], d) || Format(d.format).depth || d.cube ||
        d.depth != 1) {
        return false;
    }
    auto di = targets_.find(TargetMapKey(mem::VirtualFromPhysical(s.rt.data), true));
    auto si = targets_.find(TargetMapKey(d.data, true));
    if (di == targets_.end() || si == targets_.end() || di == si) return false;
    Target& dst = di->second;
    Target& src = si->second;
    Image* from = Flat(src);
    if (!from) return false;
    if (cap_) fprintf(cap_, "%04u depth copy %08X to %08X\n", cap_i_++, src.data, dst.data);
    if (!DepthPass(from, dst.res.get(), dst.w, dst.h)) return false;
    dst.drawn = true;
    return true;
}

// DepthPass's pipeline, kept in blit_psos_: writes depth only, always passing; null when it can't be made
VkPipeline VulkanBackend::EnsureDepthCopyPipeline(bool ms, uint32_t samples) {
    VkPipeline& pso = blit_psos_[~(uint64_t(ms) | uint64_t(samples) << 1)];
    if (pso) return pso;
    std::string hlsl = std::string(ms ? "#define MS\n" : "") + kDepthHlsl;
    uint64_t hash;
    VkShaderModule vs = Compile(hlsl, "vs", "vs_5_0", hash);
    VkShaderModule ps = Compile(hlsl, "psmain", "ps_5_0", hash);
    if (!vs || !ps) return VK_NULL_HANDLE;
    pso = CreatePassPipeline({.vs = vs, .ps = ps, .vs_entry = "vs", .ps_entry = "psmain", .depth = depth_fmt_,
                              .samples = samples});
    return pso;
}

// src's depth over the w x h depth target dst, a multisampled src's first sample
bool VulkanBackend::DepthPass(Image* src, Image* dst, uint32_t w, uint32_t h) {
    VkPipeline pso = EnsureDepthCopyPipeline(src->samples > 1, dst->samples);
    if (!pso) return false;
    Transition(*src, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    SetAttachments(nullptr, dst);
    SetFullViewport(w, h);
    Bindings b;
    float zero[64] = {};
    b.cb[0] = b.cb[1] = ConstantBuffer(zero, sizeof(zero));
    b.views[0] = src->view;
    DrawFullScreen(pso, b);
    bound_rt_ = bound_ds_ = nullptr;
    return true;
}

// multiplies occlusion from the scene's depth into its colour, p as d3d::Occlusion sends it
void VulkanBackend::Occlusion(const State& s, const float p[4 * kOcclusionRects]) {
    Target* rt = FindTarget(s.rt, false);
    Target* ds = FindTarget(s.zs, true);
    if (!OcclusionFits(rt, ds)) return;
    const Viewport& v = s.viewport;
    Rect rect = ViewRect(v, *rt);
    if (rect.right <= rect.left || rect.bottom <= rect.top || v.max_z <= v.min_z) return;
    if (!EnsureOcclusionTargets(rt->w, rt->h)) return;
    ao_frame_ = presents_ + 1;
    if (cap_) {
        CapStep(s.rt);
        fprintf(cap_, "%04u occlusion rt %08X ds %08X\n", cap_i_++, s.rt.data, s.zs.data);
    }
    uint32_t samples = Ms(*ds);
    LOG_FIRST(1, "vulkan: ambient occlusion %s %ux%u, %u samples, projection %.4f %.4f, view z %.6f / (depth - %.8f)",
              p[6] ? "GTAO" : "SSAO", rt->w, rt->h, samples, 1 / p[0], 1 / p[1], p[4], p[5]);
    float c[kAoConstants];
    OcclusionConstants(v, rect, p, c);
    const char* entry = p[6] ? "gtao" : "occlusion";
    AoPass raw{.entry = entry, .rtv = ao_[0].get(), .depth = ds->res.get(), .ms_depth = samples > 1};
    AoPass across{.entry = "occlusion_blur", .rtv = ao_[1].get(), .src = ao_[0].get(), .step_x = 1};
    AoPass down{.entry = "occlusion_apply", .out = rt->fmt, .samples = Ms(*rt), .blend = true, .rtv = rt->res.get(),
                .src = ao_[1].get(), .step_y = 1};
    bound_rt_ = bound_ds_ = nullptr;
    Transition(*ds->res, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    bool ok = OcclusionPass(raw, c, rect, rt->w, rt->h);
    // a depth target left readable would pass for one whose single sample copy is current
    Transition(*ds->res, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    if (!ok) return;
    if (!OcclusionPass(across, c, rect, rt->w, rt->h)) return;
    if (OcclusionPass(down, c, rect, rt->w, rt->h)) rt->drawn = true;
}

// the frame's own scene only: views drawn to smaller targets are left
bool VulkanBackend::OcclusionFits(const Target* rt, const Target* ds) const {
    if (!rt || !ds || rt->gw != guest_w_ || rt->gh != guest_h_) return false;
    return ds->w == rt->w && ds->h == rt->h && Ms(*ds) == Ms(*rt);
}

bool VulkanBackend::EnsureOcclusionTargets(uint32_t w, uint32_t h) {
    if (ao_[kAoTargets - 1] && ao_w_ == w && ao_h_ == h) return true;
    for (int i = 0; i < kAoTargets; i++) {
        Release(std::move(ao_[i]));
        ao_[i] = CreateImage(kAoFormat, w, h, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        if (!ao_[i]) {
            LOGE("vulkan: no %ux%u ambient occlusion target", w, h);
            return false;
        }
    }
    ao_w_ = w;
    ao_h_ = h;
    return true;
}

VkPipeline VulkanBackend::EnsureOcclusionPipeline(const AoPass& a) {
    uint64_t bits = uint64_t(a.out) | uint64_t(a.ms_depth) << 32 | uint64_t(a.samples) << 40;
    VkPipeline& pso = blit_psos_[Hash(a.entry, std::strlen(a.entry), bits)];
    if (pso) return pso;
    // a stencil view reads the stencil in its first channel, D3D's X24G8 in its second
    std::string text = std::string(a.ms_depth ? "#define MS 1\n" : "") + kAoHlsl;
    for (size_t at; (at = text.find(".g & 0x80")) != std::string::npos;) text.replace(at, 2, ".r");
    uint64_t h;
    VkShaderModule vs = Compile(text, "vs", "vs_5_0", h);
    VkShaderModule ps = Compile(text, a.entry, "ps_5_0", h);
    if (!vs || !ps) return VK_NULL_HANDLE;
    pso = CreatePassPipeline({.vs = vs, .ps = ps, .vs_entry = "vs", .ps_entry = a.entry, .color = a.out,
                              .samples = a.samples, .blend = a.blend ? PassBlend::Occlusion : PassBlend::None});
    return pso;
}

// kAoHlsl's opt.zw: the pass's blur step
constexpr int kAoStepX = 14;
constexpr int kAoStepY = 15;

// t0 the scene's depth, t1 the pass before, t2 the scene's stencil
bool VulkanBackend::OcclusionPass(const AoPass& a, float c[kAoConstants], const Rect& rect, uint32_t w, uint32_t h) {
    VkPipeline pso = EnsureOcclusionPipeline(a);
    if (!pso) return false;
    if (a.src) Transition(*a.src, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    c[kAoStepX] = a.step_x;
    c[kAoStepY] = a.step_y;
    Bindings b;
    b.cb[0] = b.cb[1] = ConstantBuffer(c, 256);
    b.views[0] = a.depth ? a.depth->view : nullptr;
    b.views[1] = a.src ? a.src->view : nullptr;
    b.views[2] = a.depth ? a.depth->stencil : nullptr;
    SetAttachments(a.rtv, nullptr);
    SetViewport(0, 0, float(w), float(h));
    SetScissor(rect);
    DrawFullScreen(pso, b);
    return true;
}

// the pixels the CPU changed since the last Sync, opaque, the rest 0; t.read takes them
std::vector<uint8_t> LatchCpuChanges(Target& t) {
    std::vector<uint8_t> px(t.read.size(), 0);
    uint32_t guest_pitch = GuestPitch(t.gw);
    const uint8_t* guest = static_cast<const uint8_t*>(kernel::HostPtr(t.data));
    for (uint32_t y = 0; y < t.gh; y++) {
        for (uint32_t x = 0; x < t.gw; x++) {
            const uint8_t* c = guest + size_t(y) * guest_pitch + x * 4;
            size_t i = (size_t(y) * t.gw + x) * 4;
            if (!std::memcmp(c, &t.read[i], 4)) continue;
            std::memcpy(&t.read[i], c, 4);
            std::memcpy(&px[i], c, 3);
            px[i + 3] = 255;
        }
    }
    return px;
}

// BGRA pixels into a new w x h texture, left readable
ImageRef VulkanBackend::CreatePixelTexture(const std::vector<uint8_t>& px, uint32_t pitch, uint32_t w, uint32_t h) {
    ImageRef tex = CreateImage(kBgra, w, h, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    if (!tex) return nullptr;
    uint32_t row = w * 4;
    Alloc a = Upload(uint64_t(row) * h, 16);
    for (uint32_t y = 0; y < h; y++) std::memcpy(a.cpu + size_t(y) * row, &px[size_t(y) * pitch], row);
    Transition(*tex, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy r{};
    r.bufferOffset = a.off;
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {w, h, 1};
    vkCmdCopyBufferToImage(cl_, a.buffer, tex->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    Transition(*tex, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return tex;
}

// RGBA pixels as an ImGui texture: copied into a BGRA image left readable, the sampler made once, the image kept
ImTextureID VulkanBackend::AddUiTexture(const uint8_t* rgba, uint32_t w, uint32_t h) {
    std::vector<uint8_t> bgra(size_t(w) * h * 4);
    for (size_t i = 0; i < bgra.size(); i += 4) {
        bgra[i] = rgba[i + 2];
        bgra[i + 1] = rgba[i + 1];
        bgra[i + 2] = rgba[i];
        bgra[i + 3] = rgba[i + 3];
    }
    ImageRef tex = CreatePixelTexture(bgra, w * 4, w, h);
    if (!tex) return ImTextureID();
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.25f;
    VkSampler sampler = EnsureSampler(si, nullptr);
    ImTextureID id = ImGui_ImplVulkan_AddTexture(sampler, tex->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    ui_images_.push_back(std::move(tex));
    return id;
}

// CPU pixels of a color surface into its host target, scaled
void VulkanBackend::UploadTarget(Target& t, const Surface& s) {
    g_perf.rtups++;
    TextureDesc d;
    if (!Describe(s.header, d) || Format(d.format).depth) return;
    std::vector<uint8_t> px;
    uint32_t pitch;
    // after a Sync only the pixels the CPU changed go over the picture, which stays at render scale
    bool over = !t.drawn && d.width == t.gw && d.height == t.gh && t.read.size() == size_t(t.gw) * t.gh * 4;
    if (over) {
        LOG_FIRST(4, "vulkan: CPU pixels drawn over target %08X", t.data);
        pitch = t.gw * 4;
        px = LatchCpuChanges(t);
    } else {
        DecodeLevel(d, 0, 0, static_cast<const uint8_t*>(kernel::HostPtr(d.data)), nullptr, px, pitch);
    }
    ImageRef tex = CreatePixelTexture(px, pitch, d.width, d.height);
    if (!tex) return;
    SetBlitTarget(t.res.get(), t.w, t.h);
    float uv[4] = {0, 0, 1, 1};
    float mul = 1 / t.range;
    Blit(tex.get(), uv, {.alpha = true, .mul = mul, .rtv = t.fmt, .over = over, .samples = Ms(t)});
    Release(tex);
    t.epoch = DirtyEpoch(t.data);
    t.sig = MemSig(t.data, t.bytes);
}

// samplers by their whole description, made once
VkSampler VulkanBackend::EnsureSampler(const VkSamplerCreateInfo& info,
                                       const VkSamplerCustomBorderColorCreateInfoEXT* border) {
    uint64_t key = Hash(&info.magFilter, offsetof(VkSamplerCreateInfo, unnormalizedCoordinates) -
                                             offsetof(VkSamplerCreateInfo, magFilter));
    if (border) key = Hash(&border->customBorderColor, sizeof(border->customBorderColor), key);
    auto it = samplers_.find(key);
    if (it != samplers_.end()) return it->second;
    VkSamplerCreateInfo ci = info;
    ci.pNext = border;
    VkSampler s = VK_NULL_HANDLE;
    VK_CHECK(vkCreateSampler(dev_, &ci, nullptr, &s));
    samplers_[key] = s;
    return s;
}

// four clamp samplers: linear at slot 0, point at slots 1 to 3
void VulkanBackend::BlitSamplers(VkSampler (&out)[4]) {
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = VK_LOD_CLAMP_NONE;
    for (int i = 0; i < 4; i++) {
        VkFilter f = i == 0 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        si.magFilter = si.minFilter = f;
        si.mipmapMode = i == 0 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
        out[i] = EnsureSampler(si, nullptr);
    }
}

// the full screen triangle with b's resources; without samplers set the passes take the blit samplers
void VulkanBackend::DrawFullScreen(VkPipeline pso, Bindings& b, bool samplers_set) {
    if (!samplers_set) BlitSamplers(b.samplers);
    BindPipeline(pso);
    PushBindings(b);
    SetTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    EnsureRendering();
    vkCmdDraw(cl_, 3, 1, 0, 0);
}

void VulkanBackend::Blit(Image* src, const float uvrect[4], const BlitOptions& o) {
    VkPipeline pso = EnsureBlitPipeline(o.rtv, o.over, o.samples);
    if (!pso) return;
    Transition(*src, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    Transition(*lut_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    float c[8] = {uvrect[0], uvrect[1], uvrect[2], uvrect[3],
                  o.gamma ? 1.0f : 0.0f, o.alpha ? 1.0f : 0.0f, float(o.taps), o.mul};
    Bindings b;
    b.cb[0] = b.cb[1] = ConstantBuffer(c, sizeof(c));
    b.views[0] = src->view;
    b.views[1] = lut_->view;
    DrawFullScreen(pso, b);
}

VkShaderModule VulkanBackend::Compile(const std::string& src, const char* entry, const char* profile, uint64_t& hash) {
    // one source can hold several entry points
    uint64_t profile_seed = Hash(profile, std::strlen(profile));
    uint64_t entry_seed = Hash(entry, std::strlen(entry), profile_seed);
    hash = Hash(src.data(), src.size(), entry_seed);
    auto it = blobs_.find(hash);
    if (it != blobs_.end()) return it->second;
    PerfScope perf_compile{g_perf.compile};
    g_perf.shaders++;
    bool vertex = profile[0] == 'v';
    // HLSL registers as bindings: b0 b1 at 0, t0 to t3 at 2, s0 to s3 at 6; cbuffers packed as HLSL packs them
    std::vector<uint32_t> spirv;
    VkShaderModule module = VK_NULL_HANDLE;
    if (!vkshader::Compile(src, entry, vertex ? vkshader::Stage::Vertex : vkshader::Stage::Fragment, hash, spirv)) {
        LOGE("vulkan: %s compile failed", profile);
        LOGE("%s", src.c_str());
    } else {
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        mi.codeSize = spirv.size() * sizeof(uint32_t);
        mi.pCode = spirv.data();
        VK_CHECK(vkCreateShaderModule(dev_, &mi, nullptr, &module));
    }
    blobs_[hash] = module;
    return module;
}

// what every pipeline here shares: dynamic viewport, scissor, blend constants, stencil reference and topology,
// one colour target at most
struct PipelineParts {
    VkPipelineShaderStageCreateInfo stages[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                                 {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    VkVertexInputBindingDescription bindings[16];
    VkVertexInputAttributeDescription attributes[16];
    VkPipelineVertexInputStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState blend_att{};
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    VkDynamicState dynamic_states[6] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                        VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
                                        VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
                                        VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    VkFormat color_format = VK_FORMAT_UNDEFINED;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};

    PipelineParts(VkShaderModule vs, VkShaderModule ps, VkPipelineLayout layout, VkFormat color, VkFormat depth_format,
                  uint32_t samples, bool vertex_strides) {
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vs;
        stages[0].pName = "main";
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = ps;
        stages[1].pName = "main";
        input.pVertexBindingDescriptions = bindings;
        input.pVertexAttributeDescriptions = attributes;
        assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_CLOCKWISE;
        raster.lineWidth = 1;
        multisample.rasterizationSamples = VkSampleCountFlagBits(std::max(samples, 1u));
        blend_att.colorWriteMask = 0xF;
        blend.attachmentCount = color != VK_FORMAT_UNDEFINED ? 1 : 0;
        blend.pAttachments = &blend_att;
        dynamic.dynamicStateCount = vertex_strides ? 6 : 5;
        dynamic.pDynamicStates = dynamic_states;
        color_format = color;
        rendering.colorAttachmentCount = blend.attachmentCount;
        rendering.pColorAttachmentFormats = &color_format;
        rendering.depthAttachmentFormat = depth_format;
        rendering.stencilAttachmentFormat = HasStencil(depth_format) ? depth_format : VK_FORMAT_UNDEFINED;
        info.pNext = &rendering;
        info.stageCount = 2;
        info.pStages = stages;
        info.pVertexInputState = &input;
        info.pInputAssemblyState = &assembly;
        info.pViewportState = &viewport;
        info.pRasterizationState = &raster;
        info.pMultisampleState = &multisample;
        info.pDepthStencilState = &depth;
        info.pColorBlendState = &blend;
        info.pDynamicState = &dynamic;
        info.layout = layout;
    }
};

// what the driver lacks taken out of a pipeline: the topology and stride as dynamic state, the rendering formats for a
// compatible render pass
void VulkanBackend::CompletePipeline(PipelineParts& p) {
    if (!dynamic_state_) p.dynamic.dynamicStateCount = 4;
    if (!dynamic_rendering_) {
        p.info.pNext = nullptr;
        p.info.renderPass = CompatiblePass(p.rendering.colorAttachmentCount ? p.color_format : VK_FORMAT_UNDEFINED,
                                           p.rendering.depthAttachmentFormat, p.multisample.rasterizationSamples);
    }
}

VkPipeline VulkanBackend::CreatePassPipeline(const PassPipe& p) {
    // vk_shader names every SPIR-V entry point "main"; the HLSL names given here were taken by Mali-G57's driver but
    // made Mali-G615's fail the pipeline (-3), the blit that shows the frame among them: a black screen
    PipelineParts parts(p.vs, p.ps, layout_, p.color, p.depth, p.samples, false);
    if (p.depth != VK_FORMAT_UNDEFINED) {
        parts.depth.depthTestEnable = VK_TRUE;
        parts.depth.depthWriteEnable = VK_TRUE;
        parts.depth.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    }
    auto& b = parts.blend_att;
    b.colorWriteMask = p.wmask;
    switch (p.blend) {
        case PassBlend::None:
            break;
        case PassBlend::Over:
            // premultiplied colour, the target keeps its alpha
            b.blendEnable = VK_TRUE;
            b.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            b.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            b.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            b.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            break;
        case PassBlend::Factor:
            b.blendEnable = VK_TRUE;
            b.srcColorBlendFactor = b.srcAlphaBlendFactor = VK_BLEND_FACTOR_CONSTANT_COLOR;
            b.dstColorBlendFactor = b.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
            break;
        case PassBlend::Occlusion:
            // dst = dst * src.a + src; the alpha the game's blends read stays
            b.blendEnable = VK_TRUE;
            b.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            b.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            b.dstColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            b.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            break;
    }
    VkPipeline pso = VK_NULL_HANDLE;
    g_perf.psos++;
    CompletePipeline(parts);
    VkResult r = vkCreateGraphicsPipelines(dev_, pipe_cache_, 1, &parts.info, nullptr, &pso);
    if (r != VK_SUCCESS) {
        LOGE("vulkan: pass pipeline %s/%s creation failed (%d)", p.vs_entry, p.ps_entry, int(r));
        return VK_NULL_HANDLE;
    }
    return pso;
}

VkPipeline VulkanBackend::EnsureBlitPipeline(VkFormat rtv, bool over, uint32_t samples) {
    VkPipeline& pso = blit_psos_[uint64_t(rtv) | uint64_t(over) << 32 | uint64_t(samples) << 40];
    if (pso) return pso;
    uint64_t h;
    VkShaderModule vs = Compile(kBlitHlsl, "vs", "vs_5_0", h);
    VkShaderModule ps = Compile(kBlitHlsl, "psmain", "ps_5_0", h);
    if (!vs || !ps) return VK_NULL_HANDLE;
    pso = CreatePassPipeline({.vs = vs, .ps = ps, .vs_entry = "vs", .ps_entry = "psmain", .color = rtv,
                              .samples = samples, .blend = over ? PassBlend::Over : PassBlend::None});
    return pso;
}

// a partial colour clear's pipeline, kept in blit_psos_ under keys of their own
VkPipeline VulkanBackend::EnsureClearPipeline(VkFormat rtv, uint32_t samples, uint32_t wmask) {
    constexpr uint64_t kClearKey = 1ull << 63;
    VkPipeline& pso = blit_psos_[kClearKey | uint64_t(rtv) | uint64_t(samples) << 40 | uint64_t(wmask) << 48];
    if (pso) return pso;
    uint64_t h;
    VkShaderModule vs = Compile(kClearHlsl, "vs", "vs_5_0", h);
    VkShaderModule ps = Compile(kClearHlsl, "psmain", "ps_5_0", h);
    if (!vs || !ps) return VK_NULL_HANDLE;
    pso = CreatePassPipeline({.vs = vs, .ps = ps, .vs_entry = "vs", .ps_entry = "psmain", .color = rtv,
                              .samples = samples, .wmask = wmask});
    return pso;
}

// Pass's pipeline for the shader, kept in blit_psos_; null when it can't be made
VkPipeline VulkanBackend::EnsurePassPipeline(const char* hlsl, const char* vs_entry, const char* entry, VkFormat rtv,
                                             float blend) {
    uint64_t seed = uint64_t(rtv) << 1 | (blend >= 0 ? 1 : 0);
    uint64_t source_seed = Hash(hlsl, std::strlen(hlsl), seed);
    uint64_t entry_seed = Hash(entry, std::strlen(entry), source_seed);
    VkPipeline& pso = blit_psos_[Hash(vs_entry, std::strlen(vs_entry), entry_seed)];
    if (pso) return pso;
    uint64_t hv;
    uint64_t hp;
    VkShaderModule vs = Compile(hlsl, vs_entry, "vs_5_0", hv);
    VkShaderModule ps = Compile(hlsl, entry, "ps_5_0", hp);
    if (!vs || !ps) return VK_NULL_HANDLE;
    pso = CreatePassPipeline({.vs = vs, .ps = ps, .vs_entry = vs_entry, .ps_entry = entry, .color = rtv,
                              .blend = blend >= 0 ? PassBlend::Factor : PassBlend::None});
    return pso;
}

bool VulkanBackend::Pass(const char* hlsl, const char* vs_entry, const char* entry, VkFormat rtv, Image* rh,
                         uint32_t w, uint32_t h, Image* const src[3], const float p[8], float blend) {
    VkPipeline pso = EnsurePassPipeline(hlsl, vs_entry, entry, rtv, blend);
    if (!pso) return false;
    Bindings b;
    for (int i = 0; i < 3; i++) {
        if (!src[i]) continue;
        Transition(*src[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        b.views[i] = src[i]->view;
    }
    b.cb[0] = b.cb[1] = ConstantBuffer(p, 8 * sizeof(float));
    SetAttachments(rh, nullptr);
    SetFullViewport(w, h);
    if (blend >= 0) {
        float f[4] = {blend, blend, blend, blend};
        SetBlendConstants(f);
    }
    DrawFullScreen(pso, b);
    bound_rt_ = bound_ds_ = nullptr;
    return true;
}

// a single sample copy of the target to read: while drawing to it, or of a multisampled one
Image* VulkanBackend::Feedback(Target& t) {
    uint32_t samples = Ms(t);
    // every write takes a multisampled target out of this layout: the last copy still holds
    VkImageLayout read = t.depth ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    if (t.fb && samples > 1 && t.res->layout == read) {
        Transition(*t.fb, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        return t.fb.get();
    }
    if (!t.fb) {
        constexpr VkImageUsageFlags kCommon = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                              VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        t.fb = CreateImage(t.res->format, t.w, t.h,
                           kCommon | (t.depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                                              : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT));
        if (!t.fb) return nullptr;
    }
    if (t.depth) {
        if (!DepthPass(t.res.get(), t.fb.get(), t.w, t.h)) {
            Release(std::move(t.fb));
            return nullptr;
        }
        Transition(*t.fb, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        return t.fb.get();
    }
    Transition(*t.res, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    Transition(*t.fb, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageSubresourceLayers layer{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    if (samples > 1) {
        VkImageResolve r{layer, {0, 0, 0}, layer, {0, 0, 0}, {t.w, t.h, 1}};
        vkCmdResolveImage(cl_, t.res->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, t.fb->image,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    } else {
        VkImageCopy c{layer, {0, 0, 0}, layer, {0, 0, 0}, {t.w, t.h, 1}};
        vkCmdCopyImage(cl_, t.res->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, t.fb->image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    }
    Transition(*t.fb, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return t.fb.get();
}

// what a read of the target takes: the target itself, or a multisampled one's single sample copy
Image* VulkanBackend::Flat(Target& t) { return Ms(t) > 1 ? Feedback(t) : t.res.get(); }


// the key's vertex attributes in register order, each at the location of its register: vk_shader.cpp's Adapt puts
// ATTRn at location n (glslang would number them 0 up, which a register gap shifts); the strides come with the buffers
uint32_t FillInputLayout(const PsoKey& k, PipelineParts& p) {
    uint32_t n = 0;
    uint32_t buffers = 0;
    for (uint32_t r = 0; r < 16; r++) {
        if (!((k.regs >> r) & 1)) continue;
        uint32_t e = k.elements[r];
        uint32_t buffer = (e >> 8) & 0xFF;
        p.attributes[n] = {r, buffer, VertexFormat(e & 0xFF), e >> 16};
        n++;
        buffers |= 1u << buffer;
    }
    uint32_t b = 0;
    for (uint32_t i = 0; i < 16; i++)
        if ((buffers >> i) & 1) p.bindings[b++] = {i, k.strides[i], VK_VERTEX_INPUT_RATE_VERTEX};
    p.input.vertexBindingDescriptionCount = b;
    p.input.vertexAttributeDescriptionCount = n;
    return n;
}

VkPrimitiveTopology TopologyClass(uint32_t topo) {
    switch (topo) {
        case 0: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        case 1: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        default: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
}

constexpr uint32_t kFillWireframe = 0x1B01;  // D3DFILL_WIREFRAME
constexpr uint32_t kCullCw = 0x900;          // D3DCULL_CW

void FillDepthStencil(const PsoKey& k, VkPipelineDepthStencilStateCreateInfo& ds) {
    ds.depthTestEnable = k.zen;
    ds.depthWriteEnable = k.zen && k.zwrite;
    ds.depthCompareOp = CompareOp(k.zfunc);
    ds.stencilTestEnable = k.sten;
    VkStencilOpState so{StencilOp(k.sfail), StencilOp(k.spass), StencilOp(k.szfail), CompareOp(k.sfunc), k.smask & 0xFF,
                        k.swmask & 0xFF, 0};
    ds.front = ds.back = so;
}

// the console's state: the D3D12 pipeline description's fields, as Vulkan names them
VkPipeline VulkanBackend::EnsurePipeline(const PsoKey& k, VkShaderModule vs, VkShaderModule ps) {
    // most draws take the pipeline the one before took: the key compared costs less than hashing and finding it
    if (last_pso_ && !std::memcmp(&k, &last_pso_key_, sizeof(k))) return last_pso_;
    uint64_t h = Hash(&k, sizeof(k));
    auto it = psos_.find(h);
    if (it != psos_.end()) {
        last_pso_key_ = k;
        last_pso_ = it->second;
        return it->second;
    }
    VkFormat color = k.has_rt ? VkFormat(k.rtfmt) : VK_FORMAT_UNDEFINED;
    VkFormat depth = k.has_ds ? depth_fmt_ : VK_FORMAT_UNDEFINED;
    PipelineParts p(vs, ps, layout_, color, depth, std::max(k.msaa, 1u), true);
    FillInputLayout(k, p);
    p.assembly.topology = TopologyClass(k.topo);
    if (!dynamic_state_ && k.strip && k.topo != 0)
        p.assembly.topology = k.topo == 1 ? VK_PRIMITIVE_TOPOLOGY_LINE_STRIP : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    auto& b = p.blend_att;
    b.blendEnable = k.blend;
    b.srcColorBlendFactor = Blend(k.src, false);
    b.dstColorBlendFactor = Blend(k.dst, false);
    b.colorBlendOp = BlendOp(k.op);
    b.srcAlphaBlendFactor = Blend(k.src, true);
    b.dstAlphaBlendFactor = Blend(k.dst, true);
    b.alphaBlendOp = BlendOp(k.op);
    b.colorWriteMask = k.wmask;
    p.raster.polygonMode = k.fill == kFillWireframe ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
    p.raster.cullMode = k.cull ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
    p.raster.frontFace = k.ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
    p.raster.depthBiasEnable = k.zbias != 0 || k.zslope != 0;
    p.raster.depthBiasConstantFactor = float(k.zbias);
    p.raster.depthBiasSlopeFactor = k.zslope;
    // no clipping at the near and far planes, depth clamped to the viewport's range: D3D's DepthClipEnable off
    p.raster.depthClampEnable = k.unclipped && depth_clamp_;
    FillDepthStencil(k, p.depth);
    VkPipeline pso = VK_NULL_HANDLE;
    g_perf.psos++;
    CompletePipeline(p);
    if (vkCreateGraphicsPipelines(dev_, pipe_cache_, 1, &p.info, nullptr, &pso) != VK_SUCCESS) {
        LOGE("vulkan: pipeline creation failed");
    } else if (++pipe_new_ >= kPipeCacheSave) {
        // the cache on disk is only as good as the last write: Android may end the process without a destructor
        pipe_new_ = 0;
        SavePipelineCache();
    }
    psos_[h] = pso;
    return pso;
}

TexKind TextureKind(const TextureDesc& d) {
    if (d.cube) return kTexCube;
    if (d.depth > 1) return kTex3D;
    return kTex2D;
}

constexpr float kUpscaledMin = 1.01f;  // host texels per console texel past rounding: the target is upscaled

Image* VulkanBackend::BindTargetTexture(const State& s, int stage, const TextureDesc& d, Target& t, bool box,
                                        DrawTextures& stages) {
    TexInfo& info = stages.info[stage];
    if (&t == draw_ds_) return nullptr;
    bool feedback = &t == draw_rt_;
    stages.rt[stage] = true;
    stages.gw[stage] = t.depth ? 0 : t.gw;
    info.flags |= kTexTarget;
    if (!t.depth && LatchCpuWrite(t)) {
        Surface su;
        su.header = s.textures[size_t(stage)];
        UploadTarget(t, su);
    }
    info.kind = kTex2D;
    // a pass shrinking it averages each tap over a console texel, else upscaled taps skip texels and ghost
    Target* rt = FindTarget(s.rt, false);
    float k = float(t.w) / (t.gw * stages.k[stage]);
    // a blur between the small targets reads texel for texel, whatever their sizes
    bool blur = rt && t.gw < guest_w_ && rt->gw < guest_w_;
    if (!t.depth && box && k > kUpscaledMin && rt && rt != &t && (blur || t.gw > rt->gw || t.gh > rt->gh)) {
        stages.wide[stage] = k;
        stages.blur[stage] = blur;
        info.flags |= kTexWide;
    }
    // the console reads the pixels it is about to overwrite: give the draw a copy
    if (feedback) return Feedback(t);
    if (t.depth) info.shadow = ShadowDepthKind(d.format);
    info.kind = kTex2D;
    Image* r = Flat(t);
    if (r) Transition(*r, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return r;
}

bool SameTexture(const Tex& tx, const TextureDesc& d) {
    return tx.res && tx.d.format == d.format && tx.d.width == d.width && tx.d.height == d.height &&
           tx.d.depth == d.depth && tx.d.levels == d.levels && tx.d.cube == d.cube && tx.d.pitch == d.pitch;
}

// false when the texture could not be made, its entry dropped
bool VulkanBackend::CreateTexture(Tex& tx, const TextureDesc& d, const FormatInfo& fi, uint64_t key) {
    g_perf.newtex++;
    Release(std::move(tx.res));
    tx = Tex{};
    tx.d = d;
    tx.fmt = HostFormat(fi.host, fi.depth);
    bool bc = fi.compressed;
    if (BlockBytes(tx.fmt) && !bc_) {
        tx.decode = fi.host;
        tx.fmt = kBgra;
    }
    uint32_t w = d.width;
    uint32_t h = d.height;
    tx.levels = d.levels;
    if (bc && ((w & 3) || (h & 3))) {
        w = (w + 3) & ~3u;
        h = (h + 3) & ~3u;
        tx.levels = 1;
    }
    // Anisotropic filtering needs levels to filter across, and the title's ground textures have one: seen at a
    // grazing angle in the distance they alias and smear. With the option on, a single level 2D colour texture
    // (any P8 one, decoded to BGRA) gets a full chain made from it on the GPU.
    constexpr uint32_t kMinMipSize = 8;
    if (app::Get().aniso && tx.levels == 1 && !d.cube && d.depth == 1 && !fi.depth && tx.fmt == kBgra &&
        std::max(w, h) >= kMinMipSize) {
        tx.gen_mips = true;
        tx.levels = uint32_t(std::bit_width(std::max(w, h)));
    }
    tx.res = CreateImage(tx.fmt, w, h, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                         1, tx.levels, d.cube ? 1 : d.depth, d.cube);
    if (!tx.res) {
        textures_.erase(key);
        return false;
    }
    return true;
}

// one level of one face from the decoded rows, missing rows zero
void VulkanBackend::UploadTextureLevel(Tex& tx, const TextureDesc& d, uint32_t f, uint32_t l,
                                       const std::vector<uint8_t>& px, uint32_t pitch) {
    const Image& img = *tx.res;
    uint32_t w = std::max(img.w >> l, 1u);
    uint32_t h = std::max(img.h >> l, 1u);
    uint32_t slices = d.cube ? 1 : std::max(img.depth >> l, 1u);
    uint32_t block = BlockBytes(tx.fmt);
    uint32_t rows = block ? (h + 3) / 4 : h;
    uint64_t row_size = block ? uint64_t((w + 3) / 4) * block : uint64_t(w) * TexelBytes(tx.fmt);
    Alloc a = Upload(row_size * rows * slices, 16);
    size_t src_rows = pitch ? px.size() / pitch / std::max(slices, 1u) : 0;
    for (uint32_t z = 0; z < slices; z++) {
        for (uint32_t y = 0; y < rows; y++) {
            uint8_t* dst = a.cpu + (size_t(z) * rows + y) * row_size;
            if (y < src_rows) {
                std::memcpy(dst, &px[(size_t(z) * src_rows + y) * pitch], std::min<size_t>(pitch, row_size));
            } else {
                std::memset(dst, 0, size_t(row_size));
            }
        }
    }
    VkBufferImageCopy r{};
    r.bufferOffset = a.off;
    r.imageSubresource = {img.aspect, l, f, 1};
    r.imageExtent = {w, h, slices};
    vkCmdCopyBufferToImage(cl_, a.buffer, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
}

void VulkanBackend::UploadTexture(Tex& tx, const TextureDesc& d, const uint8_t* p, const uint32_t* colours) {
    Transition(*tx.res, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    uint32_t faces = d.cube ? 6 : 1;
    std::vector<uint8_t> px, decoded;
    for (uint32_t f = 0; f < faces; f++) {
        for (uint32_t l = 0; l < (tx.gen_mips ? 1 : tx.levels); l++) {
            uint32_t pitch;
            DecodeLevel(d, l, f, p, colours, px, pitch);
            if (tx.decode != Host::None) {
                const Image& img = *tx.res;
                uint32_t w = std::max(img.w >> l, 1u), h = std::max(img.h >> l, 1u);
                uint32_t slices = d.cube ? 1 : std::max(img.depth >> l, 1u);
                DecodeDxt(tx.decode, px, pitch, w, h, slices, decoded);
                std::swap(px, decoded);
                pitch = w * 4;
            }
            UploadTextureLevel(tx, d, f, l, px, pitch);
        }
    }
    if (tx.gen_mips) GenerateMips(*tx.res);
    Transition(*tx.res, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

// each level halved from the one above with a linear blit; the image comes in with level 0 written and every level
// TRANSFER_DST, and leaves that way so Transition moves them together
void VulkanBackend::GenerateMips(Image& img) {
    EndRendering();
    auto barrier = [&](uint32_t level, VkImageLayout from, VkImageLayout to) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        b.oldLayout = from;
        b.newLayout = to;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img.image;
        b.subresourceRange = {img.aspect, level, 1, 0, img.layers};
        vkCmdPipelineBarrier(cl_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &b);
    };
    for (uint32_t l = 1; l < img.levels; l++) {
        barrier(l - 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkImageBlit r{};
        r.srcSubresource = {img.aspect, l - 1, 0, img.layers};
        r.srcOffsets[1] = {int32_t(std::max(img.w >> (l - 1), 1u)), int32_t(std::max(img.h >> (l - 1), 1u)), 1};
        r.dstSubresource = {img.aspect, l, 0, img.layers};
        r.dstOffsets[1] = {int32_t(std::max(img.w >> l, 1u)), int32_t(std::max(img.h >> l, 1u)), 1};
        vkCmdBlitImage(cl_, img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r,
                       VK_FILTER_LINEAR);
        barrier(l - 1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    }
}

// a P8 texture's palette: its colours, their address (part of the texture's key) and their hash, all 0 without one
struct StagePalette {
    uint32_t colours[256] = {};
    uint32_t va = 0;
    uint64_t hash = 0;
};

// a P8 texture is decoded with the stage's palette: 256 colours, fewer by the size code in the top two bits
void ReadStagePalette(const State& s, int stage, const TextureDesc& d, StagePalette& palette) {
    constexpr uint32_t kFmtP8 = 0x0B;          // D3DFMT_P8
    constexpr uint32_t kPaletteCommon = 0;     // D3DPalette: Common, its top two bits the size code
    constexpr uint32_t kPaletteData = 4;       // D3DPalette: Data, physical address of the colours
    constexpr uint32_t kPaletteSizeShift = 30;
    if (d.format != kFmtP8 || !s.palettes[size_t(stage)]) return;
    uint32_t header = s.palettes[size_t(stage)];
    palette.va = mem::VirtualFromPhysical(kernel::R32(header + kPaletteData));
    uint32_t size_code = kernel::R32(header + kPaletteCommon) >> kPaletteSizeShift;
    size_t bytes = (std::size(palette.colours) >> size_code) * sizeof(uint32_t);
    std::memcpy(palette.colours, kernel::HostPtr(palette.va), bytes);
    palette.hash = Hash(palette.colours, sizeof(palette.colours));
}

// a texture in guest memory: made again when its layout changed, uploaded again when its bytes did
Image* VulkanBackend::BindGuestTexture(const State& s, int stage, const TextureDesc& d, const FormatInfo& fi,
                                       DrawTextures& stages) {
    TexInfo& info = stages.info[stage];
    StagePalette palette;
    ReadStagePalette(s, stage, d, palette);
    const uint64_t key = uint64_t(palette.va) << 32 | d.data;
    Tex& tx = textures_[key];
    uint32_t total = TotalSize(d);
    bool same = SameTexture(tx, d);
    if (!same && !CreateTexture(tx, d, fi, key)) return nullptr;
    // cheap change check once per frame: lock epochs plus a sample of the bytes
    const uint32_t epoch = g_texture_epoch.load(std::memory_order_acquire);
    if (tx.frame != presents_ + 1 || !same || tx.palette != palette.hash || tx.epoch != epoch) {
        tx.frame = presents_ + 1;
        tx.epoch = epoch;
        tx.palette = palette.hash;
        const uint8_t* p = static_cast<const uint8_t*>(kernel::HostPtr(d.data));
        auto perf_t = PerfClock::now();
        uint64_t sig = MemSig(d.data, total) ^ DirtyEpoch(d.data) ^ palette.hash;
        g_perf.sig += PerfMs(perf_t);
        if (sig != tx.sig || !same) {
            PerfScope perf_up{g_perf.texup};
            g_perf.texups++;
            g_perf.texup_bytes += total;
            tx.sig = sig;
            UploadTexture(tx, d, p, palette.va ? palette.colours : nullptr);
        }
    }
    stages.levels[stage] = tx.levels;
    stages.gen_mips[stage] = tx.gen_mips;
    if (fi.depth) info.shadow = ShadowDepthKind(d.format);
    Transition(*tx.res, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return tx.res.get();
}

Image* VulkanBackend::BindTexture(const State& s, int stage, DrawTextures& stages) {
    TexInfo& info = stages.info[stage];
    info = {};
    stages.rt[stage] = false;
    stages.gw[stage] = 0;
    stages.k[stage] = 1;
    stages.wide[stage] = 0;
    stages.levels[stage] = 1;
    stages.gen_mips[stage] = false;
    TextureDesc d;
    if (!s.textures[size_t(stage)] || !Describe(s.textures[size_t(stage)], d)) return nullptr;
    const FormatInfo& fi = Format(d.format);
    info.kind = TextureKind(d);
    info.rect = d.linear;

    // rendered by the GPU: sample the target itself; DXT data loaded over a target's memory is not it
    auto tit = targets_.find(TargetMapKey(d.data, fi.depth));
    if (tit != targets_.end() && !d.cube && d.depth == 1 && !fi.compressed) {
        return BindTargetTexture(s, stage, d, tit->second, true, stages);
    }
    return BindGuestTexture(s, stage, d, fi, stages);
}

// the format can be sampled with a linear filter
bool VulkanBackend::Linear(VkFormat f) {
    auto it = linear_.find(f);
    if (it != linear_.end()) return it->second;
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(phys_, f, &fp);
    bool on = fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    return linear_[f] = on;
}

// D3DTEXF: 1 point, 2 linear, 3 anisotropic
constexpr uint32_t kTexfLinear = 2;
constexpr uint32_t kTexfAnisotropic = 3;

// the stage's sampler as D3D12's StageSampler describes it
void VulkanBackend::StageSamplers(const State& s, const DrawTextures& stages, bool clamp, VkSampler (&out)[4]) {
    uint32_t aniso = app::Get().aniso;
    for (int i = 0; i < 4; i++) {
        VkSamplerCreateInfo sd{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        const auto& t = s.tss[size_t(i)];
        uint32_t mag = t[kTssMagFilter];
        uint32_t min = t[kTssMinFilter];
        uint32_t mip = t[kTssMipFilter];
        if (stages.gen_mips[i] && min >= kTexfLinear) mip = kTexfLinear;
        auto lin = [](uint32_t f) { return f >= kTexfLinear; };
        sd.minFilter = lin(min) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        sd.magFilter = lin(mag) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        sd.mipmapMode = mip == kTexfLinear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
        bool anisotropic = min == kTexfAnisotropic || mag == kTexfAnisotropic;
        uint32_t an = anisotropic ? std::max<uint32_t>(t[kTssMaxAnisotropy], 1) : 1;
        // forced only where the title filters across mip levels and the texture has them
        bool mipped = mip && stages.levels[i] > 1;
        if (aniso && lin(min) && mipped && !stages.info[i].shadow) an = aniso;
        if (an > 1 && max_aniso_ > 1) {
            // D3D12's anisotropic filter is linear all the way
            sd.minFilter = sd.magFilter = VK_FILTER_LINEAR;
            sd.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            sd.anisotropyEnable = VK_TRUE;
            sd.maxAnisotropy = std::min(float(std::min<uint32_t>(an, 16)), max_aniso_);
        }
        // clamp: the draw's coordinates all lie in 0..1 (UnitTexcoords), so wrap is read as clamp
        uint32_t u = t[kTssAddressU];
        uint32_t v = t[kTssAddressV];
        sd.addressModeU = Address(clamp && u == kTaddressWrap ? kTaddressClamp : u);
        sd.addressModeV = Address(clamp && v == kTaddressWrap ? kTaddressClamp : v);
        sd.addressModeW = Address(t[kTssAddressW]);
        sd.mipLodBias = Bits(t[kTssMipLodBias]);
        sd.minLod = float(t[kTssMaxMipLevel]);
        sd.maxLod = mip ? VK_LOD_CLAMP_NONE : sd.minLod;
        float bias_limit = props_.limits.maxSamplerLodBias;
        sd.mipLodBias = std::clamp(sd.mipLodBias, -bias_limit, bias_limit);
        sd.minLod = std::min(sd.minLod, float(stages.levels[i] - 1));
        if (!mip) sd.maxLod = sd.minLod;
        if (stages.info[i].shadow) {
            // the console tests texel OP r, Vulkan tests r OP texel
            static const uint8_t swap[8] = {0, 4, 2, 6, 1, 5, 3, 7};
            sd.minFilter = sd.magFilter = lin(min) || lin(mag) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
            sd.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sd.compareEnable = VK_TRUE;
            sd.compareOp = VkCompareOp(swap[Cmp(s.rs[kRsShadowFunc]) - 1]);
            sd.anisotropyEnable = VK_FALSE;
            sd.maxAnisotropy = 1;
        }
        // R32F and depth have no linear filtering on many phones (Black-Android): nearest there, not undefined reads
        if (stages.res[i] && !Linear(stages.res[i]->format)) {
            sd.minFilter = sd.magFilter = VK_FILTER_NEAREST;
            sd.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sd.anisotropyEnable = VK_FALSE;
            sd.maxAnisotropy = 1;
        }
        bool bordered = sd.addressModeU == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
                        sd.addressModeV == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
                        sd.addressModeW == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        VkSamplerCustomBorderColorCreateInfoEXT border{VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT};
        float rgba[4];
        Color(t[kTssBorderColor], rgba);
        if (bordered && custom_border_) {
            sd.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
            std::memcpy(border.customBorderColor.float32, rgba, sizeof(rgba));
            out[i] = EnsureSampler(sd, &border);
            continue;
        }
        // without custom border colours the nearest of the three Vulkan has
        bool white = rgba[0] + rgba[1] + rgba[2] > 1.5f;
        sd.borderColor = rgba[3] < 0.5f ? VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK
                         : white        ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE
                                        : VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        out[i] = EnsureSampler(sd, nullptr);
    }
}

uint32_t SimpleState(const State& s, uint32_t method) {
    return s.simple[method >> 2];
}

constexpr uint32_t kQuadVertices = 6;  // a quad drawn as two triangles

DrawShaders ClassifyShaders(const State& s, const Assembled& a) {
    DrawShaders sh;
    // a shader made from a declaration alone (no program) is the fixed function pipeline on those inputs
    sh.prog = (s.vs_handle & 1) && s.vs && !s.vs->program.empty();
    sh.fixed_transform = !sh.prog && !a.pretransformed;
    if (!sh.prog) sh.fk = MakeFixedKey(s);
    return sh;
}

// the program's hash is cached in the guest shader object
void EnsureProgramHash(const State& s) {
    if (!s.vs->hash) s.vs->hash = Hash(s.vs->program.data(), s.vs->program.size() * 4) | 1;
}

// vertex shader cache key salts
constexpr uint64_t kDeclarationSalt = 0x10000;    // fixed function seed, above the 16 bit register mask
constexpr uint64_t kPretransformedSalt = 0x5A5A;  // screen space fixed function vertices, apart from untransformed

uint64_t VertexKey(const State& s, const Assembled& a, const DrawShaders& sh) {
    uint64_t seed = sh.prog ? s.vs->hash ^ a.regs : Hash(&s.vs_handle, 4, a.regs | kDeclarationSalt);
    uint64_t vkey = Hash(&a.decode, 8, seed);
    if (!sh.prog && a.pretransformed) vkey ^= kPretransformedSalt;
    if (!sh.prog) vkey = Hash(&sh.fk, sizeof(sh.fk), vkey);
    return vkey;
}

std::string VertexHlsl(const State& s, const Assembled& a, const DrawShaders& sh) {
    if (sh.prog) return VertexProgram(*s.vs, a.regs, a.decode);
    return FixedVertexHlsl(a.regs, a.pretransformed, a.decode, sh.fk);
}

// sets sh.vh and sh.vs_code, compiling the shader on a cache miss
void VulkanBackend::EnsureVertexShader(const State& s, const Assembled& a, DrawShaders& sh) {
    auto vit = shader_src_.find(sh.vkey);
    if (vit != shader_src_.end()) {
        sh.vh = vit->second;
        sh.vs_code = blobs_[sh.vh];
        return;
    }
    sh.vs_code = Compile(VertexHlsl(s, a, sh), "main", "vs_5_0", sh.vh);
    shader_src_[sh.vkey] = sh.vh;
}

// textures may have recorded copies that unbound the targets or moved them out of their attachment layouts
bool VulkanBackend::BindStageTexturesAndTargets(const State& s, DrawTextures& stages) {
    auto perf_t = PerfClock::now();
    draw_rt_ = bound_rt_;
    draw_ds_ = bound_ds_;
    for (int i = 0; i < 4; i++) stages.res[i] = BindTexture(s, i, stages);
    g_perf.bind += PerfMs(perf_t);
    if (bound_rt_ != draw_rt_ || bound_ds_ != draw_ds_) return BindTargets(s);
    SetAttachments(bound_rt_ ? bound_rt_->res.get() : nullptr, bound_ds_ ? bound_ds_->res.get() : nullptr);
    return true;
}

constexpr uint64_t kPixelKeySalt = 0x9E3779B97F4A7C15ull;
constexpr uint64_t kNearDepthSalt = 0xC2B2AE3D27D4EB4Full;

// c-38, the viewport scale; its w the plane d3d8.cpp NearDepth set, 0 without
constexpr uint32_t kViewportScaleRow = 58;

bool NearDepth(const State& s) { return s.vsc[kViewportScaleRow][3] > 0; }

// The pixel shader writes the depth. Past the near plane it is the rasterizer's, squeezed behind a small front share
// of the range; nearer, the parts the plane would clip fill that share linearly in view depth (pos.w, which a
// Vulkan fragment position holds as its reciprocal), in order
std::string NearDepthHlsl(std::string src) {
    const std::string head = "float4 main(VOut i, bool front : SV_IsFrontFace) : SV_Target {\n";
    const std::string with_depth =
        "float4 main(VOut i, bool front : SV_IsFrontFace, out float depth : SV_Depth) : SV_Target {\n"
        "    const float front_share = 1.0 / 4096;\n"
        "    float plane = c[" + std::to_string(kViewportScaleRow) + "].w;\n"
        "    float past = front_share + (1 - front_share) * i.pos.z;\n"
        "    float view_w = 1.0 / i.pos.w;\n"
        "    depth = view_w < plane ? front_share * view_w / plane : past;\n";
    size_t at = src.find(head);
    if (at != std::string::npos) src.replace(at, head.size(), with_depth);
    return src;
}

// sets sh.ph and sh.ps_code, compiling the shader on a cache miss
void VulkanBackend::EnsurePixelShader(const PixelKey& pk, DrawShaders& sh) {
    // pixel shaders share the map with vertex shaders, the salt keeps their keys apart
    auto pit = shader_src_.find(sh.pkey ^ kPixelKeySalt);
    if (pit != shader_src_.end()) {
        sh.ph = pit->second;
        sh.ps_code = blobs_[sh.ph];
        return;
    }
    std::string src = PixelHlsl(pk);
    if (sh.near_depth) src = NearDepthHlsl(std::move(src));
    sh.ps_code = Compile(src, "main", "ps_5_0", sh.ph);
    shader_src_[sh.pkey ^ kPixelKeySalt] = sh.ph;
}

void KeyBlend(const State& s, PsoKey& k) {
    k.blend = SimpleState(s, kMBlendEnable) != 0;
    if (k.blend) {
        k.src = SimpleState(s, kMSrcBlend);
        k.dst = SimpleState(s, kMDestBlend);
        k.op = SimpleState(s, kMBlendOp);
    }
    uint32_t cw = SimpleState(s, kMColorWriteEnable);
    // console ARGB bytes to D3D12 RGBA bits
    k.wmask = ((cw >> 16) & 1) | ((cw >> 8) & 1) << 1 | (cw & 1) << 2 | ((cw >> 24) & 1) << 3;
}

constexpr float kDepth24Max = 16777215.0f;  // 2^24 minus 1: a 24 bit depth buffer's units per 1.0

// stencil bit 7 marks a blended or alpha tested surface for the occlusion pass; the game never tests stencil
constexpr uint32_t kAoStencilBit = 0x80;
constexpr uint32_t kStencilAlways = 0x207;    // D3DCMP_ALWAYS
constexpr uint32_t kStencilKeep = 0x1E00;     // D3DSTENCILOP_KEEP
constexpr uint32_t kStencilReplace = 0x1E01;  // D3DSTENCILOP_REPLACE
constexpr uint32_t kAlphaKillEnable = 4;      // D3DTALPHAKILL_ENABLE

bool VulkanBackend::MarksOcclusion(const PsoKey& k) const {
    if (ao_frame_ == 0 || ao_frame_ != presents_ || k.sten || !k.zen || !k.zwrite) return false;
    return bound_ds_ && bound_ds_->gw == guest_w_ && bound_ds_->gh == guest_h_;
}

void KeyOcclusionMark(PsoKey& k) {
    k.ao_mark = 1;
    k.sten = 1;
    k.sfunc = kStencilAlways;
    k.swmask = kAoStencilBit;
    k.sfail = kStencilKeep;
    k.szfail = kStencilKeep;
    k.spass = kStencilReplace;
}

bool Translucent(const State& s, const PsoKey& k) {
    if (k.blend || SimpleState(s, kMAlphaTestEnable)) return true;
    for (int i = 0; i < 4; i++) {
        if (s.textures[i] && s.tss[size_t(i)][kTssAlphaKill] == kAlphaKillEnable) return true;
    }
    return false;
}

void VulkanBackend::KeyDepthStencil(const State& s, const Assembled& a, PsoKey& k) {
    k.zen = s.rs[kRsZEnable] != 0;
    if (k.zen) {
        k.zwrite = SimpleState(s, kMZWriteEnable) != 0;
        k.zfunc = SimpleState(s, kMZFunc);
    }
    k.sten = s.rs[kRsStencilEnable] != 0;
    if (k.sten) {
        k.sfunc = SimpleState(s, kMStencilFunc);
        k.smask = SimpleState(s, kMStencilMask) & 0xFF;
        k.swmask = SimpleState(s, kMStencilWriteMask) & 0xFF;
        k.sfail = s.rs[kRsStencilFail];
        k.szfail = SimpleState(s, kMStencilZFail);
        k.spass = SimpleState(s, kMStencilPass);
    }
    if (MarksOcclusion(k)) KeyOcclusionMark(k);
    // polygon offset: bias in depth buffer units, the target is always 24 bit; the slope is per console pixel,
    // and an upscaled target's pixels are smaller
    if (a.topology == Topology::Triangles && SimpleState(s, kMSolidOffsetEnable)) {
        k.zbias = int32_t(std::lround(Bits(SimpleState(s, kMDepthOffset)) * kDepth24Max / s.zscale));
        k.zslope = Bits(SimpleState(s, kMSlopeScale)) * (bound_ds_ ? float(bound_ds_->w) / bound_ds_->gw : 1.0f);
    }
}

void KeyRasterizer(const State& s, const Assembled& a, PsoKey& k) {
    uint32_t cull = s.rs[kRsCullMode];
    k.cull = cull != 0 && a.topology == Topology::Triangles;
    k.ccw = cull == kCullCw;  // culling clockwise leaves counter clockwise as the front
    k.fill = s.rs[kRsFillMode];
    // the console's near and far culling off (SET_ZMIN_MAX_CONTROL bit 0 clear): Dynasty Warriors 5 draws its ground
    // and sky dome past both planes, which clipping left as holes
    constexpr uint32_t kCullNearFar = 1;
    k.unclipped = NearDepth(s) || !(SimpleState(s, kMDepthClipControl) & kCullNearFar);
}

void VulkanBackend::BuildPipelineKey(const State& s, const Assembled& a, uint64_t vh, uint64_t ph, PsoKey& k) {
    std::memset(&k, 0, sizeof(k));
    k.vs = vh;
    k.ps = ph;
    k.regs = a.regs;
    for (int r = 0; r < 16; r++) {
        if ((a.regs >> r) & 1) {
            const VertexElement& e = a.elements[size_t(r)];
            k.elements[r] = e.format | uint32_t(e.buffer) << 8 | uint32_t(e.offset) << 16;
        }
    }
    k.topo = uint32_t(a.topology);
    if (!dynamic_state_) {
        k.strip = a.strip;
        for (uint32_t i = 0; i < a.buffer_count; i++) k.strides[i] = a.buffers[i].stride;
    }
    k.has_rt = bound_rt_ != nullptr;
    k.has_ds = bound_ds_ != nullptr;
    k.rtfmt = bound_rt_ ? bound_rt_->fmt : 0;
    uint32_t samples = Ms(bound_rt_ ? *bound_rt_ : *bound_ds_);
    k.msaa = samples > 1 ? samples : 0;
    KeyBlend(s, k);
    if (k.has_ds) KeyDepthStencil(s, a, k);
    KeyRasterizer(s, a, k);
}


// alpha test off: compare always, in Cmp() minus 1 terms like the enabled test
constexpr float kAlphaTestAlways = 7;

// FOG off leaves the constants as fog disabled: the shaders' fog factor is then 1, the colour as drawn
void FillFogConstants(const State& s, DrawConstants& c) {
    if (!s.rs[kRsFogEnable] || !game::Opt().fog) return;
    c.fog[0] = Bits(s.rs[kRsFogStart]);
    c.fog[1] = Bits(s.rs[kRsFogEnd]);
    c.fog[2] = Bits(s.rs[kRsFogDensity]);
    c.fog[3] = float(s.rs[kRsFogTableMode]);
}

void FillSurfaceConstants(const State& s, GuestExtent size, DrawConstants& c) {
    c.surf[0] = float(size.w);
    c.surf[1] = float(size.h);
    c.surf[2] = 1.0f / s.zscale;
    Color(s.rs[kRsFogColor], c.fog_color);
    c.misc[0] = float(SimpleState(s, kMAlphaRef) & 0xFF);
    c.misc[1] = SimpleState(s, kMAlphaTestEnable) ? float(Cmp(SimpleState(s, kMAlphaFunc)) - 1) : kAlphaTestAlways;
    c.misc[2] = float(Cmp(s.rs[kRsShadowFunc]) - 1);
    c.misc[3] = s.rs[kRsSpecularEnable] ? 1.0f : 0.0f;
}

void FillCombinerConstants(const State& s, DrawConstants& c) {
    for (int i = 0; i < 8; i++) {
        Color(SimpleState(s, kMPsConstant0 + i * 4), c.pc[i * 2]);
        Color(SimpleState(s, kMPsConstant1 + i * 4), c.pc[i * 2 + 1]);
    }
    Color(SimpleState(s, kMPsFinalConstant0), c.pc[16]);
    Color(SimpleState(s, kMPsFinalConstant1), c.pc[17]);
    Color(s.rs[kRsTextureFactor], c.tfactor);
}

constexpr float kGlowBoxTexels = 3.0f;  // console texels across the box each wide tap averages
constexpr float kBlurTapTexels = 7.0f / 6;  // console texels between a blur's taps

void VulkanBackend::FillTexelConstants(int i, const TextureDesc& d, const DrawTextures& stages, DrawConstants& c) {
    c.texsize[i][0] = float(d.width);
    c.texsize[i][1] = float(d.height);
    if (!stages.rt[i]) return;
    float k = stages.k[i];
    // a 3 console texel box per tap keeps the console chain's glow width and brightness; a blur's taps
    // are 7/6 texel apart, so each covers that much, a sample per host texel
    if (stages.wide[i] > 0) {
        bool blur = stages.blur[i];
        float fw = blur ? kBlurTapTexels : kGlowBoxTexels;
        float per_texel = blur ? 1.0f : 0.5f;
        float n = std::ceil(fw * stages.wide[i] * per_texel);
        c.spread[i][0] = fw / (n * float(d.width) * k);
        c.spread[i][1] = fw / (n * float(d.height) * k);
        c.spread[i][2] = n;
    }
}

// A pretransformed quad's stage 0 reads held to its own texels. The console samples a quad drawn texel for texel
// from its first texel centre to its last, one texel short of the far coordinates; drawn larger here, the bilinear
// reads past that take in the neighbouring cell of an atlas (the font's next row and column). Across, the reads go to
// the centre of the last texel the far u reaches instead, still none of the next cell: a texel short cut the last
// column, the outline, off Dynasty Warriors 5's menu words. Any other draw is left alone.
void UvClamp(const State& s, const DrawTextures& stages, DrawConstants& c) {
    constexpr float kOpen = 1e30f;
    c.uvclamp[0] = c.uvclamp[1] = -kOpen;
    c.uvclamp[2] = c.uvclamp[3] = kOpen;
    TextureDesc d;
    if (!s.quad_uv_set || stages.rt[0] || stages.info[0].rect || stages.info[0].kind != kTex2D || !s.textures[0] ||
        !Describe(s.textures[0], d) || !d.width || !d.height)
        return;
    const auto& q = s.quad_uv;
    const float w = float(d.width);
    c.uvclamp[0] = q[0];
    c.uvclamp[1] = q[1];
    c.uvclamp[2] = std::max(q[0], (std::ceil(q[2] * w - 1e-3f) - 0.5f) / w);
    c.uvclamp[3] = std::max(q[1], q[3] - 1.0f / float(d.height));
}

void VulkanBackend::FillTextureConstants(const State& s, const DrawShaders& sh, const DrawTextures& stages,
                                        DrawConstants& c) {
    for (int i = 0; i < 4; i++) {
        TextureDesc d;
        if (s.textures[size_t(i)] && Describe(s.textures[size_t(i)], d)) FillTexelConstants(i, d, stages, c);
        if (stages.rt[i]) c.rtshift[i] = RenderTargetShift(s, sh);
        // a pretransformed draw's reads half a console pixel up and left, a target's on top of its own shift
        if (stages.info[i].flags & kTexShift) c.rtshift[i] += -0.5f * TargetScale();
        const auto& t = s.tss[size_t(i)];
        c.bump[i][0] = Bits(t[kTssBumpEnvMat00]);
        c.bump[i][1] = Bits(t[kTssBumpEnvMat01]);
        c.bump[i][2] = Bits(t[kTssBumpEnvMat10]);
        c.bump[i][3] = Bits(t[kTssBumpEnvMat11]);
        c.bumpl[i][0] = Bits(t[kTssBumpEnvLScale]);
        c.bumpl[i][1] = Bits(t[kTssBumpEnvLOffset]);
    }
    UvClamp(s, stages, c);
}

void MatMul(const float* x, const float* y, float* o) {
    for (int r = 0; r < 4; r++) {
        for (int col = 0; col < 4; col++) {
            o[r * 4 + col] = x[r * 4] * y[col] + x[r * 4 + 1] * y[4 + col] + x[r * 4 + 2] * y[8 + col] +
                             x[r * 4 + 3] * y[12 + col];
        }
    }
}

// HLSL mul(row vector, matrix) with column major packing wants the transpose
void Transpose(const float* m, float* o) {
    for (int r = 0; r < 4; r++) {
        for (int col = 0; col < 4; col++) o[col * 4 + r] = m[r * 4 + col];
    }
}

void InverseTranspose3x3(const float* wv, float* n) {
    float det = wv[0] * (wv[5] * wv[10] - wv[6] * wv[9]) - wv[1] * (wv[4] * wv[10] - wv[6] * wv[8]) +
                wv[2] * (wv[4] * wv[9] - wv[5] * wv[8]);
    float id = det != 0 ? 1.0f / det : 0.0f;
    for (int r = 0; r < 3; r++) {
        for (int col = 0; col < 3; col++) {
            int r1 = (r + 1) % 3;
            int r2 = (r + 2) % 3;
            int c1 = (col + 1) % 3;
            int c2 = (col + 2) % 3;
            n[r * 4 + col] = (wv[r1 * 4 + c1] * wv[r2 * 4 + c2] - wv[r1 * 4 + c2] * wv[r2 * 4 + c1]) * id;
        }
    }
}

// D3DTRANSFORMSTATETYPE indices into s.transforms
constexpr size_t kTsView = 2;
constexpr size_t kTsProjection = 3;
constexpr size_t kTsTexture0 = 16;
constexpr size_t kTsWorld = 256;

void FillTransformConstants(const State& s, GuestExtent size, DrawConstants& c) {
    uint32_t gw = size.w;
    uint32_t gh = size.h;
    // world, view, projection, then the viewport inside the target
    float wv[16];
    float wvp[16];
    MatMul(s.transforms[kTsWorld].data(), s.transforms[kTsView].data(), wv);
    MatMul(wv, s.transforms[kTsProjection].data(), wvp);
    const auto& v = s.viewport;
    float sx = float(v.w) / gw;
    float sy = float(v.h) / gh;
    float ox = (2.0f * v.x + v.w) / gw - 1;
    float oy = 1 - (2.0f * v.y + v.h) / gh;
    float vpm[16] = {sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, v.max_z - v.min_z, 0, ox, oy, v.min_z, 1};
    float full[16];
    MatMul(wvp, vpm, full);
    Transpose(full, c.wvp);
    Transpose(wv, c.wv);
    // normals take the inverse transpose of the upper 3x3
    float n[16] = {};
    InverseTranspose3x3(wv, n);
    Transpose(n, c.wvn);
    for (int t = 0; t < 4; t++) Transpose(s.transforms[kTsTexture0 + size_t(t)].data(), c.tm[t]);
}

// D3DLIGHT8 as floats: Type, Diffuse[4], Specular[4], Ambient[4], Position[3], Direction[3], Range, Falloff,
// Attenuation0 to 2, Theta, Phi
constexpr int kLightFloats = 26;
constexpr int kLightDiffuse = 1;
constexpr int kLightSpecular = 5;
constexpr int kLightAmbient = 9;
constexpr int kLightPosition = 13;
constexpr int kLightDirection = 16;
constexpr int kLightRange = 19;
constexpr int kLightFalloff = 20;
constexpr int kLightAttenuation0 = 21;
constexpr int kLightAttenuation1 = 22;
constexpr int kLightAttenuation2 = 23;
constexpr int kLightTheta = 24;
constexpr int kLightPhi = 25;

void LightToView(const std::array<uint32_t, kLightFloats>& l, const float* vm, float (&o)[7][4]) {
    float f[kLightFloats];
    std::memcpy(f, l.data(), sizeof(f));
    for (int j = 0; j < 4; j++) {
        o[0][j] = f[kLightDiffuse + j];
        o[1][j] = f[kLightSpecular + j];
        o[2][j] = f[kLightAmbient + j];
    }
    const float* pos = f + kLightPosition;
    const float* dir = f + kLightDirection;
    for (int j = 0; j < 3; j++) {
        o[3][j] = pos[0] * vm[j] + pos[1] * vm[4 + j] + pos[2] * vm[8 + j] + vm[12 + j];
        o[4][j] = dir[0] * vm[j] + dir[1] * vm[4 + j] + dir[2] * vm[8 + j];
    }
    o[3][3] = f[kLightRange];
    o[4][3] = f[kLightFalloff];
    o[5][0] = f[kLightAttenuation0];
    o[5][1] = f[kLightAttenuation1];
    o[5][2] = f[kLightAttenuation2];
    o[6][0] = std::cos(f[kLightTheta] * 0.5f);
    o[6][1] = std::cos(f[kLightPhi] * 0.5f);
}

void FillLightConstants(const State& s, const FixedKey& fk, DrawConstants& c) {
    std::memcpy(c.mat, s.material.data(), sizeof(float) * 16);
    c.mat[4][0] = s.material[16];
    Color(s.rs[kRsAmbient], c.amb);
    // lights to view space
    const float* vm = s.transforms[kTsView].data();
    for (int i = 0; i < 8; i++) {
        if (!((fk.types >> (i * 2)) & 3)) continue;
        LightToView(s.lights[size_t(i)], vm, c.lt[i]);
    }
}

float VulkanBackend::TargetScale() const {
    const Target* t = bound_rt_ ? bound_rt_ : bound_ds_;
    if (!t || !t->gw) return 1;
    return float(t->w) / float(t->gw);
}

// the frame: the back buffer and the 512x448 scene Dynasty Warriors 5 draws before copying it there. The game's smaller
// targets (shadows, the glow and blur chain) are read back stretched to the frame and stay as drawn
bool VulkanBackend::WideDraw() const {
    const Target* t = bound_rt_ ? bound_rt_ : bound_ds_;
    return wide_ > 1 && t && t->gw * 2 > guest_w_ && t->gh * 2 > guest_h_;
}

// the XDK's screen space offset puts the console's positions half a pixel right and down of the host's, unless a
// program writes them without c-37; render target reads move back that far (TexUv)
float VulkanBackend::RenderTargetShift(const State& s, const DrawShaders& sh) const {
    bool offset = !sh.prog || ReadsViewportOffset(s.vs->program);
    return offset ? -0.5f * TargetScale() : 0.0f;
}

void VulkanBackend::UploadConstants(const State& s, const DrawShaders& sh, const PsoKey& k, const DrawTextures& stages,
                                    GuestExtent size, DrawConstants& c, Bindings& b) {
    // the vertex constants change far less often than the draws: the last upload is bound again while they are the
    // same, and with it the push descriptors are too
    if (!bound_.vsc_set || std::memcmp(s.vsc.data(), bound_.vsc.data(), sizeof(s.vsc))) {
        bound_.vsc = s.vsc;
        bound_.vsc_cb = ConstantBuffer(s.vsc.data(), sizeof(s.vsc));
        bound_.vsc_set = true;
    }
    b.cb[0] = bound_.vsc_cb;
    FillSurfaceConstants(s, size, c);
    FillFogConstants(s, c);
    FillCombinerConstants(s, c);
    FillTextureConstants(s, sh, stages, c);
    if (sh.fixed_transform) {
        FillTransformConstants(s, size, c);
        FillLightConstants(s, sh.fk, c);
    }
    if (WideDraw()) {
        c.wide[0] = wide_;
        c.wide[1] = !sh.prog && !sh.fixed_transform ? 1.0f : 0.0f;
    }
    (void)k;
    b.cb[1] = ConstantBuffer(&c, sizeof(c));
}

void VulkanBackend::CaptureTextures(const State& s, const DrawTextures& stages) {
    for (int i = 0; i < 4; i++) {
        Image* res = stages.res[i];
        TextureDesc d;
        if (!s.textures[size_t(i)] || !Describe(s.textures[size_t(i)], d)) continue;
        bool tg = targets_.count(TargetMapKey(d.data, Format(d.format).depth));
        fprintf(cap_, " | t%d %08X fmt %X %ux%u%s%s", i, d.data, d.format, d.width, d.height, tg ? " target" : "",
                res ? "" : " unbound");
        if (res && !tg && stages.info[i].kind == kTex2D && res->format == kBgra && cap_seen_.insert(d.data).second) {
            // a P8 texture is keyed with its palette's address above the data's
            for (auto& [key, tx] : textures_)
                if (uint32_t(key) == d.data && tx.res.get() == res) {
                    cap_tex_.push_back({tx.res, d.data, d.width, d.height});
                    break;
                }
        }
    }
}

// s.vsc holds the 192 vertex shader constants in hardware order: D3D register + 96
constexpr int kVsConstants = 192;
constexpr int kVsConstantBias = 96;

void VulkanBackend::CaptureInputs(const State& s, const Assembled& a, const DrawConstants& c) {
    fprintf(cap_, "\n     verts");
    uint32_t n = 0;
    for (int r = 0; r < 16; r++) n += (a.regs >> r) & 1;
    for (size_t v = 0; v < std::min<size_t>(4, n ? a.vertices.size() / (n * 4) : 0); v++) {
        fprintf(cap_, " [");
        for (uint32_t k = 0; k < n * 4; k++) fprintf(cap_, " %g", a.vertices[v * n * 4 + k]);
        fprintf(cap_, " ]");
    }
    for (int r = 0; r < kVsConstants; r++) {
        const auto& v = s.vsc[r];
        if (v[0] || v[1] || v[2] || v[3]) {
            fprintf(cap_, "\n     c%d %g %g %g %g", r - kVsConstantBias, v[0], v[1], v[2], v[3]);
        }
    }
    fprintf(cap_, "\n     pc");
    for (size_t r = 0; r < std::size(c.pc); r++) {
        fprintf(cap_, " (%.3f %.3f %.3f %.3f)", c.pc[r][0], c.pc[r][1], c.pc[r][2], c.pc[r][3]);
    }
}

// the draw's line: its state, then its textures
void VulkanBackend::CaptureDraw(const State& s, uint32_t prim, uint32_t count, const PsoKey& k, const DrawConstants& c,
                               const DrawTextures& stages) {
    fprintf(cap_,
            "%04u draw rt %08X ds %08X prim %u count %u vs %08X ps %08X blend %u %X %X zen %u %X zwrite %u zbias %d %.2f cull %u cw %X c0 %.3f %.3f %.3f %.3f",
            cap_i_++, s.rt.data, bound_ds_ ? s.zs.data : 0, prim, count, s.vs_handle, s.ps ? s.ps->handle : 0, k.blend,
            k.src, k.dst, k.zen, k.zfunc, k.zwrite, k.zbias, k.zslope, k.cull, k.wmask, c.pc[0][0], c.pc[0][1],
            c.pc[0][2], c.pc[0][3]);
    CaptureTextures(s, stages);
}

// small draws: their inputs and constants; then ends the draw's line
void VulkanBackend::EndDrawCapture(const State& s, const Assembled& a, uint32_t count, const DrawConstants& c) {
    if (count <= kQuadVertices) CaptureInputs(s, a, c);
    fputc('\n', cap_);
}

// each shader once, as translated
void VulkanBackend::CaptureShader(const char* kind, uint64_t key, const std::string& src) {
    if (!cap_sh_.insert(key).second) return;
    char name[48];
    snprintf(name, sizeof(name), "%s_%016llX.hlsl", kind, (unsigned long long)key);
    if (FILE* f = std::fopen((cap_dir_ / name).c_str(), "w")) {
        fwrite(src.data(), 1, src.size(), f);
        fclose(f);
    }
    fprintf(cap_, "     %s\n", name);
}

void VulkanBackend::CaptureShaders(const State& s, const Assembled& a, const DrawShaders& sh, const PixelKey& pk) {
    CaptureShader("vs", sh.vkey, VertexHlsl(s, a, sh));
    CaptureShader("ps", sh.pkey, PixelHlsl(pk));
}

// a stage without a texture reads a black one of its kind; the samplers read the views after
void VulkanBackend::SetTextureViewsAndSamplers(const State& s, const Assembled& a, DrawTextures& stages, Bindings& b) {
    for (int i = 0; i < 4; i++) {
        if (stages.res[i]) {
            b.views[i] = stages.res[i]->view;
            continue;
        }
        uint8_t kind = stages.info[i].kind;
        b.views[i] = dummy_[kind == kTex3D ? 1 : kind == kTexCube ? 2 : 0]->view;
    }
    StageSamplers(s, stages, AnyStageWraps(s) && UnitTexcoords(a), b.samplers);
}

// viewport covers the target, the guest viewport is in the shader; scissor clips to it. The console's viewport only
// places transformed vertices, so XYZRHW draws are left whole: Dynasty Warriors 5 draws a cutscene's black letterbox
// with the viewport still on the picture between the bars, and cut to it the bars were never drawn.
void VulkanBackend::SetViewportScissor(const State& s, uint32_t gw, bool pretransformed) {
    uint32_t W = bound_rt_ ? bound_rt_->w : bound_ds_->w;
    uint32_t H = bound_rt_ ? bound_rt_->h : bound_ds_->h;
    SetViewport(0, 0, float(W), float(H));
    float sc = float(W) / float(gw);
    // ultrawide: an edge inside the target pulled in as the 2D layer is (kVsEpilogue), the sides left where they are
    bool wide = WideDraw();
    auto gx = [&](int32_t x) {
        double mid = gw / 2.0;
        return !wide || x <= 0 || x >= int32_t(gw) ? double(x) : mid + (x - mid) / wide_;
    };
    int32_t x0 = 0, y0 = 0, x1 = int32_t(W), y1 = int32_t(H);
    if (!pretransformed) {
        x0 = HostEdge(gx(int32_t(s.viewport.x)), sc, W, false);
        y0 = HostEdge(int32_t(s.viewport.y), sc, H, false);
        x1 = HostEdge(gx(int32_t(s.viewport.x + s.viewport.w)), sc, W, true);
        y1 = HostEdge(int32_t(s.viewport.y + s.viewport.h), sc, H, true);
    }
    // a scissor rect's far corner is its last pixel (the whole screen is 0 0 511 447): taken as the edge past it, the
    // last row of a cutscene's picture went undrawn, a line of the clear colour above the bottom letterbox bar
    if (s.scissor_count) {
        x0 = std::max(x0, HostEdge(gx(int32_t(s.scissors[0])), sc, W, false));
        y0 = std::max(y0, HostEdge(int32_t(s.scissors[1]), sc, H, false));
        x1 = std::min(x1, HostEdge(gx(int32_t(s.scissors[2]) + 1), sc, W, true));
        y1 = std::min(y1, HostEdge(int32_t(s.scissors[3]) + 1, sc, H, true));
    }
    SetScissor({x0, y0, x1, y1});
}

void VulkanBackend::SetBlendFactorStencilRef(const State& s, const PsoKey& k) {
    float bf[4];
    Color(SimpleState(s, kMBlendColor), bf);
    SetBlendConstants(bf);
    uint32_t ref = SimpleState(s, kMStencilRef) & 0xFF;
    if (k.ao_mark) ref = Translucent(s, k) ? kAoStencilBit : 0;
    SetStencilReference(ref);
}

VkPrimitiveTopology PrimitiveTopology(const Assembled& a) {
    if (a.topology == Topology::Points) return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    if (a.topology == Topology::Lines) return a.strip ? VK_PRIMITIVE_TOPOLOGY_LINE_STRIP : VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    return a.strip ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
}

void VulkanBackend::DrawGeometry(const Assembled& a) {
    VkBuffer buffers[16];
    VkDeviceSize offsets[16];
    VkDeviceSize sizes[16];
    VkDeviceSize strides[16];
    for (uint32_t i = 0; i < a.buffer_count; i++) {
        const VertexBuffer& b = a.buffers[i];
        Alloc vb = Upload(std::max<uint32_t>(b.bytes, 4), 16);
        if (b.bytes) std::memcpy(vb.cpu, b.src, b.bytes);
        buffers[i] = vb.buffer;
        offsets[i] = vb.off;
        sizes[i] = std::max<uint32_t>(b.bytes, 4);
        strides[i] = b.stride;
    }
    if (dynamic_state_) {
        if (a.buffer_count) vkCmdBindVertexBuffers2(cl_, 0, a.buffer_count, buffers, offsets, sizes, strides);
        SetTopology(PrimitiveTopology(a));
    } else if (a.buffer_count) {
        vkCmdBindVertexBuffers(cl_, 0, a.buffer_count, buffers, offsets);  // strides and topology in the pipeline
    }
    if (a.index_count) {
        uint32_t ibytes = a.index_count * (a.index16 ? 2 : 4);
        Alloc ib = Upload(ibytes, 4);
        std::memcpy(ib.cpu, a.index_src, ibytes);
        vkCmdBindIndexBuffer(cl_, ib.buffer, ib.off, a.index16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
        EnsureRendering();
        vkCmdDrawIndexed(cl_, a.index_count, 1, 0, a.base_vertex, 0);
    } else {
        EnsureRendering();
        vkCmdDraw(cl_, a.draw_count, 1, 0, 0);
    }
}

// Assemble, timed and counted
bool VulkanBackend::AssembleDraw(const State& s, uint32_t prim, uint32_t start, uint32_t count,
                                const uint8_t* indices, Assembled& a) {
    auto perf_t = PerfClock::now();
    bool assembled = Assemble(s, prim, start, count, indices, a);
    g_perf.assemble += PerfMs(perf_t);
    g_perf.verts += a.vertex_count;
    g_perf.raw += a.vertices.empty();
    return assembled;
}

void VulkanBackend::Draw(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices) {
    PerfScope perf_draw{g_perf.draw};
    g_perf.draws++;
    if (cap_) CapStep(s.rt);
    if (DepthCopy(s)) return;
    Assembled& a = asm_;
    if (!AssembleDraw(s, prim, start, count, indices, a)) return;
    if (!BindTargets(s)) return;

    // vertex shader
    DrawShaders sh = ClassifyShaders(s, a);
    if (sh.prog) EnsureProgramHash(s);
    sh.vkey = VertexKey(s, a, sh);
    EnsureVertexShader(s, a, sh);

    // stage textures, then the pixel shader made for them
    DrawTextures stages;
    if (!BindStageTexturesAndTargets(s, stages)) return;
    if (a.pretransformed)
        for (auto& t : stages.info)
            if (t.kind) t.flags |= kTexShift;
    PixelKey pk = MakePixelKey(s, stages.info);
    sh.near_depth = NearDepth(s);
    sh.pkey = Hash(&pk, sizeof(pk)) ^ (sh.near_depth ? kNearDepthSalt : 0);
    EnsurePixelShader(pk, sh);
    if (!sh.vs_code || !sh.ps_code) return;

    // pipeline
    PsoKey k;
    BuildPipelineKey(s, a, sh.vh, sh.ph, k);
    VkPipeline pso = EnsurePipeline(k, sh.vs_code, sh.ps_code);
    if (!pso) return;
    BindPipeline(pso);

    // constants
    GuestExtent size;
    SurfaceSize(bound_rt_ ? s.rt : s.zs, size.w, size.h);
    DrawConstants c{};
    Bindings b;
    UploadConstants(s, sh, k, stages, size, c, b);

    if (cap_) {
        CaptureDraw(s, prim, count, k, c, stages);
        EndDrawCapture(s, a, count, c);
        CaptureShaders(s, a, sh, pk);
    }

    // texture views and samplers, viewport and scissor, blend factor and stencil reference
    SetTextureViewsAndSamplers(s, a, stages, b);
    PushBindings(b);
    SetViewportScissor(s, size.w, a.pretransformed);
    SetBlendFactorStencilRef(s, k);

    // geometry
    DrawGeometry(a);
}


// a guest rect scaled to the target and clamped to it
VkClearRect ScaledRect(const Target& t, int x0, int y0, int x1, int y1) {
    float sc = float(t.w) / float(t.gw);
    int32_t l = HostEdge(x0, sc, t.w, false);
    int32_t tp = HostEdge(y0, sc, t.h, false);
    int32_t r = HostEdge(x1, sc, t.w, true);
    int32_t b = HostEdge(y1, sc, t.h, true);
    return {{{l, tp}, {uint32_t(std::max(r - l, 0)), uint32_t(std::max(b - tp, 0))}}, 0, 1};
}

// D3DCLEAR_TARGET_R, G, B and A each clear their channel alone, which a clear attachment can't: Dynasty Warriors 5
// clears the alpha of its finished scene for the glow that follows, and clearing every channel blacked the scene out
void VulkanBackend::ClearChannels(Target& t, const float colour[4], uint32_t wmask, const std::vector<VkClearRect>& rs) {
    VkPipeline pso = EnsureClearPipeline(t.res->format, Ms(t), wmask);
    if (!pso) return;
    float c[8] = {colour[0], colour[1], colour[2], colour[3]};
    Bindings b;
    b.cb[0] = b.cb[1] = ConstantBuffer(c, sizeof(c));
    SetViewport(0, 0, float(t.w), float(t.h));
    for (const VkClearRect& r : rs) {
        const VkRect2D& rc = r.rect;
        SetScissor({rc.offset.x, rc.offset.y, rc.offset.x + int32_t(rc.extent.width), rc.offset.y + int32_t(rc.extent.height)});
        DrawFullScreen(pso, b);
    }
}

// rs: the rects buffer, reused across targets
void VulkanBackend::ClearTarget(Target& t, const State& s, const ClearArgs& c, std::vector<VkClearRect>& rs) {
    rs.clear();
    auto add = [&](VkClearRect r) {
        if (r.rect.extent.width && r.rect.extent.height) rs.push_back(r);
    };
    if (c.count && c.rects) {
        for (uint32_t i = 0; i < c.count; i++) {
            int32_t r[4];
            std::memcpy(r, c.rects + size_t(i) * sizeof(r), sizeof(r));
            add(ScaledRect(t, r[0], r[1], r[2], r[3]));
        }
    } else {
        const auto& v = s.viewport;
        add(ScaledRect(t, int(v.x), int(v.y), int(v.x + v.w), int(v.y + v.h)));
    }
    VkClearAttachment ca{};
    if (t.depth) {
        ca.aspectMask = ((c.flags & kClearZ) ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) |
                        ((c.flags & kClearStencil) && HasStencil(t.res->format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
        ca.clearValue.depthStencil = {std::clamp(c.z, 0.0f, 1.0f), c.stencil & 0xFF};
        SetAttachments(nullptr, t.res.get());
    } else {
        // pixels the CPU wrote before this clear go in first, a later upload would paint over the clear
        if (LatchCpuWrite(t)) UploadTarget(t, s.rt);
        ca.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        float col[4];
        Color(c.color, col);
        for (int i = 0; t.range != 1 && i < 3; i++) col[i] /= t.range;
        std::memcpy(ca.clearValue.color.float32, col, sizeof(col));
        SetAttachments(t.res.get(), nullptr);
        // the D3DCLEAR channel bits from 0x10 up are red, green, blue, alpha: VkColorComponentFlags' order
        uint32_t wmask = (c.flags & kClearTarget) >> 4;
        if (wmask != 0xF) {
            bound_rt_ = bound_ds_ = nullptr;
            if (!rs.empty()) ClearChannels(t, col, wmask, rs);
            t.drawn = true;
            return;
        }
    }
    // the draws' targets are bound again
    bound_rt_ = bound_ds_ = nullptr;
    if (!rs.empty() && ca.aspectMask) {
        EnsureRendering();
        vkCmdClearAttachments(cl_, 1, &ca, uint32_t(rs.size()), rs.data());
    }
    t.drawn = true;
}

void VulkanBackend::Clear(const State& s, uint32_t count, const uint8_t* rects, uint32_t flags, uint32_t color, float z,
                          uint32_t stencil) {
    if (flags & kClearOcclusion) {
        if (count != kOcclusionRects || !rects) return;
        float p[4 * kOcclusionRects];
        std::memcpy(p, rects, sizeof(p));
        Occlusion(s, p);
        return;
    }
    PerfScope perf_clear{g_perf.clear};
    if (cap_) {
        CapStep(s.rt);
        fprintf(cap_, "%04u clear rt %08X ds %08X flags %X color %08X z %.3f\n", cap_i_++, s.rt.data, s.zs.data, flags,
                color, z);
    }
    Target* rt = (flags & kClearTarget) ? EnsureTarget(s.rt, false) : nullptr;
    Target* ds = (flags & (kClearZ | kClearStencil)) ? EnsureTarget(s.zs, true) : nullptr;
    std::vector<VkClearRect> rs;
    ClearArgs c{.count = count, .rects = rects, .flags = flags, .color = color, .z = z, .stencil = stencil};
    for (Target* t : {rt, ds}) {
        if (t) ClearTarget(*t, s, c, rs);
    }
}

// CopyRects' guest lists: src_rects RECTs, dst_points POINTs, all int32
constexpr uint32_t kGuestRectBytes = 16;
constexpr uint32_t kRectLeft = 0;
constexpr uint32_t kRectTop = 4;
constexpr uint32_t kRectRight = 8;
constexpr uint32_t kRectBottom = 12;
constexpr uint32_t kGuestPointBytes = 8;
constexpr uint32_t kPointX = 0;
constexpr uint32_t kPointY = 4;

struct GuestRect {
    int32_t left;
    int32_t top;
    int32_t right;
    int32_t bottom;
};

struct GuestPoint {
    int32_t x;
    int32_t y;
};

GuestRect ReadGuestRect(uint32_t rects, uint32_t i) {
    uint32_t r = rects + i * kGuestRectBytes;
    GuestRect g;
    g.left = int32_t(kernel::R32(r + kRectLeft));
    g.top = int32_t(kernel::R32(r + kRectTop));
    g.right = int32_t(kernel::R32(r + kRectRight));
    g.bottom = int32_t(kernel::R32(r + kRectBottom));
    return g;
}

GuestPoint ReadGuestPoint(uint32_t points, uint32_t i) {
    uint32_t p = points + i * kGuestPointBytes;
    GuestPoint g;
    g.x = int32_t(kernel::R32(p + kPointX));
    g.y = int32_t(kernel::R32(p + kPointY));
    return g;
}

// a drawn copy's rect in guest pixels: the source corners and where it lands
struct DrawnRect {
    float x0 = 0;
    float y0 = 0;
    float x1 = 0;
    float y1 = 0;
    float dx = 0;
    float dy = 0;
};

// without dst_points the rect lands where it is
DrawnRect ReadDrawnRect(uint32_t src_rects, uint32_t dst_points, uint32_t i) {
    GuestRect g = ReadGuestRect(src_rects, i);
    DrawnRect r;
    r.x0 = float(g.left);
    r.y0 = float(g.top);
    r.x1 = float(g.right);
    r.y1 = float(g.bottom);
    r.dx = r.x0;
    r.dy = r.y0;
    if (dst_points) {
        GuestPoint p = ReadGuestPoint(dst_points, i);
        r.dx = float(p.x);
        r.dy = float(p.y);
    }
    return r;
}

// a drawn copy's taps round down a ratio a hair over a whole number
constexpr float kTapsSlack = 1e-3f;
// the most taps a box filter takes per axis: 8 covers an 8x render scale, and 64 samples per pixel is already more
// than a read-back can spend
constexpr uint32_t kMaxTaps = 8;

// one Blit per rect: for formats that differ or a multisampled target
void VulkanBackend::CopyByDraw(const Target& src_t, Image* src_res, Target& dst_t, const CopyList& l) {
    Transition(*src_res, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    SetAttachments(dst_t.res.get(), nullptr);
    float src_scale = float(src_t.w) / src_t.gw;
    float dst_scale = float(dst_t.w) / dst_t.gw;
    bool listed = l.count && l.src_rects;
    uint32_t n = listed ? l.count : 1;
    for (uint32_t i = 0; i < n; i++) {
        float whole_w = float(std::min(src_t.gw, dst_t.gw));
        float whole_h = float(std::min(src_t.gh, dst_t.gh));
        DrawnRect r{.x0 = 0, .y0 = 0, .x1 = whole_w, .y1 = whole_h};
        if (listed) r = ReadDrawnRect(l.src_rects, l.dst_points, i);
        if (r.x1 <= r.x0 || r.y1 <= r.y0) continue;
        float vx = r.dx * dst_scale;
        float vy = r.dy * dst_scale;
        float vw = (r.x1 - r.x0) * dst_scale;
        float vh = (r.y1 - r.y0) * dst_scale;
        // the scissor covers every pixel the viewport touches: a pixel cut off here is a line of what was under it
        int32_t left = std::clamp<int32_t>(int32_t(std::floor(vx)), 0, int32_t(dst_t.w));
        int32_t top = std::clamp<int32_t>(int32_t(std::floor(vy)), 0, int32_t(dst_t.h));
        int32_t right = std::clamp<int32_t>(int32_t(std::ceil(vx + vw)), 0, int32_t(dst_t.w));
        int32_t bottom = std::clamp<int32_t>(int32_t(std::ceil(vy + vh)), 0, int32_t(dst_t.h));
        Rect scissor{left, top, right, bottom};
        if (scissor.right <= scissor.left || scissor.bottom <= scissor.top) continue;
        SetViewport(vx, vy, vw, vh);
        SetScissor(scissor);
        float uv[4] = {r.x0 / src_t.gw, r.y0 / src_t.gh, (r.x1 - r.x0) / src_t.gw, (r.y1 - r.y0) / src_t.gh};
        uint32_t taps = uint32_t(std::ceil(std::max(src_scale / dst_scale, 1.0f) - kTapsSlack));
        float mul = 1 / dst_t.range;
        BlitOptions o{.alpha = true, .taps = taps, .mul = mul, .rtv = dst_t.fmt, .samples = Ms(dst_t)};
        Blit(src_res, uv, o);
    }
    dst_t.drawn = true;
    bound_rt_ = bound_ds_ = nullptr;
}

// one image copy per rect, clamped to both targets
void VulkanBackend::CopyByRegion(const Target& src_t, Image* src_res, Target& dst_t, const CopyList& l) {
    Transition(*src_res, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    Transition(*dst_t.res, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    // the source rect is in the source target's pixels and the destination point in the destination's: each its own
    // scale, which are the same until a target outlives a render scale change
    float src_scale = float(src_t.w) / float(src_t.gw);
    float dst_scale = float(dst_t.w) / float(dst_t.gw);
    bool listed = l.count && l.src_rects;
    uint32_t n = listed ? l.count : 1;
    for (uint32_t i = 0; i < n; i++) {
        int32_t left = 0;
        int32_t top = 0;
        int32_t right = int32_t(std::min(src_t.w, dst_t.w));
        int32_t bottom = int32_t(std::min(src_t.h, dst_t.h));
        int32_t dx = 0;
        int32_t dy = 0;
        if (listed) {
            GuestRect r = ReadGuestRect(l.src_rects, i);
            left = HostEdge(r.left, src_scale, src_t.w, false);
            top = HostEdge(r.top, src_scale, src_t.h, false);
            right = HostEdge(r.right, src_scale, src_t.w, true);
            bottom = HostEdge(r.bottom, src_scale, src_t.h, true);
            dx = HostEdge(r.left, dst_scale, dst_t.w, false);
            dy = HostEdge(r.top, dst_scale, dst_t.h, false);
            if (l.dst_points) {
                GuestPoint p = ReadGuestPoint(l.dst_points, i);
                dx = HostEdge(p.x, dst_scale, dst_t.w, false);
                dy = HostEdge(p.y, dst_scale, dst_t.h, false);
            }
        }
        if (right <= left || bottom <= top || dx >= int32_t(dst_t.w) || dy >= int32_t(dst_t.h)) continue;
        // the copy keeps its size: a rect running off the destination is cut at both ends
        right = std::min(right, left + int32_t(dst_t.w) - dx);
        bottom = std::min(bottom, top + int32_t(dst_t.h) - dy);
        VkImageSubresourceLayers layer{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        VkImageCopy c{layer, {left, top, 0}, layer, {dx, dy, 0},
                      {uint32_t(right - left), uint32_t(bottom - top), 1}};
        vkCmdCopyImage(cl_, src_res->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst_t.res->image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    }
    dst_t.drawn = true;
    bound_rt_ = bound_ds_ = nullptr;
}

void VulkanBackend::Copy(const Surface& src, const Surface& dst, uint32_t count, uint32_t src_rects,
                         uint32_t dst_points) {
    PerfScope perf_copy{g_perf.copy};
    Target* src_t = FindTarget(src, false);
    if (cap_) {
        const char* from = "memory";
        if (src_t) from = src_t->drawn ? "target" : "idle target";
        fprintf(cap_, "%04u copy %08X %s to %08X\n", cap_i_++, src.data, from, dst.data);
    }
    // pixels the CPU made: the copy happens in memory, textures and targets pick it up from there
    if (!src_t || !src_t->drawn) {
        if (!CopySurfaceMemory(src, dst, count, src_rects, dst_points)) {
            LOG_FIRST(8, "vulkan: CopyRects %08X to %08X skipped", src.data, dst.data);
        } else if (Target* dst_t = FindTarget(dst, false)) {
            UploadTarget(*dst_t, dst);
        }
        return;
    }
    Target* dst_t = EnsureTarget(dst, false);
    if (!dst_t || src_t == dst_t) return;
    Image* from = Flat(*src_t);
    if (!from) return;
    CopyList l{.count = count, .src_rects = src_rects, .dst_points = dst_points};
    float src_scale = float(src_t->w) / float(src_t->gw);
    float dst_scale = float(dst_t->w) / float(dst_t->gw);
    if (src_t->fmt != dst_t->fmt || Ms(*dst_t) > 1 || std::fabs(src_scale - dst_scale) > 1e-4f) {
        // formats differ, a multisampled target takes no copy, or the two are at different scales: drawn across
        CopyByDraw(*src_t, from, *dst_t, l);
        return;
    }
    CopyByRegion(*src_t, from, *dst_t, l);
    if (cap_) {
        char name[32];
        snprintf(name, sizeof(name), "%04u_copy_%08X", cap_i_, dst_t->data);
        CapSave(*dst_t, name, dst_t->range);
    }
}

// HDR10 when the window's display has HDR headroom (SDL reads it from the compositor's preferred image description)
// and the surface takes PQ over BT.2020; the peak is the headroom over SDR white. XBE_HDR_PAPER_NITS and
// XBE_HDR_PEAK_NITS override what the compositor tells
void VulkanBackend::UpdateHdrDisplay() {
    if (!surface_) return;  // between windows (EnsureSurface)
    uint32_t n = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys_, surface_, &n, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys_, surface_, &n, formats.data());
    bool pq = std::any_of(formats.begin(), formats.end(), [](const VkSurfaceFormatKHR& f) {
        return f.format == kHdr10Format && f.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT;
    });
    SDL_PropertiesID props = SDL_GetWindowProperties(static_cast<SDL_Window*>(app::Window()));
    bool headroom_on = SDL_GetBooleanProperty(props, SDL_PROP_WINDOW_HDR_ENABLED_BOOLEAN, false);
    float headroom = SDL_GetFloatProperty(props, SDL_PROP_WINDOW_HDR_HEADROOM_FLOAT, 1.0f);
    auto nits = [](const char* var, float fallback) {
        const char* v = std::getenv(var);
        float f = v ? float(std::atof(v)) : 0.0f;
        return f > 0 ? f : fallback;
    };
    paper_nits_ = nits("XBE_HDR_PAPER_NITS", kDefaultPaperNits);
    float peak = headroom > 1 ? paper_nits_ * headroom : kDefaultPeakNits;
    peak_nits_ = std::min(nits("XBE_HDR_PEAK_NITS", peak), kMaxPeakNits);
    bool hdr = pq && headroom_on;
    static bool logged = false;
    if (hdr != hdr_display_ || !logged) {
        logged = true;
        LOGR("Display HDR: %s, peak %.0f nits, SDR white %.0f nits", hdr ? "on" : "off", peak_nits_, paper_nits_);
    }
    hdr_display_ = hdr;
    game::g_hdr_display = hdr_display_;
}

void VulkanBackend::DestroySwapchainImages() {
    for (VkSemaphore s : rendered_) vkDestroySemaphore(dev_, s, nullptr);
    rendered_.clear();
    for (auto& b : bb_) {
        if (b && b->view) vkDestroyImageView(dev_, b->view, nullptr);
        if (b) b->view = b->attach = VK_NULL_HANDLE;
    }
    bb_.clear();
}

// false when the surface takes no swap chain in that format; the old one is replaced either way
bool VulkanBackend::CreateSwapchain(uint32_t w, uint32_t h, bool hdr) {
    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys_, surface_, &caps);
    uint32_t n = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys_, surface_, &n, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys_, surface_, &n, formats.data());
    VkSurfaceFormatKHR want = hdr ? VkSurfaceFormatKHR{kHdr10Format, VK_COLOR_SPACE_HDR10_ST2084_EXT}
                                  : VkSurfaceFormatKHR{kBgra, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    auto offered = [&](const VkSurfaceFormatKHR& w) {
        return std::any_of(formats.begin(), formats.end(), [&](const VkSurfaceFormatKHR& f) {
            return f.format == w.format && f.colorSpace == w.colorSpace;
        });
    };
    bool found = offered(want);
    // older Android window systems have no BGRA: RGBA, which the frame's blit and the panel draw into as well
    if (!found && !hdr) {
        want.format = VK_FORMAT_R8G8B8A8_UNORM;
        found = offered(want);
    }
    if (!found) return false;
    vkGetPhysicalDeviceSurfacePresentModesKHR(phys_, surface_, &n, nullptr);
    std::vector<VkPresentModeKHR> modes(n);
    vkGetPhysicalDeviceSurfacePresentModesKHR(phys_, surface_, &n, modes.data());
    auto has = [&](VkPresentModeKHR m) { return std::find(modes.begin(), modes.end(), m) != modes.end(); };
    // without vsync the frame shows at once, torn, as D3D12's ALLOW_TEARING; mailbox when the window system has no
    // immediate mode
    present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
    if (!vsync_ && has(VK_PRESENT_MODE_IMMEDIATE_KHR)) present_mode_ = VK_PRESENT_MODE_IMMEDIATE_KHR;
    else if (!vsync_ && has(VK_PRESENT_MODE_MAILBOX_KHR)) present_mode_ = VK_PRESENT_MODE_MAILBOX_KHR;
    // the window's own size, landscape: Android's currentExtent is the panel's native (portrait) one, and with the
    // identity transform below the compositor turns the picture with the screen
    VkExtent2D extent = w && h ? VkExtent2D{w, h} : caps.currentExtent;
    if (extent.width == UINT32_MAX) {
        extent.width = std::clamp(w, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height = std::clamp(h, caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    if (!extent.width || !extent.height) return false;  // minimized
    uint32_t count = std::max<uint32_t>(caps.minImageCount, kFrames);
    if (caps.maxImageCount) count = std::min(count, caps.maxImageCount);
    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = surface_;
    ci.minImageCount = count;
    ci.imageFormat = want.format;
    ci.imageColorSpace = want.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    // copied out for XBE_SNAPSHOT where the surface allows it
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                    (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
                          ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
                          : caps.currentTransform;
    ci.compositeAlpha = (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
                            ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                            : VkCompositeAlphaFlagBitsKHR(caps.supportedCompositeAlpha & -caps.supportedCompositeAlpha);
    ci.presentMode = present_mode_;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = swap_;
    VkSwapchainKHR swap = VK_NULL_HANDLE;
    VkResult r = vkCreateSwapchainKHR(dev_, &ci, nullptr, &swap);
    DestroySwapchainImages();
    if (swap_) vkDestroySwapchainKHR(dev_, swap_, nullptr);
    swap_ = swap;
    if (r != VK_SUCCESS) {
        if (r == VK_ERROR_SURFACE_LOST_KHR || r == VK_ERROR_NATIVE_WINDOW_IN_USE_KHR) surface_lost_ = true;
        LOGE("vulkan: no swap chain (%d)", int(r));
        return false;
    }
    vkGetSwapchainImagesKHR(dev_, swap_, &n, nullptr);
    std::vector<VkImage> images(n);
    vkGetSwapchainImagesKHR(dev_, swap_, &n, images.data());
    for (VkImage image : images) {
        auto img = std::make_shared<Image>();
        img->image = image;
        img->format = want.format;
        img->w = extent.width;
        img->h = extent.height;
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = want.format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(dev_, &vi, nullptr, &img->view));
        img->attach = img->view;
        bb_.push_back(img);
        VkSemaphore s = VK_NULL_HANDLE;
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(dev_, &si, nullptr, &s));
        rendered_.push_back(s);
    }
    swap_fmt_ = want.format;
    bb_w_ = extent.width;
    bb_h_ = extent.height;
    swap_dirty_ = false;
    if (hdr && hdr_ext_) {
        // BT.2020 primaries, D65, the peak the frame is stretched to
        auto set_hdr = reinterpret_cast<PFN_vkSetHdrMetadataEXT>(vkGetDeviceProcAddr(dev_, "vkSetHdrMetadataEXT"));
        VkHdrMetadataEXT md{VK_STRUCTURE_TYPE_HDR_METADATA_EXT};
        md.displayPrimaryRed = {0.708f, 0.292f};
        md.displayPrimaryGreen = {0.170f, 0.797f};
        md.displayPrimaryBlue = {0.131f, 0.046f};
        md.whitePoint = {0.3127f, 0.3290f};
        md.maxLuminance = peak_nits_;
        md.minLuminance = 0.001f;
        md.maxContentLightLevel = peak_nits_;
        md.maxFrameAverageLightLevel = paper_nits_;
        if (set_hdr) set_hdr(dev_, 1, &swap_, &md);
    }
    return true;
}

// the window's size, HDR or vsync changed: the swap chain made again, once the GPU is done with the old one
// the epoch is taken once the surface is made: until then each present tries again (Black-Android's way)
bool VulkanBackend::CreateSurface() {
    uint32_t epoch = host::SurfaceEpoch();
    // Android has not handed the window back yet: wait for it, logging the outage once
    SDL_PropertiesID props = SDL_GetWindowProperties(host::Window());
    if (!SDL_GetPointerProperty(props, SDL_PROP_WINDOW_ANDROID_WINDOW_POINTER, nullptr)) {
        if (surface_failures_++ == 0) LOGR("vulkan: waiting for Android's window");
        return false;
    }
    if (!SDL_Vulkan_CreateSurface(host::Window(), instance_, nullptr, &surface_)) {
        surface_ = VK_NULL_HANDLE;
        if (surface_failures_++ == 0) LOGE("vulkan: no surface: %s", SDL_GetError());
        return false;
    }
    surface_failures_ = 0;
    surface_epoch_ = epoch;
    surface_lost_ = false;
    return true;
}

// a new window after the app came back to the front, or one the driver lost: the swap chain goes with the old
// surface, both are made again; false while there is nothing to show on
bool VulkanBackend::EnsureSurface() {
    if (surface_ && !surface_lost_ && host::SurfaceEpoch() == surface_epoch_) return true;
    if (surface_) {
        Flush();
        vkQueueWaitIdle(queue_);
        DestroySwapchainImages();
        if (swap_) vkDestroySwapchainKHR(dev_, swap_, nullptr);
        swap_ = VK_NULL_HANDLE;
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
    }
    if (!CreateSurface()) return false;
    swap_dirty_ = true;
    return true;
}

void VulkanBackend::ResizeSwap() {
    if (!EnsureSurface()) return;
    app::Mode px = app::WindowPixels();
    bool hdr = hdr_display_ && game::Opt().hdr_output;
    bool vsync = app::Get().vsync;
    bool same = px.width == bb_w_ && px.height == bb_h_ && hdr == hdr_want_ && vsync == vsync_;
    if (same && !swap_dirty_ && swap_) return;
    Flush();
    vkQueueWaitIdle(queue_);
    hdr_want_ = hdr;
    vsync_ = vsync;
    hdr_out_ = hdr && CreateSwapchain(px.width, px.height, true);
    if (!hdr_out_) CreateSwapchain(px.width, px.height, false);
    if (hdr && !hdr_out_) LOGR("Display HDR: the swap chain can't show HDR10, SDR it is");
}

// the next swap chain image; false when there is none to draw (minimized, or the chain is out of date)
bool VulkanBackend::Acquire(uint32_t& index) {
    if (!swap_ || !surface_) return false;
    // a bounded wait: a window Android is taking away may never hand an image back
    constexpr uint64_t kAcquireTimeout = 500'000'000;
    VkResult r = vkAcquireNextImageKHR(dev_, swap_, kAcquireTimeout, CurrentFrame().acquired, VK_NULL_HANDLE, &index);
    if (r == VK_ERROR_SURFACE_LOST_KHR) surface_lost_ = true;
    // SUBOPTIMAL is what Android gives for the identity transform on a turned screen: the compositor turns it, and a
    // new swap chain would be the same; a size change comes through the window instead
    if (r == VK_ERROR_OUT_OF_DATE_KHR) swap_dirty_ = true;
    return r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR;
}

void VulkanBackend::DrawFrame(Target* t, uint32_t bi) {
    Image* r = PrepareShownFrame(t);
    Image& bb = *bb_[bi];
    // the last frame's picture is not kept
    bb.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    // without its targets an HDR10 back buffer stays black: the SDR pipelines don't match its format
    if (!hdr_out_) {
        DrawSdrFrame(t, r, &bb);
    } else if (EnsureOutTargets()) {
        DrawHdrFrame(t, r, &bb);
    } else {
        Transition(bb, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkClearColorValue black{};
        black.float32[3] = 1;
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cl_, bb.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
    }
    Transition(bb, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
}

// what the frame shows: the target made readable, SMAA's blend of it when that is on
Image* VulkanBackend::PrepareShownFrame(Target* t) {
    if (!t) return nullptr;
    Image* r = Flat(*t);
    if (!r) return nullptr;
    Transition(*r, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (game::Opt().smaa) {
        if (Image* aa = Smaa(r, t->fmt, t->w, t->h)) r = aa;
    }
    return r;
}

// src readable: SMAA's blend of it, null when it can't run
Image* VulkanBackend::Smaa(Image* src, VkFormat fmt, uint32_t w, uint32_t h) {
    static const std::string hlsl = std::string(kSmaaConfig) + kSmaaHlsl + kSmaaPasses;
    if (!EnsureSmaaLookups()) return nullptr;
    if (!EnsureSmaaTargets(fmt, w, h)) return nullptr;
    if (!DrawSmaaPasses(src, fmt, w, h, hlsl.c_str())) return nullptr;
    return smaa_[2].get();
}

bool VulkanBackend::EnsureSmaaLookups() {
    // smaa_textures.h: area 160 x 560 in two channels, search 64 x 16 in one
    constexpr SmaaLookup area{VK_FORMAT_R8G8_UNORM, 160, 560, 2, kSmaaArea};
    constexpr SmaaLookup search{VK_FORMAT_R8_UNORM, 64, 16, 1, kSmaaSearch};
    static_assert(sizeof(kSmaaArea) == area.w * area.h * area.bpp);
    static_assert(sizeof(kSmaaSearch) == search.w * search.h * search.bpp);
    if (smaa_lut_[1]) return true;
    smaa_lut_[0] = CreateLookupTexture(area);
    smaa_lut_[1] = CreateLookupTexture(search);
    if (!smaa_lut_[0] || !smaa_lut_[1]) {
        smaa_lut_[0].reset();
        smaa_lut_[1].reset();
        return false;
    }
    return true;
}

// a lookup texture filled from l's data, left readable
ImageRef VulkanBackend::CreateLookupTexture(const SmaaLookup& l) {
    ImageRef r = CreateImage(l.fmt, l.w, l.h, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    if (!r) return nullptr;
    uint32_t row = l.w * l.bpp;
    Alloc a = Upload(uint64_t(row) * l.h, 16);
    std::memcpy(a.cpu, l.data, size_t(row) * l.h);
    Transition(*r, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy c{};
    c.bufferOffset = a.off;
    c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.imageExtent = {l.w, l.h, 1};
    vkCmdCopyBufferToImage(cl_, a.buffer, r->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    Transition(*r, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return r;
}

// SMAA's own targets: edges, then blend weights, then the blended frame in the frame's format
constexpr VkFormat kSmaaEdgeFormat = VK_FORMAT_R8G8_UNORM;
constexpr VkFormat kSmaaWeightFormat = VK_FORMAT_R8G8B8A8_UNORM;

bool VulkanBackend::EnsureSmaaTargets(VkFormat fmt, uint32_t w, uint32_t h) {
    if (smaa_[2] && smaa_w_ == w && smaa_h_ == h && smaa_fmt_ == fmt) return true;
    smaa_w_ = smaa_h_ = 0;
    const VkFormat f[kSmaaTargets] = {kSmaaEdgeFormat, kSmaaWeightFormat, fmt};
    for (int i = 0; i < kSmaaTargets; i++) {
        Release(std::move(smaa_[i]));
        smaa_[i] = CreateImage(f[i], w, h, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                               VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        if (!smaa_[i]) {
            for (auto& r : smaa_) Release(std::move(r));
            return false;
        }
    }
    smaa_w_ = w;
    smaa_h_ = h;
    smaa_fmt_ = fmt;
    LOGI("vulkan: SMAA %ux%u", w, h);
    return true;
}

// edges, blend weights, then the blend of src into smaa_[2], left readable
bool VulkanBackend::DrawSmaaPasses(Image* src, VkFormat fmt, uint32_t w, uint32_t h, const char* hlsl) {
    // edge detection discards where there is no edge, so the edges and weights start clear each frame
    VkClearColorValue clear{};
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    for (int i = 0; i < 2; i++) {
        Transition(*smaa_[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkCmdClearColorImage(cl_, smaa_[i]->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
    }
    float p[8] = {1.0f / w, 1.0f / h, float(w), float(h)};
    Image* const edges[3] = {src, nullptr, nullptr};
    if (!Pass(hlsl, "vs_edges", "smaa_edges", kSmaaEdgeFormat, smaa_[0].get(), w, h, edges, p, -1)) return false;
    Image* const weights[3] = {smaa_[0].get(), smaa_lut_[0].get(), smaa_lut_[1].get()};
    if (!Pass(hlsl, "vs_weights", "smaa_weights", kSmaaWeightFormat, smaa_[1].get(), w, h, weights, p, -1)) return false;
    Image* const blend[3] = {src, smaa_[1].get(), nullptr};
    if (!Pass(hlsl, "vs_blend", "smaa_blend", fmt, smaa_[2].get(), w, h, blend, p, -1)) return false;
    Transition(*smaa_[2], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return true;
}

// where the picture lands in the back buffer: the shape the console puts out, 16:9 in its widescreen mode, pillarboxed
// rather than stretched across the phone's wider screen, or in WIDESCREEN the whole screen
VulkanBackend::FramePlace VulkanBackend::PlaceFrame(const Target& t) {
    float tv = app::Widescreen() ? 16.0f / 9.0f : 4.0f / 3.0f;
    if (game::Opt().aspect) tv *= wide_;  // widescreen: the picture the widened camera drew is the screen's shape
    // only the rows the title drew: the black it leaves under its picture is not shown, and the rows kept are as
    // tall as on the TV
    bool cut = shown_rows_ && shown_rows_ < t.gh;
    float aspect = cut ? tv * float(t.gh) / float(shown_rows_) : tv;
    float w = float(bb_w_);
    float h = float(bb_h_);
    FramePlace p;
    p.w = std::min(w, h * aspect);
    p.h = p.w / aspect;
    p.x = (w - p.w) / 2;
    p.y = (h - p.h) / 2;
    p.rows = cut ? float(shown_rows_) / float(t.gh) : 1.0f;
    return p;
}

// the attachment must be rendering: the bars are cleared inside the pass, so the surface is never loaded or written
// whole for a picture that covers most of it
void VulkanBackend::ClearBars(Image& bb, const FramePlace& p, const VkClearColorValue& colour) {
    // the picture rounded inward: a pixel the blit may not cover is cleared first and drawn over after, never left as
    // whatever the surface held last frame
    int32_t x0 = std::clamp<int32_t>(int32_t(std::floor(p.x)), 0, int32_t(bb.w));
    int32_t y0 = std::clamp<int32_t>(int32_t(std::floor(p.y)), 0, int32_t(bb.h));
    int32_t x1 = std::clamp<int32_t>(int32_t(std::floor(p.x + p.w)), 0, int32_t(bb.w));
    int32_t y1 = std::clamp<int32_t>(int32_t(std::floor(p.y + p.h)), 0, int32_t(bb.h));
    VkClearRect rs[4];
    uint32_t n = 0;
    auto add = [&](int32_t l, int32_t tp, int32_t r, int32_t b) {
        if (r <= l || b <= tp) return;
        rs[n++] = {{{l, tp}, {uint32_t(r - l), uint32_t(b - tp)}}, 0, 1};
    };
    add(0, 0, x0, int32_t(bb.h));
    add(x1, 0, int32_t(bb.w), int32_t(bb.h));
    add(x0, 0, x1, y0);
    add(x0, y1, x1, int32_t(bb.h));
    if (!n) return;
    VkClearAttachment ca{};
    ca.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ca.colorAttachment = 0;
    ca.clearValue.color = colour;
    SetScissor({0, 0, int32_t(bb.w), int32_t(bb.h)});
    vkCmdClearAttachments(cl_, 1, &ca, n, rs);
}

void VulkanBackend::DrawSdrFrame(Target* t, Image* r, Image* bb) {
    VkClearColorValue black{};
    black.float32[3] = 1;
    if (t && r) {
        // the frame and the UI cover the surface but for the bars, which are cleared inside the pass: nothing reads
        // last frame's picture, so the pass loads none of it
        SetAttachments(bb, nullptr, VK_ATTACHMENT_LOAD_OP_DONT_CARE);
        EnsureRendering();
        ClearBars(*bb, PlaceFrame(*t), black);
    } else {
        Transition(*bb, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cl_, bb->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
        SetAttachments(bb, nullptr);
    }
    bound_rt_ = bound_ds_ = nullptr;
    if (t) BlitFrame(*t, r, swap_fmt_);
    RenderUi();
}

bool VulkanBackend::EnsureOutTargets() {
    if (out_[kOutFrame] && out_[kOutPanel] && out_w_ == bb_w_ && out_h_ == bb_h_) return true;
    out_w_ = bb_w_;
    out_h_ = bb_h_;
    const VkFormat fmt[kOutTargets] = {kOutFrameFormat, ui_fmt_};
    for (int i = 0; i < kOutTargets; i++) {
        Release(std::move(out_[i]));
        out_[i] = CreateImage(fmt[i], bb_w_, bb_h_, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                                        VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        if (!out_[i]) {
            LOGE("vulkan: no %ux%u HDR10 output target", bb_w_, bb_h_);
            return false;
        }
    }
    return true;
}

// frame and panel into their own targets, then both into the back buffer as HDR10, panel at SDR white
void VulkanBackend::DrawHdrFrame(Target* t, Image* r, Image* bb) {
    VkClearColorValue clear{};
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    // the panel is transparent where the UI is not, and the UI is not everywhere
    Transition(*out_[kOutPanel], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkCmdClearColorImage(cl_, out_[kOutPanel]->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
    // the frame's target only needs its bars cleared: BlitFrame draws the rest
    if (t && r) {
        SetAttachments(out_[kOutFrame].get(), nullptr, VK_ATTACHMENT_LOAD_OP_DONT_CARE);
        EnsureRendering();
        ClearBars(*out_[kOutFrame], PlaceFrame(*t), clear);
    } else {
        Transition(*out_[kOutFrame], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkCmdClearColorImage(cl_, out_[kOutFrame]->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
        SetAttachments(out_[kOutFrame].get(), nullptr);
    }
    bound_rt_ = bound_ds_ = nullptr;
    if (t) BlitFrame(*t, r, kOutFrameFormat);
    SetAttachments(out_[kOutPanel].get(), nullptr);
    RenderUi();
    Image* const src[3] = {out_[kOutFrame].get(), out_[kOutPanel].get(), nullptr};
    float p[8] = {paper_nits_, peak_nits_, 0, 0};
    Pass(kHdrHlsl, "vs", "hdr10", kHdr10Format, bb, bb_w_, bb_h_, src, p, -1);
}

// the frame letterboxed to the game's aspect into the bound target
void VulkanBackend::BlitFrame(Target& t, Image* r, VkFormat rtv) {
    FramePlace p = PlaceFrame(t);
    float uv[4] = {0, 0, 1, p.rows};
    SetViewport(p.x, p.y, p.w, p.h);
    SetScissor({0, 0, int32_t(bb_w_), int32_t(bb_h_)});
    if (r) Blit(r, uv, {.gamma = gamma_, .rtv = rtv});
}

// the panel over the bound target; ImGui binds its own pipeline and viewport
void VulkanBackend::RenderUi() {
    std::lock_guard lock(app::UiLock());
    ImGui_ImplVulkan_NewFrame();
    app::UiFrame();
    ImGui::NewFrame();
    if (app::kGame.ui) app::kGame.ui(fps_);
    ImGui::Render();
    ImDrawData* dd = ImGui::GetDrawData();
    if (!dd || !dd->CmdListsCount) return;
    EnsureRendering();
    ImGui_ImplVulkan_RenderDrawData(dd, cl_);
    ForgetState();
}


void VulkanBackend::StartCapture() {
    std::time_t now = std::time(nullptr);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%H%M%S", std::localtime(&now));
    std::error_code ec;
    cap_dir_ = std::filesystem::weakly_canonical(app::ExeDir() / diag::CaptureDir() / stamp, ec);
    std::filesystem::create_directories(cap_dir_, ec);
    LOGI("capture: dir %s", cap_dir_.c_str());
    cap_ = std::fopen((cap_dir_ / "frame.txt").c_str(), "w");
    cap_i_ = cap_rt_ = 0;
    cap_seen_.clear();
    cap_sh_.clear();
}

// ramp_ holds the red, green and blue ramps, 256 entries each
constexpr int kRampRed = 0;
constexpr int kRampGreen = 256;
constexpr int kRampBlue = 512;

void VulkanBackend::UploadGammaRamp() {
    constexpr uint32_t kRampBytes = 256 * 4;  // one BGRA texel per ramp entry
    ramp_dirty_ = false;
    Alloc a = Upload(kRampBytes, 16);
    bool identity = true;
    for (int i = 0; i < 256; i++) {
        uint8_t* p = a.cpu + i * 4;
        p[0] = ramp_[kRampBlue + i];
        p[1] = ramp_[kRampGreen + i];
        p[2] = ramp_[kRampRed + i];
        p[3] = 255;
        identity &= ramp_[kRampRed + i] == i && ramp_[kRampGreen + i] == i && ramp_[kRampBlue + i] == i;
    }
    gamma_ = !identity;
    Transition(*lut_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy c{};
    c.bufferOffset = a.off;
    c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.imageExtent = {256, 1, 1};
    vkCmdCopyBufferToImage(cl_, a.buffer, lut_->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
}

void VulkanBackend::ReleaseOldTextures() {
    constexpr uint64_t kSweepEvery = 60;    // presents between sweeps
    constexpr size_t kTexturesKept = 2048;  // no sweep below this many
    constexpr uint64_t kIdleFrames = 600;
    // textures unused for ten seconds go, levels stream theirs in and out
    if (presents_ % kSweepEvery != 0 || textures_.size() <= kTexturesKept) return;
    for (auto it = textures_.begin(); it != textures_.end();) {
        if (it->second.frame + kIdleFrames < presents_) {
            Release(std::move(it->second.res));
            it = textures_.erase(it);
        } else {
            ++it;
        }
    }
}

void VulkanBackend::ResetFrame() {
    Frame& f = CurrentFrame();
    if (f.fence && CompletedFence() < f.fence) {
        auto perf_wait = PerfClock::now();
        WaitFence(f.fence);
        g_perf.wait += PerfMs(perf_wait);
    }
    vkResetCommandPool(dev_, f.pool, 0);
    ResetPools(f);
    f.garbage.clear();
    f.garbage_buffers.clear();
    f.used = 0;
    cl_ = f.cmd;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cl_, &bi);
    Begin();
}

void VulkanBackend::CountFps() {
    fps_n_++;
    auto now = std::chrono::steady_clock::now();
    double el = std::chrono::duration<double>(now - fps_t_).count();
    if (el >= 0.5) {
        fps_ = fps_n_ / el;
        fps_n_ = 0;
        fps_t_ = now;
    }
}

void LogPerf() {
    static PerfClock::time_point perf_start = PerfClock::now();
    static uint64_t perf_frames = 0;
    perf_frames++;
    double el = PerfMs(perf_start);
    if (el >= 1000.0) {
        Perf& p = g_perf;
        double n = double(perf_frames);
        double host = p.draw + p.clear + p.copy + p.present + p.sync;
        LOGI("perf: %.1f fps %.2f ms/frame | host %.2f (draw %.2f: assemble %.2f bind %.2f [check %.2f, upload %.2f] compile %.2f; "
             "clear %.2f copy %.2f sync %.2f present %.2f [swap %.2f wait %.2f]) other %.2f",
             n * 1000.0 / el, el / n, host / n, p.draw / n, p.assemble / n, p.bind / n, p.sig / n, p.texup / n,
             p.compile / n, p.clear / n, p.copy / n, p.sync / n, p.present / n, p.swap / n, p.wait / n,
             (el - host) / n);
        LOGI("perf: title thread per frame: recording %.2f ms, waiting %.2f ms (%.1f waits: read-back %.2f resource %.2f fence %.2f "
             "flush %.2f copy %.2f, next frame %.2f) | read-backs waited for: %.1f lock, %.1f NOOVERWRITE",
             g_perf_capture_ns.exchange(0) / n / 1e6, g_perf_wait_ns.exchange(0) / n / 1e6,
             g_perf_waits.exchange(0) / n, g_perf_wait_kind_ns[0].exchange(0) / n / 1e6,
             g_perf_wait_kind_ns[1].exchange(0) / n / 1e6, g_perf_wait_kind_ns[2].exchange(0) / n / 1e6,
             g_perf_wait_kind_ns[3].exchange(0) / n / 1e6, g_perf_wait_kind_ns[4].exchange(0) / n / 1e6,
             g_perf_wait_kind_ns[5].exchange(0) / n / 1e6, g_perf_sync_block.exchange(0) / n,
             g_perf_sync_late.exchange(0) / n);
        LOGI("perf: per frame %.0f draws (%.0f raw) %.0fk verts %.1f MB upload %.1f overflow | %.0f checks %.2f MB hashed | %.1f tex uploads "
             "%.2f MB, %.1f new tex, %.1f new targets, %.1f target uploads, %.1f shrinks, %.1f read-backs %.1f late | %llu shaders %llu psos this second",
             p.draws / n, p.raw / n, p.verts / n / 1000.0, p.upload / n / 1048576.0, p.overflow / n, p.sigs / n,
             p.sig_bytes / n / 1048576.0, p.texups / n, p.texup_bytes / n / 1048576.0, p.newtex / n,
             p.targets / n, p.rtups / n, p.shrinks / n, p.syncs / n, p.lates / n,
             (unsigned long long)p.shaders, (unsigned long long)p.psos);
        g_perf = Perf{};
        perf_start = PerfClock::now();
        perf_frames = 0;
    }
}

// the frame's commands, then the swap chain image they drew to the screen
void VulkanBackend::SubmitPresent(bool acquired, uint32_t bi) {
    EndRendering();
    vkEndCommandBuffer(cl_);
    uint64_t value = ++fence_value_;
    uint64_t values[2] = {value, 0};
    // without the timeline only the present semaphore is signalled, and a fence tells when the GPU is past
    VkSemaphore signal[2] = {fence_, acquired ? rendered_[bi] : VK_NULL_HANDLE};
    uint32_t first = timeline_ ? 0 : 1;
    VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    uint64_t wait_value = 0;
    ts.waitSemaphoreValueCount = acquired ? 1 : 0;
    ts.pWaitSemaphoreValues = &wait_value;
    ts.signalSemaphoreValueCount = acquired ? 2 : 1;
    ts.pSignalSemaphoreValues = values;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, timeline_ ? &ts : nullptr};
    si.waitSemaphoreCount = acquired ? 1 : 0;
    si.pWaitSemaphores = &CurrentFrame().acquired;
    si.pWaitDstStageMask = &stage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cl_;
    si.signalSemaphoreCount = (acquired ? 2 : 1) - first;
    si.pSignalSemaphores = signal + first;
    if (timeline_) {
        VK_CHECK(vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE));
    } else {
        std::lock_guard lock(fence_lock_);
        VK_CHECK(vkQueueSubmit(queue_, 1, &si, SubmitFence(value)));
    }
    CurrentFrame().fence = fence_value_;
    if (!acquired) return;
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &rendered_[bi];
    pi.swapchainCount = 1;
    pi.pSwapchains = &swap_;
    pi.pImageIndices = &bi;
    VkResult r = vkQueuePresentKHR(queue_, &pi);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) swap_dirty_ = true;
    if (r == VK_ERROR_SURFACE_LOST_KHR) surface_lost_ = true;
}

void VulkanBackend::Present(const Surface& front) {
    // the activity is behind another app: hold the frame until it is in front, so the swapchain is not touched
    while (!host::Foreground()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    PerfScope perf_present{g_perf.present};
    if (cap_) {
        FinishCapture(front);
    } else if (g_capture.exchange(false) && CaptureOn()) {
        LOGI("capture: start at present %llu", (unsigned long long)presents_);
        StartCapture();
    }
    if (presents_ % 60 == 0) UpdateHdrDisplay();
    shown_rows_ = front.rows;
    ResizeSwap();
    Target* t = EnsureTarget(front, false);
    if (t && LatchCpuWrite(*t)) UploadTarget(*t, front);
    if (ramp_dirty_) UploadGammaRamp();
    Transition(*lut_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    uint32_t bi = 0;
    bool acquired = Acquire(bi);
    if (acquired) DrawFrame(t, bi);
    BufferRef snapshot = acquired ? RecordSnapshot(bi) : nullptr;
    present_fence_[presents_ % (kFrames + 1)].store(fence_value_ + 1);
    auto perf_swap = PerfClock::now();
    SubmitPresent(acquired, bi);
    g_perf.swap += PerfMs(perf_swap);
    if (snapshot) {
        WaitFence(fence_value_);
        SaveSnapshot(*snapshot);
    }

    presents_++;
    presents_seen_.store(presents_);
    frame_++;
    ReleaseOldTextures();
    ResetFrame();
    CountFps();
    if (diag::Perf()) LogPerf();
    ApplySettings();
}

void VulkanBackend::ApplySettings() {
    // widescreen: the camera hook widens the next frame's projection by as much as the screen is wider than 16:9
    // against the picture's 16:9 as PlaceFrame shows it: only the rows the title drew (448 of 480), as tall as on the
    // TV, so the cut picture is wider than 16:9 and the factor that fills the screen with it that much smaller
    float was = wide_;
    wide_ = 1;
    float tv = 16.0f / 9.0f;
    if (shown_rows_ && shown_rows_ < guest_h_) tv *= float(guest_h_) / float(shown_rows_);
    if (game::Opt().aspect && app::Widescreen() && bb_h_)
        wide_ = std::max(1.0f, float(bb_w_) / float(bb_h_) / tv);
    game::g_wide = wide_;
    if (wide_ != was)
        LOGR("ultrawide: %.4f, guest %ux%u, back buffer %ux%u", wide_, guest_w_, guest_h_, bb_w_, bb_h_);
    float want = app::RenderScale(guest_h_);
    uint32_t msaa = game::Opt().msaa;
    if (std::fabs(want - scale_) > 1e-4f || msaa != msaa_set_) {
        Flush();
        targets_.clear();
        late_.clear();
        scale_ = want;
        if (msaa != msaa_set_) {
            msaa_set_ = msaa;
            msaa_ = Samples(msaa_set_);
        }
        bound_rt_ = bound_ds_ = nullptr;
        SetAttachments(nullptr, nullptr);
    }
}


// the target resolved, then shrunk with o when it is not guest sized BGRA; res null on failure
VulkanBackend::CopySource VulkanBackend::PrepareCopySource(Target& t, const ShrinkOptions& o) {
    CopySource out{Flat(t), nullptr};
    if (!out.res) return out;
    if (t.w != t.gw || t.h != t.gh || t.fmt != kBgra) {
        ShrinkOptions s = o;
        if (s.taps <= 1) {
            // one bilinear tap reads four host texels: past a render scale of 2 it drops whole rows and columns of the
            // target, and the title reads its text back with strokes missing. Taps two texels apart cover the same
            // footprint at half the samples, the sampler averaging four texels of its own
            float sc = std::max(float(t.w) / float(std::max(t.gw, 1u)), float(t.h) / float(std::max(t.gh, 1u)));
            s.taps = uint32_t(std::clamp(std::ceil(sc * 0.5f - kTapsSlack), 1.0f, float(kMaxTaps)));
        }
        out.down = Shrink(out.res, t.gw, t.gh, s);
        out.res = out.down.get();
    }
    if (out.res) Transition(*out.res, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    return out;
}

// the CPU reads a surface the GPU drew: bring the pixels back
void VulkanBackend::Sync(uint32_t data, uint32_t) {
    auto it = targets_.find(TargetMapKey(data, false));
    if (it == targets_.end() || !it->second.drawn) return;
    PerfScope perf_sync{g_perf.sync};
    g_perf.syncs++;
    if (diag::Perf()) {
        LOG_FIRST(16, "perf: read-back of %08X (%ux%u guest, %ux%u host)", data, it->second.gw, it->second.gh,
                  it->second.w, it->second.h);
    }
    Target& t = it->second;
    // src.down, when there is one, stays alive until ReadBack is done
    CopySource src = PrepareCopySource(t, {});
    if (!src.res) return;
    std::vector<uint8_t> px;
    if (!ReadBack(src.res, t.gw, t.gh, px)) return;
    // color targets are 32 bit linear in guest memory, the pitch rounds to 64 bytes
    uint32_t pitch = GuestPitch(t.gw);
    uint8_t* guest = static_cast<uint8_t*>(kernel::HostPtr(data));
    for (uint32_t y = 0; y < t.gh; y++) std::memcpy(guest + size_t(y) * pitch, &px[size_t(y) * t.gw * 4], t.gw * 4);
    t.read = std::move(px);
    t.drawn = false;
    t.sig = MemSig(t.data, t.bytes);
}

// A NOOVERWRITE lock: the console reads whatever the GPU has finished, so a surface locked every frame gets the newest
// finished copy (a frame or two old) and the CPU never waits on the GPU; one locked now and then still waits in Sync
void VulkanBackend::SyncLate(uint32_t data) {
    if (!DeliverLate(data)) Sync(data, 0);
    RecordLate(data);
}

bool VulkanBackend::DeliverLate(uint32_t data) {
    std::lock_guard lock(late_lock_);
    auto it = late_.find(data);
    if (it == late_.end()) return false;
    Late& l = it->second;
    uint64_t done = CompletedFence();
    uint64_t now = presents_seen_.load();
    int best = -1;
    for (int i = 0; i < kFrames; i++) {
        bool finished = l.fence[i] && l.fence[i] <= done;
        bool recent = now - l.frame[i] <= uint64_t(kFrames) + 1;
        bool newest = best < 0 || l.fence[i] > l.fence[best];
        if (finished && recent && newest) best = i;
    }
    if (best < 0) {
        if (diag::Perf()) {
            LOG_FIRST(16, "perf: late %08X nothing finished: done %llu now %llu, slots %llu/%llu %llu/%llu %llu/%llu",
                      data, (unsigned long long)done, (unsigned long long)now, (unsigned long long)l.fence[0],
                      (unsigned long long)l.frame[0], (unsigned long long)l.fence[1], (unsigned long long)l.frame[1],
                      (unsigned long long)l.fence[2], (unsigned long long)l.frame[2]);
        }
        return false;
    }
    if (l.fence[best] > l.delivered) {
        const Buffer& rb = *l.rb[best];
        vmaInvalidateAllocation(vma_, rb.alloc, 0, VK_WHOLE_SIZE);
        uint32_t pitch = GuestPitch(l.gw);
        uint8_t* dst = static_cast<uint8_t*>(kernel::HostPtr(data));
        for (uint32_t y = 0; y < l.gh; y++) std::memcpy(dst + size_t(y) * pitch, rb.map + size_t(y) * l.gw * 4, l.gw * 4);
        l.delivered = l.fence[best];
    }
    return true;
}

// a slot set for another size: its buffers go with the frame
void VulkanBackend::ResetLate(Late& l, const Target& t) {
    for (auto& r : l.rb) {
        if (r) CurrentFrame().garbage_buffers.push_back(r);
    }
    Release(l.down);
    l = Late{};
    l.gw = t.gw;
    l.gh = t.gh;
}

// the target as a guest sized BGRA copy source, shrunk into l.down when it is not one already
Image* VulkanBackend::PrepareLateSource(Target& t, Late& l) {
    ShrinkOptions reuse{.into = l.down};
    CopySource src = PrepareCopySource(t, reuse);
    if (src.down) l.down = src.down;
    return src.res;
}

// src into the readback buffer of the slot, made on first use
void VulkanBackend::RecordLateCopy(Late& l, uint32_t slot, Image* src) {
    if (!l.rb[slot]) {
        l.rb[slot] = CreateBuffer(uint64_t(l.gw) * l.gh * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
        if (!l.rb[slot]) return;
    }
    CopyImageToBuffer(src, l.rb[slot]->buffer, l.gw, l.gh);
    HostReadBarrier();
    l.fence[slot] = fence_value_ + 1;  // the next submit carries it
    l.frame[slot] = presents_;
    l.next = (slot + 1) % kFrames;
}

void VulkanBackend::RecordLate(uint32_t data) {
    auto it = targets_.find(TargetMapKey(data, false));
    if (it == targets_.end()) return;
    Target& t = it->second;
    PerfScope perf_sync{g_perf.sync};
    g_perf.lates++;
    // what DeliverLate or Sync left in guest memory is no CPU write to upload back
    t.sig = MemSig(t.data, t.bytes);
    t.read.clear();
    std::lock_guard lock(late_lock_);
    Late& l = late_[data];
    if (l.gw != t.gw || l.gh != t.gh) {
        if (diag::Perf()) LOG_FIRST(16, "perf: late %08X ring reset, %ux%u -> %ux%u", data, l.gw, l.gh, t.gw, t.gh);
        ResetLate(l, t);
    }
    // a copy for a later lock, unless the GPU is a whole ring behind
    uint32_t slot = l.next;
    if (l.fence[slot] > CompletedFence()) return;
    Image* src = PrepareLateSource(t, l);
    if (!src) return;
    RecordLateCopy(l, slot, src);
}

// filters src into a w x h BGRA target, new or o.into, left as a copy source
ImageRef VulkanBackend::Shrink(Image* src, uint32_t w, uint32_t h, const ShrinkOptions& o) {
    g_perf.shrinks++;
    ImageRef down = o.into;
    if (!down || down->w != w || down->h != h) {
        down = CreateImage(kBgra, w, h, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                            VK_IMAGE_USAGE_SAMPLED_BIT);
        if (!down) return nullptr;
    }
    SetBlitTarget(down.get(), w, h);
    float uv[4] = {0, 0, 1, 1};
    Blit(src, uv, {.alpha = true, .taps = o.taps, .mul = o.mul});
    Transition(*down, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    return down;
}

// a guest sized BGRA image into a buffer, rows packed
void VulkanBackend::CopyImageToBuffer(Image* src, VkBuffer dst, uint32_t w, uint32_t h) {
    Transition(*src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy c{};
    c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.imageExtent = {std::min(w, src->w), std::min(h, src->h), 1};
    c.bufferRowLength = w;
    vkCmdCopyImageToBuffer(cl_, src->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, 1, &c);
}

// a BGRA copy source to the CPU, rows packed; waits for the GPU
bool VulkanBackend::ReadBack(Image* res, uint32_t w, uint32_t h, std::vector<uint8_t>& out) {
    BufferRef rb = CreateBuffer(uint64_t(w) * h * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
    if (!rb) return false;
    CopyImageToBuffer(res, rb->buffer, w, h);
    HostReadBarrier();
    Flush();
    vmaInvalidateAllocation(vma_, rb->alloc, 0, VK_WHOLE_SIZE);
    out.assign(rb->map, rb->map + size_t(w) * h * 4);
    return true;
}

// BMP file and info header: byte offsets of the fields CapSave writes
constexpr int kBmpHeaderBytes = 54;
constexpr int kBmpFileSize = 2;
constexpr int kBmpPixelOffset = 10;
constexpr int kBmpInfoSize = 14;
constexpr int kBmpWidth = 18;
constexpr int kBmpHeight = 22;  // negative: rows top down
constexpr int kBmpPlanes = 26;
constexpr int kBmpBitCount = 28;
constexpr int kBmpImageSize = 34;
constexpr uint32_t kBmpInfoBytes = 40;

void VulkanBackend::CapSave(const CapImage& img, const std::filesystem::path& file) {
    ImageRef down = Shrink(img.res, img.w, img.h, {.mul = img.mul});
    std::vector<uint8_t> px;
    if (!down || !ReadBack(down.get(), img.w, img.h, px)) return;
    FILE* f = std::fopen(file.c_str(), "wb");
    if (!f) return;
    uint8_t hd[kBmpHeaderBytes] = {'B', 'M'};  // BITMAPFILEHEADER and BITMAPINFOHEADER, top down 32 bit
    auto put = [&](int o, uint32_t v) { std::memcpy(hd + o, &v, 4); };
    put(kBmpFileSize, kBmpHeaderBytes + uint32_t(px.size()));
    put(kBmpPixelOffset, kBmpHeaderBytes);
    put(kBmpInfoSize, kBmpInfoBytes);
    put(kBmpWidth, img.w);
    put(kBmpHeight, uint32_t(-int32_t(img.h)));
    hd[kBmpPlanes] = 1;
    hd[kBmpBitCount] = 32;
    put(kBmpImageSize, uint32_t(px.size()));
    fwrite(hd, 1, kBmpHeaderBytes, f);
    fwrite(px.data(), 1, px.size(), f);
    fclose(f);
}

// XBE_SNAPSHOT=N: the window's picture every N seconds to snapshots/NNNN.bmp beside the exe, for a run with no screen;
// the back buffer is copied with the frame and written once the GPU is past it
BufferRef VulkanBackend::RecordSnapshot(uint32_t bi) {
    static const double every = [] {
        const char* v = std::getenv("XBE_SNAPSHOT");
        return v ? std::atof(v) : 0.0;
    }();
    static auto next = std::chrono::steady_clock::now();
    if (every <= 0 || swap_fmt_ != kBgra || std::chrono::steady_clock::now() < next) return nullptr;
    next += std::chrono::milliseconds(int64_t(every * 1000));
    Image& bb = *bb_[bi];
    BufferRef b = CreateBuffer(uint64_t(bb.w) * bb.h * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
    if (!b) return nullptr;
    CopyImageToBuffer(&bb, b->buffer, bb.w, bb.h);
    HostReadBarrier();
    Transition(bb, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    return b;
}

void VulkanBackend::SaveSnapshot(const Buffer& b) {
    static uint32_t n = 0;
    std::error_code ec;
    auto dir = app::ExeDir() / "snapshots";
    std::filesystem::create_directories(dir, ec);
    char name[32];
    snprintf(name, sizeof(name), "%04u.bmp", n++);
    vmaInvalidateAllocation(vma_, b.alloc, 0, VK_WHOLE_SIZE);
    FILE* f = std::fopen((dir / name).c_str(), "wb");
    if (!f) return;
    uint32_t bytes = bb_w_ * bb_h_ * 4;
    uint8_t hd[kBmpHeaderBytes] = {'B', 'M'};
    auto put = [&](int o, uint32_t v) { std::memcpy(hd + o, &v, 4); };
    put(kBmpFileSize, kBmpHeaderBytes + bytes);
    put(kBmpPixelOffset, kBmpHeaderBytes);
    put(kBmpInfoSize, kBmpInfoBytes);
    put(kBmpWidth, bb_w_);
    put(kBmpHeight, uint32_t(-int32_t(bb_h_)));
    hd[kBmpPlanes] = 1;
    hd[kBmpBitCount] = 32;
    put(kBmpImageSize, bytes);
    fwrite(hd, 1, kBmpHeaderBytes, f);
    fwrite(b.map, 1, bytes, f);
    fclose(f);
}

// depth is shown as colour through its depth view
void VulkanBackend::CapSave(Target& t, const char* name, float mul) {
    Image* r = Flat(t);
    if (!r) return;
    CapSave({.res = r, .w = t.gw, .h = t.gh, .mul = mul}, name);
}

void VulkanBackend::CapStep(const Surface& rt) {
    uint32_t d = rt.header ? mem::VirtualFromPhysical(rt.data) : 0;
    // XBE_CAPTURE_EVERY=N: the target also every N steps, to find the draw that spoils it
    static const uint32_t every = [] {
        const char* v = std::getenv("XBE_CAPTURE_EVERY");
        return v ? uint32_t(std::atoi(v)) : 0u;
    }();
    if (d == cap_rt_ && !(every && cap_i_ && cap_i_ % every == 0)) return;
    auto it = targets_.find(TargetMapKey(cap_rt_, false));
    if (cap_rt_ && it != targets_.end()) {
        char name[32];
        snprintf(name, sizeof(name), "%04u_%08X", cap_i_, cap_rt_);
        CapSave(it->second, name, it->second.range);
    }
    auto z = targets_.find(TargetMapKey(cap_rt_, true));
    if (cap_rt_ && z != targets_.end()) {
        char name[32];
        snprintf(name, sizeof(name), "%04u_%08X_z", cap_i_, cap_rt_);
        CapSave(z->second, name, 1);
    }
    cap_rt_ = d;
}

void VulkanBackend::FinishCapture(const Surface& front) {
    CapStep(Surface{});
    if (Target* f = FindTarget(front, false)) CapSave(*f, "final", f->range);
    for (auto& t : cap_tex_) {
        char name[32];
        snprintf(name, sizeof(name), "tex_%08X", t.data);
        CapSave({.res = t.res.get(), .w = t.w, .h = t.h}, name);
    }
    fclose(cap_);
    cap_ = nullptr;
    cap_tex_.clear();
    LOGI("vulkan: frame captured to %s", cap_dir_.c_str());
}

void VulkanBackend::SetGamma(const uint8_t* ramp) {
    std::memcpy(ramp_, ramp, sizeof(ramp_));
    ramp_dirty_ = true;
}

// the slot is not reused before the caller's wait: the title runs at most one present ahead
void VulkanBackend::WaitGpuFrame(uint64_t n) {
    uint64_t fence = present_fence_[n % (kFrames + 1)].load();
    if (fence > CompletedFence()) WaitFence(fence);
}

VulkanBackend* g_vulkan = nullptr;

}  // namespace

std::unique_ptr<Backend> CreateBackend() {
    auto backend = std::make_unique<VulkanBackend>();
    g_vulkan = backend.get();
    return backend;
}

void WaitGpuFrame(uint64_t n) {
    if (g_vulkan) g_vulkan->WaitGpuFrame(n);
}

}  // namespace gpu

// the touch controls' art as ImGui textures: the backend makes them on the frame it first draws them
ImTextureID vkui::Texture(const uint8_t* rgba, uint32_t w, uint32_t h) {
    return gpu::g_vulkan ? gpu::g_vulkan->AddUiTexture(rgba, w, h) : ImTextureID();
}
