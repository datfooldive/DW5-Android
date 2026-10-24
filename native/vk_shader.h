#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

// HLSL to SPIR-V through glslang, with a cache on disk: the Android NDK ships no shaderc
namespace vkshader {

// HLSL registers as bindings: b0 b1 at 0, t0 to t3 at 2, s0 to s3 at 6, as the renderer's descriptor layout has them
constexpr uint32_t kBindingCb = 0;
constexpr uint32_t kBindingTex = 2;
constexpr uint32_t kBindingSampler = 6;

enum class Stage { Vertex, Fragment };

// clip_distance false: the vertex shader writes no SV_ClipDistance (Mali's compiler crashes on it, Mali-G57 driver
// 0D801000, and a GPU without shaderClipDistance cannot take it)
void Init(const std::filesystem::path& cache_dir, bool clip_distance);
// hash names the module in the cache: the renderer's hash of the source, entry and profile
bool Compile(const std::string& hlsl, const char* entry, Stage stage, uint64_t hash, std::vector<uint32_t>& spirv);

}  // namespace vkshader
