# Dynasty Warriors 5 (original Xbox), statically recompiled, as an Android library: libmain.so, loaded by SDL's
# activity. The x86 code is recompiled on the PC first (build.sh), this build takes generated/ as it is, with the SDK's
# portable runtime, the game's sources in src and the Android host in native.

cmake_minimum_required(VERSION 3.22)
project(dw5_android LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(REPO ${CMAKE_CURRENT_SOURCE_DIR})
set(NATIVE ${REPO}/native)
set(GAME ${REPO}/src)
set(SDK ${REPO}/sdk)

if(NOT EXISTS ${SDK}/runtime/gpu/backend.h)
    message(FATAL_ERROR "The SDK is missing from ${SDK}")
endif()

# the recompiled code and the xbe it came from
set(XBE_GENERATED ${REPO}/generated/game CACHE PATH "the recompiled code of default.xbe")
set(XBE_IMAGE ${REPO}/assets/default.xbe CACHE FILEPATH "the xbe it was recompiled from")
set(GEN ${XBE_GENERATED})
set(XBE ${XBE_IMAGE})
if(NOT EXISTS ${GEN}/x86_func_mapping.cpp OR NOT EXISTS ${XBE})
    message(FATAL_ERROR "No recompiled code: put the game in assets/ and run build.sh, it writes generated/game")
endif()
file(STRINGS ${REPO}/config/game.toml GAME_PARTITIONS REGEX "^partitions *=")
string(REGEX MATCH "[0-9]+" GAME_PARTITIONS "${GAME_PARTITIONS}")
math(EXPR last "${GAME_PARTITIONS} - 1")
set(GEN_SOURCES ${GEN}/x86_func_mapping.cpp)
foreach(i RANGE ${last})
    list(APPEND GEN_SOURCES ${GEN}/x86_recomp.${i}.cpp)
endforeach()
file(SHA1 ${XBE} xbe_sha1)
file(CONFIGURE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/image_sha1.cpp
     CONTENT "extern const char kImageSha1[] = \"${xbe_sha1}\";\n")

# SDL3 from the checkout Gradle makes (its Java side goes into the APK), the rest fetched
set(SDL_DIR ${REPO}/third_party/SDL)
if(NOT EXISTS ${SDL_DIR}/CMakeLists.txt)
    message(FATAL_ERROR "SDL3 is missing from ${SDL_DIR}: the Gradle build clones it")
endif()
set(SDL_SHARED ON CACHE BOOL "" FORCE)
set(SDL_STATIC OFF CACHE BOOL "" FORCE)
set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
set(SDL_RENDER OFF CACHE BOOL "" FORCE)
set(SDL_GPU OFF CACHE BOOL "" FORCE)
set(SDL_CAMERA OFF CACHE BOOL "" FORCE)
add_subdirectory(${SDL_DIR} ${CMAKE_CURRENT_BINARY_DIR}/SDL EXCLUDE_FROM_ALL)

include(FetchContent)
set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
set(ENABLE_HLSL ON CACHE BOOL "" FORCE)
set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
set(ENABLE_SPVREMAPPER OFF CACHE BOOL "" FORCE)
set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
set(GLSLANG_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BUILD_EXTERNAL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glslang GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git GIT_TAG 16.0.0 GIT_SHALLOW TRUE)
FetchContent_Declare(tomlplusplus GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git GIT_TAG v3.4.0 GIT_SHALLOW TRUE)
FetchContent_Declare(imgui GIT_REPOSITORY https://github.com/ocornut/imgui.git GIT_TAG v1.91.0 GIT_SHALLOW TRUE)
# the header alone: SOURCE_SUBDIR names no folder, so its own CMake project is not added
FetchContent_Declare(vma GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
                     GIT_TAG v3.1.0 GIT_SHALLOW TRUE SOURCE_SUBDIR none)
# a checkout in third_party/ is used as it is: the build does not download what is already here
foreach(dep glslang tomlplusplus imgui vma)
    string(TOUPPER ${dep} DEP)
    if(EXISTS ${REPO}/third_party/${dep}/CMakeLists.txt OR EXISTS ${REPO}/third_party/${dep}/imgui.h)
        set(FETCHCONTENT_SOURCE_DIR_${DEP} ${REPO}/third_party/${dep})
    endif()
endforeach()
FetchContent_MakeAvailable(glslang tomlplusplus imgui vma)

# the SDK's runtime without its desktop host: native/ is the window, the loop and the Vulkan backend here
file(GLOB RUNTIME_SOURCES ${SDK}/runtime/*/*.cpp)
list(FILTER RUNTIME_SOURCES EXCLUDE REGEX
     "/runtime/app/(main|splash|system_info|game_code|handover)(_linux)?\\.cpp$|/runtime/app/game_files\\.cpp$|/runtime/gpu/(d3d12|render_thread)\\.cpp$|/runtime/audio/xaudio2\\.cpp$")
# the game's sources, its own Vulkan backend and Linux game_files left out
file(GLOB GAME_HOST_SOURCES CONFIGURE_DEPENDS ${GAME}/*/*.cpp)
list(FILTER GAME_HOST_SOURCES EXCLUDE REGEX "/src/d3d/vulkan_linux\\.cpp$|/src/common/game_files\\.cpp$")
file(GLOB ANDROID_SOURCES CONFIGURE_DEPENDS ${NATIVE}/*.cpp)
set(IMGUI_SOURCES ${imgui_SOURCE_DIR}/imgui.cpp ${imgui_SOURCE_DIR}/imgui_draw.cpp ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp)
# the backend loads its Vulkan functions through the renderer (VulkanBackend::InitUi): the KHR rendering entry points a
# Vulkan 1.1 device has are not reliably given out by the Android loader's vkGetInstanceProcAddr
set_source_files_properties(${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp PROPERTIES
    COMPILE_DEFINITIONS IMGUI_IMPL_VULKAN_NO_PROTOTYPES)

add_library(main SHARED ${RUNTIME_SOURCES} ${GAME_HOST_SOURCES} ${ANDROID_SOURCES} ${GEN_SOURCES} ${IMGUI_SOURCES}
            ${CMAKE_CURRENT_BINARY_DIR}/image_sha1.cpp)
target_include_directories(main PRIVATE ${NATIVE}/compat ${SDK}/harness/compat ${NATIVE} ${GAME} ${SDK}/runtime ${GEN}
                           ${CMAKE_CURRENT_BINARY_DIR} ${glslang_SOURCE_DIR})
target_include_directories(main SYSTEM PRIVATE ${imgui_SOURCE_DIR} ${vma_SOURCE_DIR}/include)
# x86 chars are signed, as the game's code and the runtime were written for
target_compile_options(main PRIVATE -fno-strict-aliasing -fsigned-char)
set(HOST_SOURCES ${RUNTIME_SOURCES} ${GAME_HOST_SOURCES} ${ANDROID_SOURCES})
set_source_files_properties(${HOST_SOURCES} PROPERTIES COMPILE_OPTIONS "-include;${NATIVE}/compat/crt.h")
set_source_files_properties(${GEN_SOURCES} PROPERTIES
    COMPILE_OPTIONS "-O2;-w;-fno-strict-aliasing;-fsigned-char")
# glslang's HLSL front end is behind ENABLE_HLSL, which its own target keeps to itself
set_source_files_properties(${NATIVE}/vk_shader.cpp PROPERTIES COMPILE_DEFINITIONS ENABLE_HLSL)
target_link_libraries(main PRIVATE SDL3::SDL3 tomlplusplus::tomlplusplus glslang glslang-default-resource-limits
                      vulkan android log)
# 16 KB pages, as Android 15 devices may have
target_link_options(main PRIVATE -Wl,-z,max-page-size=16384)
