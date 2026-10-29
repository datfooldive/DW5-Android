// The settings screen is the phone's own, not the game's: DW5Activity draws it in Java (SettingsDialog.java), so a
// value is read once when the panel opens and written one at a time as it is changed, across JNI rather than every
// frame. The gear in native/touch.cpp asks for it through host::ShowSettings.
#include <jni.h>

#include <algorithm>
#include <cstdint>

#include <SDL3/SDL.h>
#include <SDL3/SDL_system.h>

#include "app/app.h"
#include "common/options.h"
#include "host.h"

namespace {

// the order the dialog's values are in, the same numbers SettingsDialog.java has
enum Key : int {
    kResolution, kAntiAlias, kAniso, kVsync, kFrameSkip, kAspect, kFpsCounter, kHdr, kHdrAvailable, kControlsScale,
    kControlsOpacity, kOutWidth, kOutHeight, kKeyCount
};

// the one index the panel shows for the two settings anti-aliasing is made of: 0 off, 1 SMAA, 2 MSAA 2X, 3 MSAA 4X
int AntiAliasIndex(const game::Options& o) {
    if (o.msaa >= 4) return 3;
    if (o.msaa >= 2) return 2;
    return o.smaa ? 1 : 0;
}

}  // namespace

void host::ShowSettings() {
    JNIEnv* env = static_cast<JNIEnv*>(SDL_GetAndroidJNIEnv());
    if (!env) return;
    jobject activity = static_cast<jobject>(SDL_GetAndroidActivity());
    if (!activity) return;
    jclass cls = env->GetObjectClass(activity);
    if (!cls) return;
    jmethodID mid = env->GetMethodID(cls, "showSettings", "()V");
    if (mid) env->CallVoidMethod(activity, mid);
    env->DeleteLocalRef(cls);
}

// the panel's values as one array, read when it opens
extern "C" JNIEXPORT jintArray JNICALL Java_com_datfooldive_dw5_DW5Activity_nativeSettings(JNIEnv* env, jclass) {
    const app::Settings& s = app::Get();
    const game::Options& o = game::Opt();
    const app::Mode out = app::Output();
    jint v[kKeyCount];
    v[kResolution] = jint(s.render_percent);
    v[kAntiAlias] = AntiAliasIndex(o);
    v[kAniso] = jint(s.aniso);
    v[kVsync] = s.vsync ? 1 : 0;
    v[kFrameSkip] = o.frame_skip ? 1 : 0;
    v[kAspect] = jint(o.aspect);
    v[kFpsCounter] = o.fps_counter ? 1 : 0;
    v[kHdr] = o.hdr_output ? 1 : 0;
    v[kHdrAvailable] = game::g_hdr_display.load() ? 1 : 0;
    v[kControlsScale] = jint(o.controls_scale);
    v[kControlsOpacity] = jint(o.controls_opacity);
    v[kOutWidth] = jint(out.width);
    v[kOutHeight] = jint(out.height);
    jintArray arr = env->NewIntArray(kKeyCount);
    if (arr) env->SetIntArrayRegion(arr, 0, kKeyCount, v);
    return arr;
}

// one value as the panel moves it; commit writes the file, which a slider only does when it is let go
extern "C" JNIEXPORT void JNICALL Java_com_datfooldive_dw5_DW5Activity_nativeSettingsSet(JNIEnv*, jclass, jint key,
                                                                                       jint value, jboolean commit) {
    app::Settings& s = app::Get();
    game::Options& o = game::Opt();
    switch (key) {
        case kResolution: s.render_percent = uint32_t(std::clamp<jint>(value, 20, 200)); break;
        case kAntiAlias: {
            const int i = std::clamp<jint>(value, 0, 3);
            o.smaa = i == 1;
            o.msaa = i == 2 ? 2 : i == 3 ? 4 : 0;
            break;
        }
        case kAniso: s.aniso = uint32_t(std::clamp<jint>(value, 0, 16)); break;
        case kVsync: s.vsync = value != 0; break;
        case kFrameSkip: o.frame_skip = value != 0; break;
        case kAspect: o.aspect = uint32_t(std::clamp<jint>(value, 0, 4)); break;
        case kFpsCounter: o.fps_counter = value != 0; break;
        case kHdr: o.hdr_output = value != 0 && game::g_hdr_display.load(); break;
        case kControlsScale: o.controls_scale = uint32_t(std::clamp<jint>(value, 50, 150)); break;
        case kControlsOpacity: o.controls_opacity = uint32_t(std::clamp<jint>(value, 20, 100)); break;
        default: return;
    }
    if (commit) app::Save();
}

// the settings back to a fresh install's, the controls' layout with them
extern "C" JNIEXPORT void JNICALL Java_com_datfooldive_dw5_DW5Activity_nativeSettingsDefaults(JNIEnv*, jclass) {
    app::Settings& s = app::Get();
    game::Options& o = game::Opt();
    s.render_percent = 67;
    s.aniso = 0;
    s.vsync = true;
    o.smaa = false;
    o.msaa = 0;
    o.frame_skip = true;
    o.aspect = 4;
    o.fps_counter = true;
    o.hdr_output = false;
    o.controls_scale = 100;
    o.controls_opacity = 100;
    touch::EditReset();
    app::Save();
}

// the panel's way into the layout editor: it closes, and a finger moves the controls instead of pressing them
extern "C" JNIEXPORT void JNICALL Java_com_datfooldive_dw5_DW5Activity_nativeEditLayout(JNIEnv*, jclass) {
    touch::EditBegin();
}

extern "C" JNIEXPORT void JNICALL Java_com_datfooldive_dw5_DW5Activity_nativeEditEnd(JNIEnv*, jclass) {
    touch::EditEnd();
}

extern "C" JNIEXPORT void JNICALL Java_com_datfooldive_dw5_DW5Activity_nativeEditReset(JNIEnv*, jclass) {
    touch::EditReset();
}
