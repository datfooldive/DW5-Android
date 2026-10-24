#!/usr/bin/env bash
# DW5 Android: recompiles default.xbe on this PC, then builds the APK. Everything runs on Linux: the recompiler is
# built with the host's clang, the APK with the NDK Gradle points at.
set -e
GAME_TITLE="Dynasty Warriors 5"
GAME_XBE=default.xbe
cd "$(dirname "$0")"

echo
echo "  $GAME_TITLE"
echo "  =================="
echo

fail() {
    echo
    echo "  BUILD FAILED. $1"
    echo
    exit 1
}

if [ ! -f "assets/$GAME_XBE" ]; then
    echo "  Put the game in the assets folder first:"
    echo "    $PWD/assets"
    echo "  It needs $GAME_XBE, or the disc image (.iso) beside it."
    fail ""
fi

export ANDROID_HOME="${ANDROID_HOME:-${HOME}/Android/Sdk}"
export ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-${ANDROID_HOME}}"
[ -f local.properties ] || echo "sdk.dir=${ANDROID_HOME}" > local.properties

# ---------------------------------------------------------------- recompiled code
echo "[1/2] Recompiled code"
if [ -f generated/game/x86_func_mapping.cpp ]; then
    echo "      generated/game is there"
else
    echo "      building xbe_recomp and recompiling $GAME_XBE: 15 to 60 minutes ..."
    cmake -S tools/host -B out/host -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null || fail "Read the last message above."
    cmake --build out/host || fail "Read the last message above."
    ./out/host/xbe_recomp config/game.toml --xdk sdk/xdk || fail "Read the last message above."
fi

# ---------------------------------------------------------------- apk
echo "[2/2] APK"
./gradlew assembleRelease || fail "Read the last message above."

echo
echo "  $GAME_TITLE built."
echo "  app/build/outputs/apk/release/app-release.apk"
echo
