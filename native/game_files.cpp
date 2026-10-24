// The game on the phone: its folder on the shared storage or in the app's own, the extracted files or a disc image.
// The xbe must be the one this executable was recompiled from; another release's would run its own data under this code.
#include <algorithm>
#include <cwctype>
#include <fstream>

#include "app/app.h"
#include "core/log.h"
#include "host.h"
#include "image_symbols.h"
#include "kernel/crypto.h"
#include "kernel/disc.h"

extern const char kImageSha1[];  // the xbe recompiled into this executable, written by CMake

namespace app {

extern std::filesystem::path g_game_dir;  // native/main.cpp: what host::GameDir reports

namespace {

namespace fs = std::filesystem;

// where a player's copy is looked for: a DW5 folder at the top of the phone's storage and of each card, the app's
// own files folder, and the internal one
std::vector<fs::path> Candidates() {
    std::vector<fs::path> out = {
        "/storage/emulated/0/DW5",
        "/storage/emulated/0/Android/data/com.datfooldive.dw5/files/assets",
        "/sdcard/DW5",
    };
    std::error_code ec;
    for (auto& e : fs::directory_iterator("/storage", ec))
        if (e.is_directory(ec) && e.path().filename() != "emulated" && e.path().filename() != "self")
            out.push_back(e.path() / "DW5");
    out.push_back(ExeDir() / "assets");
    return out;
}

bool ReadHostFile(const fs::path& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), {});
    return true;
}

bool HasIso(const fs::path& dir, fs::path& iso) {
    std::error_code ec;
    for (auto& e : fs::directory_iterator(dir, ec)) {
        std::wstring ext = e.path().extension().wstring();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](wchar_t c) { return wchar_t(std::towlower(c)); });
        if (ext != L".iso" || !e.is_regular_file(ec)) continue;
        iso = e.path();
        return true;
    }
    return false;
}

}  // namespace

std::string Sha1(const std::vector<uint8_t>& data) {
    alignas(8) uint8_t ctx[128];
    uint8_t digest[20];
    crypto::ShaInit(ctx);
    crypto::ShaUpdate(ctx, data.data(), data.size());
    crypto::ShaFinal(ctx, digest);
    std::string hex;
    for (uint8_t b : digest) {
        hex += "0123456789abcdef"[b >> 4];
        hex += "0123456789abcdef"[b & 15];
    }
    return hex;
}

bool FindGame(GameFiles& game) {
    std::vector<fs::path> dirs = Candidates();
    std::sort(dirs.begin(), dirs.end());
    dirs.erase(std::unique(dirs.begin(), dirs.end()), dirs.end());

    // the first copy with this xbe wins, the extracted files before any image; another release is kept to offer
    GameFiles other;
    std::unique_ptr<kernel::Disc> other_disc;
    fs::path other_name;
    auto consider = [&](const fs::path& dir, const fs::path& name, std::vector<uint8_t> xbe,
                        std::unique_ptr<kernel::Disc> disc) {
        std::string sha = Sha1(xbe);
        bool match = sha == kImageSha1;
        LOGR("Game: %s in %s, sha1 %s%s", IMAGE_XBE, name.string().c_str(), sha.c_str(), match ? "" : ", another release");
        if (match) {
            game = {dir, std::move(xbe)};
            kernel::MountDisc(std::move(disc));
            return true;
        }
        if (other_name.empty()) {
            other = {dir, std::move(xbe)};
            other_disc = std::move(disc);
            other_name = name;
        }
        return false;
    };

    for (const fs::path& dir : dirs) {
        std::error_code ec;
        if (!fs::is_directory(dir, ec)) continue;
        std::vector<uint8_t> xbe;
        if (ReadHostFile(dir / IMAGE_XBE, xbe) && consider(dir, dir / IMAGE_XBE, std::move(xbe), nullptr)) {
            g_game_dir = dir;
            return true;
        }
    }
    for (const fs::path& dir : dirs) {
        std::error_code ec;
        if (!fs::is_directory(dir, ec)) continue;
        fs::path iso;
        if (!HasIso(dir, iso)) continue;
        auto disc = std::make_unique<kernel::Disc>();
        std::string error;
        std::vector<uint8_t> xbe;
        if (!disc->Open(iso, error)) {
            LOGW("game: %s: %s", iso.string().c_str(), error.c_str());
            continue;
        }
        if (!disc->Load(IMAGE_XBE, xbe)) {
            LOGW("game: %s has no %s", iso.string().c_str(), IMAGE_XBE);
            continue;
        }
        if (consider(dir, iso, std::move(xbe), std::move(disc))) {
            g_game_dir = dir;
            return true;
        }
    }

    if (other_name.empty()) {
        ShowError(L"Put the game in a DW5 folder at the top of the phone's storage: its files, or the disc image "
                  L"(.iso). See the log in the app's files folder.");
        return false;
    }
    // another release boots its own data under this code, which usually crashes: not offered here
    ShowError(L"the game in " + std::wstring(other_name.wstring()) +
              L" is a different release than this build was made from");
    return false;
}

}  // namespace app
