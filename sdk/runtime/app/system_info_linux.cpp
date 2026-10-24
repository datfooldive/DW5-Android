// the log's opening lines: what the game runs on and with which settings
#include <sys/utsname.h>
#include <unistd.h>

#include <ctime>
#include <fstream>
#include <string>

#include "app/app.h"
#include "core/log.h"
#include "image_symbols.h"

namespace app {

namespace {

// the value of the first line of a key: value file that starts with key
std::string Field(const char* file, const std::string& key, char sep) {
    std::ifstream f(file);
    for (std::string line; std::getline(f, line);) {
        if (line.compare(0, key.size(), key) != 0) continue;
        size_t at = line.find(sep);
        if (at == std::string::npos) continue;
        std::string v = line.substr(at + 1);
        v.erase(0, v.find_first_not_of(" \t\""));
        v.erase(v.find_last_not_of(" \t\"") + 1);
        return v;
    }
    return {};
}

std::string OsVersion() {
    std::string name = Field("/etc/os-release", "PRETTY_NAME=", '=');
    utsname u{};
    if (uname(&u) == 0) name += std::string(name.empty() ? "" : ", ") + u.sysname + " " + u.release;
    if (const char* session = std::getenv("XDG_SESSION_TYPE")) name += std::string(" (") + session + ")";
    return name.empty() ? "unknown" : name;
}

}  // namespace

void LogSystem() {
    std::time_t now = std::time(nullptr);
    char date[32] = {};
    std::strftime(date, sizeof(date), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    LOGR("%s (%s), %s", Utf8(kGame.title).c_str(), IMAGE_XBE, date);
    LOGR("Path: %s", ExeDir().c_str());
    LOGR("OS: %s", OsVersion().c_str());
    std::string cpu = Field("/proc/cpuinfo", "model name", ':');
    LOGR("CPU: %s (%ld threads)", cpu.empty() ? "unknown" : cpu.c_str(), sysconf(_SC_NPROCESSORS_ONLN));
    long pages = sysconf(_SC_PHYS_PAGES);
    long page = sysconf(_SC_PAGESIZE);
    LOGR("RAM: %.1f GB", double(pages) * double(page) / 1073741824.0);
    // the GPU is the renderer's to name, once it has picked one
    Mode desk = Desktop();
    LOGR("Desktop: %ux%u", desk.width, desk.height);
    LogSettings();
}

}  // namespace app
