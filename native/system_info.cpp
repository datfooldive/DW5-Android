// the log's opening lines: what the phone runs and with which settings
#include <sys/system_properties.h>
#include <unistd.h>

#include <ctime>
#include <fstream>
#include <string>

#include "app/app.h"
#include "core/log.h"
#include "image_symbols.h"

namespace app {

namespace {

std::string Prop(const char* name) {
    char value[PROP_VALUE_MAX] = {};
    int n = __system_property_get(name, value);
    return n > 0 ? std::string(value, size_t(n)) : std::string();
}

std::string CpuName() {
    std::ifstream f("/proc/cpuinfo");
    for (std::string line; std::getline(f, line);)
        if (line.compare(0, 9, "Hardware:") == 0 || line.compare(0, 10, "model name") == 0) return line.substr(line.find(':') + 2);
    return Prop("ro.soc.model");
}

}  // namespace

void LogSystem() {
    std::time_t now = std::time(nullptr);
    char date[32] = {};
    std::strftime(date, sizeof(date), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    LOGR("%s (%s), %s", Utf8(kGame.title).c_str(), IMAGE_XBE, date);
    LOGR("Path: %s", ExeDir().c_str());
    LOGR("OS: Android %s (SDK %s), %s", Prop("ro.build.version.release").c_str(),
         Prop("ro.build.version.sdk").c_str(), Prop("ro.product.model").c_str());
    LOGR("CPU: %s (%ld threads)", CpuName().c_str(), sysconf(_SC_NPROCESSORS_ONLN));
    long pages = sysconf(_SC_PHYS_PAGES);
    long page = sysconf(_SC_PAGESIZE);
    LOGR("RAM: %.1f GB", double(pages) * double(page) / 1073741824.0);
    // the GPU is the renderer's to name, once it has picked one
    Mode desk = Desktop();
    LOGR("Screen: %ux%u", desk.width, desk.height);
    LogSettings();
}

}  // namespace app
