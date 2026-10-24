// Linux builds the recompiled code into the executable (cmake/XbeRecomp.cmake): there is nothing to build at a start
#include "app/app.h"

namespace app {

bool PrepareGameCode(const GameFiles&) { return true; }

}  // namespace app
