// A relaunch on Linux replaces the process (kernel::Relaunch), so no previous instance holds a window for this one
#include "app/app.h"

namespace app {

bool StartHandover() { return false; }
bool HandingOver() { return false; }
void FinishHandover() {}

}  // namespace app
