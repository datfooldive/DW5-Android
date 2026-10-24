#pragma once
// the few C runtime names the sources share between Windows and Linux
#ifndef _WIN32
#include <strings.h>

inline int _stricmp(const char* a, const char* b) { return strcasecmp(a, b); }
inline int _strnicmp(const char* a, const char* b, size_t n) { return strncasecmp(a, b, n); }
#endif
