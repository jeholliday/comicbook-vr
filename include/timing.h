#ifndef VRVISOR_TIMING_H
#define VRVISOR_TIMING_H

#include <chrono>

using namespace std::chrono;

#define START_TIMING() auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch());

#define STOP_TIMING(name)                                                                                                                  \
    auto end = duration_cast<milliseconds>(system_clock::now().time_since_epoch());                                                        \
    std::cout << name << ": " << (end - start).count() << " ms" << std::endl;

#endif // VRVISOR_TIMING_H
