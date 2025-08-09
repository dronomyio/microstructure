#include "hawkes_engine.h"
#include <chrono>
#include <iostream>
#include <vector>
int main() {
    std::vector<double> timestamps(10000, 1.0);
    HawkesEngine engine;
    auto start = std::chrono::high_resolution_clock::now();
    engine.processEvents(timestamps.data(), timestamps.size());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Benchmark: " << duration.count() << " microseconds" << std::endl;
    return 0;
}
