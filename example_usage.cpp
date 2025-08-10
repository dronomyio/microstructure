#include "hawkes_engine.h"
#include <iostream>
#include <vector>
int main() {
    std::vector<double> timestamps = {1.0, 1.5, 2.1, 2.8, 3.2};
        //HawkesEngine engine;
    HawkesEngineConfig config;
    config.use_gpu = true;           // ← Force GPU on
    config.gpu_threshold = 1;        // ← Use GPU for any event count
    config.max_events = 10000;       // ← Safe memory limit
    config.gpu_device_id = 0;        // ← Use first GPU

    HawkesEngine engine(config);
    // Add this after creating the engine:
auto status = engine.get_status();
std::cout << "=== GPU Configuration Debug ===" << std::endl;
std::cout << "GPU Available: " << (status.gpu_available ? "YES" : "NO") << std::endl;
std::cout << "Max Events: " << status.max_events << std::endl;
std::cout << "GPU Device ID: " << status.gpu_device_id << std::endl;
std::cout << "===============================" << std::endl;
    int result = engine.processEvents(timestamps.data(), timestamps.size());
    std::cout << "Processed " << timestamps.size() << " events, result: " << result << std::endl;
    return 0;
}
