/* #include "hawkes_engine.h"
#include <iostream>
int main() {
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

    double event = 1.0;
    int result = engine.processEvents(&event, 1);
    std::cout << "Test result: " << result << std::endl;
    return 0;
}*/

#include "hawkes_engine.h"
#include <iostream>

int main() {
    HawkesEngineConfig config;
    config.use_gpu = true;
    config.gpu_threshold = 1;
    config.max_events = 10000;       // Safe for each GPU

    // 4-GPU DISTRIBUTED CONFIGURATION:
    config.distributed_gpu = true;   // ← Enable distributed processing
    config.num_gpus = 4;            // ← Use all 4 GPUs
    config.gpu_device_ids = {0, 1, 2, 3};  // ← Specify all GPU IDs

    HawkesEngine engine(config);

    auto status = engine.get_status();
    std::cout << "=== 4-GPU Distributed Debug ===" << std::endl;
    std::cout << "GPU Available: " << (status.gpu_available ? "YES" : "NO") << std::endl;
    std::cout << "Distributed GPUs: " << status.num_gpus << std::endl;
    std::cout << "Max Events per GPU: " << status.max_events << std::endl;
    std::cout << "Total Processing Power: " << (status.max_events * status.num_gpus) << " events" << std::endl;
    std::cout << "===============================" << std::endl;

    double event = 1.0;
    int result = engine.processEvents(&event, 1);
    std::cout << "Test result: " << result << std::endl;
    return 0;
}

