#include "hawkes_engine.h"
#include <iostream>
#include <vector>
int main() {
    std::vector<double> timestamps = {1.0, 1.5, 2.1, 2.8, 3.2};
    HawkesEngine engine;
    int result = engine.processEvents(timestamps.data(), timestamps.size());
    std::cout << "Processed " << timestamps.size() << " events, result: " << result << std::endl;
    return 0;
}
