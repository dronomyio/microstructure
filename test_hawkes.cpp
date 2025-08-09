#include "hawkes_engine.h"
#include <iostream>
int main() {
    HawkesEngine engine;
    double event = 1.0;
    int result = engine.processEvents(&event, 1);
    std::cout << "Test result: " << result << std::endl;
    return 0;
}
