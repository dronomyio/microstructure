#include "hawkes_engine.h"
extern "C" {
    void* create_hawkes_engine() { return new HawkesEngine(); }
    void destroy_hawkes_engine(void* engine) { delete static_cast<HawkesEngine*>(engine); }
    int process_events(void* engine, const double* timestamps, int n_events) {
        return static_cast<HawkesEngine*>(engine)->processEvents(timestamps, n_events);
    }
}
