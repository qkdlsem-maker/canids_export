#include <stdint.h>

/*
 * Frozen final V2 hybrid IDS benchmark.
 *
 * Inputs are deterministic representative vectors taken from the
 * V2 HCRL test region and correctly classified by frozen lgbm_v2.
 *
 * Feature order:
 *   0 freq_in_window
 *   1 unique_ids_in_window
 *   2 entropy
 *   3 mean_byte
 *   4 global_delta_zscore
 *   5 global_value_zscore
 */

#define NUM_CASES 5
#define NUM_FEATURES 6

extern int can_ids_predict(const double *x);

/*
 * Case index:
 *   0 = DoS
 *   1 = Fuzzy
 *   2 = R (Normal)
 *   3 = RPM
 *   4 = gear
 */
static const double samples[NUM_CASES][NUM_FEATURES] = {
    {
        0.15000000596046448,
        18.0,
        -1.4428232913976657e-12,
        0.0,
        -0.52184444665908813,
        0.81975430250167847
    },
    {
        0.05000000074505806,
        20.0,
        3.0,
        208.0,
        5.7932701110839844,
        3.5824184417724609
    },
    {
        0.10000000149011612,
        19.0,
        0.54356443881988525,
        2.5,
        -0.52184444665908813,
        0.81975430250167847
    },
    {
        0.30000001192092896,
        13.0,
        2.25,
        91.625,
        -0.52184444665908813,
        3.0012404918670654
    },
    {
        0.15000000596046448,
        16.0,
        2.4056391716003418,
        66.0,
        1.0774377584457397,
        2.0576333999633789
    }
};

/*
 * These globals are intentionally volatile so Renode can control/read them.
 */
volatile uint32_t g_case_index = 0;
volatile uint32_t g_ready_flag = 0;
volatile uint32_t g_start_flag = 0;
volatile uint32_t g_done_flag = 0;
volatile int32_t  g_result = -1;

int main(void)
{
    /*
     * Renode control point.
     * Hook immediately after this store, then write g_case_index.
     */
    g_ready_flag = 1;

    uint32_t idx = g_case_index;

    if (idx >= NUM_CASES) {
        idx = 0;
    }

    g_start_flag = 1;

    /*
     * Measurement boundary:
     * start immediately before this call,
     * stop immediately after it returns.
     */
    g_result = can_ids_predict(samples[idx]);

    g_done_flag = 1;

    while (1) {
    }

    return 0;
}
