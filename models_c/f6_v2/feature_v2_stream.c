#include "feature_v2_stream.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define V2_NUM_CAN_IDS 2048u
#define V2_VALID_BYTES (V2_NUM_CAN_IDS / 8u)

/*
 * Frozen statistics from:
 *   models/idagnostic_stats.pkl
 *
 * SHA256:
 *   04f3af021d6c9e3d52e6060c93d8fe14fd2a381242d8bbbeb93a0aee11d0f37b
 */
static const double GLOBAL_DELTA_MEAN = 38.177002161911595;
static const double GLOBAL_DELTA_STD  = 73.15781974453239;

static const double BYTE_POS_MEAN[8] = {
    59.83983448956544,
    43.563562607886,
    36.51474799623342,
    63.44021874034496,
    49.585436009696956,
    64.42127572524171,
    25.779640930089347,
    45.624118274161454
};

static const double BYTE_POS_STD[8] = {
    93.74467133819213,
    53.9960477801128,
    57.805561856829684,
    93.09714231996038,
    75.4460376368916,
    78.58607603834241,
    56.210402274924256,
    69.76311117028881
};

/*
 * Python:
 *
 *   id_window = []
 *
 * The array contains only the valid entries. Once full,
 * the oldest ID is overwritten as a circular buffer.
 */
static uint16_t id_window[V2_WINDOW_SIZE];
static uint32_t id_window_count;
static uint32_t id_window_next;

/*
 * Exact state corresponding to:
 *
 *   last_payload_per_id = {}
 *
 * Standard 11-bit CAN IDs allow direct indexing without
 * eviction or collisions.
 */
static uint8_t last_payload[V2_NUM_CAN_IDS][8];
static uint8_t last_payload_valid[V2_VALID_BYTES];

static int payload_was_seen(uint16_t can_id)
{
    uint32_t byte_index = ((uint32_t)can_id) >> 3;
    uint32_t bit_index  = ((uint32_t)can_id) & 7u;

    return (last_payload_valid[byte_index] &
            (uint8_t)(1u << bit_index)) != 0u;
}

static void mark_payload_seen(uint16_t can_id)
{
    uint32_t byte_index = ((uint32_t)can_id) >> 3;
    uint32_t bit_index  = ((uint32_t)can_id) & 7u;

    last_payload_valid[byte_index] |=
        (uint8_t)(1u << bit_index);
}

static double payload_entropy(const uint8_t payload[8])
{
    /*
     * Equivalent feature definition to:
     *
     *   v, c = np.unique(b, return_counts=True)
     *   p = c / c.sum()
     *   -np.sum(p * np.log2(p + 1e-12))
     *
     * There are exactly eight payload positions.
     *
     * We count equal byte values directly. The 1e-12 term
     * from Python is retained for numerical parity.
     */
    uint8_t visited[8] = {0};
    double entropy = 0.0;

    for (uint32_t i = 0; i < 8u; ++i) {
        uint32_t count = 0u;

        if (visited[i]) {
            continue;
        }

        for (uint32_t j = i; j < 8u; ++j) {
            if (payload[j] == payload[i]) {
                visited[j] = 1u;
                count++;
            }
        }

        {
            double p = (double)count / 8.0;
            entropy -= p * (log(p + 1e-12) / log(2.0));
        }
    }

    return entropy;
}

void feature_v2_reset(void)
{
    memset(id_window, 0, sizeof(id_window));
    id_window_count = 0u;
    id_window_next = 0u;

    memset(last_payload, 0, sizeof(last_payload));
    memset(last_payload_valid, 0, sizeof(last_payload_valid));
}

int feature_v2_process(
    uint16_t can_id,
    const uint8_t payload[8],
    double out[V2_NUM_FEATURES])
{
    uint32_t current_count;
    uint32_t same_id_count = 0u;
    uint32_t unique_count = 0u;

    double mean_byte = 0.0;
    double delta = 0.0;
    double global_delta_zscore;
    double global_value_zscore = 0.0;

    if (can_id >= V2_NUM_CAN_IDS || payload == NULL || out == NULL) {
        return -1;
    }

    /*
     * Python semantics:
     *
     *   id_window.append(cid)
     *   if len(id_window) > WINDOW:
     *       id_window.pop(0)
     *
     * The current frame is therefore included before
     * freq/unique features are calculated.
     */
    id_window[id_window_next] = can_id;
    id_window_next = (id_window_next + 1u) % V2_WINDOW_SIZE;

    if (id_window_count < V2_WINDOW_SIZE) {
        id_window_count++;
    }

    current_count = id_window_count;

    /*
     * freq_in_window
     */
    for (uint32_t i = 0; i < current_count; ++i) {
        if (id_window[i] == can_id) {
            same_id_count++;
        }
    }

    out[0] = (double)same_id_count / (double)current_count;

    /*
     * unique_ids_in_window
     *
     * WINDOW is only 20, so an O(WINDOW^2) implementation
     * avoids allocating another 2048-ID temporary structure.
     */
    for (uint32_t i = 0; i < current_count; ++i) {
        int appeared_before = 0;

        for (uint32_t j = 0; j < i; ++j) {
            if (id_window[j] == id_window[i]) {
                appeared_before = 1;
                break;
            }
        }

        if (!appeared_before) {
            unique_count++;
        }
    }

    out[1] = (double)unique_count;

    /*
     * entropy
     */
    out[2] = payload_entropy(payload);

    /*
     * mean_byte
     */
    for (uint32_t i = 0; i < 8u; ++i) {
        mean_byte += (double)payload[i];
    }

    out[3] = mean_byte / 8.0;

    /*
     * global_delta_zscore
     *
     * First occurrence of a CAN ID uses delta = 0,
     * exactly as in the Python V2 builder.
     */
    if (payload_was_seen(can_id)) {
        for (uint32_t i = 0; i < 8u; ++i) {
            int d = (int)payload[i] -
                    (int)last_payload[can_id][i];

            if (d < 0) {
                d = -d;
            }

            delta += (double)d;
        }
    }

    memcpy(last_payload[can_id], payload, 8u);
    mark_payload_seen(can_id);

    global_delta_zscore =
        (delta - GLOBAL_DELTA_MEAN) / GLOBAL_DELTA_STD;

    out[4] = global_delta_zscore;

    /*
     * global_value_zscore
     */
    for (uint32_t i = 0; i < 8u; ++i) {
        double z =
            ((double)payload[i] - BYTE_POS_MEAN[i]) /
            BYTE_POS_STD[i];

        if (z < 0.0) {
            z = -z;
        }

        if (z > global_value_zscore) {
            global_value_zscore = z;
        }
    }

    out[5] = global_value_zscore;

    return 0;
}
