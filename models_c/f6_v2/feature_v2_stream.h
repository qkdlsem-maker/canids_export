#ifndef FEATURE_V2_STREAM_H
#define FEATURE_V2_STREAM_H

#include <stdint.h>

#define V2_NUM_FEATURES 6
#define V2_WINDOW_SIZE 20

/*
 * Reset all streaming feature state.
 *
 * This reproduces the initial state of
 * src/13_build_features_v2.py:
 *
 *   id_window = []
 *   last_payload_per_id = {}
 */
void feature_v2_reset(void);

/*
 * Process one standard 11-bit CAN frame.
 *
 * can_id:
 *   Standard CAN identifier, 0x000..0x7FF.
 *
 * payload:
 *   Exactly 8 bytes.
 *   Frames with DLC < 8 must be zero-padded by the caller,
 *   matching the Python V2 preprocessing.
 *
 * out:
 *   [0] freq_in_window
 *   [1] unique_ids_in_window
 *   [2] entropy
 *   [3] mean_byte
 *   [4] global_delta_zscore
 *   [5] global_value_zscore
 *
 * Returns:
 *   0  success
 *  -1  invalid CAN ID
 */
int feature_v2_process(
    uint16_t can_id,
    const uint8_t payload[8],
    double out[V2_NUM_FEATURES]
);

#endif
