JARVIS Physical MCU F5 - 100,000 Inference Test
================================================
Target: NUCLEO-F446RE / STM32F446RE / Cortex-M4
Clock: 84 MHz (project clock configuration unchanged)
Frozen model: existing can_ids_embedded_v2 model in this project

Change from the previous F5 package:
- 8 fixed validation vectors are unchanged.
- Expected outputs are unchanged: {0, 1, 5, 5, 0, 2, 0, 2}.
- Repeats changed from 128 to 12,500.
- Total measured physical inference calls = 12,500 x 8 = 100,000.
- Warm-up calls are not included in f5_measurements.

CubeIDE debugger variables to record after LD2 turns ON:
- f5_system_core_clock_hz
- f5_measurements              (expected: 100000)
- f5_parity_failures           (target: 0)
- f5_cycles_min
- f5_cycles_max
- f5_cycles_sum
- f5_cycles_last
- f5_last_prediction
- f5_last_maha_distance

Mean cycles = f5_cycles_sum / f5_measurements
Latency (seconds) = cycles / f5_system_core_clock_hz
At 84 MHz: latency_ms = cycles / 84000.0

Interpretation note:
This is a 100,000-run endurance/repeatability test over 8 fixed validation vectors.
It is NOT 100,000 distinct CAN samples.
