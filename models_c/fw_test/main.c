#include <stdint.h>
#include "can_ids_embedded.c"

#define SYST_CSR (*(volatile uint32_t*)0xE000E010)
#define SYST_RVR (*(volatile uint32_t*)0xE000E014)
#define SYST_CVR (*(volatile uint32_t*)0xE000E018)

volatile uint32_t g_isr_count = 0;
volatile uint32_t g_isr_during_ids = 0;
volatile uint32_t g_ids_running = 0;
volatile int g_mismatch = 0;

void SysTick_Handler(void) {
    g_isr_count++;
    if (g_ids_running) {
        g_isr_during_ids++;
    }
}

void main(void) {
    SYST_RVR = 2000;
    SYST_CVR = 0;
    SYST_CSR = 0x7;

    double sample_r[8]    = {0.050000, 13.000000, 0.000000, 1.548795, 28.875000, -0.904808, 1.262812, 0.000977};
    double sample_dos[8]  = {0.700000, 7.000000, 1.000000, -0.000000, 0.000000, -0.400000, 0.000000, 0.000000};
    double sample_gear[8] = {0.250000, 16.000000, 0.000000, 2.405639, 66.000000, -0.210796, 5000.000000, 0.530762};

    for (int i = 0; i < 100; i++) {
        g_ids_running = 1;
        int r = can_ids_predict(sample_r);
        g_ids_running = 0;
        if (r != 0) g_mismatch = 1;

        g_ids_running = 1;
        int d = can_ids_predict(sample_dos);
        g_ids_running = 0;
        if (d != 1) g_mismatch = 1;

        g_ids_running = 1;
        int g = can_ids_predict(sample_gear);
        g_ids_running = 0;
        if (g != 4) g_mismatch = 1;
    }

    while (1) { }
}
