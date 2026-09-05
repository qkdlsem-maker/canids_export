#include <stdint.h>
#include "can_ids_embedded.c"

#define RCC_APB1ENR (*(volatile uint32_t*)0x40023840)

#define CAN1_BASE 0x40006400
#define CAN_MCR   (*(volatile uint32_t*)(CAN1_BASE + 0x00))
#define CAN_MSR   (*(volatile uint32_t*)(CAN1_BASE + 0x04))
#define CAN_RF0R  (*(volatile uint32_t*)(CAN1_BASE + 0x0C))
#define CAN_BTR   (*(volatile uint32_t*)(CAN1_BASE + 0x1C))
#define CAN_RI0R  (*(volatile uint32_t*)(CAN1_BASE + 0x1B0))
#define CAN_RDTR0 (*(volatile uint32_t*)(CAN1_BASE + 0x1B4))
#define CAN_RDLR0 (*(volatile uint32_t*)(CAN1_BASE + 0x1B8))
#define CAN_RDHR0 (*(volatile uint32_t*)(CAN1_BASE + 0x1BC))

#define CAN_FMR   (*(volatile uint32_t*)(CAN1_BASE + 0x200))
#define CAN_FM1R  (*(volatile uint32_t*)(CAN1_BASE + 0x204))
#define CAN_FS1R  (*(volatile uint32_t*)(CAN1_BASE + 0x20C))
#define CAN_FFA1R (*(volatile uint32_t*)(CAN1_BASE + 0x214))
#define CAN_FA1R  (*(volatile uint32_t*)(CAN1_BASE + 0x21C))
#define CAN_F0R1  (*(volatile uint32_t*)(CAN1_BASE + 0x240))
#define CAN_F0R2  (*(volatile uint32_t*)(CAN1_BASE + 0x244))

volatile uint32_t g_msgs_received = 0;
volatile uint32_t g_overrun_count = 0;
volatile uint32_t g_ids_calls = 0;
volatile uint32_t g_done = 0;

#ifndef DRAIN_FIFO
#define DRAIN_FIFO 1
#endif

static void can1_init(void) {
    RCC_APB1ENR |= (1u << 25);

    CAN_MCR = (CAN_MCR & ~(1u << 1)) | 0x1;
    while (!(CAN_MSR & 0x1)) { }

    CAN_BTR = 0x001c0000;

    CAN_MCR &= ~0x1;
    while (CAN_MSR & 0x1) { }

    CAN_FMR |= 0x1;
    CAN_FA1R &= ~0x1;
    CAN_FM1R &= ~0x1;
    CAN_FS1R |= 0x1;
    CAN_F0R1 = 0x0;
    CAN_F0R2 = 0x0;
    CAN_FFA1R &= ~0x1;
    CAN_FA1R |= 0x1;
    CAN_FMR &= ~0x1;
}

void main(void) {
    can1_init();

    double sample[8] = {0.050000, 13.000000, 0.000000, 1.548795, 28.875000, -0.904808, 1.262812, 0.000977};

    while (1) {
        uint32_t rf0r = CAN_RF0R;
        if (rf0r & (1u << 4)) {
            g_overrun_count++;
            CAN_RF0R = rf0r;
        }

        uint32_t fmp0 = rf0r & 0x3;
        if (fmp0 > 0) {
#if DRAIN_FIFO
            volatile uint32_t id = CAN_RI0R;
            volatile uint32_t dlc = CAN_RDTR0;
            volatile uint32_t dl = CAN_RDLR0;
            volatile uint32_t dh = CAN_RDHR0;
            (void)id; (void)dlc; (void)dl; (void)dh;

            g_ids_calls++;
            can_ids_predict(sample);

            CAN_RF0R |= (1u << 5);
            g_msgs_received++;
#endif
        }

        g_done++;
        if (g_done > 2000000000U) {
            break;
        }
    }
    while (1) { }
}
