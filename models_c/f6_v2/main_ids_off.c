#include <stdint.h>
#include "feature_v2_stream.h"

/*
 * Frozen final V2 hybrid IDS exported from:
 *   models/lgbm_v2.txt
 *   models/maha_v2.npz
 */
int can_ids_predict(const double *x);

#define REG32(a) (*(volatile uint32_t *)(a))

#define RCC_APB1ENR REG32(0x40023840)

#define CAN1_BASE 0x40006400u

#define CAN_MCR   REG32(CAN1_BASE + 0x00)
#define CAN_MSR   REG32(CAN1_BASE + 0x04)
#define CAN_IER   REG32(CAN1_BASE + 0x14)
#define CAN_BTR   REG32(CAN1_BASE + 0x1C)

#define CAN_RF0R  REG32(CAN1_BASE + 0x0C)
#define CAN_RI0R  REG32(CAN1_BASE + 0x1B0)
#define CAN_RDTR0 REG32(CAN1_BASE + 0x1B4)
#define CAN_RDLR0 REG32(CAN1_BASE + 0x1B8)
#define CAN_RDHR0 REG32(CAN1_BASE + 0x1BC)

#define CAN_FMR   REG32(CAN1_BASE + 0x200)
#define CAN_FM1R  REG32(CAN1_BASE + 0x204)
#define CAN_FS1R  REG32(CAN1_BASE + 0x20C)
#define CAN_FFA1R REG32(CAN1_BASE + 0x214)
#define CAN_FA1R  REG32(CAN1_BASE + 0x21C)
#define CAN_F0R1  REG32(CAN1_BASE + 0x240)
#define CAN_F0R2  REG32(CAN1_BASE + 0x244)

/* Cortex-M NVIC */
#define NVIC_ISER0 REG32(0xE000E100)

/* Cortex-M SysTick */
#define SYST_CSR   REG32(0xE000E010)
#define SYST_RVR   REG32(0xE000E014)
#define SYST_CVR   REG32(0xE000E018)

/*
 * Renode platform:
 *   systickFrequency = 72,000,000 Hz
 *
 * 720,000 ticks = 10 ms = 100 Hz.
 */
#define CONTROL_SYSTICK_RELOAD 719999u

/*
 * Fixed synthetic ECU-control workload.
 *
 * This is intentionally deterministic and is used only
 * to measure scheduling/interference under identical load.
 */
#define CONTROL_WORK_ITERS 256u

/*
 * F6 software RX queue.
 *
 * Producer: CAN1_RX0_IRQHandler()
 * Consumer: main()
 *
 * One slot is deliberately left unused so that
 * head == tail always means empty.
 *
 * Therefore:
 *   storage entries   = 16
 *   usable capacity   = 15 CAN frames
 */
#define RX_QUEUE_SIZE 16u
#define RX_QUEUE_MASK (RX_QUEUE_SIZE - 1u)

typedef struct {
    uint32_t id;
    uint32_t dlc;
    uint32_t low;
    uint32_t high;
} can_frame_t;

static volatile can_frame_t g_rx_queue[RX_QUEUE_SIZE];

static volatile uint32_t g_q_head = 0;
static volatile uint32_t g_q_tail = 0;

/* Hardware / ISR accounting. */
volatile uint32_t g_irq_count = 0;
volatile uint32_t g_hw_rx_count = 0;
volatile uint32_t g_hw_overrun_count = 0;

/* Software queue accounting. */
volatile uint32_t g_queue_push_count = 0;
volatile uint32_t g_queue_overflow_count = 0;
volatile uint32_t g_processed_count = 0;
volatile uint32_t g_queue_max_depth = 0;

/* Last frame processed by foreground/main. */
volatile uint32_t g_last_processed_id = 0;
volatile uint32_t g_last_processed_dlc = 0;
volatile uint32_t g_last_processed_low = 0;
volatile uint32_t g_last_processed_high = 0;

/*
 * Last V2 feature vector produced by foreground/main.
 *
 * Exposed as globals so Renode can inspect the exact
 * double values after a real CAN frame traverses:
 *
 * SocketCAN -> bxCAN RX IRQ -> software queue ->
 * DLC normalization -> V2 streaming feature generator.
 */
volatile double g_last_features[V2_NUM_FEATURES] = {0.0};
volatile uint32_t g_feature_count = 0;
volatile uint32_t g_feature_error_count = 0;

/* Frozen V2 Hybrid IDS accounting. */
volatile uint32_t g_ids_call_count = 0;
volatile uint32_t g_last_prediction = 0;
volatile uint32_t g_normal_count = 0;
volatile uint32_t g_known_attack_count = 0;
volatile uint32_t g_unknown_attack_count = 0;

/*
 * F6 periodic ECU-control accounting.
 *
 * Deadline model:
 *   period   = 10 ms
 *   deadline = period
 *
 * If a new release occurs while an earlier control job
 * has not completed, that earlier job missed its deadline.
 */
volatile uint32_t g_control_tick_count = 0;
volatile uint32_t g_control_release_count = 0;
volatile uint32_t g_control_complete_count = 0;
volatile uint32_t g_control_deadline_miss_count = 0;
volatile uint32_t g_control_max_backlog = 0;

/*
 * Observable state/result prevents the deterministic
 * synthetic control computation from being optimized away.
 */
volatile uint32_t g_control_state = 0x13579BDFu;
volatile uint32_t g_control_output = 0u;

/*
 * CAN receive timing in units of the 10 ms SysTick period.
 *
 * These timestamps make the measured CAN-arrival span
 * independent of manual Renode start/pause timing.
 */
volatile uint32_t g_first_rx_tick = 0u;
volatile uint32_t g_last_rx_tick = 0u;

static void systick_init(void)
{
    SYST_CSR = 0u;
    SYST_RVR = CONTROL_SYSTICK_RELOAD;
    SYST_CVR = 0u;

    /*
     * ENABLE    bit 0
     * TICKINT   bit 1
     * CLKSOURCE bit 2 = processor clock
     */
    SYST_CSR = 7u;
}

void SysTick_Handler(void)
{
    uint32_t releases;
    uint32_t completed;
    uint32_t backlog;

    g_control_tick_count++;

    releases = g_control_release_count + 1u;
    g_control_release_count = releases;

    completed = g_control_complete_count;
    backlog = releases - completed;

    /*
     * backlog == 1:
     *   newly released job only, no missed deadline.
     *
     * backlog > 1:
     *   at least one earlier periodic job was still
     *   incomplete at this release boundary.
     */
    if (backlog > 1u) {
        g_control_deadline_miss_count++;
    }

    if (backlog > g_control_max_backlog) {
        g_control_max_backlog = backlog;
    }
}

static void run_control_workload(void)
{
    uint32_t x = g_control_state;

    /*
     * Deterministic integer workload representing a
     * periodic ECU-control computation.
     */
    for (uint32_t i = 0; i < CONTROL_WORK_ITERS; ++i) {
        x = x * 1664525u + 1013904223u;
        x ^= (x >> 13);
        x += (i * 2654435761u);
        x ^= (x << 7);
    }

    g_control_state = x;
    g_control_output = x ^ (x >> 16);
}

static void can1_init(void)
{
    /* CAN1 peripheral clock */
    RCC_APB1ENR |= (1u << 25);

    /* Initialization mode */
    CAN_MCR = (CAN_MCR & ~(1u << 1)) | 1u;

    while ((CAN_MSR & 1u) == 0u) {
    }

    /*
     * Keep the same BTR used by the previous
     * Renode CAN experiment.
     */
    CAN_BTR = 0x001C0000u;

    /*
     * Filter bank 0:
     * 32-bit mask mode, FIFO0, accept all.
     */
    CAN_FMR |= 1u;

    CAN_FA1R &= ~1u;
    CAN_FM1R &= ~1u;
    CAN_FS1R |= 1u;
    CAN_FFA1R &= ~1u;

    CAN_F0R1 = 0u;
    CAN_F0R2 = 0u;

    CAN_FA1R |= 1u;
    CAN_FMR &= ~1u;

    /*
     * FIFO0 message-pending interrupt.
     * FMPIE0 = bit 1.
     */
    CAN_IER |= (1u << 1);

    /* NVIC IRQ20 = CAN1 RX0 */
    NVIC_ISER0 = (1u << 20);

    /* Leave initialization mode */
    CAN_MCR &= ~1u;

    while ((CAN_MSR & 1u) != 0u) {
    }
}

static uint32_t queue_depth(uint32_t head, uint32_t tail)
{
    return (head - tail) & RX_QUEUE_MASK;
}

void CAN1_RX0_IRQHandler(void)
{
    uint32_t rf0r;
    uint32_t id;
    uint32_t dlc;
    uint32_t low;
    uint32_t high;
    uint32_t head;
    uint32_t next;
    uint32_t tail;
    uint32_t depth;

    g_irq_count++;

    rf0r = CAN_RF0R;

    if (rf0r & (1u << 4)) {
        g_hw_overrun_count++;

        /* Clear hardware FIFO overrun flag. */
        CAN_RF0R = (1u << 4);
    }

    if ((rf0r & 0x3u) != 0u) {
        /*
         * Capture the complete FIFO mailbox before
         * releasing it.
         */
        id   = CAN_RI0R;
        dlc  = CAN_RDTR0;
        low  = CAN_RDLR0;
        high = CAN_RDHR0;

        g_hw_rx_count++;

        /*
         * Record CAN1 RX timing using the same 10 ms
         * SysTick time base as the periodic control task.
         */
        if (g_hw_rx_count == 1u) {
            g_first_rx_tick = g_control_tick_count;
        }

        g_last_rx_tick = g_control_tick_count;

        head = g_q_head;
        next = (head + 1u) & RX_QUEUE_MASK;
        tail = g_q_tail;

        if (next != tail) {
            g_rx_queue[head].id   = id;
            g_rx_queue[head].dlc  = dlc;
            g_rx_queue[head].low  = low;
            g_rx_queue[head].high = high;

            /*
             * Publish head only after all frame fields
             * have been written.
             */
            __asm volatile ("" ::: "memory");
            g_q_head = next;

            g_queue_push_count++;

            depth = queue_depth(next, tail);
            if (depth > g_queue_max_depth) {
                g_queue_max_depth = depth;
            }
        } else {
            /*
             * CAN hardware received the frame, but the
             * software queue had no free entry.
             */
            g_queue_overflow_count++;
        }

        /* Release FIFO0 output mailbox. */
        CAN_RF0R = (1u << 5);
    }
}

void main(void)
{
    can_frame_t frame;
    uint32_t tail;
    uint32_t raw_id;
    uint32_t dlc;
    uint16_t can_id;
    uint8_t payload[8];
    double features[V2_NUM_FEATURES];

    feature_v2_reset();
    can1_init();
    systick_init();

    while (1) {
        /*
         * Periodic ECU-control work has priority over
         * foreground IDS processing.
         *
         * SysTick/CAN IRQs can still preempt this code.
         */
        if (g_control_complete_count <
            g_control_release_count) {

            run_control_workload();
            g_control_complete_count++;
            continue;
        }

        tail = g_q_tail;

        if (tail != g_q_head) {
            /*
             * Copy the queued frame before releasing
             * this queue entry back to the ISR.
             */
            frame.id   = g_rx_queue[tail].id;
            frame.dlc  = g_rx_queue[tail].dlc;
            frame.low  = g_rx_queue[tail].low;
            frame.high = g_rx_queue[tail].high;

            __asm volatile ("" ::: "memory");
            g_q_tail = (tail + 1u) & RX_QUEUE_MASK;

            g_last_processed_id   = frame.id;
            g_last_processed_dlc  = frame.dlc;
            g_last_processed_low  = frame.low;
            g_last_processed_high = frame.high;

            /*
             * Decode only standard 11-bit CAN frames.
             *
             * bxCAN RIxR:
             *   IDE = bit 2
             *   STID = bits 31:21
             */
            raw_id = frame.id;

            if ((raw_id & (1u << 2)) == 0u) {
                can_id = (uint16_t)((raw_id >> 21) & 0x7FFu);

                /*
                 * Python V2 preprocessing parses DLC bytes
                 * and zero-pads the payload to exactly 8.
                 */
                dlc = frame.dlc & 0xFu;
                if (dlc > 8u) {
                    dlc = 8u;
                }

                payload[0] = (uint8_t)(frame.low);
                payload[1] = (uint8_t)(frame.low >> 8);
                payload[2] = (uint8_t)(frame.low >> 16);
                payload[3] = (uint8_t)(frame.low >> 24);

                payload[4] = (uint8_t)(frame.high);
                payload[5] = (uint8_t)(frame.high >> 8);
                payload[6] = (uint8_t)(frame.high >> 16);
                payload[7] = (uint8_t)(frame.high >> 24);

                for (uint32_t i = dlc; i < 8u; ++i) {
                    payload[i] = 0u;
                }

                if (feature_v2_process(
                        can_id,
                        payload,
                        features) == 0) {

                    int prediction;

                    for (uint32_t i = 0;
                         i < V2_NUM_FEATURES;
                         ++i) {
                        g_last_features[i] = features[i];
                    }

                    g_feature_count++;

                    /*
                     * Frozen final V2 hybrid inference.
                     *
                     * Return codes:
                     *   0 = normal
                     *   1..4 = known attack
                     *   5 = Mahalanobis unknown anomaly
                     */
                    /*
                     * F6 IDS-OFF A/B baseline.
                     * V2 feature extraction remains active.
                     * Hybrid LightGBM + Mahalanobis inference is disabled.
                     */
                    prediction = 0;
                    g_last_prediction = 0u;
                } else {
                    g_feature_error_count++;
                }
            } else {
                /*
                 * Current exact V2 state implementation is
                 * intentionally limited to standard 11-bit IDs.
                 */
                g_feature_error_count++;
            }

            g_processed_count++;
        } else {
            __asm volatile ("nop");
        }
    }
}
