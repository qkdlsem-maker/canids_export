/* Simulation transport/observability only. The existing IDS sources are unchanged. */
#define main original_f6_main
#define CAN1_RX0_IRQHandler original_can_irq
#include "../models_c/f6_v2/main.c"
#undef main
#undef CAN1_RX0_IRQHandler

static volatile uint32_t arrival_tick[RX_QUEUE_SIZE];
static uint32_t rx_sequence, tx_sequence;
volatile uint32_t sim_tx_ok, sim_tx_busy;
static uint32_t pending_tx;
static struct { uint32_t id, dlc, low, high; } tx_queue[8];
static unsigned tx_head, tx_tail;

static void uart_char(char c)
{
    REG32(0x4000440Cu) = 0x2008u; /* USART2 enable, TX enable */
    REG32(0x40004404u) = (uint32_t)c;
}

static void hex(uint32_t x)
{
    const char *digits = "0123456789abcdef";
    for (int i = 28; i >= 0; i -= 4) uart_char(digits[(x >> i) & 15u]);
}

static void record(char kind, uint32_t seq, uint32_t id, uint32_t dlc,
                   uint32_t low, uint32_t high, uint32_t pred,
                   uint32_t arrival, uint32_t end)
{
    uart_char(ECU_ROLE == 1 ? 'A' : 'B'); uart_char(','); uart_char(kind);
    uint32_t fields[] = {seq, id, dlc, low, high, pred, arrival, end};
    for (unsigned i = 0; i < 8; ++i) { uart_char(','); hex(fields[i]); }
    uart_char('\n');
}

void CAN1_RX0_IRQHandler(void)
{
    arrival_tick[g_q_head] = g_control_tick_count;
    original_can_irq();
}

static void tx_poll(void)
{
    uint32_t status = REG32(CAN1_BASE + 8);
    if (pending_tx && (status & 1u)) {
        if (status & 2u) sim_tx_ok++;
        record((status & 2u) ? 'T' : 'E', pending_tx, 0, 0, 0, 0, 0,
               g_control_tick_count, g_control_tick_count);
        REG32(CAN1_BASE + 8) = 1u;
        pending_tx = 0;
    }
    if (!pending_tx && tx_head != tx_tail && (status & (1u << 26))) {
        uint32_t id = tx_queue[tx_tail].id, dlc = tx_queue[tx_tail].dlc;
        uint32_t low = tx_queue[tx_tail].low, high = tx_queue[tx_tail].high;
        tx_tail = (tx_tail + 1u) % 8u;
        pending_tx = ++tx_sequence;
        REG32(CAN1_BASE + 0x184) = dlc;
        REG32(CAN1_BASE + 0x188) = low;
        REG32(CAN1_BASE + 0x18C) = high;
        record('Q', pending_tx, id, dlc, low, high, 0, g_control_tick_count, g_control_tick_count);
        REG32(CAN1_BASE + 0x180) = (id << 21) | 1u;
    }
}

static int send_frame(uint32_t id, uint32_t dlc, uint32_t low, uint32_t high)
{
    unsigned next = (tx_head + 1u) % 8u;
    if (next == tx_tail) {
        sim_tx_busy++;
        record('D', 0, id, dlc, low, high, 0, g_control_tick_count, g_control_tick_count);
        return 0;
    }
    tx_queue[tx_head].id = id; tx_queue[tx_head].dlc = dlc;
    tx_queue[tx_head].low = low; tx_queue[tx_head].high = high;
    tx_head = next;
    tx_poll();
    return 1;
}

#include "sl_application.h"

void main(void)
{
    feature_v2_reset(); can1_init(); systick_init();
    record('S', 0, 0, 0, 0, 2, ECU_ROLE, 0, 0);
    for (;;) {
        tx_poll();
        application_step();
        if (g_control_complete_count < g_control_release_count) {
            run_control_workload(); g_control_complete_count++;
            if ((g_control_tick_count % 100u) == 0u)
                record('Z', g_processed_count, g_hw_rx_count, g_queue_overflow_count,
                       g_control_deadline_miss_count, g_feature_error_count, g_ids_call_count,
                       g_control_tick_count, sim_tx_busy);
        }
        uint32_t tail = g_q_tail;
        if (tail == g_q_head) {
            __asm volatile ("cpsid i" ::: "memory");
            if (g_q_tail == g_q_head && g_control_complete_count == g_control_release_count)
                __asm volatile ("wfi" ::: "memory");
            __asm volatile ("cpsie i" ::: "memory");
            continue;
        }
        can_frame_t f = g_rx_queue[tail];
        uint32_t arrival = arrival_tick[tail];
        __asm volatile ("" ::: "memory");
        g_q_tail = (tail + 1u) & RX_QUEUE_MASK;
        uint32_t cid = (f.id >> 21) & 0x7FFu, dlc = f.dlc & 15u;
        uint32_t prediction = 0xFFFFFFFFu; /* inference not executed */
        if ((f.id & 6u) || dlc > 8u) {
            g_feature_error_count++;
        } else {
            uint8_t payload[8] = {0}; double features[V2_NUM_FEATURES];
            for (uint32_t i = 0; i < dlc; ++i)
                payload[i] = (uint8_t)((i < 4 ? f.low : f.high) >> ((i % 4) * 8));
            application_receive(cid, dlc, f.low, f.high);
            if (ECU_ROLE == 2) {
                if (!feature_v2_process((uint16_t)cid, payload, features)) {
                    g_feature_count++;
                    prediction = (uint32_t)can_ids_predict(features);
                    g_last_prediction = prediction; g_ids_call_count++;
                    if (!prediction) g_normal_count++;
                    else if (prediction == 5) g_unknown_attack_count++;
                    else g_known_attack_count++;
                } else g_feature_error_count++;
            }
        }
        rx_sequence++;
        record('R', rx_sequence, cid, dlc, f.low, f.high, prediction,
               arrival, g_control_tick_count);
        g_processed_count++;
    }
}
