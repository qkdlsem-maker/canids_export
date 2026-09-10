/* SL-derived functional simulation, NOT a validated SL wire implementation.
 * See SL_PROTOCOL.md for missing/contradictory source definitions and assumptions.
 * A aggregates BMS/HPC. B is a simplified converter with the original IDS.
 * No capture rows or IDS predictions influence application traffic generation.
 */
static uint32_t app_tick, phase_tick, command_mode, observed_mode;
static uint32_t bms_seen, hpc_seen, status_seen, bms_tick, hpc_tick, status_tick;
static uint32_t enable, relay, ready, reported_ready, reported_hv;
static uint32_t acnt1, acnt2, acnt3, chkid;
static int lv_target = 1330, hv_target = 2670, lv_voltage = 1330, hv_voltage = 1330;
static int lv_current, temperature = 250;

static int approach(int value, int target, int limit)
{
    if (value < target) return value + limit < target ? value + limit : target;
    if (value > target) return value - limit > target ? value - limit : target;
    return value;
}

static void send_bits(uint32_t id, uint64_t bits)
{
    send_frame(id, 8, (uint32_t)bits, (uint32_t)(bits >> 32));
}

static void application_receive(uint32_t id, uint32_t dlc, uint32_t low, uint32_t high)
{
    if (dlc != 8) return;
    uint64_t bits = low | ((uint64_t)high << 32);
    uint32_t now = g_control_tick_count;
    if (ECU_ROLE == 1) {
        if (id == 0x0D2u) reported_hv = (uint32_t)((bits >> 24) & 65535u);
        if (id == 0x1D2u) reported_ready = (uint32_t)((bits >> 53) & 1u);
        if (id == 0x2D2u) {
            observed_mode = (low >> 16) & 255u;
            status_seen = 1; status_tick = now;
            /* K is a matching periodic status, NOT a per-request ACK or RTT. */
            if (observed_mode == command_mode)
                record('K', rx_sequence + 1u, id, dlc, low, high, 0, now, now);
        }
    } else {
        if (id == 0x0D5u) {
            uint32_t en = low & 255u, rly = (low >> 8) & 255u;
            int target = (int)(low >> 16);
            int capability = (int)(int16_t)(high & 65535u);
            if (en > 1 || rly > 1 || target < 1170 || target > 1500 ||
                capability < -2550 || capability > 2550 || (high >> 16)) return;
            enable = en; relay = rly; lv_target = target;
            bms_seen = 1; bms_tick = now;
        }
        if (id == 0x0EFu) {
            uint32_t mode = (low >> 16) & 255u, target = low >> 24;
            if ((low & 255u) > 1 || ((low >> 8) & 255u) > 1 ||
                mode > 3 || target > 33 || high) return;
            command_mode = mode;
            /* SIMULATION ASSUMPTION: 8-bit raw voltage has a +250 V offset. */
            hv_target = (250 + (int)target) * 10;
            hpc_seen = 1; hpc_tick = now;
        }
    }
}

static void application_step(void)
{
    uint32_t now = g_control_tick_count;
    if (now == app_tick) return;
    app_tick = now; /* Skip missed releases; never burst to catch up. */
    if (now < 100u) return; /* UART/observer startup, one virtual second. */
    if (ECU_ROLE == 1) {
        if (!phase_tick) phase_tick = now;
        uint32_t age = now - phase_tick;
        /* A converter command timeout clears readiness. Re-enter precharge,
         * rather than repeatedly asking an unready B for Boost/Buck forever. */
        if (now % 10u == 0 && status_seen && now - status_tick <= 30u &&
            command_mode >= 2 && observed_mode == 0 && age >= 30u) {
            command_mode = 0; phase_tick = now; age = 0;
        }
        if (now % 10u == 0 && status_seen && now - status_tick <= 30u && observed_mode == command_mode) {
            uint32_t next = command_mode;
            if (command_mode == 0 && age >= 400u) next = 1;
            if (command_mode == 1 && reported_ready && reported_hv >= 2650u) next = 2;
            if (command_mode == 2 && age >= 300u) next = 3;
            if (command_mode == 3 && age >= 300u) next = 0;
            if (next != command_mode) { command_mode = next; phase_tick = now; }
        }
        /* BMS target varies smoothly; response values are computed by B. */
        int ramp = (int)(now % 400u);
        lv_target = 1320 + (ramp < 200 ? ramp : 400 - ramp) / 10;
        enable = relay = command_mode >= 2;
        int limit = command_mode == 2 ? -2000 : 2000;
        send_bits(0x0D5u, enable | ((uint64_t)relay << 8) |
                  ((uint64_t)lv_target << 16) | ((uint64_t)(uint16_t)limit << 32));
        if (now % 10u == 0)
            send_bits(0x0EFu, (command_mode != 0) | ((uint64_t)reported_ready << 8) |
                      ((uint64_t)command_mode << 16) | ((uint64_t)17u << 24));
    } else {
        uint32_t mode = command_mode;
        if (!bms_seen || !hpc_seen || now - bms_tick > 10u || now - hpc_tick > 30u)
            mode = 0;
        if (mode >= 2 && (!enable || !relay || !ready)) mode = 0;
        hv_voltage = approach(hv_voltage, mode ? hv_target : 1330, 4);
        lv_voltage = approach(lv_voltage, lv_target, 2);
        ready = mode != 0 && hv_voltage >= hv_target - 20;
        int load = 1000 + (int)((now / 10u) % 40u) * 10;
        int target_current = mode == 2 ? -load : mode == 3 ? load : 0;
        lv_current = approach(lv_current, target_current, 20);
        /* Complete the sign transition through zero; never report wrong polarity. */
        if ((mode == 2 && lv_current > 0) || (mode == 3 && lv_current < 0) || mode < 2)
            lv_current = 0;
        temperature = approach(temperature, mode >= 2 ? 450 : 250, 1);
        int hv_current = -lv_current / 20; /* Approximate 2:1 voltage ratio, no plant model. */
        send_bits(0x0D2u, acnt1 | ((uint64_t)ready << 5) |
                  ((uint64_t)lv_voltage << 8) | ((uint64_t)hv_voltage << 24) |
                  ((uint64_t)(uint16_t)lv_current << 40));
        send_bits(0x1D2u, acnt2 | ((uint64_t)200u << 4) |
                  ((uint64_t)(uint8_t)hv_current << 12) | ((uint64_t)temperature << 20) |
                  ((uint64_t)(temperature - 20) << 36) | ((uint64_t)ready << 53));
        acnt1 = (acnt1 + 1u) % 15u; acnt2 = (acnt2 + 1u) % 15u;
        if (now % 10u == 0) {
            /* CHKGRP=0 is an unverified placeholder; source whitelist is absent. */
            send_bits(0x2D2u, acnt3 | ((uint64_t)chkid << 8) |
                      ((uint64_t)mode << 16) | ((uint64_t)mode << 24));
            acnt3 = (acnt3 + 1u) % 15u; chkid = (chkid + 1u) % 255u;
        }
    }
}
