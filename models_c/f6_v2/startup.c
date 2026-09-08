#include <stdint.h>

extern uint32_t _sidata, _sdata, _edata;
extern uint32_t _sbss, _ebss, _estack;

void Reset_Handler(void);
void Default_Handler(void);
void SysTick_Handler(void);
void CAN1_RX0_IRQHandler(void);
void main(void);

/*
 * Cortex-M4 vector table:
 *   0      initial SP
 *   1-15   core exceptions
 *   16+n   external IRQ n
 *
 * STM32F4 / Renode:
 *   CAN1 RX0 = IRQ20
 *   vector index = 16 + 20 = 36
 */
__attribute__((section(".isr_vector")))
void (* const vector_table[37])(void) = {
    [0]  = (void (*)(void))&_estack,
    [1]  = Reset_Handler,

    [2]  = Default_Handler,  /* NMI */
    [3]  = Default_Handler,  /* HardFault */
    [4]  = Default_Handler,  /* MemManage */
    [5]  = Default_Handler,  /* BusFault */
    [6]  = Default_Handler,  /* UsageFault */

    [11] = Default_Handler,  /* SVCall */
    [12] = Default_Handler,  /* DebugMon */
    [14] = Default_Handler,  /* PendSV */
    [15] = SysTick_Handler,  /* SysTick */

    /* External IRQ20: CAN1 RX FIFO0 */
    [36] = CAN1_RX0_IRQHandler,
};

void Reset_Handler(void)
{
    uint32_t *src = &_sidata;
    uint32_t *dst = &_sdata;

    while (dst < &_edata) {
        *dst++ = *src++;
    }

    dst = &_sbss;

    while (dst < &_ebss) {
        *dst++ = 0;
    }

    main();

    while (1) { }
}

void Default_Handler(void)
{
    while (1) { }
}
