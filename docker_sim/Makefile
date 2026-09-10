.DEFAULT_GOAL := all
CROSS_COMPILE ?= arm-none-eabi-
CC := $(CROSS_COMPILE)gcc
FLAGS := -mcpu=cortex-m4 -mthumb -O2 -g -ffreestanding -ffunction-sections -fdata-sections -Wall -Wextra
SOURCES := firmware.c ../models_c/f6_v2/startup.c ../models_c/f6_v2/feature_v2_stream.c ../models_c/can_ids_embedded_v2.c
all: build/ecu_a.elf build/ecu_b.elf
build/ecu_a.elf: $(SOURCES) sl_application.h ../models_c/f6_v2/main.c ../models_c/f6_v2/feature_v2_stream.h ../models_c/f6_v2/linker.ld Makefile
	@mkdir -p build
	$(CC) $(FLAGS) -DECU_ROLE=1 -T../models_c/f6_v2/linker.ld -Wl,--gc-sections --specs=nosys.specs $(SOURCES) -lm -o $@
build/ecu_b.elf: $(SOURCES) sl_application.h ../models_c/f6_v2/main.c ../models_c/f6_v2/feature_v2_stream.h ../models_c/f6_v2/linker.ld Makefile
	@mkdir -p build
	$(CC) $(FLAGS) -DECU_ROLE=2 -T../models_c/f6_v2/linker.ld -Wl,--gc-sections --specs=nosys.specs $(SOURCES) -lm -o $@
