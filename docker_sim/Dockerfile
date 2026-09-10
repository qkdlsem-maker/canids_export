FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 iproute2 libicu74 libssl3t64 zlib1g libunwind8 make gcc-arm-none-eabi libnewlib-arm-none-eabi && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY models_c/renode_portable models_c/renode_portable
COPY models_c/f6_v2 models_c/f6_v2
COPY models_c/can_ids_embedded_v2.c models_c/can_ids_embedded_v2.c
COPY docker_sim docker_sim
WORKDIR /app/docker_sim
RUN rm -f build/ecu_a.elf build/ecu_b.elf && make -j2
ENV XDG_CONFIG_HOME=/tmp/jarvis-renode-config
ENTRYPOINT ["sh", "entrypoint.sh"]
