#include <stdio.h>
#include "can_ids_embedded.c"

int main() {
    /* 정상 트래픽과 유사한 샘플 (freq_in_window, unique_ids, is_unknown, entropy, mean_byte, delta_z, value_z, norm_id) */
    double normal_sample[8] = {0.15, 8.0, 0.0, 2.1, 90.0, 0.5, 0.5, 0.4};
    /* DoS 공격과 유사한 샘플 (freq 매우 높음, unique_ids 낮음, entropy 낮음) */
    double dos_sample[8] = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    int r1 = can_ids_predict(normal_sample);
    int r2 = can_ids_predict(dos_sample);

    const char *labels[] = {"정상(R)", "DoS", "Fuzzy", "RPM", "gear", "미지 이상패턴(Zero-day)"};
    printf("정상 샘플 판정: %s (code=%d)\n", labels[r1], r1);
    printf("DoS 유사 샘플 판정: %s (code=%d)\n", labels[r2], r2);
    return 0;
}
