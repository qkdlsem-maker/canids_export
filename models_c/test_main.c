#include <stdio.h>
#include "can_ids_embedded.c"

int main() {
    double normal_sample[8] = {0.050000, 13.000000, 0.000000, 2.000000, 57.375000, -0.023866, 1.332871, 0.617188};
    double dos_sample[8]    = {0.400000, 10.000000, 1.000000, -0.000000, 0.000000, -0.400000, 0.000000, 0.000000};
    double gear_sample[8]   = {0.450000, 10.000000, 0.000000, 2.405639, 66.000000, -0.485131, 5000.000000, 0.530762};

    const char *labels[] = {"정상(R)", "DoS", "Fuzzy", "RPM", "gear", "미지 이상패턴(Zero-day)"};

    int r1 = can_ids_predict(normal_sample);
    int r2 = can_ids_predict(dos_sample);
    int r3 = can_ids_predict(gear_sample);

    printf("정상 샘플 판정: %s (code=%d)\n", labels[r1], r1);
    printf("DoS 샘플 판정: %s (code=%d)\n", labels[r2], r2);
    printf("gear 샘플 판정: %s (code=%d)\n", labels[r3], r3);
    return 0;
}
