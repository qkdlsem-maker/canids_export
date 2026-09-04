#include <stdio.h>
#include "can_ids_embedded.c"

int main() {
    double samples[3][8] = {
        {0.050000, 13.000000, 0.000000, 2.000000, 57.375000, -0.023866, 1.332871, 0.617188},  // R
        {0.400000, 10.000000, 1.000000, -0.000000, 0.000000, -0.400000, 0.000000, 0.000000},  // DoS
        {0.450000, 10.000000, 0.000000, 2.405639, 66.000000, -0.485131, 5000.000000, 0.530762} // gear
    };
    const char *names[] = {"R", "DoS", "gear"};

    for (int s = 0; s < 3; s++) {
        double out[5];
        score(samples[s], out);
        printf("%s -> [%.4f, %.4f, %.4f, %.4f, %.4f]\n", names[s], out[0], out[1], out[2], out[3], out[4]);
    }
    return 0;
}
