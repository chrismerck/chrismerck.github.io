/**
 * section2.c
 * Simple Implementation of "Demonstrating Superposition"
 * from Toy Models of Superposition (Anthropic, Harvard, 2022)
 * by Chris Merck (c) May 2025
 */

/*
 We implement the model x' = ReLU(W^T W x + b)
 */

#include "stdio.h"
#include "math.h"
#include "string.h"

#define N 20
#define M 5

typedef struct {
    float W[M][N];
    float b[N];
} params_t;

void forward(params_t * p, float x[N], float * xp) {
    float h[M];
    memset(h, 0, sizeof(h));
    // hidden layer
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            h[m] += p->W[m][n] * x[n];
        }
    }
    // output layer
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            xp[n] += p->W[m][n] * h[m];
        }
        xp[n] += p->b[n];
        // ReLU activation
        xp[n] = xp[n] > 0 ? xp[n] : 0;
    }
}

int main() {
    params_t p;
    memset(&p, 0, sizeof(p));
    // initialize with indentity (in upper corner)
    for (int m = 0; m < M; m++) {
        p.W[m][m] = 1;
    }
    // set a bias
    p.b[19] = 1;
    // some test points
    float x[N];
    memset(x, 0, sizeof(x));
    x[4] = 1;
    float xp[N];
    memset(xp, 0, sizeof(xp));
    forward(&p, x, xp);
    printf("xp: ");
    for (int n = 0; n < N; n++) {
        printf("%1.02f ", xp[n]);
    }
    printf("\n");
    return 0;
}
