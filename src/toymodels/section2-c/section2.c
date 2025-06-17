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
#include "stdlib.h" // for random

#define SEED 22

#define N 5
#define M 2
const long n_samples = 100000;
float importance(int n) { return pow(0.5, n); }
float eta = 1e-2;
float alpha = 1e-10;
const int runs = 10e3;
const long batch_size = 1024;
float S_ = 0.01;

#define DEBUG 0

#define RELU 0 // 0 for linear

typedef struct {
    float W[M][N];
    float b[N];
} params_t;

void printmat(char * tag, int rows, int cols, float A[rows][cols]) {
    printf("%s: [\n", tag);
    for (int m = 0; m < rows; m++) {
        for (int n = 0; n < cols; n++) {
            if (A[m][n]) {
                printf(" % 01.03f ", A[m][n]);
            } else {
                printf(" 0     ");
            }
        }
        printf("\n");
    }
    printf("]\n");
}

void forward(params_t * p, float x[N], float * y) {
    float hk[M];
    memset(hk, 0, sizeof(hk));
    // hidden layer
    for (int k = 0; k < M; k++) {
        for (int i = 0; i < N; i++) {
            hk[k] += p->W[k][i] * x[i];
        }
    }
    // output layer
    for (int j = 0; j < N; j++) {
        y[j] += p->b[j];
        for (int k = 0; k < M; k++) {
            y[j] += p->W[k][j] * hk[k];
        }
        // ReLU activation
        y[j] = y[j] > 0 ? y[j] : 0;
    }
}

float gradient(const params_t * p, const float x[N], float alpha, params_t * grad) {
    // unlike the forward pass, we keep track of intermediate
    // values that appear in the gradient
    // our toy model is so small that all this fits comfortably 
    // in the thread stack
    // alpha = L2 regularization co-efficient
    // grad is a pointer to the gradient accumulator
    // returns loss
    float wkj_xj[M][N];
    float hk[M];
    float y[N];
    float delta[N];
    float dL_wkj[M][N];
    memset(wkj_xj, 0, sizeof(wkj_xj));
    memset(hk, 0, sizeof(hk));
    memset(y, 0, sizeof(y));
    memset(delta, 0, sizeof(delta));
    memset(dL_wkj, 0, sizeof(dL_wkj));
    // hidden layer
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            wkj_xj[m][n] = p->W[m][n] * x[n];
            hk[m] += wkj_xj[m][n];
        }
    }
    // output layer
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            y[n] += p->W[m][n] * hk[m];
        }
        y[n] += p->b[n];
        // ReLU activation
        #if RELU
        y[n] = y[n] > 0 ? y[n] : 0;
        #endif
        // compute delta
        delta[n] = y[n] - x[n];
    }
    // compute error
    float L = 0;
    for (int n = 0; n < N; n++) {
        float Ij = importance(n);
        L += Ij * delta[n] * delta[n];
    }
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            L += alpha * p->W[m][n] * p->W[m][n];
        }
        L += alpha * p->b[n] * p->b[n];
    }
    L /= 2;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            #if RELU
            if (y[n] <= 0) continue;
            #endif
            dL_wkj[m][n] = importance(n) * delta[n] * (hk[m] + wkj_xj[m][n]) + alpha * p->W[m][n];
        }
    }
    // add to gradient accumulator
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            grad->W[m][n] -= dL_wkj[m][n];
        }
    }
    for (int n = 0; n < N; n++) {
        #if RELU
        if (y[n] <= 0) continue;
        #endif
        grad->b[n] -= delta[n] + alpha * p->b[n];
    }
    return L;
}

void update(params_t * p, params_t * grad, float eta) {
    /// update parameters p += eta * grad
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            p->W[m][n] += eta * grad->W[m][n];
        }
    }
    for (int n = 0; n < N; n++) {
        p->b[n] += eta * grad->b[n];
    }
}

float X[n_samples][N];

float frand() {
    return (random()/ (float) RAND_MAX);
}

void synthesize(int n, long count, float X[count][n], float S_) {
    // sparsity S in [0, 1), S_ is 1-S
    while (1) {
        for (long c = 0; c < count; c++) {
            for (int i = 0; i < n; i++) {
                if (frand() < S_) {
                    X[c][i] = frand();
                }
            }
        }
        // make sure we have at least one non-zero sample
        for (long c = 0; c < count; c++) {
            for (int i = 0; i < n; i++) {
                if (X[c][i] != 0) {
                    return;
                }
            }
        }
    }
}

void batch_indices(long batch_size, long indices[batch_size]) {
    /// return n randomly-selected indices in [0, count)
    for (long i = 0; i < batch_size; i++) {
        indices[i] = frand() * n_samples;
        // no need to check for duplicates because batch_size << n_samples
    }
}


int main() {
    srandom(SEED);
    memset(X, 0, sizeof(X));
    synthesize(N, n_samples, X, S_);

    params_t p;
    memset(&p, 0, sizeof(p));
    // initialize with random weights and biases
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < M; k++) {
            p.W[k][j] = frand() * 0.1;
        }
        int l = (j * 19) % N;
        p.W[0][j] += sin(l * 2 * M_PI / N) * importance(l);
        p.W[1][j] += cos(l * 2 * M_PI / N) * importance(l);
        p.b[j] += frand() * 0.01;
    }
    params_t grad;
    for (int r = 0; r < runs; r++) {
        memset(&grad, 0, sizeof(grad));
        float L = 0;
        long batch[batch_size];
        batch_indices(batch_size, batch);
        for (long c = 0; c < batch_size; c++) {
            L += gradient(&p, X[batch[c]], alpha, &grad);
        }
        update(&p, &grad, eta / batch_size);
        printf("run: %d\n", r);
        printf("L: %1.04f\n", L / batch_size);
        if (r % 100 == 99) {
            // compute W^T W and print it
            float WTW[N][N];
            memset(WTW, 0, sizeof(WTW));
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < M; k++) {
                    for (int i = 0; i < N; i++) {
                        WTW[j][i] += p.W[k][j] * p.W[k][i];
                    }
                }
            }
            printmat("W^T W", N, N, WTW);
            // print b
            printmat("b", 1, N, p.b);
            // print W
            printmat("W", M, N, p.W);
            // print grad w and b
            printmat("grad w", M, N, grad.W);
            printmat("grad b", 1, N, grad.b);
        }
        fflush(stdout);
    }
    return 0;
}
