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

#define N 5
#define M 2
const long n_samples = 100000;
float importance(int n) { return pow(0.9, n); }
float eta = 1;
float alpha = 0;
const int runs = 10000;
const long batch_size = 1024;
float S_ = 0.01;

#define DEBUG 0

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

/// gradient descent on single sample, learning rate eta
float gradient(const params_t * p, const float x[N], float alpha, params_t * grad) {
    // unlike the forward pass, we keep track of intermediate
    // values that appear in the gradient
    // our toy model is so small that all this fits comfortably 
    // in the thread stack
    // alpha = L1 regularization co-efficient
    // returns loss
    // adds to grad, but does not update
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
        y[n] = y[n] > 0 ? y[n] : 0;
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
        if (y[n] <= 0) continue;
        for (int m = 0; m < M; m++) {
            L += alpha * fabs(p->W[m][n]);
        }
        L += alpha * fabs(p->b[n]);
    }
    L /= 2;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            if (y[n] <= 0) continue;
            dL_wkj[m][n] = importance(n) * delta[n] * (hk[m] + wkj_xj[m][n]) + alpha * (p->W[m][n] > 0 ? 1 : -1);
        }
    }
    // add to gradient accumulator
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            grad->W[m][n] -= dL_wkj[m][n];
        }
    }
    for (int n = 0; n < N; n++) {
        if (y[n] <= 0) continue;
        grad->b[n] -= delta[n] + alpha * (p->b[n] > 0 ? 1 : -1);
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
    srandom(1);
    memset(X, 0, sizeof(X));
    synthesize(N, n_samples, X, S_);

    params_t p;
    memset(&p, 0, sizeof(p));
    // initialize with random weights and biases
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < M; k++) {
            p.W[k][j] = frand() * 0.001;
        }
        p.b[j] = frand() * 0.001;
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
