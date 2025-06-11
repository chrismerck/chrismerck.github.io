---
title: Demonstrating Superposition
date: 2025-05-14
description: Reproducing the intro figure from Toy Models of Superposition
categories:
  - tmos
---

# Demonstrating Superposition

Wherein I implement a toy model of feature superposition by hand in C as a remedial exercise,
and create a video showing the model learning a suboptimal representation.

---

> In the unlikely scenario where all of this makes total sense and you feel like you're ready to make contributions, [...]  
> - Scott Alexander [Nov 2023](https://www.astralcodexten.com/p/god-help-us-lets-try-to-understand)

---

In this post, I will manually reproduce the intro figure from [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
without using anything but the C standard library,
so as not to hide any of the details.
The paper comes with a PyTorch implementation,
but autograds do so much work I feel I need to earn the right to use them
by working out the toy model math and code myself.

The basic result is this little animation showing how the model learns the pentagonal representation
from the paper's intro:

<video controls width="100%">
  <source src="/assets/tmos_sec2_pentagon.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## The Data

First off, we need to generate the synthetic data. We want samples with dimension $n=5$,
where features are _sparse_ (being non-zero with probability $1 - S$) and uniformly 
distributed on the unit interval when they do appear, which we can write down as a 
mixture of two distributions:

$$x_i \sim \sum \begin{cases}
\delta(0) & S \\
\text{U}(0, 1) & (1 - S)
\end{cases}$$

Here $\delta$ is the Dirac delta function, i.e. the point mass distribution.

### Synthesizing Data in C

The C stdlib doesn't have a uniform random function so I wrote one (1) and used it to generate the data:
{ .annotate }

1. 
```c
#include <stdlib.h>

/// get a float uniformly distributed on U[0, 1)
float frand() {
    return (random() / (float) RAND_MAX);
}
```
macOS manpages implore us to use the cryptographically secure RNG `arc4random()`, 
but I think the polynomial PRNG is good enough for this application, 
and I like that we can use `srandom(0)` to force reproducibility.

```c
void synthesize(int n, long count, float X[n][count], float S_) {
    // sparsity S in [0, 1), S_ is 1-S
    for (long c = 0; c < count; c++) {
        for (int i = 0; i < n; i++) {
            if (frand() < S_) {
                X[i][c] = frand();
            }
        }
    }
}
```

Now we can generate some samples with sparsity 1-S = 0.1, 
using a little `printmat` function (1) to check our work.
Below we see the result for four 5-D samples.
{ .annotate }

1. 
```c
void printmat(char * tag, int rows, int cols, float A[rows][cols]) {
    printf("%s: [\n", tag);
    for (int m = 0; m < rows; m++) {
        for (int n = 0; n < cols; n++) {
            if (A[m][n]) {
                printf("  %1.03f ", A[m][n]);
            } else {
                printf("  0     ");
            }
        }
        printf("\n");
    }
    printf("]\n");
}

const int count = 4;
srandom(0);
memset(X, 0, sizeof(X));
synthesize((float *) X, count, 0.1);
printmat("X", (float *) X, N, count);
```

$$
X=\begin{bmatrix}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0.522 & 0 & 0 & 0 \\
0 & 0.568 & 0 & 0 \\
0 & 0 & 0 & 0 \\
\end{bmatrix}
$$

Here we see that only ~2 of the 20 elements are non-zero,
as expected with this sparsity level.


## The Model

The model is a 2-layer feedforward network,
where the hidden layer maps down from $n=5$ to $m=2$ dimensions
without any activation function,
and then the output layer uses the transpose of the hidden-layer weights
plus a bias term and a ReLU activation function. 
This is, as far as I can tell, basically an autoencoder.

In matrix notation we have:

$$y = \verb|ReLU|(W^T W x + b).$$

### The Forward Pass

Breaking down into steps with indecies we have:

$$
\begin{aligned}
h_k &= \sum_{i=1}^n w_{ki} x_i \\
a_j &= b_j + \sum_{k=1}^m h_k w_{kj} \\
y_j &= \max(0, a_j),
\end{aligned}
$$

from which follows a first C implementation of the forward pass:

```c
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
```

### Importance and Loss

We want the model to prioritize representation of certain dimensions,
so we assign an _importance_ $I_i$ to each dimension, which we 
make decrease geometrically: $I_i = 0.7^i$.
A weighted least-squares loss is then:

$$L = \frac{1}{2} \sum_i I_i (y_i - x_i)^2 + \alpha \sum_{k,j} w_{kj}^2.$$

And our goal is to optimize the parameters $W$ and $b$ to minimize this loss.
We then should be able to visualize the weights and see feature superposition
emerging as a function of sparsity.

Note that in the paper they do not specify any regularization.
I threw in the L2 regularization term because I saw that a weight-decay 
optimizer was used in the paper's code example on CoLab,
but it turns out to be totally unnecessary if we pick the learning rate right.

## Training

### Finding the Gradient

As I'm a bit rusty on my calculus, I'll go step by step through the gradient
computation.
Taking the derivative with respect to an arbitrary weight
and pushing the derivative inside the sums as far as it will go,
applying the chain and power rules, 
and using $\delta_j$ to denote the error in the $j$th output,
we have:

$$
\begin{align*}
\frac{\partial L}{\partial w_{kj}}
&= \frac{1}{2} \sum_i I_i \frac{\partial}{\partial w_{kj}}
      \bigl(y_i - x_i\bigr)^2
  + \alpha \sum_{k}\sum_{j'} \frac{\partial}{\partial w_{kj}} w_{kj'}^2, \\[1ex]
&= \sum_i I_i\,\delta_i\,\frac{\partial y_i}{\partial w_{kj}}
  + \alpha\,w_{kj}.
\end{align*}
$$

Note that in the regularization term we've used the fact that the only
summand that depends on $w_{kj}$ is the one where $k' = k$ and $j' = j$,
so the primes drop off the indices.


Now focusing on the derivative of the output layer, for the case where $y_j$ is non-zero, we have:

$$
\begin{align*}
\frac{\partial y_{j}}{\partial w_{k i}}
&= \frac{\partial}{\partial w_{k i}}
   \sum_{i'}\sum_{k'} w_{k' j}\,w_{k i'}\,x_{i'} \\[1ex]
&= \sum_{i'} x_{i'}\,
   \sum_{k'} \frac{\partial}{\partial w_{k i}}
   \bigl(w_{k' j}\,w_{k i'}\bigr) \\[1ex]
&= \sum_{i'} x_{i'} \sum_{k'}
   \begin{cases}
     2\,w_{k i'}            & k'=k \wedge i'=j=i,\\
     w_{k i'}              & k'=k \wedge (i' \ne j \wedge i' = i),\\
     0                     & \text{otherwise}
   \end{cases} \\[1ex]
&= \sum_{i'} x_{i'}\,w_{k i'} \;+\; x_{j}\,w_{k j} \\[1ex]
&= h_{k} \;+\; x_{j}\,w_{k j}.
\end{align*}
$$

Let's do an intuition check on this derivative.
The weight $w_{k i}$ appears in two places:
once in the hidden layer as $x_i$'s contribution to $h_k$,
and once in the output layer as $h_k$'s contribution to $y_j$.
So increasing $w_{k i}$ will increase the output proportionally 
to the value of $h_k$, but then we need to add in the fact
that $h_k$ itself is also increased proportional to both the
$i$th input and the current value of the weight.
So our calculation seems intuitively correct.

### Computing the Gradient in C

To compute the gradient in C, 
we implement a gradient function that adds to a gradient accumulator:

```c
float gradient(const params_t * p, const float x[N], float alpha, params_t * grad);
```

The simplest way is just to take the forward pass and keep track of temporary variables that appear
in the gradient expression above and then add them together as prescribed.
For example the hidden layer computation now looks like:

```c
for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
        wkj_xj[m][n] = p->W[m][n] * x[n];
        hk[m] += wkj_xj[m][n];
    }
}
```

And so on (1) as we compute the gradient, add to the accumulator, and return the loss.
{ .annotate } 

1. 
```c
float gradient(const params_t * p, const float x[N], float alpha, params_t * grad) {
    // unlike the forward pass, we keep track of intermediate
    // values that appear in the gradient
    // our toy model is so small that all this fits comfortably 
    // in the thread stack
    // alpha = L1 regularization coefficient
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
        for (int m = 0; m < M; m++) {
            L += alpha * fabs(p->W[m][n]);
        }
        L += alpha * fabs(p->b[n]);
    }
    L /= 2;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            if (y[n] <= 0) continue;
            dL_wkj[m][n] = importance(n) * delta[n] * (hk[m] + wkj_xj[m][n])
                + alpha * (p->W[m][n] > 0 ? 1 : -1);
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
```

### The Training Loop

Now we put it all together (1), adding in a random batch of size 1024 which provides
some stochasticity to the gradient descent.
Note that I'm not using any optimizer,
and I've got regularization turned off.
{ .annotate }

1. 
```c
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
```

This C program outputs a long log of the weights and loss during the training run.
Runs take about 10 seconds for 10000 batches, which is enough to fully converge.

It took only a little bit of trial and error to get the learning rate right.

I then asked `o3` (1) to take the outputted log and make an animation, resulting in the video shared at the top.
Here was the prompt that got me 90% of the way to a working animation:
{ .annotate }

1. 
One of the hardest parts of learning to use LLMs I find is knowing when and when _not_
to use them. For building visualizations, I find LLMs incredibly helpful,
while for _learning_, it's best to battle through the details oneself.

> Please write a Python program that takes a file called log with this format:
> (pasted example log snippet)
> and uses matplotlib to render loss as a function of the run,
> and the W matrix showing how each unit input vector is mapped to the hidden dimensions (2d) which should be a plot with one scatter dot for each of the 5 input unit vectors. Make this an animation showing how the points migrated over time, keeping the xy limits fixed so it is stable. Include a moving average of the loss as a line plot.

## Intiution about Local Minima 

I'll close with an animation of the same model but with $n=32$ features,
and importance decaying as $0.9^i$. Notice how **it converges to a suboptimal solution**!

<video controls width="100%">
  <source src="/assets/tmos_sec2_hexagon.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

You can see that the pentagon quickly forms out of features 0 through 4,
and then features 5, 6, and 7 start to emerge,
but then it is the less important feature 7 which pushes its way out to form a stable hexagon. 
Why is that? It seems to be because features 5 and 6 
were unlucky enough to be on the side of the pentagon shared with higher-importance features
while 7 had the good fortune of being near the relatively weaker feature 4
which it could push out of the way.

Bottom line: there are non-trivial local minima even in simple models _and_ we can actually have some hope of gaining
intuition about them.
