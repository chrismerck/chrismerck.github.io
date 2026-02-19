---
title: Humpty Dumpty Successfully Annealed with Aid of Hungarian Algorithm and SVD
date: 2026-02-19
description: Solving a Jane Street puzzle where a neural network's layers are scrambled and you have to put them back together
categories:
  - puzzles
  - ml
---

I was listening to Dwarkesh's podcast and he referenced some interesting-sounding Jane Street problems. I figured I'd give one a shot.

The problem: you have a neural network's weights, but the layers are scrambled. You need to put Humpty Dumpty back together.

<!-- more -->

!!! note "Honor Statement"
    I did not reference anything online. I did not look at anyone else's solution to this problem. I used Claude Opus 4.6 extensively, but I instructed it not to do any web search -- with one exception for a research dead end I'll get into later. Nothing I used was outside of what was already in Opus's training data. As a Stevens grad, I take the honor code seriously.

---

**TL;DR:**

- The pairing problem (which input layer goes with which output layer) can be solved *exactly* using SVD subspace alignment + the Hungarian algorithm -- no forward passes needed, runs in seconds
- The ordering problem (what sequence do the 48 blocks go in) yields to simulated annealing in roughly 10,000 steps[^sa-steps], seeded by sorting on weight norms
- SA is insanely effective for combinatorial optimization -- it beat every elaborate scheme I tried (genetic algorithms, Sinkhorn gradient descent, Birkhoff polytope tricks)
- The winning workflow: run a simple SA on one machine while exploring the problem analytically on another, feeding insights back in

[^sa-steps]: Actually 13,000 steps in my run, but I'd like to think that with the right temperature schedule it could be done in 10,000.

## Understanding the Problem

We have a bunch of historical data with inputs and output predictions, and a pile of weight-matrix "pieces" in three different shapes. First thing: I make the assumption that those predictions are the predictions of *this* actual network. So we'll know we've reconstructed it correctly when the predictions of our reassembled network precisely match the historical data, up to floating point error.

Second thing: we've got to figure out the geometry of this network. Thankfully, we were given a little piece of Python code showing us the architecture. It's a residual network with feed-forward blocks -- each block has a ReLU and a residual connection. Looks like just the feed-forward part of a transformer:

$$x \leftarrow x + W_\text{out} \cdot \text{ReLU}(W_\text{inp} \cdot x + b_\text{inp}) + b_\text{out}$$

We can see three different piece shapes: inputs, outputs, and then one final linear layer that gives us the output prediction. We weren't given any information about how this network was trained, or if it was even trained at all. If this is just a random initialization, the problem is going to be very difficult, if not impossible. But if it's trained -- even only partially trained -- there should be a lot of structure in the weights. The key to solving this thing is going to be leveraging that structure.

## The Brute Force Attempt

When I first set this up, I just immediately started running simulated annealing with all the pieces. We don't know which input and output layers go together, and we don't know what order the blocks go in. Just started running SA on the whole thing. I had a hunch that if I threw enough compute at it, it would probably just solve it. But all I had in front of me was a pretty inexpensive computer. And like the Project Euler problems, I know that good puzzles, if you look at them right, require surprisingly little compute to solve.

So while that big SA was running, I got thinking about how to break this down.

## The Key Insight: SVD Subspace Alignment

The key insight is that when a network is trained, the input and output layers of one block become coupled. We can see this correlation in the singular value decomposition of the weight matrices.

Each block's input layer is an encoder into a 96-dimensional hidden space, and the output layer is a decoder back out. They were trained together, so they share a private "codebook." Pair the wrong encoder with the wrong decoder, and the decoder tries to read a codebook it was never taught.

The SVD reveals the directions each matrix "uses" in the hidden space. For the input weight matrix, the left singular vectors are the *write directions* -- the directions the encoder activates. For the output weight matrix, the right singular vectors are the *read directions* -- the directions the decoder is sensitive to. For a correct pair, these should align. The score:

$$\text{score}(\text{inp}, \text{out}) = \| \text{diag}(\Sigma_\text{inp}) \cdot U_\text{inp}^T \cdot V_\text{out} \cdot \text{diag}(\Sigma_\text{out}) \|_F$$

The singular-value weighting emphasizes directions where both pieces have large singular values -- the "important" subspace dimensions. Higher score means better alignment. I explored a couple of different statistics for matching up the layers, but this one gave the strongest signal.

## The Hungarian Algorithm: From Noisy Scores to Perfect Pairings

Now, these SVD agreement scores are quite noisy. If we took a greedy approach -- lining them up one at a time -- we'd obviously get suboptimal solutions. This is a perfect place to apply the Hungarian algorithm, which finds the globally optimal one-to-one assignment in polynomial time.

And here's the thing that surprised me: the per-row signal is *really* noisy. Only 4 of 48 true pairs are the top match in their row. The median rank of the true partner is 12th out of 48. The worst is 22nd. A greedy approach would get maybe 4 out of 48.

But the Hungarian algorithm exploits a global constraint: every input must match exactly one output, no sharing. Even though any individual score is ambiguous, there's only one global assignment where all 48 scores are simultaneously high. The Hungarian algorithm finds it. SVD alignment plus Hungarian matching perfectly recovers all 48 pairings.

This takes a couple of seconds to run. And to be clear -- we're not even running the neural net. This is purely from the weight matrices.

## Ordering the Blocks: SA in 10,000 Steps

Now that we know which input blocks go with which output blocks, we can run simulated annealing on just the ordering problem. I ran it for a million steps and left it overnight. But I realized that if I played with the temperature parameters, it actually only needed about 10,000 steps[^sa-steps]. That's an insanely small number, given that we still have a search space of $48!$.

I did one further thing to improve the initial seed so I could start with a lower temperature: I used a heuristic of sorting blocks by their weight norms in ascending order. Higher-norm weights tend to come from later layers, because as the residual stream norm grows through the network, you tend to get larger weight values as well. This gave SA a reasonable starting point to refine from.

## What I Learned

**Simulated annealing is insanely effective.** I used to be a big fan of genetic algorithms -- they're a really cool analogy. But for a lot of problems, they just seem to be so much less effective. I did try a genetic algorithm here without much success. This was quite a lesson: SA was making constant, steady progress while I was over on my other machine experimenting with elaborate genetic algorithm schemes.

**The workflow that worked.** Tactically, here's how I approached this: I got a simple SA running on one machine, making steady progress. On my main machine, I'm experimenting with different possibilities -- exploring the network, looking at residuals, looking at the norms of the residual stream, just seeing what's going on. I could also inject certain swaps back into the SA run based on my findings from the SVD analysis. That feedback loop between analytical insight and stochastic search was the real workflow.

## Things That Didn't Work (But I'm Curious About)

### Birkhoff Polytope and Genetic Algorithm Diversity

I tried a genetic algorithm with permutations embedded as matrices in the Birkhoff polytope -- the idea being to use Euclidean distance in that continuous space to repulse population members away from the centroid and boost biodiversity. It sort of worked to break up the cluster, but it was just one of those tangents I was exploring while my naive SA was running on the other machine, making steady progress. Lesson learned.

### Sinkhorn Optimization

I also tried casting the discrete permutation space into a continuous doubly-stochastic[^doubly-stochastic] matrix space and doing gradient descent on the assignments (not the neural network) using the Sinkhorn operator. It got trapped in the same local minima as SA -- the Metropolis random walk is just remarkably better at escaping basins than smooth gradient descent. I'm curious if anybody's managed to make this work.

### Fine-Tuning Individual Layers

I have a hunch that you could retrain the network while adding a cost function that penalizes the distance of each layer from one of the provided piece files. This would let the network kind of organically coalesce into the correct positions. I haven't tried it, but I'm curious if somebody's been successful with that approach.

[^doubly-stochastic]: "Doubly stochastic" is a somewhat confusing term -- there's nothing stochastic about it, really. It just means the matrix has rows and columns that each sum to one.
