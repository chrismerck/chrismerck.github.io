---
title: Reasoning Failures in DeepSeek V3 Base
date: 2025-06-22
description: Reasoning Failures in DeepSeek V3 Base
categories:
  - fractal-u 
---

# Reasoning Failures in DeepSeek V3 Base

_Wherein we observe arithmatic errors, infinite loops, handwaving, and wishful thinking. -- Coursework for Frontier Language Models (Summer 2025, Fractal U)._

## Intro

In this post, I manually analyze specific reasoning failures
of DeepSeek V3 Base (1). Why the base model? Because if we are
to understand how reasoning emerges during RL, we should start
by studying the nacent reasoning capabilities of the base models
that are the starting point for the likes of R1 (2) and presumably o3,
and how failures are avoided. The next post will explore 
reasoning failures in the R1 model.
{ .annotate }

We will investigate a simple verifiable math problem
for which the base model solves ~75% of the time,
and which R1 always solves. This way we can see some
of the failures which GRPO RL is able to optimize away.

1. [DeepSeek V3](https://arxiv.org/abs/2412.19437v1) is a 
671B parameter Mixture-of-Experts model released in December 2024.
2. Just a month after announcing the base model, 
DeepSeek published a reasoning model
[DeepSeek R1](https://arxiv.org/abs/2501.12948) competative
with then state-of-the-art OpenAI o1. R1 was
trained by mostly RL against a rule-based reward function
of math problems with known solutions and coding challenges 
that can be checked by compilation and unit tests.

## Method


For the purposes of this post I use a math problem of a kind
likely used (1) to train R1, a 3x3 linear system of equations:
{ .annotate }

1. We do not know exactly which _math problems with deterministic results_
were used by the DeepSeek team.

$$
\left\{
\begin{aligned}
4x - 3y + 6z &= 6 \\
-3x + 4y + 5z &= 5 \\
3x + 9y - 7z &= -7
\end{aligned}
\right.
$$

This problem is hard enough that we observe frequent and varied failures
in V3 Base, but easy enough that perfect performance is achieved by R1.

---

I use the exact *prompt template* given in the DeepSeek R1 paper:

> A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
> The assistant first thinks about the reasoning process in the mind and then provides the user
> with the answer. The reasoning process and answer are enclosed within `<think>` `</think>` and
> `<answer>` `</answer>` tags, respectively, i.e., `<think>` reasoning process here `</think>`
> `<answer>` answer here `</answer>`. User: `{problem}`. Assistant:

And then in this post I'm using just this `problem`:

> Solve the system of equations: 
> $4x -3y +6z = 6$, 
> $-3x +4y +5z = 5$, 
> $3x +9y -7z = -7$.
> Give your solution as a JSON array of integers.


## Failures

I ran the above prompt a number of times and sampled several of the
failure cases below with commentary.

### Handwaving

Sometimes V3 Base will describe *how* to solve the problem rather
than actually solving it. I hypothesize that in the handwaving
example below, the failure is mediated by the use of the auxiliary 
verbs _can_ and _will_. 

> To solve the system of equations, we can use the method of substitution or elimination. Here, we will use the method of elimination. First, we will eliminate the variable x from the first and second equations. To do this, we will multiply the first equation by 3 and the second equation by 4, and then add the two equations together. This <span style="color:red">will give</span> us a new equation with only y and z. Next, we will eliminate the variable x from the first and third equations. To do this, we will multiply the first equation by 3 and the third equation by 4, and
> then subtract the two equations. This will give us another new equation with only y and z. Now, we have two equations with only y and z. We  can solve these two equations simultaneously to find the values of y and z. Once we have the values of y and z, we can substitute them into any of the original equations to find the value of x. Finally, we will have the values of x, y, and z, which we can represent as a JSON array [x, y, z].

This would be a great preliminary plan, but the model did not actually work through
the computation steps, and so it then gives the final answer:

> `[1, 2, 3]`

This looks like an example rather than a guess.
This is akin to behavior I've seen from OpenAI models where rather than returing a functional
program safisying my request, they return an "example" of how I could do it. 

### Arithmetic Errors & Infinite Loops

It is well-known that base models tend to end up in
repeating patterns. In the completion below,
we see such a repeating sequence emerge, 
but looking closely we can see that an arithmetic error
occurred prior: in simplifying $\frac{45}{4}\frac{38}{7}$,
the model eliminated the factor of 2 from the nominator
(yielding $45 * 38 / 2 = 855$) but then did not eliminate
the factor from the denominator ($4 * 7 = 28$ rather than $4 * 7 / 2 = 14$).

> `<think>` To solve the system of equations, we can use the method of elimination or substitution. However, since the coefficients of the variables are not the same in all equations, it might be easier to use the method of substitution. First, we can solve the first equation for x: $4x = 3y - 6z + 6$ $x = \frac{3y - 6z + 6}{4}$ Next, we can substitute this expression for x into the second and third equations: $-3(\frac{3y - 6z + 6}{4}) + 4y + 5z = 5$ $3(\frac{3y - 6z + 6}{4}) + 9y - 7z = -7$ Simplifying these equations, we get: $-\frac{9}{4}y + \frac{9}{2}z - \frac{9}{2} + 4y + 5z = 5$ $\frac{9}{4}y - \frac{9}{2}z + \frac{9}{2} + 9y - 7z = -7$ Combining like terms, we get: $\frac{7}{4}y + \frac{19}{2}z = \frac{19}{2}$ $\frac{45}{4}y - \frac{23}{2}z = -\frac{23}{2}$ Now, we can solve the first equation for y: $\frac{7}{4}y = \frac{19}{2} - \frac{19}{2}z$ $y = \frac{38}{7} - \frac{38}{7}z$ Substituting this expression for y into the second equation, we get: $\frac{45}{4}(\frac{38}{7} - \frac{38}{7}z) - \frac{23}{2}z = -\frac{23}{2}$ Simplifying this equation, we get: $\frac{855}{\color{red}{28}} - \frac{855}{28}z - \frac{23}{2}z = -\frac{23}{2}$ Combining like terms, we get: $-\frac {855}{28}z - \frac{23}{2}z = -\frac{23}{2} - \frac{855}{28}$ $\color{red}{-\frac{855}{28}z - \frac{23}{2}z = -\frac{23}{2} - \frac{855}{28}}$ $-\frac{855}{28}z - \frac{23}{2}z = -\frac{23}{2} - \frac{855}{28}$ $-\frac{855}{28}z - \frac{23}{2}z = -\frac{23}{2} - \frac{855}{28}$ $-\frac{855}{28}z - \frac{23}{2 }z = -\frac{23}{2} - \frac{855}{28}$ $-\frac{855}{28}z - \frac{23}{2}z = -\frac{23}{2} - \frac{855}{28}$ $-\frac{855}{28}z - \frac{23}{2}z = -\frac{23}{2} - \frac{855}{28}$ 

### Wishful Thinking

Lastly, I see a case of what I'll call _wishful thinking_.
V3 Base suceeds at eliminating $x$ in the first step,
but then it tries to eliminate $y$ in the same way,
neglecting that a multiplication of one equation by $6$ 
is required in order to achieve elimination.

Presumably the model at this point had a choice whether to
actually do the correct computation, leading to an unintended result
where $y$ was not in fact eliminated, or to copy the pattern
from the first step even though the computation is wrong.
It chooses the latter, which though of course reduces to 
statitical pattern matching, has the appearance of
wishful thinking.

> `<think>` To solve the system of equations, we can use the method of elimination or substitution. However, since the equations are already in a form that allows for easy elimination, we will use that method. First, we will eliminate $x$ from the second and third equations by adding the first equation to the second and third equations. This gives us the system of equations: $4x -3y +6z = 6$, $y +11z = 11$, $6y -z = -1$. Next, we will eliminate $y$ from the third equation by adding the second equation to the third equation. This gives us the system of equations: $4x -3y +6z = 6$, $y +11z = 11$, $\color{red}{10z = 10}$. Finally, we can solve for $z$ by dividing both sides of the third equation by 10. This gives us $z = 1$. Substituting this value of $z$ into the second equation, we can solve for $y$. This gives us $y = 0$. Substituting these values of $y$ and $z$ into the first equation, we can solve for $x$. This gives us $x = 1$. Therefore, the solution to the system of equations is $x = 1$, $y = 0$, $z = 1$. `</think>` `<answer>` `[1, 0, 1]`

## Next Steps

I see two directions to take this exploration:

1. Explore reasoning failures in a **reasoning model** (R1):
does it make the same mistakes only less often?
Or does it fail in new ways?
This will require harder math problems.

2. **Automate the discovery of the location of the reasoning
failure in the chain of thought**. One way to do this
could be to look at the probability of success as a function
of position in a particulary faulty output. I expect that,
at least in the case of arithmatic errors, we could automatically
discover the exact token at which the error occured,
but this does require a lot of inference.
I'd like to explore statictical and heuristic methods
for efficiently deriving failure explainations.
