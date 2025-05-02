---
title: Research
---

# Research Notes Overview

This section collects informal research experiments, prototypes, and thought pieces.

> "Research is what I'm doing when I don't know what I'm doing." — *Wernher von Braun*

Expect messy code, half-baked ideas, and plenty of TODOs. 

$$
P(W) = \prod_{i=1}^n P(w_i | w_1, w_2, ..., w_{i-1})
$$

```python
import numpy as np
from tinygrad.tensor import Tensor

# Create a tinygrad tensor
x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
```