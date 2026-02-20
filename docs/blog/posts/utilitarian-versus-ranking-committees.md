---
title: Utilitarian versus Ranking Committees
date: 2013-10-13
description: Comparing ranking-based and utility-based committee decision models, with connections to voting theory
categories:
  - math
tags:
  - math
  - econ
---

In [Manolo Martinez](http://www.manolomartinez.net/aboutme/)'s [recent talk](https://philosophy.commons.gc.cuny.edu/event/cogsci-manolo-martinez/) at CUNY on the Rationalizing Role of Pains, he presented a model describing the decision-making process of a mind faced with multiple conflicting sensory messages of varying reliability. This model was by no means the focus of his talk, but it is what I'd like to focus on here.

<!-- more -->

I present an alternative model and then restrict myself to posing a few mathematical questions as to the relative power of the models. Questions of cognitive plausibility and interpretation are left unprovoked.

Please note that what follows is based on my imperfect recollection of his talk, and so all errors and misinterpretations are my own.

## Ranking Committees

Let $A$ denote the set of possible worlds and $M$ the set of messages. In Martinez's model, each message $m$ induces a ranking $R_m$ of possible worlds. This ranking is a [total ordering](https://en.wikipedia.org/wiki/Total_order) of $A$, and we use $0$ and $1$ to represent false and true, respectively. So, we interpret $R_m(a,b) = 1$ to mean that message $m$ tells the mind to prefer world $a$ to world $b$.

Now, the mind is bombarded with many messages, each inducing their own ranking, from which it must ultimately make a decision as to which possible world is preferred. According to the model, the mind resolves this problem by assigning a real weight $w_m$ to each message, perhaps based on the reliability of the information source. These weights are then used to compute an ultimate ranking:

$$R^* = \sum_m w_m R_m > 0.$$

That is, each message is given a weighted vote for each pair of alternative worlds, and the ultimate ranking simply counts up those votes. Now, when presented with some alternatives, the mind can apply $R^*$ to make a decision informed by the committee of messages with weight.

*[Edit: I realize two days later that this definition for $R^*$ does not yield a total ordering. The natural definition seems to be equivalent to an [instant-runoff vote](https://en.wikipedia.org/wiki/Instant-runoff_voting).]*

## Utility Committees

Being accustomed to [utility theory](https://en.wikipedia.org/wiki/Utility) from economics, I was surprised to see Martinez's model only allow messages to induce rankings, rather than full [utility functions](https://en.wikipedia.org/wiki/Utility#Utility_function). In response to a comment to this effect, he said that this restriction was intentional. So, I ask, how much more restricted is a committee of ranking voters than a committee of utility voters? Also, I sense that any ranking committee may be emulated by a utility committee, but this remains to be demonstrated.

My utility committee model is similar in structure to the model above but with the difference that each message induces a real-valued utility function $u_m$ rather than a ranking over possible worlds. It may be helpful to restrict the range of utility to the unit interval so that messages cannot arbitrarily dominate the committee. To obtain the ultimate utility function, we sum with weights:

$$u^* = \sum_m w_m \, u_m.$$

Finally, to compare the results of the utility model with the ranking model, we can use $u^*$ to induce a ranking in the obvious way: $R_u(a,b) = (u(a) > u(b))$.

## Questions

Now that we've set the stage, we can ask clear questions about the relative power of these models. Some example questions are:

1. **(strong)** Is it always the case that for a given set of rankings and weights, a set of utility functions may be chosen such that the utility committee agrees with the ranking committee?

2. **(weaker)** Is it always the case that for a given set of rankings and weights, a set of utility functions and *different weights* may be chosen such that the utility committee agrees with the ranking committee?

3. If the above do not hold in general, we may test similar claims relative to certain choices of cardinality of $A$ and $M$. (Consider a table with rows and columns giving choices of cardinality for $A$ and $M$, respectively. We should consider all integral cardinalities, plus discrete and continuous infinities.)

4. **(odd)** Given a finite set of utility functions and weights, can a (possibly infinite) set of rankings and other weights be found to agree with the utility model?

I haven't gotten around to proving any of this yet, but I'm curious about the results, not only in a purely mathematical capacity but also because of possible consequences for non-canonical utility theories in economics that might better model irrational agents as committees of arguing rational agents (cf. [some paradoxes of expected utility theory and some workarounds](http://ocw.mit.edu/courses/economics/14-123-microeconomic-theory-iii-spring-2010/lecture-notes/MIT14_123S10_notes06.pdf)).

*[Edit: Interestingly, the ranked vs. utilitarian committee comparison is formally equivalent to that between [instant-runoff voting](https://en.wikipedia.org/wiki/Instant-runoff_voting) (also called "choice voting") and [score voting](https://en.wikipedia.org/wiki/Score_voting) (also called "range voting"), respectively. [A particular activist website](http://rangevoting.org/) makes some strong arguments for the use of score voting in political elections. My intuition is with the score voting supporter, but I will need to run some simulations of my own before I'll be fully convinced.]*

---

*Originally published on [Quasiphysics](https://quasiphysics.wordpress.com/2013/10/13/utilitarian-versus-ranking-committees/).*
