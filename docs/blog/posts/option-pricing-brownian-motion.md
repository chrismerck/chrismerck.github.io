---
title: Option Pricing and Brownian Motion
date: 2011-03-29
description: Exploring connections between option pricing theory and the physics of diffusion and Brownian motion
categories:
  - physics
tags:
  - fin
  - math
  - phys
---

In the penultimate lecture of his Financial Markets course, entitled "Option Markets", Schiller discusses several important facets of financial options. I found his discussion of option pricing to be most intriguing, kindling thoughts relating to the physics of diffusion. Here I dump a few ideas I had on the subject, and hopefully I'll be able to flesh them out later on, either after doing some more research or with the help of someone knowledgeable in the subject (anyone?). My comments on diffusion are very rough, because I never studied that in school for some reason. Anyhow, here goes.

<!-- more -->

## Definitions of Options Jargon

An **option** is a contract that gives a specified party (known as the **option holder**) the right to buy or sell some specific asset from or to a specified second party (known as the **option writer**) at a predetermined price (known as the **exercise price**) on or before a predetermined date (known as the **exercise date**). Options are transferable, that is, either party may transfer their rights and obligations to another person in exchange for payment in either direction. However, we tend to think of an option from the point of view of the option holder, for whom involvement in the contract has positive value. So, when we refer to the trading of an option, we mean the purchase or sale of the holder's rights.

An option giving the holder the right to sell is called a **put option**. Conversely, a **call option** gives the holder the right to buy. Schiller explains that to understand the option pricing concept one only need consider one such type of option. Therefore, for the rest of the post when I write "option" I mean the more commonly discussed of the two: the call option.

Furthermore, some options permit the holder to exercise his right to buy or sell at the exercise price on any date up to and including the exercise date. These are known as **American options**. Other options only give the holder such a right on the exercise date, but not before. Such options are known as **European options**. I will discuss European options from now on, because they are simpler.

Lastly, an options contract must specify the asset at stake. To make the discussion more concrete, I will be specifically referring to **stock options**, that is, options to buy stock in a corporation. However, the chose of asset is not important to the logic of the discussion.

## Pricing an Option

Assume that we have a person who wishes to write options for some stock which is currently is selling for $s$ dollars, which we may write $s(0)$ to differentiate if from future prices of the stock. Further assume that the writer wishes to write the option with an exercise price of $e$ dollars for an exercise date $t$ units of time in the future. The writer of the option wants to estimate the price at which he should sell the option. This price should be close to the value of the option, and so we concern ourselves with determining the value of the option and will simply denote it $p$.

If the writer was somehow prescient so as to already know what the price of the stock would be on the exercise date, call it $s(t)$ dollars, then the value of the option would clearly be the present value of the intrinsic value of the option on the day of expiry, if it be positive. Taking the simplifying assumption of a zero interest rate (which, in real-terms, is not such a bad assumption these days), we can write algebraically that

$$p = \max(0, s(t) - e).$$

However, nobody knows what the future stock price will be. At best, we can use a combination of historical data, knowledge about the particular stock, and other financial knowledge to form an educated guess about how likely the future price will fall into certain ranges. This is quantified as a probability density function (PDF), call it $S$. That is, the writer's best estimate of the probability that the stock price at expiry will lie between $s'$ and $s''$ is found by integrating $S$ on that interval. So, given such a PDF $S$, the expected value of the option is found by integrating the product of the value of the option for each possible final stock price with the probability density for such a price:

$$p = \int_0^\infty ds' \, \max(0, s' - e) \, S(s')$$

which we can simplify to

$$p = \int_e^\infty ds' \, (s' - e) \, S(s').$$

## Conditions on an Estimation Formula for Future Stock Prices

Now that I've given the context, I can focus on the really interesting part: the properties of the future stock price PDF. Some smart analyst might announce that he has come up with a formula which gives $S$ for a particular stock given the present stock price and the expiry date. I noticed that there is a self-referential property that any such formula must have: the PDF obtained from a single application of the formula over a time interval $t$ must be equal to any composite PDF obtained by applying the formula over two time intervals that sum to $t$, e.g. $t'$ and $(t-t')$. To write this algebraically, I'll use the cleaner notation $S(s,t,s',t')$ to denote the probability density given from the analyst's formula for a stock to change from price $s$ to price $s'$ between time $t$ and time $t'$. So this self-referential property, which may very well be termed a *transitive property*, can be written:

$$\forall \, t', \quad S(s,t,s'',t'') = \int_0^\infty ds' \, S(s,t,s',t') \, S(s',t',s'',t'')$$

Another useful property that $S$ should have is *time invariance*. That is, the results should depend only on the duration between initial and final time, not on the absolute time. Algebraically,

$$\forall \, \delta, \quad S(s,t,s',t') = S(s, t+\delta, s', t'+\delta)$$

Lastly, it would be convenient to posit another invariance property for the stock price variables. Indeed this is implied by Schiller when he speaks of using a cumulative normal distribution to model the change in stock price. However, such a property would allow the stock price to become negative. To avoid this, we could either create an ad-hoc limit keeping $s$ positive, or we could posit that changes to stock price occur with respect to the logarithm of the price, not with respect to the price itself. That is, we could introduce a change of variables to work instead with the logarithm of stock price $x$ (where $x = \ln(s)$). Now it is plausible that the stock price estimator formula is invariant with respect to $x$. This has the wonderfully simplifying result that we only need to talk about the change in logarithm of stock price and the change in time. So when we want to denote the PDF for the logarithm of the stock price to change by $x$ units over a time interval of length $t$, we write simply:

$$S(x, t).$$

We can now use the invariance properties to simplify the transitive property equation to

$$\forall \, t', \quad S(s, t) = \int_0^\infty ds' \, S(s', t') \, S(s - s', t - t').$$

There are other conditions on $S$ that we could formalize, such as continuity, that $S(t=t')$ must be a Dirac delta function, that the resulting function has a well-defined integral of unity, and so on. However, the most important are the above transitive property and time and space invariance properties.

## Relation to the Physics of Diffusion

The properties that we have argued for and defined above for a formula to estimate the distribution of future stock prices are exactly the same properties that we would reasonably demand of a formula describing the movement of a particular particle, known to be at a particular point in space initially, over time as it moves around stochastically through a gas in one-dimension. Alternatively, it may describe the expected distribution of many marked particles which were initially concentrated at a single point. Lastly, this formula could be considered a Green's function for diffusion.

This relationship to physics makes me curious about the mathematics of diffusion. Perhaps there is a relationship then between the diffusion equation and option pricing? Or perhaps Einstein's Nobel Prize-winning paper on brownian motion would provide some insights. Then, from a strictly mathematical standpoint, I'm curious whether the cumulative normal function (which I believe is the PDF for brownian motion) is the only solution satisfying the properties discussed above. If not, I'd like to understand the solution space better... what else is out there? How else could prices or particles move around?

---

*Originally published on [Quasiphysics](https://quasiphysics.wordpress.com/2011/03/29/option-pricing-and-brownian-motion/).*
