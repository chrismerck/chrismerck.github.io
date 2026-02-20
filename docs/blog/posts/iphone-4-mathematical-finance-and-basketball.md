---
title: iPhone 4, Mathematical Finance, and Basketball
date: 2010-11-29
description: Using iTunes U lectures on financial markets to explore geometric variance and log-returns
categories:
  - math
tags:
  - fin
  - math
---

I've been playing around with the iPhone 4 since I took the plunge about two weeks ago. Amazing device. I really love the iTunes U feature. Access to countless hours of educational material. I've been falling asleep to The Tolkien Professor and the 99% Invisible podcast. The other day I took a long walk while listening to a lecture on the history of probability theory and found myself playing basketball (more vigorously than I'd expect). Why sit in a class, half-asleep, when you can listen to the same lecture while getting some cardio? As Steve Jobs would say, This is HUGE.

<!-- more -->

If money is sweat (and love is tears), and I plan on working my tail off (i.e., sweating a lot), I had better understand money. My weapon of choice is Robert Shiller's [Financial Markets](https://oyc.yale.edu/economics/econ-252-08) course from [Open Yale Courses](https://en.wikipedia.org/wiki/Open_Yale_Courses). My plan is to watch the video lectures (on the iPhone!), attempt to work the problem sets (implementing them as Python programs where ever possible), and to keep studying game theory with a view towards financial applications.

## Lecture 1

The first lecture is mostly an apology for making money. Not much of interest.

## Lecture 2

Here it gets interesting. I learn that [probability theory](https://en.wikipedia.org/wiki/Probability_theory) was not known in academia until the 1600s, and that the word "probable" originally meant "trustworthy." I had never heard that.

I also had an idea I want to record for later investigation: Is there a geometric analog to the [arithmetic variance](https://en.wikipedia.org/wiki/Variance)? The question comes up in considering returns on an investment. Investment returns are naturally multiplicative, so one would expect the [geometric mean](https://en.wikipedia.org/wiki/Geometric_mean) to be the relevant measure of central tendency. Let $\rho_i = \ln r_i$ be log-returns. Then the arithmetic mean of the log-returns,

$$\bar{\rho} = \frac{1}{n} \sum_{i=1}^n \rho_i,$$

yields the geometric mean of the returns:

$$\bar{r} = \left(\prod_{i=1}^n r_i\right)^{1/n}.$$

What about the variance? Writing the variance of the log-returns,

$$\sigma^2_\rho = \sum_{i=1}^n (\rho_i - \bar{\rho})^2,$$

we can substitute back in to get

$$\sigma^2_\rho = \sum_{i=1}^n \left(\ln r_i - \ln \bar{r}\right)^2 = \sum_{i=1}^n \left(\ln \frac{r_i}{\bar{r}}\right)^2.$$

However, it is not apparent to me how to proceed further. Is this last form sufficient to be called a "geometric variance"?

## Lecture 3

The third lecture covered the basics of insurance and communism. Not too much here in terms of mathematics.

---

*Originally published on [Quasiphysics](https://quasiphysics.wordpress.com/2010/11/29/iphone-4-mathematical-finance-and-basketball/).*
