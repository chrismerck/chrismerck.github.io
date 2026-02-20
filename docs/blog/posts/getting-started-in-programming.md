---
title: Getting Started in Programming
date: 2010-12-29
description: A guide to getting started in programming across four domains — philosophical, game, embedded, and data-driven — with recommended languages and resources for each
categories:
  - programming
tags:
  - comp
---

As with [natural languages](https://en.wikipedia.org/wiki/Natural_language), most of the tools are poor, most of the methods are ineffective, and the books unreadable. Successful language learners gather quality resources and use proven references. The same applies to programming. Here I organize programming into four application areas with my recommended resources for each.

<!-- more -->

## 1. Philosophical Programming

This is for those interested in math, logic, and language. The recommended language is [Scheme](https://en.wikipedia.org/wiki/Scheme_(programming_language)). It's like learning to program with Yoda.

**Reading Materials:**

- [*The Little Schemer*](http://www.amazon.com/Little-Schemer-Daniel-P-Friedman/dp/0262560992) by Daniel P. Friedman
- [*Structure and Interpretation of Computer Programs* (SICP)](http://mitpress.mit.edu/sicp/)

**Implementation:**

- [Racket](http://racket-lang.org/) (formerly PLT Scheme)
- [MIT Scheme](http://www.gnu.org/software/mit-scheme/)

After mastering Scheme, I recommend transitioning to [Python](https://en.wikipedia.org/wiki/Python_(programming_language)) for practical applications.

## 2. Game Programming

Game programming is engaging because you can immediately see and enjoy the results of your work. The recommended language is Python with the [PyGame](https://en.wikipedia.org/wiki/Pygame) library.

**Reading Materials:**

- [*Think Python*](http://www.greenteapress.com/thinkpython/thinkpython.html)
- [*Beginning Game Development with Python and PyGame*](http://www.amazon.com/Beginning-Game-Development-Python-Pygame/dp/1590598725)
- [Python Documentation](http://docs.python.org/)

**Important Note:** Use Python 2.7, not Python 3, as Python 3 is NOT compatible with most of the books, tutorials, and libraries.

**Downloads:**

- [Python](http://www.python.org/)
- [PyGame](http://www.pygame.org/download.shtml)

## 3. Embedded Programming

This is for programming [microcontrollers](https://en.wikipedia.org/wiki/Microcontroller) and electronic circuits.

**Recommended Platform:**

- [ATMEL AVR](https://en.wikipedia.org/wiki/AVR_microcontrollers) microcontrollers (specifically ATmega168)
- [Arduino](https://en.wikipedia.org/wiki/Arduino) community hardware and tutorials
- [Parallax Propeller](https://en.wikipedia.org/wiki/Parallax_Propeller) chip (alternative, more advanced option)

**Language:**

- [C](https://en.wikipedia.org/wiki/C_(programming_language)) (de-facto standard for AVR)
- SPIN (for Propeller chips)
- [Assembly language](https://en.wikipedia.org/wiki/Assembly_language) (for advanced applications)

**Reading Materials:**

- [*Practical C Programming*](http://www.amazon.com/Practical-Programming-3rd-Steve-Oualline/dp/1565923065) by Steve Oualline
- [*Introduction to Embedded Systems*](http://leeseshia.org/)
- [*Designing Embedded Hardware*](http://www.amazon.com/Designing-Embedded-Hardware-John-Catsoulis/dp/0596007558/) by John Catsoulis

## 4. Data-Driven Programming

This addresses programming that manages information through databases and user interfaces. A typical web application involves the following components:

- Hosting service
- [HTTP](https://en.wikipedia.org/wiki/Hypertext_Transfer_Protocol) daemon/server
- Database
- Server-side scripts

**My Recommended Stack:**

- Hosting: [Dreamhost](https://www.dreamhost.com/)
- HTTP daemon: [Apache](https://en.wikipedia.org/wiki/Apache_HTTP_Server)
- Database: [SQLite](https://en.wikipedia.org/wiki/SQLite)
- Framework: Python with [Django](https://en.wikipedia.org/wiki/Django_(web_framework))

**Resources:**

- [*Think Python*](http://www.greenteapress.com/thinkpython/thinkpython.html) (mentioned above)
- The official [Django book](https://www.djangoproject.com/) (available on the Django website)

---

*Originally published on [Quasiphysics](https://quasiphysics.wordpress.com/2010/12/29/getting-started-in-programming/).*
