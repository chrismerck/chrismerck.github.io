---
title: Adding Accents to Romanized Japanese
date: 2012-02-19
description: A Python utility for converting capitalized romanized Japanese into JSL-style accent-marked text
categories:
  - programming
tags:
  - ling
  - comp
  - nihongo
---

While working through [*Japanese the Spoken Language*](https://en.wikipedia.org/wiki/Japanese:_The_Spoken_Language) (JSL), I feel the need to make [Anki](https://en.wikipedia.org/wiki/Anki_(software)) cards for the spoken words that I'm having trouble recalling the meanings of. Now, I haven't yet decided exactly what card format is best, but I was dreading having to type in accents on the romanized words. (See the introduction to JSL to see what these accents are.) So, I wrote a quick program to allow easy input of accented words.

<!-- more -->

This Python function makes it easy to write JSL-romanized Japanese text with accent markings. It works by taking as input the text with capitalization indicating high-pitch, and outputs JSL-type accent marked text. This is particularly useful when entering a list of spoken Japanese words into Anki.

Examples:

- `KYOo` => `kyôo`
- `aTARASIi ZIsyo` => `atárasìi zîsyo`
- `aSITA IKIMAsu yo` => `asíta ikimàsu yo`

## The Code

```python
# -*- coding: utf-8 -*-

"""
file: jslaccent.py
desc: converts accent markings from mora capitalization to JSL accents
usage: accent('aSITA') ==> 'asíta'
note: This is for JSL romanization only!
"""

vowels = ['a','e','i','o','u']

def decompose(s):
    """Decompose romanized string into individual mora."""
    if s == '':
        return []
    # punctuation or space
    elif s[0] in [' ',';','.','!',',','?',':']:
        return [s[0]] + decompose(s[1:])
    # vowel mora
    elif s[0].lower() in vowels:
        return [s[0]] + decompose(s[1:])
    # full mora n
    elif s[0].lower() == 'n' and (len(s)==1 or s[1].lower() not in vowels):
        return [s[0]] + decompose(s[1:])
    # consonant + y
    elif s[1].lower() == 'y':
        return [s[0:3]] + decompose(s[3:])
    # long consonant
    elif s[1].lower() == s[0].lower():
        return [s[0]] + decompose(s[1:])
    # consonant + vowel
    else:
        return [s[0:2]] + decompose(s[2:])

def iscap(s):
    """Check if the vowel (or n mora) is capitalized = high-pitch."""
    if len(s) == 1:
        c = s[0]  # single letter mora (n or vowel)
    else:
        c = s[-1]  # multi-letter mora (vowel is in final position)
    return c.lower() != c

# fns to add accents to final char in string and remove capitalization
acute_dict = {'a':'á','e':'é','i':'í','o':'ó','u':'ú','n':'ń'}
grave_dict = {'a':'à','e':'è','i':'ì','o':'ò','u':'ù','n':'ǹ'}
cflex_dict = {'a':'â','e':'ê','i':'î','o':'ô','u':'û','n':'n̂'}

def add_acute(s):
    return s[:-1].lower() + acute_dict[s[-1].lower()]

def add_circumflex(s):
    return s[:-1].lower() + cflex_dict[s[-1].lower()]

def add_grave(s):
    return s[:-1].lower() + grave_dict[s[-1].lower()]

def accent(s):
    """Convert capitalization-based accent marking to JSL diacritics."""

    # decompose into mora
    mora = decompose(s)

    # add accent marks based on context
    prev_accent_level = 0
    for i in range(len(mora)):
        if mora[i] in ['.',':',';',',']:
            # punctuation resets accent phrase
            prev_accent_level = 0
        if not mora[i].isalpha():
            # skip over spaces and punctuation
            continue

        this_accent_level = iscap(mora[i])
        next_accent_level = 0
        if i+1 < len(mora) and iscap(mora[i+1]):
            next_accent_level = 1
        elif i+2 < len(mora) and not mora[i+1].isalpha() and iscap(mora[i+2]):
            next_accent_level = 1

        if this_accent_level:
            if prev_accent_level and not next_accent_level:
                mora[i] = add_grave(mora[i])
            elif not prev_accent_level and next_accent_level:
                mora[i] = add_acute(mora[i])
            elif not prev_accent_level and not next_accent_level:
                mora[i] = add_circumflex(mora[i])

        prev_accent_level = this_accent_level

    # rejoin mora
    retval = ''.join(mora)
    return retval.lower()

# examples
print(accent('aSITA'))
print(accent('sen'))
print(accent('saNMAn'))
print(accent('aTARASIi ZIsyo'))
print(accent('KYOo'))
print(accent('aSITA IKIMAsu yo.'))
print(accent('aSITA, IKIMAsu yo.'))
print(accent('TYOtto SUMIMASEn.'))
```

The program decomposes romanized text into individual [mora](https://en.wikipedia.org/wiki/Mora_(linguistics)), detects which mora are high-pitched via capitalization, and applies the appropriate [diacritical marks](https://en.wikipedia.org/wiki/Diacritic): acute (á) for rising pitch, grave (à) for falling pitch, and circumflex (â) for a single high-pitched mora surrounded by low pitch. Punctuation boundaries reset the pitch tracking, respecting [accent phrase](https://en.wikipedia.org/wiki/Japanese_pitch_accent) boundaries in Japanese prosody.

*Originally published on [Quasiphysics](https://quasiphysics.wordpress.com/2012/02/19/adding-accents-to-romanized-japanese/).*
