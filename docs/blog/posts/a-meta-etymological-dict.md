---
title: A Meta-Etymological Dict
date: 2012-07-02
description: A proposal for a polyglot tool that searches multiple etymological dictionaries simultaneously to reveal cognate relationships across languages
categories:
  - linguistics
tags:
  - etymology
  - polyglot
  - tools
---

Aspiring [polyglots](https://en.wikipedia.org/wiki/Multilingualism) face the huge challenge of massive vocabulary acquisition. Even in closely related languages, it is easy to miss [cognates](https://en.wikipedia.org/wiki/Cognate) because of sound changes and different orthographies. I have found that using etymological information from dictionary look-ups has boosted my recall for learning from a single language -- but I conjecture that this boost will carry over to other languages of interest if the dictionary would provide related words in every language of interest when they exist. However, in present form, this procedure requires searching through possibly 12 different books! I believe an electronic version would be a huge boon to the modern polyglot community.

<!-- more -->

## Intention

To build a tool with an online (mobile-friendly) interface for simultaneously searching the [etymological](https://en.wikipedia.org/wiki/Etymology) entries of many trustworthy sources and giving a digest of the relationships between words of interest in a concise manner. Hyperlinks will allow the user to jump to the original source dictionary.

## Input

Etymological dictionaries (or normal dictionaries with etymological information) in several languages.

## Output

A network of words with edges representing an etymological link as claimed by some dictionary. This may be stored in XML.

## Usage

The user searches for a word form in any language. The program finds all matching nodes, and then for each node, lists all neighboring entries, regardless of language. The program may also search several levels deep, filtering results by languages of interest. Think of a tree-like presentation.

## Example

So, for example, if I search "hill", and I have told the program that I'm also interested in French and Swedish, then I should see the Swedish "kulle" and French "colline" as results, with links showing how they are interconnected:

- Eng: "hill" <> Fr. "colline"
- Eng: "hill" <> Sv. "kulle"

Or something like that (I'm not claiming this etymology is right or whatever.) The point is that you get just enough info on the screen so you can make a mental link between the vocabularies of the various languages and learn the word as a SINGLE word, rather than having to relearn repeatedly.

Plus, I firmly believe that such interconnected knowledge is more easily retained than simple, rote pairs like "kulle=hill".

---

*Originally published on [Quasiphysics](https://quasiphysics.wordpress.com/2012/07/02/a-meta-etymological-dict/).*
