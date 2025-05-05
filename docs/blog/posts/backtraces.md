---
title: Crashes in the Field
date: 2024-05-05
description: How to collect and analyze crash data from embedded systems in the field
categories:
  - embedded
---

# Crashes in the Field

As developers, we should write software that doesn't crash.
Crashes are, at best, warts on the user experience.
Nobody likes warts, do they?

If we cannot catch every way the system can crash while
developing, we've got to have some means of pulling crash
data back from the software that we deploy.

My goal here is to demystify the 

## Desiderata

Our minimalist embedded observability solution should be:

 - **efficient**
 - **portable**
 - **accurate**
 - **maintainable**
 - and **usable**

