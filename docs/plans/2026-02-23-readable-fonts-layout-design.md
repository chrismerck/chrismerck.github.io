---
title: Readable Fonts & Side Menu Layout
subtitle: Typography and layout improvements for chrismerck.github.io
date: February 23, 2026
abstract: |
  Switch from Special Elite (weathered typewriter) to Newsreader (editorial serif)
  for improved readability, narrow the main content column to ~45rem for optimal
  line length, and enable a right-side table of contents sidebar on desktop.
---

## Font

- **Body text:** Newsreader (Google Fonts) — editorial serif designed for on-screen long-form reading
- **Code:** Fira Code (unchanged)
- Special Elite removed entirely

## Layout (Desktop)

- **Left sidebar:** Site navigation (existing)
- **Right sidebar:** Page table of contents — enabled by removing `toc.integrate` from Material features
- **Main content column:** Max-width narrowed from `60rem` to `45rem` (~720px, ~65-75 chars/line)
- Material's built-in responsive grid handles the three-column layout natively

## Card Treatment

- Keep notebook-style card (border, shadow, rounded corners) on main content area
- Adjust padding if needed at narrower width

## Mobile

- Sidebars collapse as they do now (Material handles this)
- Content goes full-width on small screens (existing responsive CSS stays)

## Changes Required

1. `mkdocs.yml`: Change `text` font from `Special Elite` to `Newsreader`; remove `toc.integrate` from features
2. `custom.css`: Change `.md-main__inner` max-width from `60rem` to `45rem`
