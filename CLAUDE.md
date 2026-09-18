# build123d

Eurorack panels as code. `panels/*.py` are the sources; `render/out/<module>/manifest.json`
carries the generated `panel_w` / `panel_h` in mm and is the authority on every own-build
panel width (HP = `panel_w / 5.08`; panels are cut slightly under nominal for clearance).

## Module inventory

Before answering anything about which modules exist, what hardware is in the rack, or
which repo a module lives in, read the canonical inventory:

**`MODULES.md`** in the [`eight4awish`](https://github.com/Eight4aWish/eight4awish) repo
— <https://github.com/Eight4aWish/eight4awish/blob/main/MODULES.md>

If that repo is checked out alongside this one, read it from disk; otherwise fetch the URL.

It covers all ten repos: the released modules, the built-but-undrafted ones, the
purchased rack with HP and function, companion software, and what is deliberately *not*
a module. No single repo sees all of it, so do not infer the full picture from this one.
