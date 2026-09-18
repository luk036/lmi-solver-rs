# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] - 2026-09-18

### Changed
- Update `ellalgo-rs` to 0.1.10

### Performance
- `LDLTMgr::factor_impl` now holds one mutable slice for the whole sweep and
  indexes it flat. The per-element `as_slice().unwrap()` contiguity check used to
  run once per `j` (O(n^2) times); it is now hoisted out. The function is also
  marked `#[inline]`.
- Replace ndarray view/dot with flat-slice loops in the LDLT inner reduction
- Add `#[inline]` to single-line-body functions across the crate

### Refactoring
- Extract a shared `factor_impl` driven by a `PivotPolicy` (Template Method)
