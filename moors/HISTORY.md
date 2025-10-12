# moors HISTORY

## moors-0.2.6 (2025-09-22)

[GitHub release](https://github.com/andresliszt/moo-rs/releases/tag/moors-0.2.6)

## What’s Changed
- Add IBEA algorithm with hyper volume estimator [#219](https://github.com/andresliszt/moo-rs/pull/219) @andresliszt

## Contributors
@andresliszt

## moors-0.2.5 (2025-08-10)

[GitHub release](https://github.com/andresliszt/moo-rs/releases/tag/moors-0.2.5)

### What's Changed

#### Fixes

* remove rayon from fds by @andresliszt in [#210](https://github.com/andresliszt/moo-rs/pull/210)

## moors-0.2.4 (2025-07-10)

[GitHub release](https://github.com/andresliszt/moo-rs/releases/tag/moors-0.2.4)

### What's Changed

#### Fixes

* Derive clone in `NoDuplicatesCleaner` and expose `cross_euclidean_distances` as public to be used in python by @andresliszt in [#206](https://github.com/andresliszt/moo-rs/pull/206)

## moors-0.2.3 (2025-07-5)

[GitHub release](https://github.com/andresliszt/moo-rs/releases/tag/moors-0.2.3)

### What's Changed

#### Fixes

* Remove context from operators by @andresliszt in [#205](https://github.com/andresliszt/moo-rs/pull/205)

## moors-0.2.2 (2025-07-5)

[GitHub release](https://github.com/andresliszt/moo-rs/releases/tag/moors-0.2.2)

### What's Changed

#### New Features

* Add arithmetic and two points crossover in [#205](https://github.com/andresliszt/moo-rs/pull/204)
* Add inversion and uniform mutation in [#205](https://github.com/andresliszt/moo-rs/pull/204)

## moors-0.2.1 (2025-07-4)

[GitHub release](https://github.com/andresliszt/moo-rs/releases/tag/moors-0.2.1)

### What's Changed

#### New Features

* Implement `FitnessConstraintsPenaltySurvival ` for single objective optimization by @andresliszt in [#201](https://github.com/andresliszt/moo-rs/pull/201)
* Simplify builders by @andresliszt in [#203](https://github.com/andresliszt/moo-rs/pull/203)

## moors-0.2.0 (2025-06-25)

[GitHub release](https://github.com/andresliszt/moo-rs/releases/tag/moors-0.2.0)

### What's Changed

#### New Features

* Add `thiserror` crate for error defs by @andresliszt in [#182](https://github.com/andresliszt/moo-rs/pull/182)
* Refactor genetic to allow generic output dimensions for fitness and constraints by @andresliszt in [#185](https://github.com/andresliszt/moo-rs/pull/185)
* Remove unuseful GeneticOperator trait by @andresliszt in [#186](https://github.com/andresliszt/moo-rs/pull/186)
* hoy fix: `FitnessSurvival` was setting the rank incorrectly by @andresliszt in [#190](https://github.com/andresliszt/moo-rs/pull/190)
* Add `impl_constraints_fn!` macro for concatenating constraints @andresliszt in [#192](https://github.com/andresliszt/moo-rs/pull/192)
* Remove generics `FDim` and `ConstrDim`. Implement derive-builder by @andresliszt in [#193](https://github.com/andresliszt/moo-rs/pull/193)
* Implement derive builder in `Evaluator`, `Evolve` and `AlgorithmContext` structs. Remove num_objectives and num_constraints params by @andresliszt in [#194](https://github.com/andresliszt/moo-rs/pull/194)
* Implement constraints macro by @andresliszt in [#197](https://github.com/andresliszt/moo-rs/pull/197)
* Add `constraint_violation_totals` to `Population` and `Individual` to be used in the Selection and Surival strategies by @andresliszt in [#199](https://github.com/andresliszt/moo-rs/pull/199)
