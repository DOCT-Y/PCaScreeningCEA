# PCaScreeningCEA

## Design philosophy

### 1) Model = a message-passing graph, not a matrix

Instead of building a transition matrix up front, the model is represented as a **tree/graph of objects**:

* `MarkovController` orchestrates time and aggregation.
* `MarkovState` represents a state.
* `StateTransition` represents an edge from one state to another (and *reports* its results back).
* `ChanceNode` / base `Node` support hierarchical branching.

Flow is computed by calling `forward()` down the graph and **sending summaries upward** via `notify_controller()`. This keeps the core logic generic: “move probability mass through nodes, accumulate outcomes.”

### 2) Probabilities are first-class objects (with PSA hooks)

All transition probabilities are `Probability` objects, which support:

* **constant values** (`ProbabilityWithRange` with default params)
* **sampling** from distributions (uniform/beta/binomial/gamma/normal/lognormal) via `sample_value()`
* **time-varying schedules** (`TimeVaringProbability` is a list indexed by cycle)
* a **complement** probability (`ComplementProbability`) so you can define “the remainder to 1” without manually computing it

**Intent:** make deterministic runs easy, and make probabilistic sensitivity analysis possible by re-sampling parameters before runs.

### 3) Outcomes are just “variables” carried along with the probability mass

Each node can carry arbitrary `**variables` (e.g., `cost=...`, `qaly=...`).

During traversal:

* variables are **summed** along the path,
* then the controller multiplies them by the selected cycle probability (“start/end/half-cycle correction style”), and discounts them.

So the engine doesn’t “know” what cost or utility is; it just propagates and aggregates named scalars.

### 4) Separation of concerns: traversal vs accounting

* Traversal: `MarkovState.start()` and `StateTransition.forward()` decide what to send.
* Accounting: `MarkovController.handle_node_message()` decides *how* to count probability within-cycle (`start`, `end`, `half`) and how to discount outcomes.

This makes the model extensible: if you wanted different counting rules or extra bookkeeping, you mostly touch the controller.

### 5) Cycle-to-cycle logic is explicit and simple

The controller runs cycles like this:

1. It holds `next_cycle_start_prob[state]` for the upcoming cycle.
2. At cycle start, it sends each state its starting probability.
3. `StateTransition` objects push probability to destinations and append the “end of cycle” prob into `next_cycle_start_prob[destination]`.
4. At cycle end, controller sums the lists into single next-cycle probabilities, stores totals, clears per-cycle caches, increments time.

This is a classic cohort Markov update, but implemented through object interactions.


## The “intended workflow” (as this demo embodies)

1. Define global settings: total_cycles, count_method, discount_rate
2. Define parameters (possibly time-varying)
3. Create states (`MarkovState`) with initial distribution and state rewards
4. Create chance nodes (`ChanceNode`) for within-cycle branching/rewards
5. Create transitions (`StateTransition`) to destination states with probs/rewards
6. Wire graph with `add_child`
7. `controller.init_prob(...)` then `controller.verify()`
8. prob_df, variable_df = `controller.run()`
