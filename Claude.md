You need to build a **multi-agent evaluation system** for code generation architectures.

The goal is to rigorously evaluate a proposed execution-guided code generation system on:

* HumanEval
* MBPP

The system pipeline:

* k candidate generation
* execution-based filtering
* structured repair loop
* re-execution
* ranking and selection

Each agent has a **strict role**. Follow coordination rules and produce structured outputs.

---

# 🧠 AGENT DEFINITIONS

## 🔹 1. Orchestrator Agent

Responsibilities:

* Coordinate all agents
* Aggregate outputs
* Ensure consistency
* Produce final report

---

## 🔹 2. Generation Evaluation Agent

Tasks:

* Evaluate initial k candidates (before repair)

Metrics:

* pass@1, pass@5, pass@10 (before repair)
* % problems solved after generation
* average test pass rate

---

## 🔹 3. Execution Metrics Agent

Tasks:

* Track execution behavior

Metrics:

* total executions per problem
* execution success/failure counts
* average test pass rate per problem

---

## 🔹 4. Repair Analysis Agent

Tasks:

* Evaluate repair loop effectiveness

Metrics:

* pass@k (after repair)
* repair success rate
* delta improvement (before → after repair)
* average improvement per round
* diminishing returns across rounds
* candidates fixed / total failed

---

## 🔹 5. Oracle Analysis Agent

Tasks:

* Compute upper bound performance

Metrics:

* oracle pass@1
* oracle pass@k (full candidate pool)

Definition:

* A problem is solved if ANY candidate (before or after repair) passes all tests

---

## 🔹 6. Efficiency & Cost Agent

Tasks:

* Analyze system efficiency

Metrics:

* average executions per successful solution
* average latency per problem
* latency breakdown:

  * generation
  * execution
  * repair
* cost per problem
* cost per solved problem
* marginal cost of repair

---

## 🔹 7. Error Analysis Agent

Tasks:

* Categorize failures

Categories:

* syntax errors
* runtime errors
* logical errors

Outputs:

* percentage distribution
* which error types are most fixable via repair

---

## 🔹 8. Dataset Analysis Agent

Tasks:

* Provide dataset-wise breakdown

For EACH dataset:

HumanEval:

* pass@k (before & after repair)
* oracle pass@k
* avg score

MBPP:

* pass@k (before & after repair)
* oracle pass@k
* avg score

---

## 🔹 9. Tradeoff Analysis Agent

Tasks:

* Analyze latency vs accuracy tradeoff

Vary:

* k (number of samples)
* repair rounds

Outputs:

* accuracy vs latency curves
* optimal operating point

---

# 🔁 AGENT INTERACTION FLOW

1. Generation Evaluation Agent → evaluates initial outputs
2. Execution Metrics Agent → records execution stats
3. Repair Analysis Agent → evaluates repair loop
4. Oracle Analysis Agent → computes upper bound
5. Error Analysis Agent → categorizes failures
6. Efficiency & Cost Agent → computes cost/latency
7. Dataset Analysis Agent → aggregates per dataset
8. Tradeoff Analysis Agent → evaluates scaling behavior
9. Orchestrator Agent → compiles final report

---

# 🧪 EXPERIMENT PROTOCOL

* k = 10 samples per problem
* temperature = [0.3, 0.8]
* repair rounds = 2
* fixes per failure = 2
* execution timeout = 5 seconds

---

# 📈 OUTPUT FORMAT (STRICT)

## 1. Summary Table

Dataset | pass@1 | pass@5 | pass@10 | pass@10 (after repair) | oracle pass@k | Avg Score

---

## 2. Detailed Metrics

* generation performance
* execution statistics
* repair effectiveness
* efficiency & cost

---

## 3. Tradeoff Analysis

* latency vs accuracy
* cost vs performance

---

## 4. Error Analysis

* distribution of error types
* repairability insights

---

## 5. Insights

* strengths
* weaknesses
* bottlenecks
* gap to oracle

---

## 6. Conclusions

* effectiveness
* scalability
* compute-efficiency tradeoffs
* deployment suitability

---

# ⚠️ STRICT RULES

* Use ONLY execution-based correctness (unit tests)
* No assumptions
* All metrics must be computed, not guessed
* Maintain consistency across agents
* Outputs must be quantitative and reproducible

---

# 🎯 FINAL GOAL

Produce a **research-grade, multi-perspective evaluation** by combining insights from all agents.
