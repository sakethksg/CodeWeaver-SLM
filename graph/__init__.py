"""
LangGraph-based evaluation graph definitions.

Modules:
  - state: TypedDict state definitions for pipeline and evaluation graphs
  - pipeline_graph: Per-problem pipeline subgraph (generate → execute → repair → rank)
  - analysis_graph: Parallel fan-out/fan-in for 8 analysis agents
  - eval_graph: Main evaluation graph (load → process → analyze → report)
"""
