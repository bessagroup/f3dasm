"""LLM backend plug-in subpackage for agentic-f3dasm.

Each backend module in this package supplies a :class:`Backend` bundle
(defined in ``base``) that the orchestrator uses to create Strategizer
and Implementer sessions.  Adding a new backend requires only a single
new module here — no changes to ``agent_runtime`` are needed.
"""
