"""Shared test setup.

The agent's scenario, rehearsal and contract tests check orchestration against one fixed demo world (the fixture pack),
so they pin every tool to the fixture backend. tests/test_agent_live.py tests the live backend directly.
Set before any test module imports the tool registry, which reads its backends once.
"""
import os

os.environ.setdefault("AGENT_BACKENDS", "fixture")
