"""In-process, AI-assisted localized repair of a recorded run.

The deterministic ``playwright_test.py`` replay is the fast path. When it *errors*
(a stale locator, a renamed control — the *test* broke, not the app), this package
re-runs the recorded steps in-process and, at the exact step that fails to resolve,
asks the agent to look at the live page and pick the element the step meant. The
corrected locator is captured back into a fresh recording, so the next replay is
deterministic again. A replay that *fails its assertions* is a genuine regression and
is never healed here.
"""

# Human-readable, step-by-step record of what a model-free run executed, written into
# the run's own directory next to report.json and served like any other artifact.
# It lives in this dependency-free module so both the suite runner and the in-process
# replayer can name it without importing each other.
REPLAY_LOG_NAME = "replay.log"
