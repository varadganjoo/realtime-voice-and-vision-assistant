"""Guardrails for the assistant's tool calls.

Two concerns live here so they can be tested without a microphone, a GPU or Windows:

1. Human confirmation for tools that change the machine (power, radios, running
   generated code). Confirmation fails closed: no registered confirmer, a timeout,
   or an error all count as "no".
2. Running model-generated Python in a separate process with a timeout, an empty
   working directory, and an environment stripped of API keys and other secrets.

The subprocess runner is isolation, not a sandbox. Generated code still runs as the
current user, which is why code execution also sits behind the confirmation gate.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
from typing import Any, Callable, Mapping, Optional

# Tools that shut the machine down, change connectivity, or run arbitrary code.
SENSITIVE_TOOLS = frozenset(
    {
        "shutdown",
        "restart",
        "sleep",
        "toggle_wifi",
        "toggle_bluetooth",
        "code_generate_and_execute",
    }
)

CONFIRMATION_TIMEOUT_SECONDS = 30
CODE_TIMEOUT_SECONDS = 60

Confirmer = Callable[[str, Any], bool]
_confirmer: Optional[Confirmer] = None


def register_confirmer(confirmer: Optional[Confirmer]) -> None:
    """Install the function that asks the user to approve a sensitive tool call."""
    global _confirmer
    _confirmer = confirmer


def request_confirmation(tool_name: str, tool_input: Any) -> bool:
    """Return True only if the user explicitly approved this call."""
    if tool_name not in SENSITIVE_TOOLS:
        return True
    if _confirmer is None:
        logging.warning("No confirmer registered; denying %s", tool_name)
        return False
    try:
        return bool(_confirmer(tool_name, tool_input))
    except Exception as exc:  # a broken prompt must never mean "yes"
        logging.error("Confirmation for %s failed: %s", tool_name, exc)
        return False


def call_tool(
    tool_name: str,
    tool_input: Any,
    registry: Mapping[str, Callable[..., Any]],
) -> str:
    """Single dispatch point for every tool the model asks for."""
    func = registry.get(tool_name)
    if func is None:
        return f"Unknown tool: {tool_name}"
    if not request_confirmation(tool_name, tool_input):
        return f"The user did not approve running {tool_name}, so it was not run."
    return func(tool_input) if tool_input else func()


def make_tk_confirmer(root, timeout: float = CONFIRMATION_TIMEOUT_SECONDS) -> Confirmer:
    """Build a confirmer that shows a Yes/No dialog on the Tk thread.

    Tool calls arrive on worker threads and Tk is not thread safe, so the dialog is
    scheduled with root.after and the worker waits on an Event. No answer within
    `timeout` seconds is treated as a refusal.
    """
    from tkinter import messagebox

    def confirm(tool_name: str, tool_input: Any) -> bool:
        answered = threading.Event()
        result = {"approved": False}
        detail = "" if tool_input in (None, "") else f"\n\nInput:\n{str(tool_input)[:500]}"

        def ask() -> None:
            try:
                result["approved"] = messagebox.askyesno(
                    "Confirm action",
                    f"The assistant wants to run '{tool_name}'.{detail}\n\nAllow it?",
                    parent=root,
                )
            finally:
                answered.set()

        root.after(0, ask)
        if not answered.wait(timeout):
            logging.warning("No answer for %s within %ss; denying", tool_name, timeout)
            return False
        return result["approved"]

    return confirm


_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH", re.IGNORECASE)


def scrubbed_env(source: Optional[Mapping[str, str]] = None) -> dict:
    """Copy of the environment with anything that looks like a secret removed."""
    source = os.environ if source is None else source
    return {name: value for name, value in source.items() if not _SECRET_NAME.search(name)}


def execute_code(code: str, timeout: float = CODE_TIMEOUT_SECONDS) -> str:
    """Run generated Python in a child process and return its combined output.

    The child gets its own empty temporary working directory, an environment without
    API keys, and a hard timeout. The directory is removed afterwards.
    """
    with tempfile.TemporaryDirectory(prefix="assistant_exec_") as workdir:
        script = os.path.join(workdir, "generated.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(code)
        try:
            result = subprocess.run(
                [sys.executable, script],
                cwd=workdir,
                env=scrubbed_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out."
        except Exception as exc:
            return f"Error during execution: {exc}"

    output = result.stdout.strip()
    errors = result.stderr.strip()
    if result.returncode != 0:
        logging.error("Generated code exited with return code %s", result.returncode)
    if output and errors:
        return f"Output:\n{output}\n\nErrors:\n{errors}"
    return output or errors or "Execution completed with no output."
