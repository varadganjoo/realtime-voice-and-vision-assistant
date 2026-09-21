import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import safety


@pytest.fixture(autouse=True)
def reset_confirmer():
    safety.register_confirmer(None)
    yield
    safety.register_confirmer(None)


def test_sensitive_tool_is_denied_without_a_confirmer():
    calls = []
    registry = {"shutdown": lambda: calls.append("ran")}
    result = safety.call_tool("shutdown", None, registry)
    assert calls == []
    assert "did not approve" in result


def test_sensitive_tool_runs_only_after_approval():
    calls = []
    registry = {"toggle_wifi": lambda state: calls.append(state) or "done"}
    safety.register_confirmer(lambda name, arg: False)
    safety.call_tool("toggle_wifi", "off", registry)
    assert calls == []
    safety.register_confirmer(lambda name, arg: True)
    assert safety.call_tool("toggle_wifi", "off", registry) == "done"
    assert calls == ["off"]


def test_confirmer_error_counts_as_denial():
    def broken(name, arg):
        raise RuntimeError("dialog failed")

    safety.register_confirmer(broken)
    assert safety.request_confirmation("restart", None) is False


def test_ordinary_tools_do_not_prompt():
    def should_not_be_called(name, arg):
        raise AssertionError("prompted for a harmless tool")

    safety.register_confirmer(should_not_be_called)
    registry = {"calculate": lambda expr: f"calc:{expr}"}
    assert safety.call_tool("calculate", "2+2", registry) == "calc:2+2"


def test_unknown_tool():
    assert safety.call_tool("format_disk", None, {}) == "Unknown tool: format_disk"


def test_code_execution_is_gated():
    assert "code_generate_and_execute" in safety.SENSITIVE_TOOLS


def test_scrubbed_env_drops_secrets():
    env = safety.scrubbed_env(
        {"GROQ_API_KEY": "x", "OPENWEATHER_API_KEY": "y", "GITHUB_TOKEN": "z", "PATH": "/bin"}
    )
    assert env == {"PATH": "/bin"}


def test_execute_code_returns_output():
    assert safety.execute_code("print(6 * 7)") == "42"


def test_execute_code_cannot_see_api_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "secret-value")
    out = safety.execute_code("import os; print(os.environ.get('GROQ_API_KEY', 'missing'))")
    assert out == "missing"


def test_execute_code_runs_in_an_empty_temp_dir():
    out = safety.execute_code("import os; print(sorted(os.listdir('.')))")
    assert out == "['generated.py']"


def test_execute_code_times_out():
    assert safety.execute_code("import time; time.sleep(5)", timeout=1) == "Error: Code execution timed out."


def test_execute_code_reports_errors():
    out = safety.execute_code("raise ValueError('boom')")
    assert "ValueError: boom" in out
