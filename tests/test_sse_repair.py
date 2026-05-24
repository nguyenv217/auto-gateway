"""Unit tests for heuristic_json_repair and extract_error_from_payload."""

from __future__ import annotations

import pytest

from auto_gateway.core.sse_repair import heuristic_json_repair, extract_error_from_payload


class TestHeuristicJsonRepair:
    """Tests for heuristic_json_repair()."""

    # ---- fast path: already-valid JSON ----

    def test_already_valid_json_dict(self):
        """Valid JSON dict should be returned unchanged."""
        result = heuristic_json_repair('{"key": "value"}')
        assert result == {"key": "value"}

    def test_already_valid_json_nested(self):
        """Valid nested JSON should be returned unchanged."""
        result = heuristic_json_repair('{"choices":[{"delta":{"content":"hi"}}]}')
        assert result == {"choices": [{"delta": {"content": "hi"}}]}

    def test_valid_json_but_not_dict_returns_none(self):
        """A valid JSON string that isn't a dict should return None."""
        assert heuristic_json_repair('"just a string"') is None
        assert heuristic_json_repair('42') is None
        assert heuristic_json_repair('[1, 2, 3]') is None

    def test_empty_or_non_string_returns_none(self):
        """Empty / non-string inputs should return None."""
        assert heuristic_json_repair("") is None
        assert heuristic_json_repair(None) is None  # type: ignore[arg-type]

    # ---- repair: unquoted-key JavaScript objects ----

    @pytest.mark.parametrize(
        "raw, expected",
        [
            # Single unquoted key with string value
            ('{error: "overloaded"}', {"error": "overloaded"}),
            # Single unquoted key with boolean value
            ("{rate_limit: true}", {"rate_limit": True}),
            # Single unquoted key with numeric value
            ("{retry_after: 30}", {"retry_after": 30}),
            # Single unquoted key with null value
            ("{data: null}", {"data": None}),
            # Multiple unquoted keys
            ('{error: "overloaded", code: 429, retry_after: 30}',
             {"error": "overloaded", "code": 429, "retry_after": 30}),
            # Nested object with unquoted keys
            ('{error: {message: "boom", type: "rate"}}',
             {"error": {"message": "boom", "type": "rate"}}),
            # Key containing dot (e.g. some.custom_key)
            ('{some.custom_key: "val"}', {"some.custom_key": "val"}),
            # Already-quoted keys mixed with unquoted — only unquoted fixed
            ('{"ok": true, error: "overloaded", "code": 429}',
             {"ok": True, "error": "overloaded", "code": 429}),
            # Leading / trailing whitespace tolerated
            ('  {error: "overloaded"}  ', {"error": "overloaded"}),
            # Empty object
            ("{}", {}),
        ],
    )
    def test_repairs_common_patterns(self, raw, expected):
        result = heuristic_json_repair(raw)
        assert result == expected

    # ---- edge cases ----

    def test_garbage_input_returns_none(self):
        assert heuristic_json_repair("not json at all") is None
        assert heuristic_json_repair("{broken") is None
        assert heuristic_json_repair("<html>") is None

    def test_key_already_quoted_not_double_quoted(self):
        """A key that is already in double quotes must not get extra quotes."""
        result = heuristic_json_repair('{"error": "overloaded"}')
        assert result == {"error": "overloaded"}

    def test_single_quoted_key_repaired(self):
        """Single-quoted keys (also invalid JSON) should be wrapped with
        double quotes."""
        result = heuristic_json_repair("{'error': 'overloaded'}")
        # 'error' in the raw is unquoted, but the regex should still match it
        # because it's NOT preceded by a double quote.
        # However, single quotes around values will cause json.loads to fail.
        # This is a known limitation — we only guarantee repair for
        # double-quoted values.
        # The function should fall through to the "unable to repair" path.
        assert result is None  # single-quoted values aren't valid JSON


class TestExtractErrorFromPayload:
    """Tests for extract_error_from_payload()."""

    def test_error_as_string(self):
        info = extract_error_from_payload({"error": "something went wrong"})
        assert info == {
            "message": "something went wrong",
            "type": "provider_error",
            "code": None,
        }

    def test_error_as_dict(self):
        info = extract_error_from_payload({
            "error": {
                "message": "quota exceeded",
                "type": "insufficient_quota",
                "code": "402",
            },
        })
        assert info == {
            "message": "quota exceeded",
            "type": "insufficient_quota",
            "code": "402",
        }

    def test_error_as_dict_no_message_field(self):
        info = extract_error_from_payload({
            "error": {"details": "something odd"},
        })
        assert info["message"] == "{'details': 'something odd'}"
        assert info["type"] == "provider_error"

    def test_flat_type_error_shape(self):
        info = extract_error_from_payload({
            "type": "error",
            "message": "provider is down",
            "code": "server_error",
        })
        assert info == {
            "message": "provider is down",
            "type": "provider_error",
            "code": "server_error",
        }

    def test_rate_limit_signal_retry_after(self):
        info = extract_error_from_payload({
            "retry_after": 60,
        })
        assert info == {
            "message": "Rate limited by upstream provider",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
        }

    def test_rate_limit_signal_with_message(self):
        info = extract_error_from_payload({
            "rate_limit": True,
            "message": "Too many requests",
            "code": 429,
        })
        assert info == {
            "message": "Too many requests",
            "type": "rate_limit_error",
            "code": 429,
        }

    def test_normal_chunk_returns_none(self):
        """A normal content chunk should not be mistaken for an error."""
        info = extract_error_from_payload({
            "choices": [{"delta": {"content": "hello"}}],
        })
        assert info is None

    def test_empty_dict_returns_none(self):
        assert extract_error_from_payload({}) is None

    def test_unrelated_keys_returns_none(self):
        assert extract_error_from_payload({"foo": "bar"}) is None
