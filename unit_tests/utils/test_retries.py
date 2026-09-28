import time
import pytest
from unittest.mock import MagicMock

import llm_val.utils.retries as retries

from llm_val.utils.retries import call_with_retries


def test_call_with_retries_success():
    func = MagicMock(return_value=['answer'])
    result = call_with_retries(func, timeout=0.1, retry_delay=0.01, max_attempts=3, func_args=(['q'],))
    assert result == ['answer']
    func.assert_called_once_with(['q'])


def test_call_with_retries_passes_kwargs():
    func = MagicMock(return_value=['answer'])
    result = call_with_retries(func, timeout=0.1, retry_delay=0.01, max_attempts=3,
                               func_args=(['q'],), func_kwargs={'model': 'name'})
    assert result == ['answer']
    func.assert_called_once_with(['q'], model='name')


def test_call_with_retries_success_after_error(monkeypatch):
    sleep_values = []
    monkeypatch.setattr(retries.time, 'sleep', lambda value: sleep_values.append(value))
    func = MagicMock(side_effect=[RuntimeError('fail'), ['answer']])
    result = call_with_retries(func, timeout=0.5, retry_delay=0.01, max_attempts=2, func_args=(['q'],))
    assert result == ['answer']
    assert func.call_count == 2
    assert sleep_values == [0.01]


def test_call_with_retries_raises_after_max_attempts(monkeypatch):
    sleep_values = []
    monkeypatch.setattr(retries.time, 'sleep', lambda value: sleep_values.append(value))
    func = MagicMock(side_effect=RuntimeError('fail'))
    with pytest.raises(RuntimeError):
        call_with_retries(func, timeout=0.5, retry_delay=0.01, max_attempts=2, func_args=(['q'],))
    assert func.call_count == 2
    assert sleep_values == [0.01]


def test_call_with_retries_max_attempts_less_than_one(monkeypatch):
    sleep_values = []
    monkeypatch.setattr(retries.time, 'sleep', lambda value: sleep_values.append(value))
    func = MagicMock(side_effect=RuntimeError('fail'))
    with pytest.raises(RuntimeError):
        call_with_retries(func, timeout=0.5, retry_delay=0.01, max_attempts=0, func_args=(['q'],))
    assert func.call_count == 1
    assert sleep_values == []


def test_call_with_retries_timeout():
    def slow_call(messages):
        time.sleep(0.05)
        return messages

    func = MagicMock(side_effect=slow_call)
    with pytest.raises(TimeoutError):
        call_with_retries(func, timeout=0.001, retry_delay=0.01, max_attempts=1, func_args=(['q'],))
    assert func.call_count == 1
