import pytest
import numpy as np

from llm_val.utils.report_helper import (semaphore_by_threshold,
                                         worst_semaphore,
                                         semaphore_by_threshold_without_yellow, 
                                         semaphore_by_threshold_without_red, 
                                         tricky_semaphore,
                                         quantile_semaphore)

SEMAPHORE_BY_THRESHOLD_PARAMS = [(0.1, (0.2, 0.4), True, 'red'),
                                 (0.1, (0.2, 0.4), False, 'green'),
                                 (0.3, (0.2, 0.4), True, 'yellow'),
                                 (0.6, (0.2, 0.4), False, 'red')]

SEMAPHORE_BY_THRESHOLD_WITHOUT_YELLOW_PARAMS = [(0.1, 0.2, True, 'red'),
                                                (0.3, 0.2, True, 'green'),
                                                (0.1, 0.2, False, 'green'),
                                                (0.3, 0.2, False, 'red')]

SEMAPHORE_BY_THRESHOLD_WITHOUT_RED_PARAMS = [(0.1, 0.2, True, 'yellow'),
                                             (0.3, 0.2, True, 'green'),
                                             (0.1, 0.2, False, 'green'),
                                             (0.3, 0.2, False, 'yellow')]

WORST_SEMAPHORE_PARAMS = [(['green', 'yellow', 'red'], 'red'),
                          (['gray', 'gray', 'gray'], 'gray'),
                          (['gray', 'yellow', 'gray'], 'yellow'),
                          (['green', 'green', 'green'], 'green')]

TRICKY_SEMAPHORE_PARAMS = [(0.86, ['green', 'yellow', 'green', 'green', 'green', 'green'], True, (0.4, 0.6), 0.2, 0.2, 'green'),
                           (0.86, [], True, (0.4, 0.6), 0.2, 0.2, 'green'),
                           (0.56, [], True, (0.4, 0.6), 0.2, 0.2, 'yellow'),
                           (0.16, [], True, (0.4, 0.6), 0.2, 0.2, 'red'),
                           (0.12, ['green', 'yellow', 'green', 'green', 'green', 'green'], True, (0.4, 0.6), 0.2, 0.2, 'red'),
                           (0.12, ['green', 'yellow', 'green', 'green', 'green', 'green'], False, (0.4, 0.6), 0.2, 0.2, 'green'),
                           (0.7, ['yellow', 'yellow', 'yellow', 'green'], True, (0.4, 0.6), 0.2, 0.2, 'yellow'),
                           (0.7, ['red', 'yellow', 'yellow', 'green'], True, (0.4, 0.6), 0.2, 0.2, 'red')]

QUANTILE_SEMAPHORE_PARAMS = [(0.5, (0.95, 0.99), True, 'green'),
                             (0.05, (0.95, 0.99), True, 'green'),
                             (0.02, (0.95, 0.99), True, 'yellow'),
                             (0.001, (0.95, 0.99), True, 'red'),
                             (0.001, (0.95, 0.99), False, 'green')]

SEMAPHORE_BY_THRESHOLD_WITHOUT_YELLOW_BORDERS_PARAMS = [(0.2, 0.2, True, 'green', True),
                                                        (0.2, 0.2, False, 'red', True),
                                                        (0.2, 0.2, True, 'red', False),
                                                        (0.2, 0.2, False, 'green', False)]

SEMAPHORE_BY_THRESHOLD_BORDERS_PARAMS = [(0.2, (0.2, 0.4), True, 'yellow', True, True),
                                         (0.2, (0.2, 0.4), True, 'red', False, False),
                                         (0.2, (0.2, 0.4), False, 'yellow', True, True),
                                         (0.2, (0.2, 0.4), False, 'green', False, False),
                                         (0.4, (0.2, 0.4), True, 'green', True, True),
                                         (0.4, (0.2, 0.4), True, 'yellow', False, False),
                                         (0.4, (0.2, 0.4), True, 'green', False, True),
                                         (0.4, (0.2, 0.4), False, 'yellow', True, False),
                                         (0.3, (0.2, 0.4), False, 'yellow', True, True),
                                         (0.3, (0.2, 0.4), True, 'yellow', False, True),
                                         (0.3, (0.2, 0.4), True, 'yellow', False, False)]


@pytest.mark.parametrize('report_params', SEMAPHORE_BY_THRESHOLD_PARAMS)
def test_semaphore_by_threshold(report_params):
    value, threshold, greater_is_better, color = report_params
    assert semaphore_by_threshold(value, threshold, greater_is_better) == color


@pytest.mark.parametrize('report_params', SEMAPHORE_BY_THRESHOLD_WITHOUT_YELLOW_PARAMS)
def test_semaphore_by_threshold_without_yellow(report_params):
    value, threshold, greater_is_better, color = report_params
    assert semaphore_by_threshold_without_yellow(value, threshold, greater_is_better) == color
    
    
@pytest.mark.parametrize('report_params', SEMAPHORE_BY_THRESHOLD_WITHOUT_RED_PARAMS)
def test_semaphore_by_threshold_without_red(report_params):
    value, threshold, greater_is_better, color = report_params
    assert semaphore_by_threshold_without_red(value, threshold, greater_is_better) == color


@pytest.mark.parametrize('report_params', WORST_SEMAPHORE_PARAMS)
def test_worst_semaphore(report_params):
    value_list, color = report_params
    assert worst_semaphore(value_list) == color


@pytest.mark.parametrize('report_params', TRICKY_SEMAPHORE_PARAMS)
def test_tricky_semaphore(report_params):
    (value,
     semaphore_list,
     greater_is_better,
     threshold_metric,
     red_freq_threshold,
     yellow_freq_threshold, color) = report_params
    assert tricky_semaphore(value,
                            semaphore_list,
                            greater_is_better,
                            threshold_metric,
                            red_freq_threshold,
                            yellow_freq_threshold) == color

    
@pytest.mark.parametrize('report_params', QUANTILE_SEMAPHORE_PARAMS)
def test_quantile_semaphore(report_params):
    value, threshold, inner_is_better, color = report_params
    assert quantile_semaphore(value, threshold, inner_is_better) == color


@pytest.mark.parametrize('report_params', SEMAPHORE_BY_THRESHOLD_WITHOUT_YELLOW_BORDERS_PARAMS)
def test_semaphore_by_threshold_without_yellow_border(report_params):
    value, threshold, greater_is_better, color, border_is_bigger = report_params
    assert semaphore_by_threshold_without_yellow(value, threshold, greater_is_better, border_is_bigger) == color


@pytest.mark.parametrize('report_params', SEMAPHORE_BY_THRESHOLD_BORDERS_PARAMS)
def test_semaphore_by_threshold_borders(report_params):
    value, threshold, greater_is_better, color, left_border_is_bigger, right_border_is_bigger = report_params
    assert semaphore_by_threshold(value, threshold, greater_is_better, left_border_is_bigger, right_border_is_bigger) == color
