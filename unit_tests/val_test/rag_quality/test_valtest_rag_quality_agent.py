import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from llm_val.sampler.Sampler import Sampler
from llm_val.val_test.rag_quality.valtest_rag_quality_agent import (
    valtest_rag_quality_agent,
    report_valtest_rag_quality_agent,
    calculate_metrics_for_query
)


def setup_test_data(with_scenario=True):
    """Подготавливает тестовый Sampler с тремя запросами."""
    sampler = MagicMock(spec=Sampler)
    data_X = pd.DataFrame({
        'context': [
            ["passage1_q1", "passage2_q1", "passage3_q1"],
            ["passage1_q2", "passage2_q2", "passage3_q2"],
            ["passage1_q3", "passage2_q3", "passage3_q3"]
        ]
    })
    if with_scenario:
        data_X['scenario'] = ['finance', 'sport', 'finance']
    data_y = pd.DataFrame({
        'passage_accuracy_metric': [
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 1]
        ]
    })
    sampler.test = {'X': data_X, 'y': data_y}
    return sampler


def test_calculate_metrics_for_query():
    """Проверяет, что для одного запроса вычисляются все запрошенные метрики."""
    passage_accuracy = [1, 0, 0]
    topk = 3
    metric_list = [
        'nDCG@k', 'MAP@k', 'MP@k',
        'AUC@k', 'ERR@k', 'MRR@k', 'HitRate@k', 'Recall@k'
    ]
    metrics = calculate_metrics_for_query(passage_accuracy, topk, metric_list)
    for metric in metric_list:
        assert metric in metrics
        assert isinstance(metrics[metric], float)


def test_report_without_groups():
    """Проверяет отчёт без группировки (нет тематик)."""
    avg_metrics = {
        'nDCG@k': 0.8,
        'MAP@k': 0.7,
        'MP@k': 0.6,
        'AUC@k': 0.9,
        'ERR@k': 0.5,
        'MRR@k': 0.4,
        'HitRate@k': 0.3,
        'Recall@k': 0.2
    }
    metric_list = list(avg_metrics.keys())
    thresholds = {'HitRate@k': (0.2, 0.5)}         
    default_thresholds = (0.0, 1.0)         
    is_info = False

    report = report_valtest_rag_quality_agent(
        avg_metrics,
        metric_list,
        thresholds,
        default_thresholds,
        groups=None,
        is_info=is_info
    )

    assert 'semaphore' in report
    assert 'result_plots' in report
    assert 'result_dataframes' in report
    res_table = report['result_dataframes'][0]
    assert list(res_table.columns) == ['Название метрики', 'Значение метрики', 'Результат теста']

    for metric in metric_list:
        assert metric in res_table['Название метрики'].values
        row = res_table[res_table['Название метрики'] == metric].iloc[0]
        assert row['Значение метрики'] == avg_metrics[metric]

    hit_row = res_table[res_table['Название метрики'] == 'HitRate@k'].iloc[0]
    assert hit_row['Результат теста'] == 'yellow'
    for metric in metric_list:
        if metric != 'HitRate@k':
            row = res_table[res_table['Название метрики'] == metric].iloc[0]
            assert row['Результат теста'] == 'yellow'
    assert report['semaphore'] == 'yellow'


def test_report_with_groups():
    """Проверяет отчёт с группировкой по тематикам."""
    avg_metrics = {
        'nDCG@k_finance': 0.8,
        'nDCG@k_sport': 0.6,
        'MAP@k_finance': 0.7,
        'MAP@k_sport': 0.5,
        'HitRate@k_finance': 0.9,
        'HitRate@k_sport': 0.4
    }
    metric_list = ['nDCG@k', 'MAP@k', 'HitRate@k']
    groups = ['finance', 'sport']
    thresholds = {
        'nDCG@k': (0.5, 0.8),
        'MAP@k': (0.5, 0.8),
        'HitRate@k': (0.5, 0.8)
    }
    default_thresholds = (0.0, 1.0)
    is_info = False

    report = report_valtest_rag_quality_agent(
        avg_metrics,
        metric_list,
        thresholds,
        default_thresholds,
        groups=groups,
        is_info=is_info
    )

    res_table = report['result_dataframes'][0]
    assert list(res_table.columns) == ['Название метрики', 'Название тематики', 'Значение метрики', 'Результат теста']
    assert len(res_table) == 6
    assert report['semaphore'] == 'red'

    row = res_table[(res_table['Название метрики'] == 'HitRate@k') & (res_table['Название тематики'] == 'sport')].iloc[0]
    assert row['Значение метрики'] == 0.4
    assert row['Результат теста'] == 'red'


def test_valtest_rag_quality_agent_without_scenario():
    """Проверяет основной тест, когда в данных нет колонки scenario."""
    sampler = setup_test_data(with_scenario=False)
    topk = 3
    metric_list = ['nDCG@k', 'MAP@k', 'HitRate@k']
    thresholds = {'HitRate@k': (0.2, 0.5)}
    default_thresholds = (0.0, 1.0)

    result = valtest_rag_quality_agent(
        sampler,
        data_type='test',
        topk=topk,
        metric_list=metric_list,
        thresholds=thresholds,
        default_thresholds=default_thresholds,
        is_info=False
    )

    assert 'report' in result
    assert 'precomputed' in result
    assert 'avg_metrics' in result['precomputed']

    report = result['report']
    assert 'semaphore' in report
    assert 'result_dataframes' in report
    res_table = report['result_dataframes'][0]

    avg_metrics = result['precomputed']['avg_metrics']
    for metric in metric_list:
        assert metric in avg_metrics
    assert report['semaphore'] == 'yellow'
    hit_row = res_table[res_table['Название метрики'] == 'HitRate@k'].iloc[0]
    assert hit_row['Результат теста'] == 'green'


def test_valtest_rag_quality_agent_with_scenario():
    """Проверяет основной тест с группировкой по тематикам."""
    sampler = setup_test_data(with_scenario=True)
    topk = 3
    metric_list = ['nDCG@k', 'MAP@k', 'HitRate@k']
    thresholds = {
        'nDCG@k': (0.5, 0.8),
        'MAP@k': (0.5, 0.8),
        'HitRate@k': (0.5, 0.8)
    }
    default_thresholds = (0.0, 1.0)

    result = valtest_rag_quality_agent(
        sampler,
        data_type='test',
        topk=topk,
        metric_list=metric_list,
        thresholds=thresholds,
        default_thresholds=default_thresholds,
        is_info=False
    )

    report = result['report']
    res_table = report['result_dataframes'][0]

    avg_metrics = result['precomputed']['avg_metrics']
    groups = ['finance', 'sport']
    for metric in metric_list:
        for group in groups:
            key = f"{metric}_{group}"
            assert key in avg_metrics
    assert len(res_table) == len(metric_list) * len(groups)
    assert report['semaphore'] in ('green', 'yellow', 'red')