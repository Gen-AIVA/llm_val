import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from llm_val.sampler.Sampler import Sampler
from llm_val.val_test.rag_quality.valtest_rag_quality import valtest_rag_quality, report_valtest_rag_quality, calculate_metrics_for_query

def setup_test_data():
    sampler = MagicMock(spec=Sampler)
    data_X = pd.DataFrame({'context': [["passage1_q1", "passage2_q1", "passage3_q1"],
                                       ["passage1_q2", "passage2_q2", "passage3_q2"],
                                       ["passage1_q3", "passage2_q3", "passage3_q3"]]
    })

    data_y = pd.DataFrame({'criterion_passage_accuracy': [[1, 0, 0],
                                                          [0, 1, 0],
                                                          [1, 1, 1]]
    })

    sampler.test = {'X': data_X, 'y': data_y}
    return sampler


def test_report_valtest_rag_quality():
    sampler = setup_test_data()
    topk = 3
    metric_list = ['nDCG@k',
                   'MAP@k',
                   'MP@k',
                   'Recall@k',
                   'AUC@k',
                   'ERR@k',
                   'MRR@k',
                   'HitRate@k']
    key_metric = 'HitRate@k'
    confidence = 0.95
    max_value = 1.0
    thresholds = (0.2, 0.4)

    result = valtest_rag_quality(sampler,
                                 'test',
                                 topk,
                                 metric_list,
                                 key_metric,
                                 confidence,
                                 max_value,
                                 thresholds)

    avg_metrics = result['precomputed']['baseline_score']
    confidence_intervals = {metric: f'{avg_metrics[metric]} - {avg_metrics[metric]}'
                            for metric in metric_list}
    
    report = report_valtest_rag_quality(avg_metrics,
                                        confidence_intervals,
                                        metric_list,
                                        key_metric,
                                        max_value,
                                        thresholds)

    assert 'semaphore' in report
    assert 'result_plots' in report
    assert 'result_dataframes' in report

    res_table = report['result_dataframes'][0]
    assert 'Название метрики' in res_table.columns
    assert 'Значение метрики' in res_table.columns
    assert 'Доверительный интервал' in res_table.columns
    assert 'Результат теста' in res_table.columns

    assert 'nDCG@k' in res_table['Название метрики'].values
    assert 'MAP@k' in res_table['Название метрики'].values


def test_calculate_metrics_for_query():
    passage_accuracy = [1, 0, 0]
    topk = 3
    metric_list = ['nDCG@k',
                   'MAP@k',
                   'MP@k',
                   'Recall@k',
                   'AUC@k',
                   'ERR@k',
                   'MRR@k',
                   'HitRate@k']

    metrics = calculate_metrics_for_query(passage_accuracy, 
                                          topk, 
                                          metric_list)

    assert 'nDCG@k' in metrics
    assert 'MAP@k' in metrics
    assert 'MP@k' in metrics
    assert 'Recall@k' in metrics
    assert 'AUC@k' in metrics
    assert 'ERR@k' in metrics
    assert 'MRR@k' in metrics
    assert 'HitRate@k' in metrics
