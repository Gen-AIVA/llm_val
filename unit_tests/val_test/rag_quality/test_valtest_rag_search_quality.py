import numpy as np
import pandas as pd
from unittest.mock import Mock
import matplotlib.pyplot as plt

from llm_val.val_test.rag_quality.valtest_rag_search_quality import valtest_rag_search_quality

def test_valtest_rag_search_quality():
    sampler = Mock()
    test_data = {
        'criterion_passage_accuracy': [
            [1, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 0]
        ],
        'criterion_accuracy': [1, 0, 1, 1]
    }
    sampler.test = {'y': pd.DataFrame(test_data)}
    
    topk = 3
    result = valtest_rag_search_quality(sampler, 
                                        topk, 
                                        data_type='test')
    
    assert 'report' in result, "Результат должен содержать ключ 'report'"
    assert 'precomputed' in result, "Результат должен содержать ключ 'precomputed'"
    report = result['report']
    assert len(report['result_dataframes']) == 1, "Отчет должен содержать один DataFrame"
    correlation_df = report['result_dataframes'][0]
    expected_rows = topk
    actual_rows = len(correlation_df)
    assert actual_rows == expected_rows, f"Количество строк в DataFrame должно быть {expected_rows}, но найдено {actual_rows}"
    
    expected_columns = ['Наименование метрики', 'Корреляция']
    assert all(col in correlation_df.columns for col in expected_columns), \
        f"DataFrame должен содержать столбцы: {expected_columns}"
    
    correlation_values = correlation_df['Корреляция'].values
    assert all(-1 <= val <= 1 for val in correlation_values), \
        "Значения корреляции должны находиться в диапазоне [-1, 1]"
    
    assert len(report['result_plots']) == 1, "Отчет должен содержать один график"
    plot = report['result_plots'][0]
    assert isinstance(plot, plt.Figure), "График должен быть объектом matplotlib.figure.Figure"