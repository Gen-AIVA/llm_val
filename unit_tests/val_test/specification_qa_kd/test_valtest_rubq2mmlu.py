import unittest
import pandas as pd
from unittest.mock import MagicMock, patch

from llm_val.val_test.specification_qa_kd.valtest_rubq2mmlu import valtest_rubq2mmlu
def setup():
    sampler = MagicMock()
    scorer = MagicMock()
    model = MagicMock()
    model_incorrect_gen = MagicMock()
    data_rubq = pd.DataFrame({'question': ['Question 1', 'Question 2'],
                              'context': ['Context 1', 'Context 2'],
                              'true_answer': [['Answer 1'], ['Answer 2']]})
    sampler.test = {'X': data_rubq}
    model.predict.return_value = ['A', 'B']
    model_incorrect_gen.predict.return_value = "Incorrect Option 1\nIncorrect Option 2\nIncorrect Option 3"
    scorer.calc.return_value = {'single_mean_metric_value': 0.5}
    return sampler, scorer, model, model_incorrect_gen


@patch('llm_val.val_test.specification_qa_kd.valtest_rubq2mmlu')
@patch('llm_val.val_test.specification_qa_kd.valtest_rubq2mmlu.report_valtest_metric_rubq2mmlu')
def test_valtest_rubq2mmlu(mock_report_valtest_metric_rubq2mmlu, 
                           mock_convert_rubq_to_mmlu):
    mock_convert_rubq_to_mmlu.return_value = (
        pd.DataFrame({'question': ['Question 1', 'Question 2'],
                      'model_answer': ['A', 'B'],
                      'true_answer': ['A', 'B']
        }),
        {'failure_rate': 0.0, 
         'failed_generation_count': 0, 
         'total_rows': 2}
    )
    mock_report_valtest_metric_rubq2mmlu.return_value = {
        'semaphore': 'green',
        'result_plots': [],
        'result_dataframes': [pd.DataFrame({'Наименование ключевой метрики': ['single_mean'],
                                            'Прирост ключевой метрики (КМ) качества': ['10.00%'],
                                            'Результат теста': ['green']
        })]
    }
    sampler, scorer, model, model_incorrect_gen = setup()

    result = valtest_rubq2mmlu(sampler, 
                               scorer, 
                               model, 
                               model_incorrect_gen, 
                               generate_prompt="Test Prompt")
    assert 'report' in result
    assert 'precomputed' in result
    report = result['report']
    assert 'semaphore' in report
    assert 'result_plots' in report
    assert 'result_dataframes' in report
    assert report['semaphore'] in ['gray', 'green', 'red']
    assert len(report['result_plots']) == 0
    assert len(report['result_dataframes']) > 0
