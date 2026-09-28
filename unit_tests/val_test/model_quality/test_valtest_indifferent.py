import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from llm_val.val_test.model_quality.valtest_indifferent import valtest_indifferent

def setup():
    sampler = MagicMock()
    scorer = MagicMock()
    model = MagicMock()
    data = pd.DataFrame({
        'question': ['Question 1', 'Question 2'],
        'true_answer': ['Answer 1', 'Answer 2']
    })
    sampler.test = {'X': data, 
                    'y': pd.Series(['Answer 1', 'Answer 2'])}
    model.predict.return_value = ['Answer 1', 'Answer 2']
    scorer.calc.return_value = {'single_mean_metric_value': 0.5}
    return sampler, scorer, model


@patch('llm_val.val_test.model_quality.valtest_indifferent.generate_random_texts')
@patch('llm_val.val_test.model_quality.valtest_indifferent.calculate_confidence_interval')
def test_valtest_indifferent(mock_calculate_confidence_interval, 
                             mock_generate_random_texts):
    mock_generate_random_texts.return_value = ['Random Text 1', 'Random Text 2']
    mock_calculate_confidence_interval.return_value = (0.4, 0.6)
    sampler, scorer, model = setup()
    
    result = valtest_indifferent(sampler, 
                                 scorer, 
                                 model)
    
    assert 'report' in result
    assert 'precomputed' in result
    report = result['report']
    assert 'semaphore' in report
    assert 'result_plots' in report
    assert 'result_dataframes' in report
    assert report['semaphore'] in ['gray', 'green', 'red']
    assert len(report['result_plots']) > 0
    assert len(report['result_dataframes']) > 0
