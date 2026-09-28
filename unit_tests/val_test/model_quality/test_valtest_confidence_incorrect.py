import pytest
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from unittest.mock import MagicMock

from llm_val.val_test.model_quality.valtest_confidence_in_incorrect_answers import plot_confidence_quality
from llm_val.val_test.model_quality.valtest_confidence_in_incorrect_answers import valtest_confidence_in_incorrect_answers
from llm_val.val_test.model_quality.valtest_confidence_in_incorrect_answers import report_valtest_confidence_in_incorrect_answers

@pytest.fixture
def test_data():
    return pd.DataFrame({
        'system': [None, None, None, None, None],
        'question': ['Какие объекты недвижимости можно приобрести',
                     'Какой минимальный срок кредита доступен при оф',
                     'Чем апартаменты отличаются от квартир?',
                     'Какой минимальный первоначальный взнос необход',
                     'Какие требования к заемщикам предусмотрены для',
        ],
        'context': [['Что такое ипотека и ипотечный кредит.'],
                    ['Кредитные карты'],
                    ['Приобретение готового жилья'],
                    ['Приобретение строящегося жилья'],
                    ['Коммерческая недвижимость'],
        ],
        'thema': [None, 
                  None, 
                  None, 
                  None, 
                  None],
        'answer_options': [{'A': 'Жилое помещение', 'B': 'Коммерческое', 'C': 'Земля', 'D': 'Все вышеперечисленное'},
                           {'A': '6 месяцев', 'B': '12 месяцев', 'C': '24 месяца', 'D': '36 месяцев'},
                           {'A': 'Возможность оформления постоянной регистрации', 'B': 'Наличие коммунальных услуг', 'C': 'Отсутствие постоянной регистрации', 'D': 'Все вышеперечисленное'},
                           {'A': '10%', 'B': '20%', 'C': '30%', 'D': '40%'},
                           {'A': 'Возраст заемщика от 21 до 75 лет.', 'B': 'Наличие гражданства РФ.', 'C': 'Наличие постоянного дохода.', 'D': 'Все вышеперечисленное'},
        ],
        'true_answer': ['D', 
                        'A', 
                        'C', 
                        'C', 
                        'A'],
        'raw_prompt': ['\nКонтекст:\nПример1 Контекста',
                       '\nКонтекст:\nКредитные карты.',
                       '\nКонтекст:\nПриобретение жилья',
                       '\nКонтекст:\nПриобретение строящегося жилья',
                       '\nКонтекст:\nКоммерческая недвижимость',
        ],
        'model_answer': ['A', 
                         'D', 
                         'C', 
                         'C', 
                         'A'],
    })


@pytest.fixture
def mock_sampler(test_data):
    sampler = MagicMock()
    sampler.test = {'X': test_data}
    return sampler


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict = MagicMock(return_value='A')
    return model


@pytest.fixture
def mock_get_embed_func():
    return MagicMock(return_value=np.array([[0.1, 0.2]]))


def test_valtest_confidence_in_incorrect_answers(mock_sampler, 
                                                 mock_model, 
                                                 mock_get_embed_func):
    
    result = valtest_confidence_in_incorrect_answers(sampler=mock_sampler,
                                                     model=mock_model,
                                                     get_embed_func=mock_get_embed_func,
                                                     data_type='test')

    assert 'report' in result
    assert 'precomputed' in result

    report = result['report']
    assert 'semaphore' in report
    assert 'result_plots' in report
    assert 'result_dataframes' in report

    precomputed = result['precomputed']
    assert 'entropies' in precomputed
    assert 'correct_labels' in precomputed

    assert isinstance(precomputed['entropies'], np.ndarray)
    assert isinstance(precomputed['correct_labels'], np.ndarray)
    assert len(precomputed['entropies']) == len(precomputed['correct_labels'])


def test_report_valtest_confidence_in_incorrect_answers():
    entropies = np.array([0.1, 0.2, 0.3, 0.4])
    correct_labels = np.array([1, 0, 1, 0])

    report = report_valtest_confidence_in_incorrect_answers(entropies, 
                                                            correct_labels)
    assert 'semaphore' in report
    assert 'result_plots' in report
    assert 'result_dataframes' in report
    assert report['semaphore'] == 'gray'
    assert isinstance(report['result_plots'][0], Figure)


def test_plot_confidence_quality():
    entropies = np.array([0.1, 0.2, 0.3, 0.4])
    correct_labels = np.array([1, 0, 1, 0])
    fig = plot_confidence_quality(entropies, 
                                  correct_labels)
    assert isinstance(fig, Figure)
