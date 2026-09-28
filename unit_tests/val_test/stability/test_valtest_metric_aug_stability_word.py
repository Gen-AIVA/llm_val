import typing as tp
import pytest
import numpy as np
import pandas as pd
from types import SimpleNamespace
from unittest.mock import MagicMock

from llm_val.scorer.UniversalScorer import UniversalScorer
from llm_val.utils.metrics import METRICS
from llm_val.val_test.stability.valtest_metric_aug_stability_word import (
    valtest_metric_aug_stability_word,
    report_valtest_metric_aug_stability_word,
    calculate_stat_significance,
    find_metric_value,
    get_sampler_data,
    get_augmented_data,
    prepare_and_evaluate,
    resolve_answer_checker,
    default_answer_checker,
)


class MockModel:
    
    def __init__(self, answers_map: tp.Optional[tp.Dict[str, str]] = None):
        self.answers_map = answers_map or {}
        self.predict_calls = []

    def predict(self, prompts: tp.List[str]) -> tp.List[str]:
        self.predict_calls.append(prompts)
        return [self.answers_map.get(p, p) for p in prompts]


class MockChatCompletions:
    
    def __init__(self, responses_map: tp.Optional[tp.Dict[str, str]] = None):
        self.responses_map = responses_map or {}
        self.call_count = 0
        self.last_model_name = None

    def create(self, model: str, messages: tp.List[tp.Dict[str, str]], **kwargs) -> tp.Any:
        self.call_count += 1
        self.last_model_name = model
        user_content = messages[-1]['content'] if messages else ''
        response_text = user_content
        for key, value in self.responses_map.items():
            if key in user_content:
                response_text = value
                break
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response_text))]
        )


class MockChat:
    
    def __init__(self, responses_map: tp.Optional[tp.Dict[str, str]] = None):
        self.completions = MockChatCompletions(responses_map)


class MockOpenAIClient:
    
    def __init__(self, responses_map: tp.Optional[tp.Dict[str, str]] = None):
        self.chat = MockChat(responses_map)


@pytest.fixture
def data_columns() -> pd.DataFrame:
    
    return pd.DataFrame({
        'input_query': ['Какая столица Франции?', 'Кто написал Войну и мир?', 'Сколько будет 2+2?'],
        'output_answer': ['Париж', 'Лев Толстой', 'Четыре'],
        'reference_answer': ['Париж', 'Лев Толстой', 'Четыре'],
    })


@pytest.fixture
def mock_sampler_columns(data_columns: pd.DataFrame) -> MagicMock:
    
    sampler = MagicMock()
    sampler.test = {'X': data_columns}
    return sampler


@pytest.fixture
def mock_model() -> MockModel:
    '''Мок валидируемой модели'''
    return MockModel({
        'Какая столица Франции?': 'Париж',
        'Кто написал Войну и мир?': 'Лев Толстой',
        'Сколько будет 2+2?': 'Четыре',
    })


@pytest.fixture
def mock_aug_client() -> MockOpenAIClient:
    '''Мок OpenAI-клиента для аугментаций'''
    return MockOpenAIClient(responses_map={
        'столица': 'Какой главный город Франции?',
        'Толстой': 'Автор романа Война и мир',
        '2+2': 'Сколько получится два плюс два?',
    })


@pytest.fixture
def scorer() -> UniversalScorer:
    return UniversalScorer(metrics=METRICS)


@pytest.fixture
def mock_scorer() -> MagicMock:
    scorer = MagicMock()
    scorer.calc.return_value = {'single_mean_value_metric': 0.8}
    return scorer


def test_valtest_full_pipeline_with_all_columns(scorer: UniversalScorer):
    '''
    Проверка полного пути расчёта, когда в сэмплере уже есть все колонки:
    input_query, output_answer, reference_answer, aug_*, pred_aug_*.
    Ни model.predict, ни aug_client не должны вызываться.
    '''
    data = pd.DataFrame({
        'input_query': ['Какая столица Франции?', 'Кто написал Войну и мир?', 'Сколько будет 2+2?'],
        'output_answer': ['Париж', 'Лев Толстой', 'Четыре'],
        'reference_answer': ['Париж', 'Лев Толстой', 'Четыре'],
        'aug_valtest_metric_aug_stability_word_paraphrase': [
            'Какой главный город Франции?',
            'Кто автор романа Война и мир?',
            'Сколько получится два плюс два?',
        ],
        'pred_aug_valtest_metric_aug_stability_word_paraphrase': [
            'Париж',
            'Лев Толстой',
            'Четыре',
        ],
    })
    sampler = MagicMock()
    sampler.test = {'X': data}

    model = MagicMock()
    aug_client = MagicMock()

    result = valtest_metric_aug_stability_word(
        sampler=sampler,
        scorer=scorer,
        main_metric='single_mean_value_metric',
        augmentations_list=['paraphrase'],
        exec_env='default',
        is_info=True,
    )

    model.predict.assert_not_called()
    aug_client.chat.completions.create.assert_not_called()

    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 'precomputed'}

    precomputed = result['precomputed']
    assert 'paraphrase' in precomputed['baseline_per_aug']
    assert precomputed['baseline_per_aug']['paraphrase']['single_mean_value_metric'] == pytest.approx(1.0)
    assert 'paraphrase' in precomputed['metrics_augmented']['single_mean_value_metric']
    assert precomputed['metrics_augmented']['single_mean_value_metric']['paraphrase'] == pytest.approx(1.0)
    assert 'paraphrase' in precomputed['p_values']


def test_valtest_no_model_no_aug_client_with_columns(scorer: UniversalScorer):
    '''
    Проверка работы без model и aug_client, если все колонки уже есть в сэмплере.
    '''
    data = pd.DataFrame({
        'input_query': ['q1', 'q2', 'q3', 'q4'],
        'output_answer': ['a1', 'a2', 'a3', 'a4'],
        'reference_answer': ['a1', 'a2', 'wrong', 'a4'],
        'aug_valtest_metric_aug_stability_word_paraphrase': ['aq1', 'aq2', 'aq3', 'aq4'],
        'pred_aug_valtest_metric_aug_stability_word_paraphrase': ['a1', 'a2', 'wrong', 'a4'],  
    })
    sampler = MagicMock()
    sampler.test = {'X': data}

    result = valtest_metric_aug_stability_word(
        sampler=sampler,
        scorer=scorer,
        main_metric='single_mean_value_metric',
        augmentations_list=['paraphrase'],
        exec_env='default',
    )

    assert isinstance(result, dict)
    assert result['precomputed']['baseline_per_aug']['paraphrase']['single_mean_value_metric'] == pytest.approx(0.75)
    assert result['precomputed']['metrics_augmented']['single_mean_value_metric']['paraphrase'] == pytest.approx(1.0)
    assert result['precomputed']['p_values']['paraphrase'] is not None

def test_valtest_no_output_answer_baseline_nan(scorer: UniversalScorer):
    '''
    Проверка: если в сэмплере нет output_answer, baseline должен быть NaN,
    но аугментированная метрика всё равно считается.
    '''
    data = pd.DataFrame({
        'input_query': ['q1', 'q2'],
        'reference_answer': ['a1', 'a2'],
        'aug_valtest_metric_aug_stability_word_paraphrase': ['aq1', 'aq2'],
        'pred_aug_valtest_metric_aug_stability_word_paraphrase': ['a1', 'a2'],
    })
    sampler = MagicMock()
    sampler.test = {'X': data}

    result = valtest_metric_aug_stability_word(
        sampler=sampler,
        scorer=scorer,
        main_metric='single_mean_value_metric',
        augmentations_list=['paraphrase'],
        exec_env='default',
    )

    assert np.isnan(result['precomputed']['baseline_per_aug']['paraphrase']['single_mean_value_metric'])
    assert result['precomputed']['metrics_augmented']['single_mean_value_metric']['paraphrase'] == pytest.approx(1.0)


def test_valtest_no_aug_column_skipped(scorer: UniversalScorer):
    '''
    Проверка: если aug-колонки нет в сэмплере и не передан aug_client,
    аугментация пропускается, метрики — NaN.
    '''
    data = pd.DataFrame({
        'input_query': ['q1', 'q2'],
        'output_answer': ['a1', 'a2'],
        'reference_answer': ['a1', 'a2'],
    })
    sampler = MagicMock()
    sampler.test = {'X': data}

    result = valtest_metric_aug_stability_word(
        sampler=sampler,
        scorer=scorer,
        main_metric='single_mean_value_metric',
        augmentations_list=['paraphrase'],
        exec_env='default',
    )

    assert np.isnan(result['precomputed']['baseline_per_aug']['paraphrase']['single_mean_value_metric'])
    assert np.isnan(result['precomputed']['metrics_augmented']['single_mean_value_metric']['paraphrase'])
    assert np.isnan(result['precomputed']['p_values']['paraphrase'])


def test_valtest_aug_column_unchanged_skipped(scorer: UniversalScorer):
    '''
    Проверка: если aug-колонка есть, но текст не изменился (равен input_query),
    аугментация пропускается, метрики — NaN.
    '''
    data = pd.DataFrame({
        'input_query': ['q1', 'q2'],
        'output_answer': ['a1', 'a2'],
        'reference_answer': ['a1', 'a2'],
        'aug_valtest_metric_aug_stability_word_paraphrase': ['q1', 'q2'],
    })
    sampler = MagicMock()
    sampler.test = {'X': data}

    result = valtest_metric_aug_stability_word(
        sampler=sampler,
        scorer=scorer,
        main_metric='single_mean_value_metric',
        augmentations_list=['paraphrase'],
        exec_env='default',
    )

    assert np.isnan(result['precomputed']['baseline_per_aug']['paraphrase']['single_mean_value_metric'])
    assert np.isnan(result['precomputed']['metrics_augmented']['single_mean_value_metric']['paraphrase'])



def test_valtest_multiple_augmentations_mixed(scorer: UniversalScorer):
    '''
    Проверка смешанного сценария: для одной аугментации есть все колонки,
    для другой — только aug-колонка (нужен model), для третьей — ничего.
    '''
    data = pd.DataFrame({
        'input_query': ['q1', 'q2'],
        'output_answer': ['a1', 'a2'],
        'reference_answer': ['a1', 'a2'],
        'aug_valtest_metric_aug_stability_word_paraphrase': ['aq1', 'aq2'],
        'pred_aug_valtest_metric_aug_stability_word_paraphrase': ['a1', 'a2'],
        'aug_valtest_metric_aug_stability_word_synonym': ['sq1', 'sq2'],
    })
    sampler = MagicMock()
    sampler.test = {'X': data}

    model = MockModel({'sq1': 'a1', 'sq2': 'a2'})

    result = valtest_metric_aug_stability_word(
        sampler=sampler,
        scorer=scorer,
        main_metric='single_mean_value_metric',
        augmentations_list=['paraphrase', 'synonym', 'translation'],
        model=model,
        exec_env='default',
    )

    assert result['precomputed']['baseline_per_aug']['paraphrase']['single_mean_value_metric'] == pytest.approx(1.0)
    assert result['precomputed']['metrics_augmented']['single_mean_value_metric']['paraphrase'] == pytest.approx(1.0)
    assert result['precomputed']['baseline_per_aug']['synonym']['single_mean_value_metric'] == pytest.approx(1.0)
    assert result['precomputed']['metrics_augmented']['single_mean_value_metric']['synonym'] == pytest.approx(1.0)
    assert np.isnan(result['precomputed']['baseline_per_aug']['translation']['single_mean_value_metric'])
    assert np.isnan(result['precomputed']['metrics_augmented']['single_mean_value_metric']['translation'])


def test_valtest_stat_significance_with_real_data(scorer: UniversalScorer):
    '''
    Проверка, что p-value и статистическая значимость корректно рассчитываются
    на реальных данных, когда baseline и augmented метрики различаются.
    '''
    data = pd.DataFrame({
        'input_query': [f'q{i}' for i in range(10)],
        'output_answer': ['correct'] * 10,
        'reference_answer': ['correct'] * 10,
        'aug_valtest_metric_aug_stability_word_paraphrase': [f'aq{i}' for i in range(10)],
        'pred_aug_valtest_metric_aug_stability_word_paraphrase': ['wrong'] * 10,
    })
    sampler = MagicMock()
    sampler.test = {'X': data}

    result = valtest_metric_aug_stability_word(
        sampler=sampler,
        scorer=scorer,
        main_metric='single_mean_value_metric',
        augmentations_list=['paraphrase'],
        exec_env='default',
    )

    assert result['precomputed']['baseline_per_aug']['paraphrase']['single_mean_value_metric'] == pytest.approx(1.0)
    assert result['precomputed']['metrics_augmented']['single_mean_value_metric']['paraphrase'] == pytest.approx(0.0)
    assert result['precomputed']['p_values']['paraphrase'] < 0.05


def test_valtest_report_structure_with_all_columns(scorer: UniversalScorer):
    '''
    Проверка структуры отчёта при полном пути расчёта с готовыми колонками.
    '''
    data = pd.DataFrame({
        'input_query': ['q1', 'q2'],
        'output_answer': ['a1', 'a2'],
        'reference_answer': ['a1', 'a2'],
        'aug_valtest_metric_aug_stability_word_paraphrase': ['aq1', 'aq2'],
        'pred_aug_valtest_metric_aug_stability_word_paraphrase': ['a1', 'a2'],
    })
    sampler = MagicMock()
    sampler.test = {'X': data}

    result = valtest_metric_aug_stability_word(
        sampler=sampler,
        scorer=scorer,
        main_metric='single_mean_value_metric',
        augmentations_list=['paraphrase'],
        exec_env='default',
    )

    report = result['report']
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert report['semaphore'] in ('green', 'yellow', 'red', 'gray')
    assert isinstance(report['result_dataframes'][0], pd.DataFrame)

    df = report['result_dataframes'][0]
    assert 'Тип преобразования' in df.columns
    assert 'p-value' in df.columns
    assert 'Доля успешно аугментированных объектов' in df.columns
    assert 'Результат теста' in df.columns
    assert len(df) == 1
    assert df['Тип преобразования'].iloc[0] == 'paraphrase'


def test_valtest_subset_fraction_with_all_columns(scorer: UniversalScorer):
    '''
    Проверка, что доля успешно аугментированных объектов корректно рассчитывается.
    '''
    data = pd.DataFrame({
        'input_query': ['q1', 'q2', 'q3', 'q4'],
        'output_answer': ['a1', 'a2', 'a3', 'a4'],
        'reference_answer': ['a1', 'a2', 'a3', 'a4'],
        'aug_valtest_metric_aug_stability_word_paraphrase': ['aq1', 'q2', 'aq3', 'q4'],
        'pred_aug_valtest_metric_aug_stability_word_paraphrase': ['a1', 'a2', 'a3', 'a4'],
    })
    sampler = MagicMock()
    sampler.test = {'X': data}

    result = valtest_metric_aug_stability_word(
        sampler=sampler,
        scorer=scorer,
        main_metric='single_mean_value_metric',
        augmentations_list=['paraphrase'],
        exec_env='default',
    )

    stats_dict_for_report = result['report']['result_dataframes'][0]
    subset_fraction = stats_dict_for_report['Доля успешно аугментированных объектов'].iloc[0]
    assert subset_fraction == pytest.approx(0.5)


def test_valtest_format_default(mock_sampler_columns: MagicMock,
                                mock_model: MockModel,
                                mock_aug_client: MockOpenAIClient,
                                mock_scorer: MagicMock):
    '''Проверка формата выхода основной функции в режиме default'''
    test_res = valtest_metric_aug_stability_word(
        sampler=mock_sampler_columns,
        scorer=mock_scorer,
        model=mock_model,
        main_metric='single_mean_value_metric',
        augmentations_list=['paraphrase'],
        aug_client=mock_aug_client,
        aug_model_name='GigaChat',
        exec_env='default',
        is_info=True,
    )
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}
    report = test_res['report']
    precomputed = test_res['precomputed']
    assert isinstance(report, dict)
    assert isinstance(precomputed, dict)
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert set(precomputed.keys()) == {'baseline_per_aug', 'metrics_augmented', 'p_values'}
    assert isinstance(report['result_dataframes'][0], pd.DataFrame)


def test_valtest_precomputed_usage(mock_sampler_columns: MagicMock,
                                   mock_scorer: MagicMock):
    '''Проверка использования предвычисленных значений из precomputed'''
    baseline_per_aug = {'paraphrase': {'single_mean_value_metric': 0.8}}
    metrics_augmented = {'single_mean_value_metric': {'paraphrase': 0.75}}
    p_values = {'paraphrase': 0.03}
    test_res = valtest_metric_aug_stability_word(
        sampler=mock_sampler_columns,
        scorer=mock_scorer,
        model=MagicMock(),
        main_metric='single_mean_value_metric',
        augmentations_list=['paraphrase'],
        baseline_per_aug=baseline_per_aug,
        metrics_augmented=metrics_augmented,
        p_values=p_values,
        is_info=True,
    )
    assert test_res['precomputed']['baseline_per_aug'] == baseline_per_aug
    assert test_res['precomputed']['metrics_augmented'] == metrics_augmented
    assert test_res['precomputed']['p_values'] == p_values


def test_valtest_skip_inference_with_pred_columns(mock_sampler_columns: MagicMock,
                                                  mock_model: MockModel,
                                                  mock_aug_client: MockOpenAIClient,
                                                  mock_scorer: MagicMock):
    '''Проверка пропуска реинференса при наличии колонок предсказаний'''
    data = mock_sampler_columns.test['X'].copy()
    data['pred_aug_valtest_metric_aug_stability_word_paraphrase'] = ['Париж', 'Лев Толстой', 'Четыре']
    data['aug_valtest_metric_aug_stability_word_paraphrase'] = [
        'Какой главный город Франции?', 'Кто автор Войны и мира?', 'Сколько два плюс два?'
    ]
    sampler = MagicMock()
    sampler.test = {'X': data}
    valtest_metric_aug_stability_word(
        sampler=sampler,
        scorer=mock_scorer,
        model=mock_model,
        main_metric='single_mean_value_metric',
        augmentations_list=['paraphrase'],
        aug_client=mock_aug_client,
        aug_model_name='GigaChat',
        exec_env='default',
        is_info=True,
    )
    assert len(mock_model.predict_calls) == 0


def test_valtest_reinference_uses_existing_aug_column(mock_model: MockModel,
                                                      mock_scorer: MagicMock):
    '''Проверка реинференса по готовой aug-колонке без повторной аугментации'''
    data = pd.DataFrame({
        'input_query': ['Привет', 'Как дела'],
        'output_answer': ['Здравствуйте', 'Отлично'],
        'reference_answer': ['Здравствуйте', 'Отлично'],
        'aug_valtest_metric_aug_stability_word_paraphrase': ['АУГ1', 'АУГ2'],
    })
    sampler = MagicMock()
    sampler.test = {'X': data}
    aug_client = MagicMock()
    valtest_metric_aug_stability_word(
        sampler=sampler,
        scorer=mock_scorer,
        model=mock_model,
        main_metric='single_mean_value_metric',
        augmentations_list=['paraphrase'],
        aug_client=aug_client,
        aug_model_name='GigaChat',
        exec_env='default',
        is_info=True,
    )


def test_valtest_custom_prompts(mock_sampler_columns: MagicMock,
                                mock_model: MockModel,
                                mock_aug_client: MockOpenAIClient,
                                mock_scorer: MagicMock):
    '''Проверка передачи кастомных промптов для аугментаций'''
    custom_prompts = {
        'paraphrase': 'Ты — редактор. Перепиши текст.',
        'synonym': 'Подбери синоним.',
    }
    test_res = valtest_metric_aug_stability_word(
        sampler=mock_sampler_columns,
        scorer=mock_scorer,
        model=mock_model,
        main_metric='single_mean_value_metric',
        augmentations_list=['paraphrase'],
        aug_client=mock_aug_client,
        aug_model_name='GigaChat',
        aug_model_prompts=custom_prompts,
        exec_env='default',
        is_info=True,
    )
    assert isinstance(test_res, dict)


def test_valtest_word_replace_ratio(mock_sampler_columns: MagicMock,
                                    mock_model: MockModel,
                                    mock_aug_client: MockOpenAIClient,
                                    mock_scorer: MagicMock):
    '''Проверка передачи параметра word_replace_ratio'''
    test_res = valtest_metric_aug_stability_word(
        sampler=mock_sampler_columns,
        scorer=mock_scorer,
        model=mock_model,
        main_metric='single_mean_value_metric',
        augmentations_list=['synonym'],
        aug_client=mock_aug_client,
        aug_model_name='GigaChat',
        word_replace_ratio=0.3,
        exec_env='default',
        is_info=True,
    )
    assert isinstance(test_res, dict)


def test_calculate_stat_significance_not_significant():
    '''Проверка расчета статистической значимости при незначимой разнице'''
    baseline = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    aug = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    is_sig, p_val = calculate_stat_significance(baseline, aug, alpha=0.05)
    assert is_sig is False
    assert p_val == 1.0


def test_calculate_stat_significance_small_sample():
    '''Проверка возврата NaN при малом размере выборки'''
    baseline = np.array([1.0])
    aug = np.array([0.5])
    is_sig, p_val = calculate_stat_significance(baseline, aug, alpha=0.05)
    assert is_sig is False
    assert np.isnan(p_val)


def test_report_valtest_metric_aug_stability_word_structure():
    '''Проверка структуры отчета'''
    baseline_per_aug = {
        'paraphrase': {'accuracy': 0.8},
        'synonym': {'accuracy': 0.8},
    }
    metrics_augmented = {
        'accuracy': {'paraphrase': 0.75, 'synonym': 0.78},
    }
    stats_dict = {
        'paraphrase': {'p_value': 0.03, 'is_significant': True, 'subset_fraction': 1.0},
        'synonym': {'p_value': 0.15, 'is_significant': False, 'subset_fraction': 1.0},
    }
    report = report_valtest_metric_aug_stability_word(
        baseline_per_aug=baseline_per_aug,
        metrics_augmented=metrics_augmented,
        main_metric='accuracy',
        augmentations_list=['paraphrase', 'synonym'],
        semaphore_threshold=(0.05, 0.10),
        greater_is_better=True,
        is_info=False,
        stats_dict=stats_dict,
    )
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(report['result_dataframes'][0], pd.DataFrame)
    df = report['result_dataframes'][0]
    assert 'Тип преобразования' in df.columns
    assert 'p-value' in df.columns
    assert 'Результат теста' in df.columns


def test_report_valtest_metric_aug_stability_word_gray_on_zero_fraction():
    '''Проверка серого цвета при нулевой доле аугментированных объектов'''
    baseline_per_aug = {'paraphrase': {'accuracy': 0.8}}
    metrics_augmented = {'accuracy': {'paraphrase': 0.8}}
    stats_dict = {'paraphrase': {'p_value': np.nan, 'is_significant': False, 'subset_fraction': 0.0}}
    report = report_valtest_metric_aug_stability_word(
        baseline_per_aug=baseline_per_aug,
        metrics_augmented=metrics_augmented,
        main_metric='accuracy',
        augmentations_list=['paraphrase'],
        stats_dict=stats_dict,
    )
    df = report['result_dataframes'][0]
    assert df['Результат теста'].iloc[0] == 'gray'


def test_report_valtest_metric_aug_stability_word_info_mode():
    '''Проверка серого цвета в информативном режиме'''
    baseline_per_aug = {'paraphrase': {'accuracy': 0.8}}
    metrics_augmented = {'accuracy': {'paraphrase': 0.5}}
    stats_dict = {'paraphrase': {'p_value': 0.01, 'is_significant': True, 'subset_fraction': 1.0}}
    report = report_valtest_metric_aug_stability_word(
        baseline_per_aug=baseline_per_aug,
        metrics_augmented=metrics_augmented,
        main_metric='accuracy',
        augmentations_list=['paraphrase'],
        stats_dict=stats_dict,
        is_info=True,
    )
    assert report['semaphore'] == 'gray'


def test_resolve_answer_checker_default():
    '''Проверка возврата дефолтного чекера при None'''
    checker = resolve_answer_checker(None)
    assert checker is default_answer_checker


def test_resolve_answer_checker_custom():
    '''Проверка возврата кастомного чекера'''
    custom = lambda q, a, r: 1.0
    checker = resolve_answer_checker(custom)
    assert checker is custom


def test_default_answer_checker_exact_match():
    '''Проверка exact match в дефолтном чекере'''
    assert default_answer_checker('q', 'Paris', 'Paris') == 1.0
    assert default_answer_checker('q', 'Paris', 'London') == 0.0


def test_default_answer_checker_no_reference():
    '''Проверка возврата 0.0 при отсутствии эталонного ответа'''
    assert default_answer_checker('q', 'Paris', None) == 0.0


def test_prepare_and_evaluate_with_reference():
    '''Проверка подготовки данных с эталонными ответами'''
    data = pd.DataFrame({
        'input_query': ['q1', 'q2'],
        'output_answer': ['a1', 'a2'],
        'reference_answer': ['a1', 'a3'],
    })
    scorer = MagicMock()
    scorer.calc.return_value = {'accuracy': 0.5}
    checker = lambda q, a, r: 1.0 if a == r else 0.0
    aggregated, raw = prepare_and_evaluate(data, scorer, 'accuracy', checker, 'test')
    assert aggregated == {'accuracy': 0.5}
    assert len(raw) == 2
    assert raw[0] == 1.0
    assert raw[1] == 0.0


def test_prepare_and_evaluate_with_pred_col():
    '''Проверка использования готовой колонки предсказаний'''
    data = pd.DataFrame({
        'input_query': ['q1', 'q2'],
        'output_answer': ['a1', 'a2'],
        'reference_answer': ['a1', 'a2'],
        'pred_col': ['a1', 'a3'],
    })
    scorer = MagicMock()
    scorer.calc.return_value = {'accuracy': 0.5}
    checker = lambda q, a, r: 1.0 if a == r else 0.0
    aggregated, raw = prepare_and_evaluate(data, scorer, 'accuracy', checker, 'test', use_pred=True, pred_col='pred_col')
    assert raw[0] == 1.0
    assert raw[1] == 0.0


def test_get_augmented_data_creates_columns(mock_sampler_columns: MagicMock,
                                            mock_aug_client: MockOpenAIClient):
    '''Проверка создания колонок с аугментированными текстами'''
    result = get_augmented_data(
        sampler=mock_sampler_columns,
        data_type='test',
        augmentations_list=['paraphrase'],
        seed=42,
        aug_client=mock_aug_client,
        aug_model_name='GigaChat',
        exec_env='default',
    )
    assert 'aug_valtest_metric_aug_stability_word_paraphrase' in result.columns
    assert len(result) == 3