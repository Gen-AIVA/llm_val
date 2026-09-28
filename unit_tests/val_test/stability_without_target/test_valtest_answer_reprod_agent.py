import typing as tp
import pytest
import numpy as np
import pandas as pd
from types import SimpleNamespace
from unittest.mock import MagicMock

import llm_val.utils.retries as retries

from llm_val.utils.report_helper import semaphore_reproducibility_agent
from llm_val.val_test.stability_without_target.valtest_answer_reprod_agent import (valtest_answer_reprod_agent,
                                                                                    get_query_predictions,
                                                                                    regroup_by_counts,
                                                                                    predictions_from_columns,
                                                                                    find_columns_by_prefix,
                                                                                    evaluate_mean_diversity,
                                                                                    compute_pairwise_diversity,
                                                                                    lines_to_ngrams,
                                                                                    mean_pairwise_lingustic_diversity,
                                                                                    mean_pairwise_semantic_diversity_emb,
                                                                                    mean_pairwise_exact_match_diversity,
                                                                                    compute_difference,
                                                                                    get_augmented_queries,
                                                                                    get_augmented_data)


class MockModel:
    '''Мок языковой модели, возвращающей детерминированные ответы по запросам'''
    def __init__(self,
                 answers_map: tp.Dict[str, str]):
        self.answers_map = answers_map

    def predict(self,
                messages: tp.List[str]) -> tp.List[str]:
        return [self.answers_map.get(message, message) for message in messages]


class MockEmbeddings:
    '''Мок ресурса embeddings OpenAI-клиента, возвращающего эмбеддинги фиксированной размерности'''
    def __init__(self,
                 dim: int=4):
        self.dim = dim
        self.last_model_name = None

    def create(self,
               model: str,
               input: tp.List[str]) -> tp.Any:
        self.last_model_name = model
        data = []
        for text in input:
            rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
            data.append(SimpleNamespace(embedding=rng.random(self.dim).tolist()))
        return SimpleNamespace(data=data)


class MockOpenAIClient:
    '''Мок OpenAI-совместимого клиента с ресурсом embeddings'''
    def __init__(self,
                 dim: int=4):
        self.embeddings = MockEmbeddings(dim)


@pytest.fixture
def data_columns() -> pd.DataFrame:
    '''Данные в формате отдельных запросов query_id/input_query/output_answer'''
    return pd.DataFrame({'session_id': [1, 2, 3],
                         'query_id': ['id1', 'id2', 'id3'],
                         'input_query': ['Привет', 'Как дела', 'Что нового'],
                         'output_answer': ['Здравствуйте', 'Отлично', 'Много интересного'],
                         'scenario': ['smalltalk', 'smalltalk', 'news']})


@pytest.fixture
def mock_sampler_columns(data_columns: pd.DataFrame) -> MagicMock:
    '''Мок Sampler для режима default с колоночным форматом'''
    sampler = MagicMock()
    sampler.test = {'X': data_columns}
    return sampler


@pytest.fixture
def mock_model() -> MockModel:
    '''Мок модели с детерминированными ответами'''
    return MockModel({'Как дела': 'Все хорошо', 'Что нового': 'Ничего особенного'})


@pytest.fixture
def mock_embedding_client() -> MockOpenAIClient:
    '''Мок OpenAI-совместимого клиента для эмбеддингов'''
    return MockOpenAIClient(dim=4)


def test_valtest_format_default(mock_sampler_columns: MagicMock,
                                mock_model: MockModel,
                                mock_embedding_client: MockOpenAIClient):
    '''Проверка формата выхода основной функции в режиме default'''
    test_res = valtest_answer_reprod_agent(sampler=mock_sampler_columns,
                                           model=mock_model,
                                           is_long_answer=True,
                                           embedding_client=mock_embedding_client,
                                           n_generated_answers=3,
                                           augmentations_list=['use_special_symbol'],
                                           exec_env='default')
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}
    report = test_res['report']
    precomputed = test_res['precomputed']
    assert isinstance(report, dict)
    assert isinstance(precomputed, dict)
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert set(precomputed.keys()) == {'orig_diversity', 'aug_diversity'}
    assert isinstance(report['result_dataframes'][0], pd.DataFrame)


def test_valtest_format_sberds(data_columns: pd.DataFrame,
                               mock_model: MockModel,
                               mock_embedding_client: MockOpenAIClient):
    '''Проверка работы с exec_env=sberds'''
    sampler = {'test': {'X': data_columns}}
    test_res = valtest_answer_reprod_agent(sampler=sampler,
                                           model=mock_model,
                                           is_long_answer=True,
                                           embedding_client=mock_embedding_client,
                                           n_generated_answers=3,
                                           use_augmentations=False,
                                           exec_env='sberds')
    assert set(test_res.keys()) == {'report', 'precomputed'}
    assert test_res['precomputed']['aug_diversity'] is None
    assert isinstance(test_res['precomputed']['orig_diversity'], dict)


def test_valtest_short_answer(mock_sampler_columns: MagicMock,
                              mock_model: MockModel):
    '''Проверка работы теста для коротких ответов (exact match) без эмбеддинг-модели'''
    test_res = valtest_answer_reprod_agent(sampler=mock_sampler_columns,
                                           model=mock_model,
                                           is_long_answer=False,
                                           n_generated_answers=4,
                                           use_augmentations=False,
                                           exec_env='default')
    orig = test_res['precomputed']['orig_diversity']
    assert orig['lexical'] == orig['semantic']


def test_valtest_default_run(mock_sampler_columns: MagicMock,
                             mock_model: MockModel,
                             mock_embedding_client: MockOpenAIClient):
    '''Проверка дефолтного запуска без явного указания augmentations_list и use_augmentations'''
    test_res = valtest_answer_reprod_agent(sampler=mock_sampler_columns,
                                           model=mock_model,
                                           is_long_answer=True,
                                           embedding_client=mock_embedding_client,
                                           n_generated_answers=3,
                                           exec_env='default')
    assert set(test_res.keys()) == {'report', 'precomputed'}
    assert test_res['precomputed']['aug_diversity'] is not None
    assert isinstance(test_res['precomputed']['orig_diversity'], dict)


def test_valtest_precomputed_usage(mock_sampler_columns: MagicMock):
    '''Проверка использования предвычисленных значений из precomputed'''
    orig = {'lexical': 0.01, 'semantic': 0.5}
    aug = {'lexical': 0.02, 'semantic': 0.55}
    test_res = valtest_answer_reprod_agent(sampler=mock_sampler_columns,
                                           is_creative=False,
                                           orig_diversity=orig,
                                           aug_diversity=aug)
    assert test_res['precomputed']['orig_diversity'] == orig
    assert test_res['precomputed']['aug_diversity'] == aug


def test_valtest_skip_orig_inference_with_pred_columns(mock_embedding_client: MockOpenAIClient):
    '''Проверка пропуска реинференса исходных данных при наличии колонок предсказаний'''
    data = pd.DataFrame({'input_query': ['Привет', 'Как дела'],
                         'output_answer': ['Здравствуйте', 'Отлично']})
    data['pred_valtest_answer_reprod_agent_0'] = ['ответ a', 'ответ x']
    data['pred_valtest_answer_reprod_agent_1'] = ['ответ b', 'ответ y']
    sampler = MagicMock()
    sampler.test = {'X': data}
    model = MagicMock()
    valtest_answer_reprod_agent(sampler=sampler,
                                model=model,
                                is_long_answer=True,
                                embedding_client=mock_embedding_client,
                                n_generated_answers=2,
                                use_augmentations=False,
                                exec_env='default')
    model.predict.assert_not_called()


def test_valtest_skip_aug_inference_with_pred_columns(mock_embedding_client: MockOpenAIClient):
    '''Проверка пропуска реинференса аугментированных данных при наличии колонок предсказаний'''
    data = pd.DataFrame({'input_query': ['Привет', 'Как дела'],
                         'output_answer': ['Здравствуйте', 'Отлично']})
    data['pred_valtest_answer_reprod_agent_0'] = ['ответ a', 'ответ x']
    data['pred_valtest_answer_reprod_agent_1'] = ['ответ b', 'ответ y']
    data['pred_aug_valtest_answer_reprod_agent_0'] = ['ауг a', 'ауг x']
    data['pred_aug_valtest_answer_reprod_agent_1'] = ['ауг b', 'ауг y']
    sampler = MagicMock()
    sampler.test = {'X': data}
    model = MagicMock()
    valtest_answer_reprod_agent(sampler=sampler,
                                model=model,
                                is_long_answer=True,
                                embedding_client=mock_embedding_client,
                                n_generated_answers=2,
                                use_augmentations=True,
                                exec_env='default')
    model.predict.assert_not_called()


def test_valtest_reinference_uses_existing_aug_column(mock_embedding_client: MockOpenAIClient):
    '''Проверка реинференса по готовой aug-колонке без повторной аугментации'''
    data = pd.DataFrame({'input_query': ['Привет', 'Как дела'],
                         'output_answer': ['Здравствуйте', 'Отлично']})
    data['pred_valtest_answer_reprod_agent_0'] = ['ответ a', 'ответ x']
    data['pred_valtest_answer_reprod_agent_1'] = ['ответ b', 'ответ y']
    data['aug_valtest_answer_reprod_agent_0'] = ['АУГ1', 'АУГ2']
    captured = []

    def predict(messages):
        captured.extend(messages)
        return list(messages)

    model = MagicMock()
    model.predict.side_effect = predict
    valtest_answer_reprod_agent(sampler=_sampler_with(data),
                                model=model,
                                is_long_answer=True,
                                embedding_client=mock_embedding_client,
                                n_generated_answers=2,
                                use_augmentations=True,
                                exec_env='default')
    assert 'АУГ1' in captured
    assert 'АУГ2' in captured


def _sampler_with(data: pd.DataFrame) -> MagicMock:
    '''Вспомогательная функция: создает мок Sampler с заданными данными'''
    sampler = MagicMock()
    sampler.test = {'X': data}
    return sampler


def test_get_query_predictions(mock_model: MockModel):
    '''Проверка получения повторных генераций по запросам'''
    predictions = get_query_predictions(mock_model, ['Как дела', 'Что нового'], n_generated_answers=3)
    assert len(predictions) == 2
    assert len(predictions[0]) == 3
    assert predictions[0][0] == 'Все хорошо'


def test_get_query_predictions_retries_model_predict(monkeypatch):
    sleep_values = []
    monkeypatch.setattr(retries.time, 'sleep', lambda value: sleep_values.append(value))
    model = MagicMock()
    model.predict.side_effect = [RuntimeError('fail'), ['answer']]
    predictions = get_query_predictions(model,
                                        ['q'],
                                        n_generated_answers=1,
                                        timeout=0.5,
                                        retry_delay=0.01,
                                        max_attempts=2)
    assert predictions == [['answer']]
    assert model.predict.call_count == 2
    assert sleep_values == [0.01]


def test_mean_pairwise_semantic_diversity_retries_embedding(monkeypatch):
    sleep_values = []
    monkeypatch.setattr(retries.time, 'sleep', lambda value: sleep_values.append(value))
    embedding_client = MagicMock()
    embedding_client.embeddings.create.side_effect = [RuntimeError('fail'),
                                                      SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0]),
                                                                            SimpleNamespace(embedding=[0.0, 1.0])])]
    diversity = mean_pairwise_semantic_diversity_emb(['a', 'b'],
                                                     embedding_client,
                                                     timeout=0.5,
                                                     retry_delay=0.01,
                                                     max_attempts=2)
    assert diversity == pytest.approx(1.0)
    assert embedding_client.embeddings.create.call_count == 2
    assert sleep_values == [0.01]


def test_regroup_by_counts():
    '''Проверка группировки плоского списка ответов по размерам групп'''
    grouped = regroup_by_counts(['a', 'b', 'c'], [2, 1])
    assert grouped == [['a', 'b'], ['c']]


def test_predictions_from_columns():
    '''Проверка восстановления предсказаний из колонок'''
    data = pd.DataFrame({'pred_valtest_answer_reprod_agent_0': ['a1', 'b1'],
                         'pred_valtest_answer_reprod_agent_1': ['c1', 'd1']})
    predictions = predictions_from_columns(data, 'pred_valtest_answer_reprod_agent_')
    assert predictions[0] == ['a1', 'c1']
    assert predictions[1] == ['b1', 'd1']


def test_find_columns_by_prefix():
    '''Проверка поиска колонок по префиксу'''
    data = pd.DataFrame(columns=['input_query',
                                 'pred_valtest_answer_reprod_agent_0',
                                 'pred_aug_valtest_answer_reprod_agent_0'])
    assert find_columns_by_prefix(data, 'pred_valtest_answer_reprod_agent_') == \
        ['pred_valtest_answer_reprod_agent_0']
    assert find_columns_by_prefix(data, 'pred_aug_valtest_answer_reprod_agent_') == \
        ['pred_aug_valtest_answer_reprod_agent_0']


def test_evaluate_mean_diversity():
    '''Проверка усреднения разнообразия по запросам'''
    predictions = [['a', 'a'], ['x', 'y']]
    diversity_func = exact_match_pair_diversity
    result = evaluate_mean_diversity(predictions, diversity_func)
    assert result['lexical'] == pytest.approx(0.5)
    assert result['semantic'] == pytest.approx(0.5)


def exact_match_pair_diversity(sentences: tp.List[str]) -> tp.Tuple[float, float]:
    '''Вспомогательная функция: возвращает exact match разнообразие как лексику и семантику'''
    value = mean_pairwise_exact_match_diversity(sentences)
    return value, value


def test_compute_pairwise_diversity_long(mock_embedding_client: MockOpenAIClient):
    '''Проверка расчета разнообразия для развернутых ответов'''
    sentences = ['красный синий зеленый', 'красный синий зеленый', 'другой текст совсем']
    lexical, semantic = compute_pairwise_diversity(sentences, True, 2, mock_embedding_client)
    assert 0.0 <= lexical <= 1.0
    assert 0.0 <= semantic <= 1.0


def test_compute_pairwise_diversity_passes_model_name(mock_embedding_client: MockOpenAIClient):
    '''Проверка прокидывания названия эмбеддинг-модели в embeddings.create'''
    sentences = ['красный синий зеленый', 'другой текст совсем']
    compute_pairwise_diversity(sentences, True, 2, mock_embedding_client, embedding_model_name='EmbeddingsGigaR')
    assert mock_embedding_client.embeddings.last_model_name == 'EmbeddingsGigaR'


def test_compute_pairwise_diversity_short():
    '''Проверка расчета разнообразия для коротких ответов'''
    sentences = ['A', 'A', 'B']
    lexical, semantic = compute_pairwise_diversity(sentences, False, 3)
    assert lexical == semantic == pytest.approx(2 / 3)


def test_compute_pairwise_diversity_long_without_embedding_client():
    '''Проверка явной ошибки для длинных ответов без эмбеддинг-клиента'''
    with pytest.raises(ValueError, match='embedding_client'):
        compute_pairwise_diversity(['ответ один', 'ответ два'], True, 3)


def test_lines_to_ngrams():
    '''Проверка построения n-грамм'''
    ngrams = lines_to_ngrams(['одно два три четыре'], n=2)
    assert ngrams[0] == [('одно', 'два'), ('два', 'три'), ('три', 'четыре')]


def test_mean_pairwise_lingustic_diversity():
    '''Проверка лексического разнообразия по n-граммам'''
    identical = mean_pairwise_lingustic_diversity(['a b c', 'a b c'], 2)
    different = mean_pairwise_lingustic_diversity(['a b c', 'x y z'], 2)
    assert identical == pytest.approx(0.0)
    assert different == pytest.approx(1.0)


def test_mean_pairwise_semantic_diversity_emb(mock_embedding_client: MockOpenAIClient):
    '''Проверка семантического разнообразия по эмбеддингам'''
    diversity = mean_pairwise_semantic_diversity_emb(['текст один', 'текст два'], mock_embedding_client)
    assert isinstance(diversity, float)


def test_mean_pairwise_exact_match_diversity():
    '''Проверка разнообразия по exact match'''
    assert mean_pairwise_exact_match_diversity(['a', 'a', 'a']) == pytest.approx(0.0)
    assert mean_pairwise_exact_match_diversity(['a', 'b']) == pytest.approx(1.0)


def test_compute_difference():
    '''Проверка расчета абсолютной и относительной разницы'''
    abs_diff, rel_diff = compute_difference(0.1, 0.25)
    assert abs_diff == pytest.approx(0.15)
    assert rel_diff == pytest.approx(1.5)
    nan_abs, nan_rel = compute_difference(0.1, np.nan)
    assert np.isnan(nan_abs) and np.isnan(nan_rel)


def test_compute_difference_zero_orig_value():
    '''Проверка относительной разницы при нулевом исходном значении'''
    zero_abs_diff, zero_rel_diff = compute_difference(0.0, 0.0)
    positive_abs_diff, positive_rel_diff = compute_difference(0.0, 0.2)
    assert zero_abs_diff == pytest.approx(0.0)
    assert zero_rel_diff == pytest.approx(0.0)
    assert positive_abs_diff == pytest.approx(0.2)
    assert positive_rel_diff == np.inf


def test_get_augmented_queries():
    '''Проверка аугментации запросов'''
    queries = ['Привет как твои дела сегодня']
    augmented_queries = get_augmented_queries(queries, ['use_special_symbol'], seed=1, p_augmented=1.0)
    assert len(augmented_queries) == 1


def test_get_augmented_queries_default_augmentations():
    '''Проверка дефолтной аугментации запросов без явного списка аугментаций'''
    queries = ['Привет как твои дела сегодня']
    augmented_queries = get_augmented_queries(queries, seed=1, p_augmented=0.5)
    assert len(augmented_queries) == 1


def test_get_augmented_data(mock_sampler_columns: MagicMock):
    '''Проверка формирования датафрейма с аугментированными запросами для реинференса'''
    result_df = get_augmented_data(mock_sampler_columns,
                                   'test',
                                   ['use_special_symbol'],
                                   seed=1,
                                   p_augmented=0.5,
                                   n_columns=2,
                                   exec_env='default')
    assert list(result_df.columns) == ['aug_valtest_answer_reprod_agent_0',
                                       'aug_valtest_answer_reprod_agent_1']
    assert len(result_df) == 3
    assert isinstance(result_df['aug_valtest_answer_reprod_agent_0'].iloc[0], str)


def test_semaphore_reproducibility_agent_precise():
    '''Проверка светофора для точной задачи'''
    assert semaphore_reproducibility_agent(0.01, 0.1, is_creative=False) == 'green'
    assert semaphore_reproducibility_agent(0.1, 0.1, is_creative=False) == 'red'
    assert semaphore_reproducibility_agent(0.04, 0.1, is_creative=False) == 'yellow'


def test_semaphore_reproducibility_agent_creative():
    '''Проверка светофора для творческой задачи'''
    assert semaphore_reproducibility_agent(0.1, 0.1, is_creative=True) == 'green'
    assert semaphore_reproducibility_agent(0.01, 0.1, is_creative=True) == 'red'
    assert semaphore_reproducibility_agent(0.04, 0.1, is_creative=True) == 'yellow'
