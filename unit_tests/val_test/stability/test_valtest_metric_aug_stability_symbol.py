import sys
import types
import pandas as pd
import pytest
from unittest.mock import MagicMock

if 'torch' not in sys.modules:
    torch_stub = types.ModuleType('torch')
    torch_stub.cuda = types.SimpleNamespace(empty_cache=lambda: None)
    sys.modules['torch'] = torch_stub
if 'pymorphy3' not in sys.modules:
    pymorphy3_stub = types.ModuleType('pymorphy3')
    pymorphy3_stub.MorphAnalyzer = lambda *args, **kwargs: None
    sys.modules['pymorphy3'] = pymorphy3_stub

from llm_val.val_test.stability.valtest_metric_aug_stability_symbol import (valtest_metric_aug_stability_symbol,
                                                                            report_valtest_metric_aug_stability_symbol,
                                                                            compute_augmented_metrics,
                                                                            compute_augmented_metric_for_method,
                                                                            compute_baseline_metric,
                                                                            get_default_augmentations,
                                                                            resolve_answer_checker,
                                                                            default_answer_checker,
                                                                            get_metric_values_from_predictions,
                                                                            compute_metric,
                                                                            get_queries_metric_values,
                                                                            apply_augmentation,
                                                                            build_rng,
                                                                            calc_metric,
                                                                            get_augmented_data,
                                                                            build_augmented_column)

MAIN_METRIC = 'single_mean_metric_value'


def make_queries_sampler(reference=True, target_values=(1, 1, 1)):
    columns = {'session_id': [0, 1, 2],
               'query_id': [0, 1, 2],
               'input_query': ['Столица Франции?', 'Где находится Париж?', 'Что такое Москва?'],
               'output_answer': ['Париж', 'Париж', 'город']}
    if reference:
        columns['reference_answer'] = ['Париж', 'Париж', 'город']
    data = pd.DataFrame(columns)
    target = pd.DataFrame({'metric_value': list(target_values)})
    sampler = MagicMock()
    sampler.test = {'X': data, 'y': target}
    return sampler, data


def make_scorer():
    scorer = MagicMock()
    scorer.calc.side_effect = lambda sampler, data_type, metric_name: \
        {MAIN_METRIC: float(sampler.test['y']['metric_value'].mean())}
    return scorer


def make_model(answer='Париж'):
    model = MagicMock()
    model.predict.side_effect = lambda questions, chat_history=None: [answer for _ in questions]
    return model


def correct_checker(query, model_answer, reference_answer=None):
    return 1.0


def test_output_format():
    sampler, _ = make_queries_sampler()
    result = valtest_metric_aug_stability_symbol(sampler=sampler,
                                                 scorer=make_scorer(),
                                                 model=make_model(),
                                                 main_metric=MAIN_METRIC,
                                                 p_augmented=0.3)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 'precomputed'}
    assert isinstance(result['report'], dict)
    assert isinstance(result['precomputed'], dict)


def test_report_keys_and_table():
    sampler, _ = make_queries_sampler()
    result = valtest_metric_aug_stability_symbol(sampler=sampler,
                                                 scorer=make_scorer(),
                                                 model=make_model(),
                                                 main_metric=MAIN_METRIC,
                                                 p_augmented=0.3)
    report = result['report']
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(report['result_dataframes'][0], pd.DataFrame)
    assert report['semaphore'] in {'gray', 'green', 'yellow', 'red'}


def test_precomputed_keys():
    sampler, _ = make_queries_sampler()
    result = valtest_metric_aug_stability_symbol(sampler=sampler,
                                                 scorer=make_scorer(),
                                                 model=make_model(),
                                                 main_metric=MAIN_METRIC,
                                                 p_augmented=0.3)
    precomputed = result['precomputed']
    assert set(precomputed.keys()) == {'metric_baseline', 'metrics_augmented', 'augmentations_list'}
    assert precomputed['augmentations_list'] == get_default_augmentations()
    assert set(precomputed['metrics_augmented'][MAIN_METRIC].keys()) == set(get_default_augmentations())


def test_baseline_taken_from_target_values():
    sampler, _ = make_queries_sampler(target_values=(1, 0, 1))
    model = make_model()
    result = valtest_metric_aug_stability_symbol(sampler=sampler,
                                                 scorer=make_scorer(),
                                                 model=model,
                                                 main_metric=MAIN_METRIC,
                                                 p_augmented=0.3)
    assert result['precomputed']['metric_baseline'][MAIN_METRIC] == pytest.approx(2 / 3)


def test_precomputed_skips_all_steps():
    metric_baseline = {MAIN_METRIC: 0.95}
    metrics_augmented = {MAIN_METRIC: {'human_level_augmentations': 0.94,
                                       'symbol_swap': 0.88,
                                       'butter_finger': 0.80}}
    sampler = MagicMock()
    type(sampler).test = property(lambda self: (_ for _ in ()).throw(AssertionError('data accessed')))
    model = make_model()
    result = valtest_metric_aug_stability_symbol(sampler=sampler,
                                                 scorer=make_scorer(),
                                                 model=model,
                                                 main_metric=MAIN_METRIC,
                                                 metric_baseline=metric_baseline,
                                                 metrics_augmented=metrics_augmented)
    model.predict.assert_not_called()
    assert result['report']['semaphore'] == 'red'


def test_input_data_not_mutated():
    sampler, data = make_queries_sampler()
    columns_before = list(data.columns)
    valtest_metric_aug_stability_symbol(sampler=sampler,
                                        scorer=make_scorer(),
                                        model=make_model(),
                                        main_metric=MAIN_METRIC,
                                        p_augmented=0.3)
    assert list(data.columns) == columns_before


def test_precomputed_skips_inference():
    sampler, _ = make_queries_sampler()
    model = make_model()
    metric_baseline = {MAIN_METRIC: 0.95}
    metrics_augmented = {MAIN_METRIC: {'human_level_augmentations': 0.94,
                                       'symbol_swap': 0.88,
                                       'butter_finger': 0.80}}
    result = valtest_metric_aug_stability_symbol(sampler=sampler,
                                                 scorer=make_scorer(),
                                                 model=model,
                                                 main_metric=MAIN_METRIC,
                                                 metric_baseline=metric_baseline,
                                                 metrics_augmented=metrics_augmented)
    model.predict.assert_not_called()
    assert result['report']['semaphore'] == 'red'


def test_skips_inference_with_prediction_columns():
    sampler, data = make_queries_sampler()
    for aug_name in get_default_augmentations():
        data['pred_aug_valtest_metric_aug_stability_symbol_' + aug_name] = ['нет', 'нет', 'нет']
    model = make_model()
    result = valtest_metric_aug_stability_symbol(sampler=sampler,
                                                 scorer=make_scorer(),
                                                 model=model,
                                                 main_metric=MAIN_METRIC,
                                                 p_augmented=0.3)
    model.predict.assert_not_called()
    precomputed = result['precomputed']
    assert precomputed['metric_baseline'][MAIN_METRIC] == pytest.approx(1.0)
    for aug_name in get_default_augmentations():
        assert precomputed['metrics_augmented'][MAIN_METRIC][aug_name] == pytest.approx(0.0)


def test_reinference_uses_existing_aug_columns():
    sampler, data = make_queries_sampler()
    captured = []

    def model_predict(questions, chat_history=None):
        captured.append(list(questions))
        return ['Париж' for _ in questions]

    model = MagicMock()
    model.predict.side_effect = model_predict
    aug_queries = ['АУГ1', 'АУГ2', 'АУГ3']
    for aug_name in get_default_augmentations():
        data['aug_valtest_metric_aug_stability_symbol_' + aug_name] = aug_queries
    valtest_metric_aug_stability_symbol(sampler=sampler,
                                        scorer=make_scorer(),
                                        model=model,
                                        main_metric=MAIN_METRIC,
                                        p_augmented=0.3)
    augmented_calls = [call for call in captured if call == aug_queries]
    assert len(augmented_calls) == len(get_default_augmentations())


def test_semaphore_thresholds():
    metric_baseline = {MAIN_METRIC: 1.0}
    metrics_augmented = {MAIN_METRIC: {'human_level_augmentations': 0.97,
                                       'symbol_swap': 0.93,
                                       'butter_finger': 0.85}}
    augmentations = ['human_level_augmentations', 'symbol_swap', 'butter_finger']
    report = report_valtest_metric_aug_stability_symbol(metric_baseline,
                                                        metrics_augmented,
                                                        MAIN_METRIC,
                                                        augmentations)
    table = report['result_dataframes'][0]
    colors = table['Результат теста'].tolist()
    assert colors == ['green', 'yellow', 'red']
    assert report['semaphore'] == 'red'


def test_is_info_returns_gray():
    metric_baseline = {MAIN_METRIC: 1.0}
    metrics_augmented = {MAIN_METRIC: {'human_level_augmentations': 0.5}}
    report = report_valtest_metric_aug_stability_symbol(metric_baseline,
                                                        metrics_augmented,
                                                        MAIN_METRIC,
                                                        ['human_level_augmentations'],
                                                        is_info=True)
    assert report['semaphore'] == 'gray'


def test_resolve_answer_checker_default():
    checker = resolve_answer_checker(None)
    assert checker is default_answer_checker


def test_resolve_answer_checker_custom():
    checker = resolve_answer_checker(correct_checker)
    assert checker is correct_checker


def test_default_answer_checker():
    assert default_answer_checker('q', 'Это Париж', 'Париж') == 1.0
    assert default_answer_checker('q', 'Это Лондон', 'Париж') == 0.0
    assert default_answer_checker('q', 'Это Лондон', None) == 0.0


def test_get_metric_values_from_predictions():
    data = pd.DataFrame({'input_query': ['q1', 'q2'],
                         'reference_answer': ['Париж', 'Лондон'],
                         'pred': ['Это Париж', 'Это Берлин']})
    values = get_metric_values_from_predictions(data, 'pred', default_answer_checker)
    assert values == [1.0, 0.0]


def test_get_metric_values_from_predictions_no_reference():
    data = pd.DataFrame({'input_query': ['q1', 'q2'],
                         'pred': ['Это Париж', 'Это Берлин']})
    values = get_metric_values_from_predictions(data, 'pred', default_answer_checker)
    assert values == [0.0, 0.0]


def test_get_metric_values_from_predictions_uses_custom_checker():
    data = pd.DataFrame({'input_query': ['q1', 'q2'],
                         'pred': ['a', 'b']})
    values = get_metric_values_from_predictions(data, 'pred', correct_checker)
    assert values == [1.0, 1.0]


def test_compute_baseline_metric_uses_target_values():
    sampler, _ = make_queries_sampler(target_values=(1, 1, 0))
    score = compute_baseline_metric(sampler, make_scorer(), 'test', 'single_mean', 'default')
    assert score[MAIN_METRIC] == pytest.approx(2 / 3)


def test_compute_baseline_metric_passes_original_sampler():
    sampler, _ = make_queries_sampler(target_values=(1, 1, 0))
    scorer = make_scorer()
    compute_baseline_metric(sampler, scorer, 'test', 'single_mean', 'default')
    passed_sampler = scorer.calc.call_args.kwargs['sampler']
    assert passed_sampler is sampler


def test_compute_baseline_metric_sberds_wraps_dict():
    target = pd.DataFrame({'metric_value': [1, 1, 0]})
    sampler = {'test': {'X': pd.DataFrame({'input_query': ['a', 'b', 'c']}), 'y': target}}
    score = compute_baseline_metric(sampler, make_scorer(), 'test', 'single_mean', 'sberds')
    assert score[MAIN_METRIC] == pytest.approx(2 / 3)


def test_compute_augmented_metric_for_method_uses_prediction_column():
    sampler, data = make_queries_sampler()
    data['pred_aug_valtest_metric_aug_stability_symbol_symbol_swap'] = ['нет', 'нет', 'нет']
    model = make_model()
    score = compute_augmented_metric_for_method(data, model, 'symbol_swap', default_answer_checker,
                                                make_scorer(), 'single_mean', 42, 0.0)
    model.predict.assert_not_called()
    assert score[MAIN_METRIC] == pytest.approx(0.0)


def test_apply_augmentation_reproducible():
    questions = ['Привет, как дела сегодня?', 'Столица Франции это Париж']
    for aug_name in get_default_augmentations():
        first = apply_augmentation(questions, aug_name, build_rng(42), 0.3)
        second = apply_augmentation(questions, aug_name, build_rng(42), 0.3)
        assert list(first) == list(second)
        assert len(first) == len(questions)


def test_apply_augmentation_none_returns_copy():
    questions = ['Привет', 'Париж']
    result = apply_augmentation(questions, None, build_rng(1), 0.5)
    assert result == questions
    assert result is not questions


def test_apply_augmentation_does_not_mutate_input():
    questions = ['Привет, как дела?', 'Столица Франции это Париж']
    snapshot = list(questions)
    apply_augmentation(questions, 'symbol_swap', build_rng(1), 0.5)
    assert questions == snapshot


def test_get_queries_metric_values():
    data = pd.DataFrame({'input_query': ['q1', 'q2'],
                         'reference_answer': ['Париж', 'Лондон']})
    model = make_model('Это Париж')
    values = get_queries_metric_values(data, model, default_answer_checker, None, None, build_rng(42), 0.0)
    assert values == [1.0, 0.0]


def test_get_queries_metric_values_no_reference():
    data = pd.DataFrame({'input_query': ['q1', 'q2']})
    model = make_model('Это Париж')
    values = get_queries_metric_values(data, model, default_answer_checker, None, None, build_rng(42), 0.0)
    assert values == [0.0, 0.0]


def test_get_queries_metric_values_uses_augmented_data():
    data = pd.DataFrame({'input_query': ['q1', 'q2'],
                         'reference_answer': ['Париж', 'Лондон']})
    captured = []

    def model_predict(questions, chat_history=None):
        captured.append(list(questions))
        return ['Это Париж' for _ in questions]

    model = MagicMock()
    model.predict.side_effect = model_predict
    augmented = ['АУГ1', 'АУГ2']
    get_queries_metric_values(data, model, default_answer_checker, 'symbol_swap', augmented, build_rng(42), 0.5)
    assert captured == [augmented]


def test_compute_metric_queries():
    sampler, data = make_queries_sampler()
    model = make_model('город')
    score = compute_metric(data, model, default_answer_checker, 'symbol_swap', None,
                           make_scorer(), 'single_mean', 42, 0.0)
    assert score[MAIN_METRIC] == pytest.approx(1 / 3)


def test_compute_augmented_metrics_keys():
    sampler, data = make_queries_sampler()
    metrics = compute_augmented_metrics(data, make_model('Париж'), get_default_augmentations(),
                                        default_answer_checker, make_scorer(), 'single_mean', 42, 0.3)
    assert set(metrics[MAIN_METRIC].keys()) == set(get_default_augmentations())


def test_calc_metric():
    score = calc_metric([1.0, 0.0, 1.0, 1.0], make_scorer(), 'single_mean')
    assert score[MAIN_METRIC] == 0.75


def test_get_augmented_data_columns_queries():
    sampler, data = make_queries_sampler()
    aug_df = get_augmented_data(sampler, p_augmented=0.3)
    expected = {'aug_valtest_metric_aug_stability_symbol_' + name for name in get_default_augmentations()}
    assert set(aug_df.columns) == expected
    assert len(aug_df) == len(data)


def test_build_augmented_column_queries():
    _, data = make_queries_sampler()
    column = build_augmented_column(data, 'symbol_swap', build_rng(42), 0.3)
    assert len(column) == len(data)
    assert all(isinstance(value, str) for value in column)
