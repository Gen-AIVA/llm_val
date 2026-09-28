import re
import random
import string
import numpy as np
import typing as tp
import pandas as pd
from copy import deepcopy
from functools import partial
from scipy.stats import wilcoxon

from llm_val.scorer.Scorer import Scorer
from llm_val.sampler.Sampler import Sampler
from llm_val.utils.api_augmentations import batch_paraphrase_api, batch_synonym_api, batch_translation_api
from llm_val.utils.custom_checkers import check_if_correct
from llm_val.utils.report_helper import semaphore_by_threshold, worst_semaphore
from llm_val.utils.retries import call_with_retries


def report_valtest_metric_aug_stability_word(baseline_per_aug: tp.Dict[str, tp.Dict[str, float]],
                                             metrics_augmented: tp.Dict[str, tp.Dict[str, float]],
                                             main_metric: str,
                                             augmentations_list: tp.List[str],
                                             data_type: str='test',
                                             semaphore_threshold: tp.Tuple[float]=(0.05, 0.10),
                                             greater_is_better: bool=False,
                                             is_info: bool=False,
                                             stats_dict: tp.Optional[tp.Dict] = None) -> tp.Dict[str, tp.Any]:
    '''
    Parameters:
        baseline_per_aug: Словарь со значениями метрик на оригинальных данных для каждой аугментации
        metrics_augmented: Словарь со значениями метрик на аугментированных данных
        main_metric: Название ключевой метрики качества
        augmentations_list: Список названий примененных аугментаций
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        semaphore_threshold: Красный и желтый порог светофора по абсолютной разнице
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: True, если тест носит исключительно информативный характер
        stats_dict: Опциональный словарь со статистическими данными (p-value, значимость, доля успешно аугментированных) для каждой аугментации
    Return:
        Словарь в соответствии с концепцией llm_val, содержащий результат светофора и датафрейм с отчетом
    '''
    baseline_vals = [find_metric_value(baseline_per_aug.get(name, {}), main_metric) for name in augmentations_list]

    metrics_augmented_vals = []
    for name in augmentations_list:
        found_value = np.nan
        for metric_key, aug_dict in metrics_augmented.items():
            if main_metric in metric_key or metric_key in main_metric:
                found_value = find_metric_value(aug_dict, name)
                if not pd.isna(found_value):
                    break
        metrics_augmented_vals.append(found_value)

    absolute_difference = [
        b - a if not (pd.isna(b) or pd.isna(a)) else np.nan
        for b, a in zip(baseline_vals, metrics_augmented_vals)
    ]
    p_values = [stats_dict.get(name, {}).get('p_value', np.nan) if stats_dict else np.nan for name in augmentations_list]
    is_significant = [stats_dict.get(name, {}).get('is_significant', False) if stats_dict else False for name in augmentations_list]
    subset_fractions = [stats_dict.get(name, {}).get('subset_fraction', 0.0) if stats_dict else 0.0 for name in augmentations_list]
    aug_changed = [not np.allclose(b, a, equal_nan=True) if not (pd.isna(b) or pd.isna(a)) else False
                   for b, a in zip(baseline_vals, metrics_augmented_vals)]

    semaphore_colors = []
    for fraction, changed, diff, pv, is_sig in zip(subset_fractions, aug_changed, absolute_difference, p_values, is_significant):
        if fraction == 0.0:
            semaphore_colors.append('gray')
        elif not changed:
            semaphore_colors.append('green')
        elif pd.isna(diff) or pd.isna(pv):
            semaphore_colors.append('gray')
        elif not is_sig:
            semaphore_colors.append('green')
        else:
            semaphore_colors.append(semaphore_by_threshold(diff, semaphore_threshold, greater_is_better=greater_is_better))

    color = 'gray' if is_info else worst_semaphore(semaphore_colors)

    sig_col = ['Да' if sig else 'Нет' if not pd.isna(pv) else 'N/A' for sig, pv in zip(is_significant, p_values)]

    results = pd.DataFrame({
        'Тип преобразования': augmentations_list,
        f'Значение {main_metric} на аугментированных данных': metrics_augmented_vals,
        f'Абсолютное изменение {main_metric}': absolute_difference,
        'Доля успешно аугментированных объектов': subset_fractions,
        'p-value': p_values,
        'Стат. значимость (падения)': sig_col,
        'Результат теста': semaphore_colors
    })
    return {
        'semaphore': color,
        'result_plots': [],
        'result_dataframes': [results]
    }


def valtest_metric_aug_stability_word(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                                      scorer: Scorer,
                                      main_metric: str,
                                      model: tp.Optional[tp.Any] = None,
                                      augmentations_list: tp.List[str] = [
                                          'paraphrase',
                                          'synonym',
                                          'translation'
                                      ],
                                      translation_language: str = 'английский',
                                      word_replace_ratio: float = 0.1,
                                      answer_checker: tp.Optional[tp.Callable[[tp.Any, str, tp.Optional[str]], float]] = None,
                                      metric_agg: str = 'single_mean',
                                      data_type: str = 'test',
                                      semaphore_threshold: tp.Tuple[float] = (0.05, 0.10),
                                      greater_is_better: bool = False,
                                      is_info: bool = False,
                                      seed: int = 42,
                                      max_attempts: int = 3,
                                      timeout: float = 60.0,
                                      retry_delay: float = 0.1,
                                      exec_env: tp.Literal['default', 'sberds'] = 'default',
                                      aug_client: tp.Optional[tp.Any] = None,
                                      aug_model_name: tp.Optional[str] = None,
                                      aug_model_prompts: tp.Optional[tp.Dict[str, str]] = None,
                                      baseline_per_aug: tp.Optional[tp.Dict[str, tp.Dict[str, float]]] = None,
                                      metrics_augmented: tp.Optional[tp.Dict[str, tp.Dict[str, float]]] = None,
                                      p_values: tp.Optional[tp.Dict[str, float]] = None,
                                      **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на стабильность прогноза модели в зависимости от словесных аугментаций с расчетом статистической значимости.
    Поддерживает работу без передачи model/aug_client, если в данных уже есть готовые aug- и pred-колонки
    с корректным неймингом (aug_valtest_metric_aug_stability_word_* и pred_aug_valtest_metric_aug_stability_word_*).
    Parameters:
        sampler: Объект класса Sampler
        scorer: Объект класса Scorer
        main_metric: Название ключевой метрики качества
        model: Валидируемая модель; имеет метод predict.
        augmentations_list: Список выполняемых аугментаций
        translation_language: Язык для аугментации translation (указанный на русском языке)
        word_replace_ratio: Доля слов для замены при синонимической аугментации
        answer_checker: Функция автооценки ответа с сигнатурой (запрос, ответ модели, эталонный ответ)
        metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        semaphore_threshold: Красный и желтый порог светофора по абсолютной разнице
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: True, если тест информативный
        seed: Начальное значение датчика случайных чисел
        max_attempts: Максимальное количество попыток аугментации для текстов, которые не удалось изменить
        timeout: Максимальное время ожидания одного вызова model.predict в секундах
        retry_delay: Базовая задержка между повторными попытками вызова model.predict в секундах
        exec_env: Среда выполнения ('default' или 'sberds')
        aug_client: OpenAI-совместимый клиент для генерации аугментаций.
        aug_model_name: Название модели для генерации аугментаций
        aug_model_prompts: Опциональный словарь с системными промптами для аугментаций.
                           Допустимые ключи: 'paraphrase', 'synonym', 'translation_fwd', 'translation_bwd'.
                           Если ключ отсутствует или словарь None, используются промпты по умолчанию.
        baseline_per_aug: Опциональный словарь с базовыми метриками
        metrics_augmented: Опциональный словарь с метриками на аугментированных данных
        p_values: Опциональный словарь с рассчитанными p-values
    Return:
        Словарь в соответствии с концепцией llm_val, содержащий отчет и предвычисленные значения
    '''
    aug_col_prefix = 'aug_valtest_metric_aug_stability_word_'
    pred_col_prefix = 'pred_aug_valtest_metric_aug_stability_word_'

    if metrics_augmented is not None and p_values is not None:
        active_baseline_per_aug = baseline_per_aug if baseline_per_aug is not None else {}
        stats_dict = {aug: {'p_value': p_values.get(aug, np.nan),
                            'is_significant': (p_values.get(aug, np.nan) < 0.05) if not pd.isna(p_values.get(aug, np.nan)) else False,
                            'subset_fraction': 0.0}
                      for aug in augmentations_list}
        report = report_valtest_metric_aug_stability_word(
            baseline_per_aug=active_baseline_per_aug, metrics_augmented=metrics_augmented,
            main_metric=main_metric, augmentations_list=augmentations_list, data_type=data_type,
            semaphore_threshold=semaphore_threshold, greater_is_better=greater_is_better,
            is_info=is_info, stats_dict=stats_dict
        )
        result = {'report': report, 'precomputed': {'baseline_per_aug': active_baseline_per_aug, 'metrics_augmented': metrics_augmented, 'p_values': p_values}}
    else:
        augmented_data = get_augmented_data(
            sampler=sampler, data_type=data_type, augmentations_list=augmentations_list,
            seed=seed, aug_client=aug_client, aug_model_name=aug_model_name,
            translation_language=translation_language, word_replace_ratio=word_replace_ratio,
            max_attempts=max_attempts, exec_env=exec_env, aug_model_prompts=aug_model_prompts
        )
        total_size = len(augmented_data)
        checker = resolve_answer_checker(answer_checker)
        baseline_per_aug = {}
        final_metrics_augmented = {}
        stats_dict = {}
        metric_keys = [main_metric]

        for aug_name in augmentations_list:
            aug_col = f'{aug_col_prefix}{aug_name}'
            pred_col = f'{pred_col_prefix}{aug_name}'

            if aug_col not in augmented_data.columns:
                baseline_per_aug[aug_name] = {key: np.nan for key in metric_keys}
                for key in metric_keys:
                    final_metrics_augmented.setdefault(key, {})[aug_name] = np.nan
                stats_dict[aug_name] = {'p_value': np.nan, 'is_significant': False, 'subset_fraction': 0.0}
                continue

            mask_changed = (
                augmented_data['input_query'].astype(str).str.strip() !=
                augmented_data[aug_col].astype(str).str.strip()
            )
            if not mask_changed.any():
                baseline_per_aug[aug_name] = {key: np.nan for key in metric_keys}
                for key in metric_keys:
                    final_metrics_augmented.setdefault(key, {})[aug_name] = np.nan
                stats_dict[aug_name] = {'p_value': np.nan, 'is_significant': False, 'subset_fraction': 0.0}
                continue

            data_changed = augmented_data[mask_changed].copy()

            if 'output_answer' in data_changed.columns:
                score_baseline, raw_baseline = prepare_and_evaluate(
                    data_changed.copy(), scorer, metric_agg, checker, data_type, use_pred=False, pred_col=None
                )
            else:
                score_baseline = {key: np.nan for key in metric_keys}
                raw_baseline = np.full(len(data_changed), np.nan)

            data_aug_temp = data_changed.copy()
            if pred_col in data_aug_temp.columns:
                data_aug_temp['output_answer'] = data_aug_temp[pred_col]
                score_aug, raw_aug = prepare_and_evaluate(
                    data_aug_temp, scorer, metric_agg, checker, data_type, use_pred=False, pred_col=None
                )
            elif model is not None:
                data_aug_temp['input_query'] = data_aug_temp[aug_col]
                data_aug_temp['output_answer'] = call_with_retries(
                    model.predict,
                    timeout=timeout,
                    retry_delay=retry_delay,
                    max_attempts=max_attempts,
                    func_args=(data_aug_temp[aug_col].tolist(),)
                )
                score_aug, raw_aug = prepare_and_evaluate(
                    data_aug_temp, scorer, metric_agg, checker, data_type, use_pred=False, pred_col=None
                )
            else:
                baseline_per_aug[aug_name] = {key: np.nan for key in metric_keys}
                for key in metric_keys:
                    final_metrics_augmented.setdefault(key, {})[aug_name] = np.nan
                stats_dict[aug_name] = {'p_value': np.nan, 'is_significant': False, 'subset_fraction': 0.0}
                continue

            metric_keys = list(score_baseline.keys())
            baseline_per_aug[aug_name] = score_baseline
            for key, value in score_aug.items():
                final_metrics_augmented.setdefault(key, {})[aug_name] = value
            is_sig, p_val = calculate_stat_significance(raw_baseline, raw_aug, alpha=0.05, greater_is_better=greater_is_better)
            subset_fraction = len(raw_baseline) / total_size if total_size > 0 else 0.0
            stats_dict[aug_name] = {'p_value': p_val, 'is_significant': is_sig, 'subset_fraction': subset_fraction}

        precomputed_result = {
            'baseline_per_aug': baseline_per_aug,
            'metrics_augmented': final_metrics_augmented,
            'p_values': {aug: stats_dict[aug]['p_value'] for aug in augmentations_list}
        }
        report = report_valtest_metric_aug_stability_word(
            baseline_per_aug=baseline_per_aug, metrics_augmented=final_metrics_augmented,
            main_metric=main_metric, augmentations_list=augmentations_list, data_type=data_type,
            semaphore_threshold=semaphore_threshold, greater_is_better=greater_is_better,
            is_info=is_info, stats_dict=stats_dict
        )
        result = {'report': report, 'precomputed': precomputed_result}

    return result


def resolve_answer_checker(answer_checker: tp.Optional[tp.Callable[[tp.Any, str, tp.Optional[str]], float]]) -> tp.Callable[[tp.Any, str, tp.Optional[str]], float]:
    '''
    Определяет функцию автооценки, подставляя exact-match по умолчанию
    Parameters:
        answer_checker: Переданная пользователем функция автооценки или None
    '''
    checker = answer_checker if answer_checker is not None else default_answer_checker
    return checker


def default_answer_checker(query: tp.Any,
                           model_answer: str,
                           reference_answer: tp.Optional[str] = None) -> float:
    '''
    Автооценка ответа методом exact-match при наличии эталонного ответа
    Parameters:
        query: Запрос, на который получен ответ
        model_answer: Ответ модели
        reference_answer: Эталонный ответ для сравнения
    Return:
        Оценка корректности ответа в диапазоне от 0 до 1
    '''
    return float(check_if_correct(model_answer, reference_answer)) if reference_answer is not None else 0.0


def calculate_stat_significance(baseline_scores: np.ndarray,
                                aug_scores: np.ndarray,
                                alpha: float = 0.05,
                                greater_is_better: bool = False) -> tp.Tuple[bool, float]:
    '''
    Расчет статистической значимости разницы между базовыми и аугментированными оценками с помощью критерия Уилкоксона
    Parameters:
        baseline_scores: Массив оценок на оригинальных данных
        aug_scores: Массив оценок на аугментированных данных
        alpha: Уровень значимости для проверки гипотезы
        greater_is_better: True, если чем больше метрика, тем лучше
    '''
    mask = ~(np.isnan(baseline_scores) | np.isnan(aug_scores))
    X = baseline_scores[mask]
    Y = aug_scores[mask]

    is_sig = False
    p_val = np.nan

    if len(X) >= 2:
        n_nonzero = int(np.sum(X != Y))
        if n_nonzero > 0:
            method = 'exact' if n_nonzero <= 25 else 'approx'
            alternative = 'greater' if greater_is_better else 'less'
            res = wilcoxon(X, Y, alternative=alternative, method=method)
            is_sig = bool(res.pvalue < alpha)
            p_val = float(res.pvalue)
        else:
            p_val = 1.0

    return is_sig, p_val


def find_metric_value(metrics_dict: tp.Dict[str, float], main_metric: str) -> float:
    '''
    Поиск значения основной метрики в словаре с учетом возможных неточностей в названии ключа
    Parameters:
        metrics_dict: Словарь с рассчитанными метриками
        main_metric: Название искомой ключевой метрики
    Return:
        Значение метрики или np.nan, если метрика не найдена
    '''
    result = np.nan
    if main_metric in metrics_dict:
        result = metrics_dict[main_metric]
    else:
        for key, value in metrics_dict.items():
            if main_metric in key:
                result = value
                break
        else:
            if metrics_dict:
                result = list(metrics_dict.values())[0]
    return result


def get_sampler_data(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                     data_type: str,
                     exec_env: tp.Literal['default', 'sberds']) -> pd.DataFrame:
    '''
    Получает данные из sampler с учетом среды выполнения теста
    Parameters:
        sampler: Объект класса Sampler 
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        exec_env: Среда выполнения 
    '''
    if exec_env == 'sberds':
        sampler_dict = tp.cast(tp.Dict[str, tp.Any], sampler)
        data = sampler_dict[data_type]['X']
    else:
        sampler_object = tp.cast(Sampler, sampler)
        data = getattr(sampler_object, data_type)['X']
    return deepcopy(data)


def get_augmented_data(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                       data_type: str,
                       augmentations_list: tp.List[str],
                       seed: int,
                       aug_client: tp.Optional[tp.Any] = None,
                       aug_model_name: tp.Optional[str] = None,
                       translation_language: str = 'английский',
                       word_replace_ratio: float = 0.1,
                       max_attempts: int = 3,
                       exec_env: tp.Literal['default', 'sberds'] = 'default',
                       aug_model_prompts: tp.Optional[tp.Dict[str, str]] = None) -> pd.DataFrame:
    '''
    Применение словесных аугментаций к исходному набору данных с использованием API-моделей.
    Parameters:
        sampler: Объект класса Sampler 
        data_type: Тип данных (train/val/test)
        augmentations_list: Список применяемых аугментаций
        seed: Начальное значение датчика случайных чисел
        aug_client: OpenAI-совместимый клиент для генерации аугментаций. Опционален, если aug-колонки
                    уже есть в данных.
        aug_model_name: Название модели для генерации аугментаций
        translation_language: Язык перевода для аугментации translation
        word_replace_ratio: Доля слов для замены при синонимической аугментации
        max_attempts: Максимальное количество попыток аугментации для текстов, которые не удалось изменить
        exec_env: Среда выполнения
        aug_model_prompts: Опциональный словарь с системными промптами для аугментаций.
    Return:
        Датафрейм с добавленными колонками, содержащими аугментированные тексты
    '''
    aug_col_prefix = 'aug_valtest_metric_aug_stability_word_'

    data = get_sampler_data(sampler, data_type, exec_env)
    rng = np.random.default_rng(seed)
    paraphrase_prompt = aug_model_prompts.get('paraphrase') if aug_model_prompts else None
    synonym_prompt = aug_model_prompts.get('synonym') if aug_model_prompts else None
    translation_fwd_prompt = aug_model_prompts.get('translation_fwd') if aug_model_prompts else None
    translation_bwd_prompt = aug_model_prompts.get('translation_bwd') if aug_model_prompts else None
    for aug_name in augmentations_list:
        col_name = f'{aug_col_prefix}{aug_name}'
        if col_name in data.columns:
            continue
        if aug_client is None or aug_model_name is None:
            continue
        original_questions = data['input_query'].tolist()
        if aug_name == 'paraphrase':
            data[col_name] = batch_paraphrase_api(
                batch=original_questions, aug_client=aug_client, aug_model_name=aug_model_name,
                system_prompt=paraphrase_prompt, max_attempts=max_attempts
            )
        elif aug_name == 'synonym':
            data[col_name] = batch_synonym_api(
                batch=original_questions, aug_client=aug_client, aug_model_name=aug_model_name,
                rng=rng, word_replace_ratio=word_replace_ratio, system_prompt=synonym_prompt,
                max_attempts=max_attempts
            )
        elif aug_name == 'translation':
            data[col_name] = batch_translation_api(
                batch=original_questions, aug_client=aug_client, aug_model_name=aug_model_name,
                target_language=translation_language,
                system_prompt_fwd=translation_fwd_prompt, system_prompt_bwd=translation_bwd_prompt,
                max_attempts=max_attempts
            )
        else:
            raise ValueError(f"Неизвестный тип аугментации: {aug_name}")
    return data


def prepare_and_evaluate(data: pd.DataFrame,
                         scorer: Scorer,
                         metric_agg: str,
                         answer_checker: tp.Callable[[tp.Any, str, tp.Optional[str]], float],
                         data_type: str,
                         use_pred: bool = False,
                         pred_col: tp.Optional[str] = None) -> tp.Tuple[tp.Dict[str, float], np.ndarray]:
    '''
    Подготовка данных и расчет агрегированной метрики, а также сырых оценок для каждого объекта выборки.
    Parameters:
        data: Исследуемый набор данных
        scorer: Объект класса Scorer
        metric_agg: Название агрегации метрики
        answer_checker: Функция автооценки ответа
        data_type: Тип данных, на которых производится расчет метрики
        use_pred: Флаг использования готовой колонки с предсказаниями
        pred_col: Название колонки с предсказаниями (если use_pred=True)
    Return:
        Кортеж, содержащий словарь с агрегированной метрикой и массив сырых оценок по каждому объекту
    '''
    data = data.copy()
    queries = data['input_query'].tolist()

    if use_pred and pred_col is not None and pred_col in data.columns:
        model_answers = data[pred_col].astype(str).tolist()
    elif 'output_answer' in data.columns:
        model_answers = data['output_answer'].astype(str).tolist()
    else:
        model_answers = None

    if model_answers is not None:
        if 'reference_answer' in data.columns:
            references = data['reference_answer'].tolist()
        else:
            references = [None] * len(queries)

        raw_scores = np.array([
            answer_checker(q, a, r)
            for q, a, r in zip(queries, model_answers, references)
        ], dtype=float)
    else:
        raw_scores = np.full(len(data), np.nan)

    y_for_scorer = pd.DataFrame({'value_metric': raw_scores})
    class _MockSampler:
        pass
    mock_sampler = _MockSampler()
    setattr(mock_sampler, data_type, {'y': y_for_scorer})
    aggregated = scorer.calc(sampler=mock_sampler, data_type=data_type, metric_name=metric_agg)
    return aggregated, raw_scores