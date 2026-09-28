import json
import logging
import typing as tp
import ast
import warnings

import pandas as pd

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import (
    semaphore_by_threshold_without_red,
    semaphore_by_threshold_without_yellow,
    worst_semaphore,
)
from llm_val.utils.retries import call_with_retries
from llm_val.utils.openai_utils import get_openai_predictions


logger = logging.getLogger(__name__)

def report_valtest_incorrect_tool_calls(per_tool_stats: pd.DataFrame,
                                        total_error_rate: float,
                                        red_total_rate_threshold: float = 0.30,
                                        yellow_per_tool_threshold: float = 0.50,
                                        is_info: bool = False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а для теста на некорректный вызов функций.
    Parameters:
        per_tool_stats: DataFrame со статистикой по каждому инструменту
        total_error_rate: общая доля некорректных вызовов среди всех инструментов
        red_total_rate_threshold: порог общей доли ошибок для красного светофора
        yellow_per_tool_threshold: порог доли ошибок для отдельного инструмента для жёлтого светофора
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val.
    '''
    final_semaphore = 'gray'
    result_text = 'Тест неприменим: в трейсах отсутствуют вызовы инструментов.'

    if not per_tool_stats.empty:
        overall_semaphore = semaphore_by_threshold_without_yellow(
            value=total_error_rate,
            threshold=red_total_rate_threshold,
            greater_is_better=False,
            border_is_bigger=False,
        )

        tool_semaphores = [
            semaphore_by_threshold_without_red(
                value=error_rate,
                threshold=yellow_per_tool_threshold,
                greater_is_better=False,
                border_is_bigger=False,
            )
            for error_rate in per_tool_stats['error_rate'].tolist()
        ]

        per_tool_semaphore = worst_semaphore(tool_semaphores)

        final_semaphore = worst_semaphore(
            [overall_semaphore, per_tool_semaphore]
        )

        bad_tools_this = per_tool_stats.loc[
            per_tool_stats['error_rate'] > yellow_per_tool_threshold,
            'tool_name',
        ].tolist()

        total_rate_text = (
            f'Общая доля некорректных вызовов: {total_error_rate:.2%}.'
        )

        if final_semaphore == 'green':
            detail_text = (
                'Все инструменты вызываются корректно '
                f'(ни один инструмент не превышает порог '
                f'{yellow_per_tool_threshold:.0%}).'
            )
        elif final_semaphore == 'yellow':
            detail_text = (
                f'Инструменты с долей ошибок выше '
                f'{yellow_per_tool_threshold:.0%}: {bad_tools_this}.'
            )
        else:
            detail_text = (
               'Общая доля некорректных вызовов превышает критический порог '
                f'{red_total_rate_threshold:.0%}.'
            )
            if bad_tools_this:
                detail_text += (
                    f' Инструменты с долей ошибок выше '
                    f'{yellow_per_tool_threshold:.0%}: {bad_tools_this}.'
                )

        result_text = f'{total_rate_text} {detail_text}'

    result_color = 'gray' if is_info else final_semaphore

    result_table = pd.DataFrame(
        {
            'Результат теста': [result_text],
            'Светофор': [final_semaphore],
        }
    )

    result = {
        'semaphore': result_color,
        'result_plots': [],
        'result_dataframes': [per_tool_stats, result_table],
    }

    return result


def valtest_incorrect_tool_calls(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                                 data_type: str = 'test',
                                 judge_client: tp.Optional[tp.Any] = None,
                                 judge_model_name: tp.Optional[str] = None,
                                 check_p3: bool = True,
                                 system_prompt: tp.Optional[str] = None,
                                 temperature: float = 0.0,
                                 max_tokens: int = 8,
                                 exec_env: str = 'default',
                                 signatures: tp.Optional[tp.Dict[str, tp.Dict[str, tp.Any]]] = None,
                                 tool_descriptions: tp.Optional[tp.Mapping[str, str]] = None,
                                 timeout: float = 60.0,
                                 retry_delay: float = 0.1,
                                 max_attempts: int = 3,
                                 is_info: bool = False,
                                 per_tool_stats: tp.Optional[pd.DataFrame] = None,
                                 red_total_rate_threshold: float = 0.30,
                                 yellow_per_tool_threshold: float = 0.50,
                                 **kwargs: tp.Any) -> tp.Dict[str, tp.Any]:
    '''
    Тест на проверку корректности вызова функций (инструментов) агентом.

    Читает трейсы AEF-формата, отфильтровывает спаны с aef_kind='tool',
    разбирает аргументы из input_text и, при однозначном совпадении
    с сигнатурой, извлекает вложенный словарь фактических аргументов
    и вычисляет долю некорректных вызовов — как по каждому инструменту,
    так и в целом.

    Дополнительно (п.3 методики): если переданы OpenAI-compatible клиент
    и model_name, тест также семантически проверяет аргументы
    вызова относительно пользовательского запроса. п.3 не влияет на
    итоговый светофор и является информативной метрикой.

    Parameters:
        sampler: объект класса Sampler
        data_type: ключ, где лежат трейсы агента
        judge_client: OpenAI-compatible клиент для сравнения фактических аргументов с требуемыми в p3
        judge_model_name: Название модели судьи
        check_p3: запускать семантическую проверку P3. По умолчанию True.
            P3 выполняется, только если также переданы judge_client и judge_model_name.
        system_prompt: системный промт для судьи
            Если параметр - None, то используется промт по умолчанию.
            Основные рекомендации при составлении промта:
            На вход судье передаются пользовательский запрос, история диалога при её наличии, имя инструмента и сериализованные аргументы вызова
            Судья оценивает только то, позволяют ли аргументы выполнить намерение пользователя
            Судья не проверяет наличие инструмента в реестре, сигнатуру, типы значений, лишние параметры и JSON-валидность
            Если в контексте недостаточно сведений для выполнения запроса, вердикт — INCORRECT
            Если аргументы включают достаточные конкретизирующие данные либо допустимое значение по умолчанию, вердикт — CORRECT
            Итоговый формат ответа строго ограничен одним словом: CORRECT или INCORRECT
        temperature: температура модели судьи
        max_tokens: максимальное количество токенов в ответе судьи
        exec_env: среда выполнения ('default' или 'sberds')
        signatures: словарь {имя_функции: {параметр: тип}} с эталонными сигнатурами
        tool_descriptions: Опциональные человекочитаемые описания назначения
            инструментов в формате {tool_name: description}. Используются
            только для формирования промта судьи для проверки в п.3.
        timeout: Максимальное время ожидания одного обращения к модели в секундах
        retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
        max_attempts: Максимальное количество попыток обращения к модели
        is_info: True, если тест информативный
        per_tool_stats: предвычисленная статистика по инструментам (опционально)
        red_total_rate_threshold: порог общей доли ошибок для красного светофора
        yellow_per_tool_threshold: порог доли ошибок для отдельного инструмента для жёлтого светофора
    Return:
        Словарь в соответствии с концепцией llm_val.
    '''
    if signatures is None:
        raise ValueError('signatures cannot be None. Pass a dict of {tool_name: {param_name: type}}.')

    if system_prompt is None:
        system_prompt = (
            'Ты — строгий судья корректности фактических аргументов вызова '
            'инструмента в диалоге AI-ассистента.\n\n'
            'Твоя задача — сравнить фактические значения аргументов вызова '
            'с требованиями пользователя, явно выраженными в диалоге, или '
            'однозначно следующими из его контекста.\n\n'
            'Вердикт CORRECT: каждый аргумент, существенный для пользовательского '
            'запроса, имеет значение, которое соответствует требованию пользователя '
            'либо не противоречит ему.\n\n'
            'Вердикт INCORRECT: в диалоге есть явное или однозначно выводимое '
            'требование к значению существенного параметра, но в вызов передано '
            'другое, противоречащее ему значение.\n\n'
            'Пример:\n'
            'Пользователь: «Хочу кредит на 100000 рублей».\n'
            'Аргументы инструмента: {"credit_amount": 100}.\n'
            'Вердикт: INCORRECT, потому что 100 не равно 100000.\n\n'
            'Не проверяй: существование инструмента; его сигнатуру; обязательность '
            'или типы параметров; JSON-формат; лишние аргументы; стиль диалога; '
            'правильность выбора инструмента.\n\n'
            'Не считай ошибкой отсутствие в диалоге информации о значении параметра: '
            'без явного или однозначно выводимого требования нельзя утверждать, '
            'что фактическое значение неверно.\n\n'
            'Ответь строго одним словом: CORRECT или INCORRECT.'
        )
    elif not system_prompt.strip():
        raise ValueError('system_prompt must be a non-empty string or None')

    if per_tool_stats is None:
        traces = get_data(sampler, data_type, 'traces', exec_env).copy()
        per_tool_stats, total_error_rate, tool_call_results = compute_tool_stats(
            traces=traces,
            signatures=signatures,
            sampler=sampler,
            judge_client=judge_client,
            judge_model_name=judge_model_name,
            check_p3=check_p3,
            system_prompt=system_prompt,
            tool_descriptions=tool_descriptions,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retry_delay=retry_delay,
            max_attempts=max_attempts,
            data_type=data_type,
            exec_env=exec_env,
        )
    else:
        total_error_rate = (
                            per_tool_stats['error_count'].sum() / per_tool_stats['total_calls'].sum()
                            if len(per_tool_stats) > 0
                            and per_tool_stats['total_calls'].sum() > 0
                            else 0.0
                        )
        tool_call_results = pd.DataFrame()

    report = report_valtest_incorrect_tool_calls(
        per_tool_stats=per_tool_stats,
        total_error_rate=total_error_rate,
        red_total_rate_threshold=red_total_rate_threshold,
        yellow_per_tool_threshold=yellow_per_tool_threshold,
        is_info=is_info,
    )

    precomputed = {
        'per_tool_stats': per_tool_stats,
        'total_error_rate': total_error_rate,
        'tool_call_results': tool_call_results,
    }

    return {'report': report, 
            'precomputed': precomputed}


def parse_to_dict(input_text: tp.Union[str, tp.Dict[str, tp.Any]]) -> tp.Optional[tp.Dict[str, tp.Any]]:
    '''
    Преобразует input_text в словарь аргументов.
    Сначала пробует разобрать корректный JSON. Если строка содержит
    Python-представление словаря, например с одинарными кавычками,
    пробует ast.literal_eval без замены символов в данных.
    Return:
        Словарь или None, если значение не удалось распознать как dict.
    '''
    result: tp.Optional[tp.Dict[str, tp.Any]] = None

    if isinstance(input_text, dict):
        result = input_text

    elif not isinstance(input_text, str):
        logger.warning(
            'input_text must be str or dict, got %s: %r',
            type(input_text).__name__,
            input_text,
        )

    else:
        parsers: tuple[tp.Callable[[str], tp.Any], ...] = (
            json.loads,
            ast.literal_eval,
        )

        for parser in parsers:
            if result is None:
                try:
                    parsed_value = parser(input_text)
                except (
                    json.JSONDecodeError,
                    SyntaxError,
                    ValueError,
                    TypeError,
                    MemoryError,
                    RecursionError,
                ):
                    continue

                if isinstance(parsed_value, dict):
                    result = parsed_value

    if result is None:
        logger.warning(
            'Failed to parse input_text as a dict: %r',
            input_text,
        )

    return result


def compute_tool_stats(traces: pd.DataFrame,
                    signatures: tp.Dict[str, tp.Dict[str, tp.Any]],
                    sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                    judge_client: tp.Optional[tp.Any] = None,
                    judge_model_name: tp.Optional[str] = None,
                    check_p3: bool = True,
                    system_prompt: tp.Optional[str] = None,
                    tool_descriptions: tp.Optional[tp.Mapping[str, str]] = None,
                    temperature: float = 0.0,
                    max_tokens: int = 8,
                    timeout: float = 60.0,
                    retry_delay: float = 0.1,
                    max_attempts: int = 3,
                    data_type: str = 'test',
                    exec_env: str = 'default') -> tp.Tuple[
                                                            pd.DataFrame,
                                                            float,
                                                            pd.DataFrame,
                                                        ]:
    '''
    Вычисляет статистику по инструментам: число вызовов, ошибки p2 и p3.
    Parameters:
        traces: датафрейм со спанами (должен содержать aef_kind, span_name, input_text, session_id, и т.д.)
        signatures: словарь эталонных сигнатур {имя инструмента: {параметр: тип}}
        sampler: объект sampler (используется в run_p3_check)
        judge_client: OpenAI-compatible клиент судьи для проверки p3
        judge_model_name: название модели судьи
        check_p3: запускать проверку п.3 и учитывать эти ошибки в итоговых error_count/error_rate для светофора.
        system_prompt: систем-промт для судьи, который должен возвращать CORRECT/INCORRECT
        tool_descriptions: Опциональные описания назначения инструментов для проверки п.3.
        temperature: температура
        max_tokens: максимальное количество токенов в ответе
        data_type: ключ данных (train/val/test)
        exec_env: среда выполнения
    Return:
        Кортеж:
        - per_tool_stats: агрегированная статистика по инструментам;
        - total_error_rate: общая доля ошибок для светофора:
            только P2 при check_p3=False, P2 + P3 при check_p3=True;
        - tool_call_results: построчные результаты P2-проверки tool calls.
    '''
    stat_columns = [
        'tool_name',
        'total_calls',
        'error_count',
        'error_rate',
        'p2_error_count',
        'p3_checked_calls',
        'p3_error_count',
        'p3_status',
    ]

    per_tool_stats = pd.DataFrame(columns=stat_columns)
    total_error_rate = 0.0

    tool_call_result_columns = [
        'session_id',
        'span_name',
        'input_text',
        'parsed_args',
        'args_normalization_status',
        'args_source_path',
        'p2_is_correct',
    ]
    tool_call_results = pd.DataFrame(columns=tool_call_result_columns)

    if 'aef_kind' in traces.columns:
        tool_spans = traces.loc[traces['aef_kind'] == 'tool'].copy()
    else:
        warnings.warn(
            'Колонка aef_kind отсутствует в трейсах.'
            ' Фильтрация не применяется — все спаны '
            'рассматриваются как вызовы инструментов.',
            stacklevel=2,
        )
        tool_spans = traces.copy()

    if not tool_spans.empty:
        parsed_results = [
            extract_args(
                input_text=row['input_text'],
                tool_name=row['span_name'],
                signatures=signatures,
            )
            for _, row in tool_spans.iterrows()
        ]

        tool_spans[
                [
                'parsed_args',
                'args_normalization_status',
                'args_source_path',
                ]
        ] = pd.DataFrame(
            parsed_results,
            index=tool_spans.index,
            )

        tool_spans['p2_is_correct'] = tool_spans.apply(
            lambda row: check_call(
                parsed_args=row['parsed_args'],
                tool_name=row['span_name'],
                signatures=signatures,
            ),
            axis=1,
        )

        results: list[dict[str, tp.Any]] = []

        for tool_name, group in tool_spans.groupby('span_name'):
            total_calls = len(group)

            p2_error_count = int((~group['p2_is_correct']).sum())

            p3_status = (
                'disabled'
                if not check_p3
                else 'not_configured'
            )

            p3_result: tp.Dict[str, tp.Any] = {
                'checked_calls': 0,
                'error_count': None,
                'status': p3_status,
            }
            p2_correct_group = group.loc[group['p2_is_correct']].copy()

            if check_p3 and judge_client is not None and judge_model_name is not None:
                p3_result = run_p3_check(
                    sampler=sampler,
                    judge_client=judge_client,
                    group=p2_correct_group,
                    tool_name=tool_name,
                    judge_model_name=judge_model_name,
                    system_prompt=system_prompt,
                    tool_descriptions=tool_descriptions,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    retry_delay=retry_delay,
                    max_attempts=max_attempts,
                    data_type=data_type,
                    exec_env=exec_env,
                )
            
            p3_error_count = p3_result['error_count']

            if check_p3 and p3_error_count is not None:
                error_count = p2_error_count + p3_error_count
            else:
                error_count = p2_error_count

            error_rate = error_count / total_calls if total_calls > 0 else 0.0

            results.append(
                {
                    'tool_name': tool_name,
                    'total_calls': total_calls,
                    'error_count': error_count,
                    'error_rate': error_rate,
                    'p2_error_count': p2_error_count,
                    'p3_checked_calls': p3_result['checked_calls'],
                    'p3_error_count': p3_error_count,
                    'p3_status': p3_result['status'],
                }
            )

        tool_call_results = tool_spans[
            tool_call_result_columns
        ].copy()

        per_tool_stats = (
            pd.DataFrame(results, columns=stat_columns)
            .sort_values('error_rate', ascending=False)
            .reset_index(drop=True)
        )

        total_calls = per_tool_stats['total_calls'].sum()
        total_errors = per_tool_stats['error_count'].sum()

        if total_calls > 0:
            total_error_rate = total_errors / total_calls

    return per_tool_stats, total_error_rate, tool_call_results


def extract_args(input_text: tp.Union[str, tp.Dict[str, tp.Any]],
                 tool_name: str,
                 signatures: tp.Mapping[str, tp.Mapping[str, tp.Any]],
                 ) -> tp.Tuple[tp.Optional[tp.Dict[str, tp.Any]],
                               str,
                               str]:
    '''
    Парсит и нормализует аргументы tool call по сигнатуре.
    Return:
        Кортеж:
        - нормализованные аргументы либо None при ошибке парсинга;
        - статус обработки;
        - путь выбранного словаря.
    '''
    raw_args = parse_to_dict(input_text)

    result_args: tp.Optional[tp.Dict[str, tp.Any]] = None
    result_status = 'invalid_json'
    result_path = '$'

    if raw_args is not None:
        signature_for_tool = signatures.get(tool_name)

        if signature_for_tool is None:
            result_args = raw_args
            result_status = 'unknown_tool'
        else:
            (
                result_args,
                result_status,
                result_path,
            ) = normalize_args_by_signature(
                raw_args=raw_args,
                signature_for_tool=signature_for_tool,
            )

    return result_args, result_status, result_path


def iter_nested_dicts(value: tp.Any,
                      path: tuple[str, ...] = (),
                      ) -> tp.Iterator[tp.Tuple[tp.Dict[str, tp.Any], tuple[str, ...]]]:
    '''Возвращает все словари внутри JSON вместе с путями до них.'''
    if isinstance(value, dict):
        yield value, path

        for key, nested_value in value.items():
            nested_path = path + (str(key),)
            yield from iter_nested_dicts(
                value=nested_value,
                path=nested_path,
            )

    elif isinstance(value, list):
        for index, nested_value in enumerate(value):
            nested_path = path + (f'[{index}]',)
            yield from iter_nested_dicts(
                value=nested_value,
                path=nested_path,
            )


def format_json_path(path: tuple[str, ...]) -> str:
    '''Преобразует путь внутри JSON в строковый вид.'''
    result = '$'

    if path:
        result += '.' + '.'.join(path)

    return result


def normalize_args_by_signature(raw_args: tp.Dict[str, tp.Any],
                                signature_for_tool: tp.Mapping[str, tp.Any],
                                ) -> tp.Tuple[tp.Dict[str, tp.Any], str, str]:
    '''
    Выбирает наиболее вероятный словарь аргументов инструмента.
    Вложенный словарь выбирается только если он единственный лучший
    кандидат и строго лучше корневого словаря по совпадению с сигнатурой.
    Return:
        Кортеж:
        - словарь для P2-проверки;
        - статус нормализации;
        - путь выбранного словаря в исходном JSON.
    '''
    expected_keys = set(signature_for_tool)
    required_keys = {
        param_name
        for param_name, expected_type in signature_for_tool.items()
        if not is_optional(expected_type)
    }

    candidates = list(iter_nested_dicts(raw_args))
    root_candidate = next(
        candidate
        for candidate in candidates
        if candidate[1] == ()
    )

    def score(candidate: tp.Tuple[tp.Dict[str, tp.Any], tuple[str, ...]],) -> tp.Tuple[int, int, int]:
        candidate_dict, _ = candidate
        candidate_keys = set(candidate_dict)

        required_matched = len(candidate_keys & required_keys)
        expected_matched = len(candidate_keys & expected_keys)
        unexpected_count = len(candidate_keys - expected_keys)

        result = (
            required_matched,
            expected_matched,
            -unexpected_count,
        )

        return result

    root_score = score(root_candidate)
    best_score = max(score(candidate) for candidate in candidates)
    best_candidates = [
        candidate
        for candidate in candidates
        if score(candidate) == best_score
    ]

    result_args = raw_args
    result_status = 'root'
    result_path = '$'

    has_single_best_candidate = len(best_candidates) == 1

    has_any_expected_key = any(
    bool(set(candidate_dict) & expected_keys)
    for candidate_dict, _ in candidates
)

    if has_single_best_candidate:
        best_args, best_path = best_candidates[0]
        is_nested = best_path != ()
        improves_root = best_score > root_score
        has_expected_argument = bool(set(best_args) & expected_keys)
        covers_required = (
            not required_keys
            or required_keys.issubset(best_args)
        )

        if (
            is_nested
            and improves_root
            and has_expected_argument
            and covers_required
        ):
            result_args = best_args
            result_status = 'unwrapped'
            result_path = format_json_path(best_path)

    elif best_score > root_score:
        result_status = 'ambiguous'

    if not has_any_expected_key:
        result_status = 'no_match'

    return result_args, result_status, result_path


def is_type_compatible(value: tp.Any, 
                      expected_type: tp.Any) -> bool:
    '''
    Проверяет, совпадает ли тип значения с ожидаемым типом.
    Поддерживает Union и Optional. Если expected_type - object или Any, всегда True.
    Parameters:
        value: значение для проверки
        expected_type: ожидаемый тип (из аннотаций)
    Return:
        True, если тип совместим, иначе False.
    '''
    result = False
    origin = tp.get_origin(expected_type)

    is_bool_for_number = (
        isinstance(value, bool)
        and expected_type in {int, float}
    )

    if is_bool_for_number:
        result = False
    elif expected_type in (object, tp.Any):
        result = True
    elif origin is tp.Union:
        union_types = tp.get_args(expected_type)

        if value is None:
            result = type(None) in union_types
        else:
            non_none_types = tuple(
                item_type
                for item_type in union_types
                if item_type is not type(None)
            )
            result = isinstance(value, non_none_types)
    else:
        result = isinstance(value, expected_type)

    return result


def is_optional(expected_type: tp.Any) -> bool:
    '''
    Проверяет, является ли ожидаемый тип Optional (или Union с None).
    Parameters:
        expected_type: ожидаемый тип
    Return:
        True, если тип optionally содержит None, иначе False.
    '''
    return tp.get_origin(expected_type) is tp.Union and type(None) in tp.get_args(expected_type)


def run_p3_check(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                 judge_client: tp.Optional[tp.Any],
                 group: pd.DataFrame,
                 tool_name: str,
                 judge_model_name: tp.Optional[str],
                 system_prompt: str,
                 tool_descriptions: tp.Optional[tp.Mapping[str, str]] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 8,
                 timeout: float = 60.0,
                 retry_delay: float = 0.1,
                 max_attempts: int = 3,
                 data_type: str = 'test',
                 exec_env: str = 'default') -> tp.Dict[str, tp.Any]:
    '''
    Запускает p3-проверку через OpenAI-compatible судью (пункт 3 теста из методики).
    Parameters:
        sampler: объект класса Sampler (или dict) с данными
        judge_client: OpenAI-compatible клиент судьи
        group: tool-спаны для одного инструмента
        tool_name: имя инструмента
        judge_model_name: название модели судьи
        system_prompt: систем-промт для судьи, который должен возвращать CORRECT/INCORRECT
        tool_descriptions: Опциональные описания назначения инструментов,
        передаваемые LLM-судье для корректной семантической интерпретации вызова.
        temperature: температура
        max_tokens: максимальное количество токенов в ответе
        data_type: ключ, где лежат данные (train/val/test)
        exec_env: среда выполнения
        signatures: словарь сигнатур инструментов
    Return:
        Словарь с результатом P3-проверки:
        - checked_calls: число вызовов с валидным вердиктом;
        - error_count: число INCORRECT среди проверенных вызовов;
        - status: статус выполнения P3-проверки.
    '''
    result: tp.Dict[str, tp.Any] = {
        'checked_calls': 0,
        'error_count': None,
        'status': 'not_configured',
    }

    if judge_client is not None and judge_model_name is not None:
        if group.empty:
            result['status'] = 'no_p2_valid_calls'
        else:
            prompt_data = build_p3_prompts(
                sampler=sampler,
                tool_spans=group,
                data_type=data_type,
                exec_env=exec_env,
                tool_name=tool_name,
                tool_descriptions=tool_descriptions,
            )

            if prompt_data.empty:
                result['status'] = 'no_input_query'
            else:
                raw_answers: list[tp.Optional[str]] = []

                for prompt in prompt_data['prompt'].tolist():
                    raw_answer: tp.Optional[str] = None

                    try:
                        answers = call_with_retries(
                            func=get_openai_predictions,
                            timeout=timeout,
                            retry_delay=retry_delay,
                            max_attempts=max_attempts,
                            func_kwargs={
                                'client': judge_client,
                                'prompts': [prompt],
                                'model': judge_model_name,
                                'system_prompt': system_prompt,
                                'temperature': temperature,
                                'max_tokens': max_tokens,
                            },
                        )

                        raw_answer = answers[0] if answers else None
                    except Exception:
                        logger.exception(
                            'P3 judge call failed after retries. tool=%s',
                            tool_name,
                        )

                    raw_answers.append(raw_answer)

                verdicts = [
                    normalize_judge_verdict(raw_answer)
                    for raw_answer in raw_answers
                ]

                valid_verdicts = [
                    verdict
                    for verdict in verdicts
                    if verdict in {'CORRECT', 'INCORRECT'}
                ]

                result['checked_calls'] = len(valid_verdicts)
                result['error_count'] = sum(
                    verdict == 'INCORRECT'
                    for verdict in valid_verdicts
                )

                if len(valid_verdicts) == len(prompt_data):
                    result['status'] = 'completed'
                elif valid_verdicts:
                    result['status'] = 'partial'
                else:
                    result['status'] = 'failed'

    return result


def dialogue_to_text(dialogue: tp.Any) -> str:
    '''
    Преобразует диалог из кортежей в текстовый формат для LLM-судьи.
    Ожидается список кортежей:
    (query_id, input_query, output_answer).
    Return:
        История диалога без query_id.
    '''
    result = ''

    if isinstance(dialogue, (list, tuple)):
        parts: list[str] = []

        for message in dialogue:
            if isinstance(message, (list, tuple)) and len(message) >= 3:
                _, question, answer = message[:3]

                if question is not None and str(question).strip():
                    parts.append(f'USER: {str(question).strip()}')

                if answer is not None and str(answer).strip():
                    parts.append(f'ASSISTANT: {str(answer).strip()}')

        result = '\n'.join(parts)

    return result


def process_x_merge(spans: pd.DataFrame, 
                    X: pd.DataFrame) -> pd.DataFrame:
    '''
    Присоединяет пользовательский контекст к tool-спанам по session_id.
    Если X содержит несколько строк для одной сессии, непустые значения
    input_query/dialogue объединяются в единый текст в исходном порядке строк.
    '''
    result = spans.copy()
    X_copy = X.copy()

    result['session_id'] = result['session_id'].astype(str)
    X_copy['session_id'] = X_copy['session_id'].astype(str)

    source_field: tp.Optional[str] = None

    if 'dialogue' in X_copy.columns:
        source_field = 'dialogue'
    elif 'input_query' in X_copy.columns:
        source_field = 'input_query'

    if source_field is None:
        logger.warning(
            'X does not contain input_query or dialogue; '
            'setting input_query to NA'
        )
        result['input_query'] = pd.NA
    else:
        X_copy = (
            X_copy
            .groupby('session_id', as_index=False, sort=False)
            .nth(0)
            .copy()
        )

        context_by_session = X_copy[
            ['session_id', source_field]
        ].copy()

        if source_field == 'dialogue':
            context_by_session[source_field] = context_by_session[
                source_field
            ].map(dialogue_to_text)
        else:
            context_by_session[source_field] = (
                context_by_session[source_field]
                .fillna('')
                .astype(str)
                .str.strip()
            )

        context_by_session = context_by_session.loc[
            context_by_session[source_field].ne('')
        ]

        merged = result.merge(
            context_by_session,
            on='session_id',
            how='left',
            indicator=True,
            validate='many_to_one',
        )

        result['input_query'] = merged[source_field].to_numpy()

        missing = merged.loc[
            merged['_merge'] == 'left_only',
            ['session_id', 'span_name'],
        ]

        if not missing.empty:
            logger.warning(
                'Tool-spans with missing %s in X: %d sessions',
                source_field,
                missing['session_id'].nunique(),
            )
            logger.warning(
                'First few missing tool-spans: %s',
                missing.head(10).to_dict('records'),
            )

        non_null_from_x = result['input_query'].notna().sum()
        logger.info(
            'Non-null input_query from X (%s): %d / %d',
            source_field,
            non_null_from_x,
            len(result),
        )

    return result


def build_p3_prompts(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                     tool_spans: pd.DataFrame,
                     data_type: str = 'test',
                     exec_env: str = 'default',
                     tool_name: tp.Optional[str] = None,
                     tool_descriptions: tp.Optional[tp.Mapping[str, str]] = None) -> pd.DataFrame:
    '''
    Формирует контекстные промпты для LLM-судьи (пункт 3 методики).
    Пытается получить input_query из `sampler.X` или dialogue из `sampler.X`.
    Затем создаёт промпты из input_query/dialogue и аргументов инструмента.
    Parameters:
        sampler: объект класса Sampler (или dict) с данными
        tool_spans: датафрейм с tool-спанами для одного инструмента
        data_type: ключ, где лежат данные (train/val/test)
        exec_env: среда выполнения
        tool_name: имя инструмента (для шаблона)
        tool_descriptions: Опциональные описания назначения инструментов
        в формате {tool_name: description}. Используются в промте судьи для проверки п.3.
    Return:
        Датафрейм с колонкой prompt.
    '''
    result = pd.DataFrame(columns=['prompt'])

    spans = tool_spans.copy()
    X = get_data(sampler, data_type, 'X', exec_env)
    spans = process_x_merge(spans, X)

    input_query = spans['input_query'].fillna('').astype(str).str.strip()

    valid_input_query = input_query.ne('')
    spans = spans.loc[valid_input_query].copy()

    if not spans.empty:
        prompts: list[str] = []

        for _, row in spans.iterrows():
            current_tool_name = tool_name or row.get('span_name', '')

            tool_description = None
            if tool_descriptions is not None:
                tool_description = tool_descriptions.get(current_tool_name)

            prompt = make_prompt_from_input_query(
                input_query=str(row['input_query']).strip(),
                tool_name=current_tool_name,
                tool_args_json=json.dumps(
                    row.get('parsed_args', {}),
                    ensure_ascii=False,
                ),
                tool_description=tool_description,
            )
            prompts.append(prompt)

        spans['prompt'] = prompts
        result = spans[['prompt']]

    return result


def make_prompt_from_input_query(input_query: str,
                                 tool_name: str,
                                 tool_args_json: str,
                                 tool_description: tp.Optional[str] = None) -> str:
    '''
    Создаёт промпт для LLM-судьи для проверки вызова инструмента.
    Промт содержит пользовательский контекст, имя инструмента,
    фактически переданные аргументы и опциональное текстовое описание
    назначения инструмента.
    Parameters:
        input_query: пользовательский запрос и беседа
        tool_name: имя инструмента
        tool_args_json: аргументы вызова в формате JSON
        tool_description: опциональное человекочитаемое назначение инструмента
    Return:
        Строка промпта для судьи.
    '''
    description_block = ''

    if tool_description is not None and tool_description.strip():
        description_block = (
            f'Назначение инструмента "{tool_name}":\n'
            f'{tool_description.strip()}\n\n'
        )

    prompt = (
        'Диалог / пользовательский контекст:\n'
        f'{input_query.strip()}\n\n'
        f'Инструмент: {tool_name}\n\n'
        f'{description_block}'
        'Фактические аргументы вызова:\n'
        f'{tool_args_json}\n\n'
        'Сравни фактические значения аргументов с требованиями пользователя. '
        'Верни только CORRECT или INCORRECT.'
    )

    return prompt
    

def normalize_judge_verdict(text: tp.Optional[str]) -> tp.Optional[str]:
    '''Нормализует ответ LLM до CORRECT / INCORRECT.'''
    result: tp.Optional[str] = None

    if text is not None:
        tokens = (
            text.strip()
            .upper()
            .replace('.', ' ')
            .replace(',', ' ')
            .replace('`', ' ')
            .split()
        )
        verdicts = [
            token
            for token in tokens
            if token in {'CORRECT', 'INCORRECT'}
        ]

        if len(verdicts) == 1:
            result = verdicts[0]

    return result


def check_call(parsed_args: tp.Optional[tp.Dict[str, tp.Any]],
               tool_name: str,
               signatures: tp.Dict[str, tp.Dict[str, tp.Any]]) -> bool:
    '''
    Проверяет один вызов инструмента на корректность.
    Правила:
      - Если инструмент отсутствует в signatures -> вызов считается некорректным.
      - Если в parsed_args есть ключи, которых нет в эталонной сигнатуре -> ошибка.
      - Каждый обязательный параметр (присутствующий в эталонной сигнатуре
        и не являющийся Optional) должен быть в parsed_args.
      - Тип переданного значения должен совпадать с ожидаемым типом.
    Parameters:
        parsed_args: плоский словарь аргументов после extract_args
        tool_name: имя вызываемого инструмента
        signatures: эталонные сигнатуры {имя: {параметр: тип}}
    Return:
        True - вызов корректен, False - есть ошибка.
    '''
    is_correct = parsed_args is not None and tool_name in signatures

    if is_correct and parsed_args is not None:
        expected_params = signatures[tool_name]
        unexpected_params = set(parsed_args) - set(expected_params)

        if unexpected_params:
            is_correct = False

        if is_correct:
            for param_name, expected_type in expected_params.items():
                has_param = param_name in parsed_args

                if not has_param and not is_optional(expected_type):
                    is_correct = False
                elif has_param and not is_type_compatible(
                    parsed_args[param_name],
                    expected_type,
                ):
                    is_correct = False

    return is_correct


def get_data(sampler_obj: tp.Union[Sampler, tp.Dict[str, tp.Any]],
             data_type: str,
             key: str,
             exec_env: str='default') -> tp.Any:
    '''
    Служебная функция для получения данных.
    Parameters:
        sampler_obj: Объект сэмплера, содержащий данные для извлечения,
        data_type: Тип данных, на которых производится расчёт метрики,
        key: Ключ или имя атрибута для извлечения конкретных данных,
        exec_env: Переменная исполняемого окружения ('default' или 'sberds'),
    Return:
        Содержимое sampler_obj из data_type по ключу key.
    '''
    result: tp.Any = None

    if exec_env == 'default':
        result = getattr(sampler_obj, data_type)[key]
    elif exec_env == 'sberds':
        dict_sampler = tp.cast(tp.Dict[str, tp.Any], sampler_obj)
        result = dict_sampler[data_type][key]
    else:
        raise ValueError(
            f'Unknown exec_env={exec_env!r}. '
            'Expected one of: default, sberds.'
        )
    return result