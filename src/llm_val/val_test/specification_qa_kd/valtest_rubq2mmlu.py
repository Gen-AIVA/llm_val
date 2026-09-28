import random
import pandas as pd
import typing as tp

from llm_val.scorer.Scorer import Scorer
from llm_val.sampler.Sampler import Sampler
from llm_val.sampler.MMLUSampler import MMLUSampler
from llm_val.utils.report_helper import semaphore_by_threshold
from llm_val.utils.custom_instructions import mmlu_instructions

def report_valtest_metric_rubq2mmlu(metric_baseline: float,
                                    mmlu_metric: float,
                                    metric_name: str,
                                    growth_threshold: tp.Tuple[float]=(0.0,5.0),
                                    greater_is_better: bool=True,
                                    is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание отчета по результатам теста.
    Parameters:
        metric_baseline: Значение метрики на исходных данных.
        mmlu_metric: Значение метрики на преобразованных данных.
        metric_name: Название метрики качества.
        growth_threshold: Пороги прироста метрики для выставления светофора в процентных пунктах.
        greater_is_better: True, если чем больше метрика, тем лучше.
        is_info: Если True, отчет будет информативным.
    Returns:
        Отчет в соответствии с концепцией llm_val.
    '''
    growth = (mmlu_metric - metric_baseline) / metric_baseline * 100
    if greater_is_better:
        color = semaphore_by_threshold(value=growth,
                                       threshold=growth_threshold,
                                       greater_is_better=True,
                                       left_border_is_bigger=True,
                                       right_border_is_bigger=True)
    else:
        color = semaphore_by_threshold(value=growth,
                                       threshold=growth_threshold,
                                       greater_is_better=False,
                                       left_border_is_bigger=False,
                                       right_border_is_bigger=False)
    color = 'gray' if is_info else color
    result_df = pd.DataFrame({'Наименование ключевой метрики': [metric_name],
                              'Прирост ключевой метрики (КМ) качества': [f'{growth:.2f}%'],
                              'Результат теста': [color]})

    report = {}
    report['semaphore'] = color
    report['result_plots'] = []
    report['result_dataframes'] = [result_df]
    return report


def valtest_rubq2mmlu(sampler: Sampler,
                      scorer: Scorer,
                      model: tp.Any,
                      model_incorrect_gen: tp.Any,
                      generate_prompt: str,
                      max_attempts: int=5,
                      num_incorrect_options: int=3,
                      metric_name: str='single_mean_metric_value',
                      metric_agg: str='single_mean',
                      growth_threshold: tp.Tuple[float]=(0.0,5.0),
                      greater_is_better: bool=True,
                      data_mmlu_test: tp.Optional[pd.DataFrame]=None,
                      is_info: bool=False,
                      **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Оценивает качество модели на тестовых данных, преобразованных из формата RuBQ в MMLU, и формирует отчет с результатами.    
    Parameters:
        sampler: Объект класса Sampler.
        scorer: Объект класса Scorer.
        model: Валидируемая модель. Выдает ответы на основе сгенерированных вариантов.
        model_incorrect_gen: Модель, используемая для генерации неправильных вариантов.
        generate_prompt: Промпт для генерации неправильных вариантов.
        max_attempts: Максимальное количество попыток для генерации неправильных вариантов.
        num_incorrect_options: Количество неправильных вариантов, которые нужно сгенерировать.
        metric_name: Название метрики качества.
        metric_agg: Название агрегации метрики.
        growth_threshold: Пороги прироста метрики для выставления светофора в процентных пунктах.
        greater_is_better: True, если чем больше метрика, тем лучше.
        data_mmlu_test: Опциональные тестовые данные в формате MMLU. Если параметр предоставлен, то данные будут использованы напрямую для тестирования.
        is_info: Если True, то тест информативный.
    Returns:
        Словарь в соответствии с концепцией llm_val.
    '''
    precomputed_info = {}
    if data_mmlu_test == None:
        data_mmlu_test, precomputed_info = convert_rubq_to_mmlu(sampler.test['X'], 
                                                                model_incorrect_gen, 
                                                                model, 
                                                                generate_prompt, 
                                                                max_attempts, 
                                                                num_incorrect_options)

        data_mmlu_test['query'] = data_mmlu_test.apply(create_query, axis=1)
        data_mmlu_test['true_answer'] = data_mmlu_test['true_answer'].apply(lambda x: f"['{x}']")
        data_mmlu_test = data_mmlu_test.rename(columns={'model_answer': 'answer'})
    
    mmlu_sampler_test = MMLUSampler(data_mmlu_test)
    mmlu_metrics = scorer.calc(mmlu_sampler_test, data_type='test', metric_name=metric_agg)
    rubq_metrics = scorer.calc(sampler, data_type='test', metric_name=metric_agg)
    mmlu_metric = mmlu_metrics.get(metric_name, None)
    rubq_metric = rubq_metrics.get(metric_name, None)
    report = report_valtest_metric_rubq2mmlu(rubq_metric, 
                                             mmlu_metric,
                                             metric_name, 
                                             growth_threshold, 
                                             greater_is_better, 
                                             is_info)
    return {'report': report, 
            'precomputed': precomputed_info}


def generate_incorrect_options(model_incorrect_gen: tp.Any,
                               generate_prompt: str,
                               question: str, 
                               correct_answer: str, 
                               max_attempts: int=5,
                               num_incorrect_options: int=3) -> tp.List[str]:
    '''
    Генерирует список неправильных вариантов ответов на основе правильного ответа.
    Parameters:
        model_incorrect_gen: Модель, используемая для генерации неправильных вариантов.
        generate_prompt: Промпт, который будет использоваться для генерации неправильных вариантов.
        question: Вопрос, на который необходимо сгенерировать неправильные варианты ответов.
        correct_answer: Правильный ответ на вопрос.
        max_attempts: Максимальное количество попыток для генерации неправильных вариантов.
        num_incorrect_options: Количество неправильных вариантов, которые нужно сгенерировать.
    Returns:
        Список из num_incorrect_options неправильных вариантов.
    '''
    generate_prompt = generate_prompt.replace('question', question).replace('correct_answer', str(correct_answer)).replace('num_incorrect_options', str(num_incorrect_options + 2))
    incorrect_options_list = []
    for _ in range(max_attempts):
        incorrect_options = model_incorrect_gen.predict([generate_prompt])[0]
        incorrect_options_list = [option.strip() for option in incorrect_options.split('\\n') if option.strip()]
        if len(incorrect_options_list) >= num_incorrect_options:
            incorrect_options_list = incorrect_options_list[:num_incorrect_options]
            break
    return incorrect_options_list


def convert_rubq_to_mmlu(data_rubq: pd.DataFrame, 
                         model_incorrect_gen: tp.Any, 
                         model: tp.Any, 
                         generate_prompt: str, 
                         max_attempts: int=5,
                         num_incorrect_options: int=3) -> tp.Tuple[pd.DataFrame, tp.Dict[str, tp.Any]]:
    '''
    Преобразует данные из формата RuBQ в MMLU.
    Parameters:
        data_rubq: Входные данные в формате RUBQ.
        model_incorrect_gen: Модель, используемая для генерации неправильных вариантов.
        model: Валидируемая модель. Выдает ответы на основе сгенерированных вариантов.
        generate_prompt: Промпт для генерации неправильных вариантов.
        max_attempts: Максимальное количество попыток для генерации неправильных вариантов.
        num_incorrect_options: Количество неправильных вариантов, которые нужно сгенерировать.
    Returns:
        Кортеж из преобразованных данных в формате MMLU и словаря с дополнительной информацией.
    '''
    mmlu_data = []
    failed_generation_count = 0  
    for index, row in data_rubq.iterrows():
        question = row['question']
        context = row['context']
        true_answer = row['true_answer']
        true_ans = random.choice(true_answer)
        options_list = generate_incorrect_options(model_incorrect_gen, 
                                                  generate_prompt,
                                                  question, 
                                                  true_ans, 
                                                  max_attempts, 
                                                  num_incorrect_options)

        options_list.insert(0, true_ans)
        random.shuffle(options_list)
        correct_answer_index = options_list.index(true_ans)
        correct_answer_letter = chr(ord('A') + correct_answer_index)
        options_prompt = "\n".join([f"{chr(ord('A') + i)} {option}" for i, option in enumerate(options_list)])
        prompt = mmlu_instructions['instruction'] + mmlu_instructions['shot'] + 'Дополнительная информация: ' + str(context) +'\n'
        options_prefix, postfix = mmlu_instructions['options_prefix'], mmlu_instructions['postfix']
        formatted_question = prompt + question + options_prefix + options_prompt + postfix
        mmlu_answer = model.predict([formatted_question])[0]
        mmlu_true_answer = f"{correct_answer_letter}"

        mmlu_data.append({'question': question,
                          'model_answer': mmlu_answer,
                          'true_answer': mmlu_true_answer})
    data_mmlu = pd.DataFrame(mmlu_data)
    failure_rate = failed_generation_count / len(data_rubq) * 100
    precomputed_info = {'failure_rate': failure_rate,
                        'failed_generation_count': failed_generation_count,
                        'total_rows': len(data_rubq)}
    return data_mmlu, precomputed_info


def create_query(row: tp.Dict[str, tp.Any]) -> str:
    question = row['question']
    query = f"Контекст:\n\nВопрос: {question}\nВарианты ответов:\nA) Вариант A\nB) Вариант B\nC) Вариант C\nD) Вариант D\nТвой ответ: [{row['model_answer']}]"
    return query