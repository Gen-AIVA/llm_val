import re
import typing as tp
import pandas as pd

from llm_val.scorer.Scorer import Scorer
from llm_val.sampler.Sampler import Sampler
from llm_val.utils.custom_checkers import check_if_correct


def report_valtest_rubq_from_mmlu_metric(metric_value_rubq_from_mmlu: tp.Dict[str, float],
                                         data_type: str='test',
                                         metric_agg: str='single_mean') -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        metric_value_rubq_from_mmlu: Словарь со значениями метрик на сгенерированном RUBQ-датасете
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
    Return:
        report в соответствии с концепцией llm_val
    '''
    res_table = pd.DataFrame(metric_value_rubq_from_mmlu, index=['Значение метрики']).T
    res_dict = {}
    res_dict['semaphore'] = 'gray'
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [res_table]
    return res_dict


def valtest_rubq_from_mmlu_metric(sampler: Sampler,
                                  scorer: Scorer,
                                  model: tp.Any,
                                  rewrite_question: tp.Callable,
                                  data_type: str='test',
                                  metric_agg: str='single_mean',
                                  check_correct_answer: tp.Callable=check_if_correct,
                                  metric_value_rubq_from_mmlu: tp.Optional[float]=None,
                                  **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест подсчета метрики на RuBQ-датасете, сформированном из MMLU-датасета
    Parameters:
        sampler: Объект класса Sampler для задачи MMLU
        scorer: Объект класса Scorer
        model: Валидируемая модель; имеет метод predict(список вопросов)
        rewrite_question: Функция для переделки вопроса MMLU в RuBQ; принимает в себя список вопросов MMLU 
            и возвращает список вопросов для RuBQ
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
        check_correct_answer: Функция для определения правильности ответа RuBQ модели
        metric_value_rubq_from_mmlu: Опциональное значение метрики теста. 
            Если не None, то подсчет метрики не производится.
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if metric_value_rubq_from_mmlu is None:
        data = getattr(sampler, data_type)['X']
        question_list = data['context'].fillna('').apply(lambda x: '. '.join(x)) + '\n' + data['question']
        question_list = rewrite_question(question_list)
        
        data_from_mmlu = pd.DataFrame(columns=['true_answer', 'model_answer'])
        
        data_from_mmlu['true_answer'] = data.apply(lambda row: row['answer_options'][row['true_answer']], axis=1)
        data_from_mmlu['model_answer'] = model.predict(question_list)

        single_metrics = data_from_mmlu.apply(lambda row: check_correct_answer(row['model_answer'], row['true_answer']), axis=1)
        dict_from_mmlu = {'X': None, 'y': pd.DataFrame({'metric_value': single_metrics})}
        tmp_rubq_sampler = Sampler(test=dict_from_mmlu)
        metric_value_rubq_from_mmlu = scorer.calc(sampler=tmp_rubq_sampler,
                                                  data_type='test',
                                                  metric_name=metric_agg)
    report = report_valtest_rubq_from_mmlu_metric(metric_value_rubq_from_mmlu=metric_value_rubq_from_mmlu,
                                                  data_type=data_type,
                                                  metric_agg=metric_agg)
    precomputed = {'metric_value_rubq_from_mmlu': metric_value_rubq_from_mmlu}
    return {'report': report,
            'precomputed': precomputed}
