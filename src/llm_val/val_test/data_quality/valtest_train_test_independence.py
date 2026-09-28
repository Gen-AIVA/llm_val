import re
import random
import string
import numpy as np
import typing as tp
import pandas as pd
import scipy.sparse
from tqdm import tqdm
from math import floor
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.custom_instructions import mask_option_instructions, mask_word_instructions
from llm_val.utils.report_helper import semaphore_by_threshold, quantile_semaphore, worst_semaphore


def report_valtest_train_test_independence(mask_option_frac: tp.Optional[float]=None,
                                           mask_word_frac: tp.Optional[float]=None,
                                           split_gini_quantile: tp.Optional[float]=None,
                                           mask_option_semaphore_threshold: tp.Tuple[float, float]=(0.01, 0.05),
                                           mask_word_semaphore_threshold: tp.Tuple[float, float]=(0.5, 0.7),
                                           split_gini_semaphore_threshold: tp.Tuple[float, float]=(0.95, 0.99),
                                           test_data_type: str='test',
                                           train_split_data_type: str='val',
                                           split_gini_fact: tp.Optional[float]=None,
                                           mask_fig: tp.Optional[tp.Any]=None,
                                           split_gini_fig: tp.Optional[tp.Any]=None,
                                           is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        mask_option_frac: Опциональное значение доли правильно восстановленных неверных вариантов ответа
        mask_word_frac: Опциональное значение доли правильно восстановленных слов в предложении
        split_gini_quantile: Опциональное значение квантиля gini модели, разделяющей test и train
        mask_option_semaphore_threshold: Границы светофора в эксперименте по маскированию неверных ответов
        mask_word_semaphore_threshold: Границы светофора в эксперименте по восстановлению слов в вопросах
        split_gini_semaphore_threshold: Границы светофора в эксперименте о случайности разбиения выборок
        test_data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        train_split_data_type: Тип данных, используемый в качестве обучающей выборки для векторизатора
        split_gini_fact: Фактическое значение gini, посчитанное на исходном разбиении на train и test
        mask_fig: График распределения доли успешно восстановленных слов
        split_gini_fig: График распределения gini в эксперименте о случайности разбиения выборок
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    checks_list, stats_list, semaphores_list, plots_list = [], [], [], []
    if mask_option_frac is not None:
        checks_list.append('Доля восстановленных маскированных вариантов ответа')
        stats_list.append(mask_option_frac)
        mask_option_semaphore = semaphore_by_threshold(mask_option_frac, 
                                                       mask_option_semaphore_threshold,
                                                       greater_is_better=False)
        semaphores_list.append(mask_option_semaphore)
    if mask_word_frac is not None:
        checks_list.append('Доля восстановленных маскированных слов в тексте вопроса')
        stats_list.append(mask_word_frac)
        mask_word_semaphore_threshold = semaphore_by_threshold(mask_word_frac, 
                                                               mask_word_semaphore_threshold,
                                                               greater_is_better=False)
        semaphores_list.append(mask_word_semaphore_threshold)
        if mask_fig:
            plots_list.append(mask_fig)
    if split_gini_quantile is not None:
        checks_list.append(f'Квантиль Gini модели, разделяющей {test_data_type} и {train_split_data_type}')
        stats_list.append(split_gini_quantile)
        split_gini_semaphore_threshold = quantile_semaphore(split_gini_quantile, 
                                                            split_gini_semaphore_threshold,
                                                            inner_is_better=True)
        semaphores_list.append(split_gini_semaphore_threshold)
        if split_gini_fig:
            plots_list.append(split_gini_fig)
    is_empty_test_list = [mask_option_frac, mask_word_frac, split_gini_quantile]
    is_empty_test = pd.isna(is_empty_test_list).all()
    if is_empty_test:
        res_table = pd.DataFrame({'Проверка': ['Проверки не проводились']})
    else:
        res_table = pd.DataFrame({'Проверка': checks_list,
                                  'Статистика': stats_list,
                                  'Светофор по проверке': semaphores_list})
    res_dict = {}
    res_dict['semaphore'] = 'gray' if is_info else worst_semaphore(semaphores_list)
    res_dict['result_plots'] = plots_list
    res_dict['result_dataframes'] = [res_table]
    return res_dict


def valtest_train_test_independence(sampler: Sampler,
                                    model: tp.Any,
                                    test_data_type: str='test',
                                    train_split_data_type: str='val',
                                    is_mask_option: bool=False,
                                    is_mask_word: bool=True,
                                    is_split_gini_quantile: bool=False,
                                    mask_word_iterations: int=100,
                                    vectorizer_min_df: tp.Union[float, int]=0.005,
                                    vectorizer_max_df: tp.Union[float, int]=0.9,
                                    indep_cv_folds: int=4,
                                    indep_n_iter: int=200,
                                    mask_option_semaphore_threshold: tp.Tuple[float, float]=(0.01, 0.05),
                                    mask_word_semaphore_threshold: tp.Tuple[float, float]=(0.5, 0.7),
                                    split_gini_semaphore_threshold: tp.Tuple[float, float]=(0.95, 0.99),
                                    is_info: bool=False,
                                    seed: int=29,
                                    mask_option_frac: tp.Optional[float]=None,
                                    mask_word_frac: tp.Optional[float]=None,
                                    split_gini_quantile: tp.Optional[float]=None,
                                    **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на независимость тестовой и обучающей выборок
    Parameters:
        sampler: Объект класса Sampler
        model: Валидируемая модель; имеет метод predict(список вопросов)
        test_data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        is_mask_option: Считается ли эксперимент по маскированию неверных ответов
        is_mask_word: Считается ли эксперимент по восстановлению слов в вопросах
        is_split_gini_quantile: Считается ли эксперимент о случайности разбиения выборок
        train_split_data_type: Тип данных, используемый в качестве обучающей выборки для векторизатора
        mask_word_iterations: Число итераций в эксперименте по восстановлению слов в вопросах
        vectorizer_min_df: Параметр min_df для TfidfVectorizer в эксперименте о случайности разбиения выборок
        vectorizer_max_df: Параметр max_df для TfidfVectorizer в эксперименте о случайности разбиения выборок
        indep_cv_folds: Число фолдов при проведении кросс-валидации в эксперименте о случайности разбиения выборок
        indep_n_iter: Число перемешиваний целевой переменной в эксперименте о случайности разбиения выборок
        mask_option_semaphore_threshold: Границы светофора в эксперименте по маскированию неверных ответов
        mask_word_semaphore_threshold: Границы светофора в эксперименте по восстановлению слов в вопросах
        split_gini_semaphore_threshold: Границы светофора в эксперименте о случайности разбиения выборок
        is_info: True, если тест информативный
        seed: Стартовое значение датчика случайных чисел
        mask_option_frac: Опциональное значение доли правильно восстановленных неверных вариантов ответа
            Если не None, то подсчет метрики не производится.
        mask_word_frac: Опциональное значение доли правильно восстановленных слов в вопросах
            Если не None, то подсчет метрики не производится.
            Обратите внимание! График распределения таким образом выведен не будет!
        split_gini_quantile: Опциональное значение квантиля Gini модели, разделяющей test и train
            Если не None, то подсчет метрики не производится.
            Обратите внимание! График распределения таким образом выведен не будет!
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    mask_fig, split_gini_fig, split_gini_fact = None, None, None
    data = getattr(sampler, test_data_type)['X']
    if (mask_option_frac is None) and (is_mask_option):
        mask_option_frac = get_masked_options_fraction(data,
                                                       model)
    if (mask_word_frac is None) and (is_mask_word):
        mask_word_frac, mask_fig = get_mask_word_percent(data,
                                                         model,
                                                         rng,
                                                         mask_word_iterations)
    if (split_gini_quantile is None) and (is_split_gini_quantile):
        gini_statistics, split_gini_fig = get_train_test_indep_stats(sampler,
                                                                     vectorizer_min_df,
                                                                     vectorizer_max_df,
                                                                     indep_cv_folds, 
                                                                     indep_n_iter,
                                                                     test_data_type,
                                                                     train_split_data_type)
        split_gini_fact = gini_statistics['gini_fact']
        split_gini_quantile = gini_statistics['gini_quantile']
    report = report_valtest_train_test_independence(mask_option_frac=mask_option_frac,
                                                    mask_word_frac=mask_word_frac,
                                                    split_gini_quantile=split_gini_quantile,
                                                    mask_option_semaphore_threshold=mask_option_semaphore_threshold,
                                                    mask_word_semaphore_threshold=mask_word_semaphore_threshold,
                                                    split_gini_semaphore_threshold=split_gini_semaphore_threshold,
                                                    test_data_type=test_data_type,
                                                    train_split_data_type=train_split_data_type,
                                                    split_gini_fact=split_gini_fact,
                                                    mask_fig=mask_fig,
                                                    split_gini_fig=split_gini_fig,
                                                    is_info=is_info)
    precomputed = {'mask_option_frac': mask_option_frac,
                   'mask_word_frac': mask_word_frac,
                   'split_gini_quantile': split_gini_quantile}
    return {'report': report,
            'precomputed': precomputed}


def get_masked_options_fraction(data: pd.DataFrame,
                                model: tp.Any) -> float:
    '''
    Проведение эксперимента по маскированию неверных ответов
    Parameters:
        data: Датасет в формате MMLU
        model: Валидируемая модель; имеет метод predict(список вопросов)
    Return:
        Средний процент корректно распознанных моделью неправильных ответов
    '''
    frame_for_masked = pd.DataFrame(columns=['question', 'true_answer'])
    instruction, shot, postfix = split_instruction(mask_option_instructions)
    options = set([key for row in data['answer_options'] for key in row])
    for cur_option in options:
        cur_names_is_exact_list = []
        cur_name_is_exact, cur_name_model_answer = f'is_exact_on_{cur_option}', f'model_answer_on_{cur_option}'
        cur_names_is_exact_list.append(cur_name_is_exact)

        cur_data = data[data['true_answer'] != cur_option].copy()
        cur_data['full_true_answer'] = cur_data.apply(lambda row: exclude_punctuation(row['answer_options'][cur_option]), axis=1)
        cur_data['masked_question'] = cur_data.apply(lambda row: create_full_question_mask_option(row, cur_option), axis=1)
        cur_data['question_to_model'] = cur_data['masked_question'].apply(lambda x: instruction + shot + x + postfix)       
        cur_data[cur_name_model_answer] = model.predict(cur_data['question_to_model'].values.tolist())
        cur_data[cur_name_model_answer] = cur_data[cur_name_model_answer].apply(lambda row: exclude_option_letter(row)).apply(lambda row: exclude_punctuation(row))
        cur_data[cur_name_is_exact] = (cur_data[cur_name_model_answer] == cur_data['full_true_answer']).astype(int)

        frame_for_masked = frame_for_masked.merge(cur_data[['question', 
                                                            'true_answer', 
                                                            cur_name_model_answer, 
                                                            cur_name_is_exact]].copy(deep=True),
                                                  how='outer',
                                                  on=['question', 'true_answer'])
    frame_for_masked['correct_guesses'] = frame_for_masked.fillna(0).apply(lambda row: sum([row[i] for i in cur_names_is_exact_list]), axis=1).astype(int)
    leaked_questions = frame_for_masked[frame_for_masked['correct_guesses'] == (len(options) - 1)]
    mask_option_frac = len(leaked_questions) / len(data)
    return mask_option_frac


def get_mask_word_percent(data: pd.DataFrame,
                          model: tp.Any,
                          rng: tp.Any,
                          mask_word_iterations: int=100) -> tp.Tuple[float, tp.Any]:
    '''
    Проведение эксперимента по восстановлению слов в вопросах
    Parameters:
        data: Датасет со столбцом 'question'
        model: Валидируемая модель; имеет метод predict(список вопросов)
        rng: Датчик случайных чисел
        mask_word_iterations: Число итераций в эксперименте по восстановлению слов в вопросах
    Return:
        Средний процент корректно восстановленных моделью слов в вопросах, 
        график распределения корректно восстановленных моделью слов в вопросах
    '''
    final_masking_result = []
    instruction, shot, postfix = split_instruction(mask_word_instructions)
    data['single_words_list'] = data['question'].apply(lambda x: x.split(' '))
    data['num_words'] = data['single_words_list'].apply(lambda x: len(x))
    for cur_iter in tqdm(range(mask_word_iterations)):
        rand_part = rng.random()
        cur_data = data.copy(deep=True)
        cur_data['masked_question'] = cur_data.apply(lambda row: create_word_in_question(row, rand_part), axis=1)
        cur_data['target_masked'] = cur_data.apply(lambda row: exclude_punctuation(row['single_words_list'][floor(rand_part * row['num_words'])]), axis=1)
        cur_data['question_to_model'] = cur_data['masked_question'].apply(lambda x: instruction + shot + x + postfix)
        cur_data['model_answer'] = model.predict(cur_data['question_to_model'].values.tolist())
        cur_data['model_answer'] = cur_data['model_answer'].apply(lambda row: exclude_punctuation(row))
        cur_data['is_exact'] = (cur_data['model_answer'] == cur_data['target_masked']).astype(int)
        final_masking_result.append(cur_data['is_exact'].mean())
    mask_word_frac = sum(final_masking_result) / len(final_masking_result)
    mask_fig = plot_masking_word(final_masking_result)
    return mask_word_frac, mask_fig


def get_train_test_indep_stats(sampler: Sampler,
                               vectorizer_min_df: tp.Union[float, int]=0.005,
                               vectorizer_max_df: tp.Union[float, int]=0.9,
                               indep_cv_folds: int=4,
                               indep_n_iter: int=200,
                               test_data_type: str='test',
                               train_split_data_type: str='val') -> tp.Dict[str, tp.Any]:
    '''
    Проведение эксперимента о случайности разбиения выборок
    Parameters:
        sampler: объект класса Sampler
        vectorizer_min_df: Параметр min_df для TfidfVectorizer в эксперименте о случайности разбиения выборок
        vectorizer_max_df: Параметр max_df для TfidfVectorizer в эксперименте о случайности разбиения выборок
        indep_cv_folds: Число фолдов при проведении кросс-валидации в эксперименте о случайности разбиения выборок
        indep_n_iter: Число перемешиваний целевой переменной в эксперименте о случайности разбиения выборок
        test_data_type: Тип данных, на которых производится расчет метрики
        train_split_data_type: Тип данных, используемый в качестве обучающей выборки для векторизатора
    Return:
        Статистики по эксперименту о случайности разбиения выборок,
        график распределения gini в эксперименте
    '''
    X_train = getattr(sampler, train_split_data_type)['X']
    X_test = getattr(sampler, test_data_type)['X']
    vectorizer = TfidfVectorizer(min_df=vectorizer_min_df,
                                 max_df=vectorizer_max_df)
    X_train_vectorized = vectorizer.fit_transform(X_train['question'])
    X_test_vectorized = vectorizer.transform(X_test['question'])
    
    gini_statistics = calculate_train_test_indep_gini(X_train_vectorized, 
                                                      X_test_vectorized, 
                                                      indep_cv_folds, 
                                                      indep_n_iter)
    split_gini_fig = plot_indepedence_hist_dist(gini_statistics['gini_fact'], 
                                                gini_statistics['gini_random_list'], 
                                                gini_statistics['gini_quantile'])
    return gini_statistics, split_gini_fig


def split_instruction(instructions: tp.Dict[str, str]) -> tp.Tuple[str, str, str]:
    '''
    Выделение частей из инструкции для формирования вопроса
    Parameters:
        instructions: Словарь с инструкцией для создания промпта
    Return:
        Части промпта для вопроса модели
    '''
    return instructions['instruction'], instructions['shot'], instructions['postfix']


def exclude_option_letter(start_string: str) -> str:
    '''
    Удаление буквы ответа из ответа модели на вопрос MMLU-формата
    Parameters:
        start_string: Строка без удаленной буквы ответа вида 'A) ', '[A] '
    Return:
        Строка с удаленной буквой ответа
    '''
    return re.sub(r'^[^\wА-Яа-я]*[А-Яа-яA-Za-z][^\wА-Яа-я]*\s*', '', start_string)


def exclude_punctuation(start_string: str) -> str:
    '''
    Удаление пунктуации из строки
    Parameters:
        start_string: Строка без удаленной пунктуации
    Return:
        Строка без пунктуации
    '''
    exclude = set(string.punctuation)
    str_wo_punct = ''.join(ch for ch in start_string if ch not in exclude)
    return str_wo_punct


def create_full_question_mask_option(row: pd.DataFrame,
                                     cur_option: str) -> str:
    '''
    Создание вопроса в формате mask_option_instructions
    Parameters:
        row: Строка из датасета формата MMLU
        cur_option: Буква маскируемого варианта ответа
    Return:
        Вопрос в формате mask_option_instructions
    '''
    masked_question = row['question'] + '\nВарианты ответов:\n'
    masked_question += '\n'.join([key + ') ' + value if key != cur_option else key + ') ' + '[...]' for key, value in row['answer_options'].items()])
    return masked_question


def create_word_in_question(row: pd.DataFrame,
                            rand_part: float) -> str:
    '''
    Создание вопроса в формате mask_word_instructions
    Parameters:
        row: Строка из датасета
        rand_part: Случайно сгенерированное число
    Return:
        Вопрос в формате mask_word_instructions    
    '''
    question_in_list = [row['single_words_list'][i] if i != floor(rand_part * row['num_words']) else '[...]' for i in range(row['num_words'])]
    return ' '.join(question_in_list)


def calculate_train_test_indep_gini(X_train: scipy.sparse.spmatrix,
                                    X_test: scipy.sparse.spmatrix,
                                    indep_cv_folds: int=4,
                                    indep_n_iter: int=200) -> tp.Dict[str, tp.Any]:
    '''
    Подсчет статистик распределения gini в эксперименте о случайности разбиения выборок:
    Фактический gini при исходном разбиении;
    Распределение gini после перемешивания обучающей и тестовой выборки;
    Квантиль, в который попадает фактический gini.
    Parameters:
        X_train: Векторизованные вопросы обучающей выборки модели
        X_test: Векторизованные вопросы тестовой выборки модели
        indep_cv_folds: Число фолдов при проведении кросс-валидации
        indep_n_iter: Число перемешиваний целевой переменной для построения распределения gini
    Return:
        Словарь с подсчитанными статистиками (фактический gini, распределение gini и квантиль фактического gini). 
    '''
    res = {}
    X, target = create_data_for_gini(X_train, X_test)
    res['gini_fact'] = calculate_cv_gini(X, target, indep_cv_folds)
    res['gini_random_list'] = calculate_gini_random_list(X, target, indep_cv_folds, indep_n_iter)
    res['gini_quantile'] = min(sum(res['gini_random_list'] <= res['gini_fact']) / len(res['gini_random_list']), 1)
    return res


def create_data_for_gini(X_train: scipy.sparse.spmatrix,
                         X_test: scipy.sparse.spmatrix) -> tp.Tuple[pd.DataFrame, pd.Series]:
    '''
    Объединяет обучающую и тестовую выборки и добавляет новую колонку с вектором из 0 и 1
    (где 0 – объект обучающей выборки, 1 – объект тестовой выборки)
    Parameters:
        X_train: Обучающая выборка
        X_test: Тестовая выборка
    Return:
        DataFrame с данными и Series таргетов
    '''    
    train_label = pd.Series(0, index=np.arange(X_train.shape[0]), name='target')
    test_label = pd.Series(1, index=np.arange(X_test.shape[0]), name='target')
    X = scipy.sparse.vstack([X_train, X_test])
    target = pd.concat([train_label, test_label])
    return X, target


def calculate_gini_random_list(X: pd.DataFrame,
                               target: pd.Series,
                               indep_cv_folds: int=4,
                               indep_n_iter: int=200) -> tp.List[float]:
    '''
    Подсчет Gini на перемешанных выборках.
    Случайным образом перемешиваем целевую переменную, сохраняя исходное соотношение между обучающей и тестовой выборкой.
    Кросс-валидацией получаем indep_cv_folds значений метрики качества модели Gini для каждой из indep_n_iter выборок.
    Parameters:
        X: Обучающая и тестовая функция
        target: Вектор из 0 и 1 (где 0 – объект обучающей выборки, 1 – объект тестовой выборки)
        indep_cv_folds: Количество фолдов перекрестной проверки
        indep_n_iter: Количество итераций(выборок)
    Return:
        Список метрик Gini для каждой выборки
    '''
    
    gini_random = []
    target = np.array(target)
    for _ in tqdm(range(indep_n_iter)):
        np.random.shuffle(target)
        score_cv = calculate_cv_gini(X, target, indep_cv_folds)
        gini_random.append(score_cv)
    gini_random = sorted(gini_random)
    return gini_random


def calculate_cv_gini(X: pd.DataFrame,
                      target: pd.Series,
                      indep_cv_folds: int=4) -> float: 
    '''
    Подсчет Gini методом перекрестной проверки (кросс-валидации)
    Parameters:
        X: Обучающая и тестовая функция
        target: Вектор из 0 и 1 (где 0 – объект обучающей выборки, 1 – объект тестовой выборки)
        indep_cv_folds: Количество фолдов перекрестной проверки
    Return:
        Значение Gini, подсчитанное через кросс-валидацию
    '''
    model = GradientBoostingClassifier()
    roc_auc_fact = cross_val_score(model,
                                   X,
                                   target,
                                   cv=indep_cv_folds,
                                   n_jobs=1,
                                   scoring='roc_auc')
    return (-1 + 2 * roc_auc_fact).mean()


def plot_masking_word(final_masking_result: tp.List) -> tp.Any:
    '''
    Построение графика распределения доли успешно восстановленных слов
    Parameters:
        final_masking_result: Список со средними значениями долей успешно восстановленных слов
    Return:
        График распределения доли успешно восстановленных слов
    '''
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.hist(final_masking_result, bins = 10, color = '#FAC205')
    ax.set_title('Распределение доли вопросов с успешно восстановленными словами')
    ax.set_xlabel('Доля вопросов, где восстановили слово')
    return fig


def plot_indepedence_hist_dist(gini_fact: float,
                               gini_random_list: tp.List[float],
                               quantile: float) -> tp.Any:
    '''
    Построение графика плотности и функции распределения Gini
    Parameters:
        gini_fact: Gini, посчитанный на исходном разбиении данных
        gini_random_list: Список Gini, посчитанных на рандомном разбиении
        quantile: Квантиль фактического Gini
    Return:
        График плотности и функции распределения Gini модели предсказания отнесения к выборке train/out-of-sample
    '''
    
    fig, ax = plt.subplots(2, figsize=(12, 9))
    fig.subplots_adjust(hspace=0.05)
    y_lim_min = -0.05
    y_lim_max = 1.05
    x_lim_min = min(gini_fact, min(gini_random_list)) * 1.05
    x_lim_max = max(gini_fact, max(gini_random_list)) * 1.05
    weights = np.ones(len(gini_random_list)) / len(gini_random_list)

    ax[0].grid(True, linestyle=':')
    y, _, _ = ax[0].hist(gini_random_list,
                         weights=weights,
                         bins=100,
                         alpha=0.5,
                         color='Orange')
    ax[0].set_ylim(0, y.max() * 1.05)
    ax[0].plot([gini_fact] * 2, [0, y.max() * 1.05], color='green', linestyle='--')
    ax[0].set_ylabel('Probability')
    ax[0].set_xlim(x_lim_min, x_lim_max)
    ax[0].set_xticks([])

    ax[1].grid(True, linestyle=':')
    ax[1].set_xlim(x_lim_min, x_lim_max)
    ax[1].set_ylim(y_lim_min, y_lim_max)
    ax[1].set_xlabel('Gini')
    ax[1].set_ylabel('Probability')
    ax[1].plot(gini_random_list, np.linspace(0, 1, len(gini_random_list)))
    ax[1].plot([gini_fact] * 2, [y_lim_min, y_lim_max], color='green', linestyle='--')
    ax[1].plot([x_lim_min, x_lim_max], [quantile] * 2, color='green', linestyle='--')
    ax[1].scatter(gini_fact, quantile, color='green', marker='x', s=200)
    return fig
