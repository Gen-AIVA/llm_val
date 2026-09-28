import copy
import torch
import random
import numpy as np
import pandas as pd
import typing as tp
import seaborn as sns
from functools import partial
import matplotlib.pyplot as plt

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.paraphraser import batch_parapraser
from llm_val.utils.word_augmentations import batch_swap_word
from llm_val.utils.human_augmentations import batch_augmentations
from llm_val.utils.report_helper import semaphore_by_threshold, worst_semaphore


def report_valtest_answer_reproducibility(diversity_values: tp.Dict[str, float],
                                          main_model: str,
                                          is_semantic_diversity: bool,
                                          data_type: str,
                                          figs: tp.List[tp.Any],
                                          aug_diversity_values: tp.Optional[tp.Dict[str, float]]=None,
                                          semaphore_threshold: tp.Tuple[float]=(0.03, 0.05),
                                          greater_is_better: bool=True,
                                          is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        diversity_values: Словарь со значениями метрик разнообразия
        main_metric: Название ключевой метрики качества. Hint: один из ключей словаря, который выдает scorer
        is_semantic_diversity: Булева переменная, указывает на то, относится ли сэмплер к творческим задачам или нет
        data_type: Тип данных, на которых сравниваются метрики. Hint: название атрибутов в sampler'е
        figs: Список графиков
        aug_diversity_values: Опциональный словарь со значениями метрик разнообразия для всех моделей и наборов данных на аугментированных данных
        semaphore_threshold: Красный и желтый порог светофора по абсолютной разнице
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: Если True, то тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    df = pd.DataFrame({'Разнообразие на исходных данных': [diversity_values[main_model]]})
    if aug_diversity_values is not None:
        df['Разнообразие на аугментированных данных'] = aug_diversity_values[main_model]
    df.index = ['Метрика смыслового разнообразия' if is_semantic_diversity else 'Метрика языкового разнообразия']
    df = df.T
    df['Результат теста'] = [semaphore_by_threshold(value=value,
                                                    threshold=semaphore_threshold,
                                                    greater_is_better=is_semantic_diversity) for value in df.iloc[:, 0]]
    res_dict = {}
    res_dict['semaphore'] = 'gray' if is_info else worst_semaphore(df['Результат теста'])
    res_dict['result_plots'] = figs
    res_dict['result_dataframes'] = [df]
    return res_dict


def valtest_answer_reproducibility(sampler: Sampler,
                                   data_type: str,
                                   models_list: tp.List[tp.Any],
                                   is_semantic_diversity: bool,
                                   n_generated_answers: int=15,
                                   n_gramms_length: int=3,
                                   semaphore_threshold: tp.Tuple[float]=(0.03, 0.05),
                                   greater_is_better: bool=True,
                                   is_info: bool=False,
                                   seed: int=42,
                                   models_names: tp.Optional[tp.List[str]]=None,
                                   main_model: tp.Optional[str]=None,
                                   use_augmentations: bool=True,
                                   augmentations_list: tp.Optional[tp.List[str]]=None,
                                   p_augmented: float=0.1,
                                   paraphrase_model: tp.Optional[tp.Any]=None,
                                   paraphrase_tokenizer: tp.Optional[tp.Any]=None,
                                   paraphrase_model_config: tp.Optional[tp.Dict[str, tp.Any]]=None,
                                   paraphrase_tokenizer_config: tp.Optional[tp.Dict[str, tp.Any]]=None,
                                   semantic_model: tp.Optional[tp.Any]=None,
                                   semantic_tokenizer: tp.Optional[tp.Any]=None,
                                   semantic_model_config: tp.Optional[tp.Dict[str, tp.Dict[str, tp.Any]]]=None,
                                   semantic_tokenizer_config: tp.Optional[tp.Dict[str, tp.Dict[str, tp.Any]]]=None,
                                   diversity_values: tp.Optional[tp.Dict[str, float]]=None,
                                   aug_diversity_values: tp.Optional[tp.Dict[str, float]]=None,
                                   **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на стабильность воспроизведения ответа 
    Parameters:
        sampler: Объект класса Sampler
        data_types: Тип данных, на которых сравниваются метрики. Hint: название атрибутов в sampler'е
        models_list: Список сравниваемых языковых моделей; каждая имеет метод predict(список вопросов)
        is_semantic_diversity: Булева переменная, указывает на то, относится ли сэмплер к творческим задачам или нет
        n_generated_answers: Число генераций ответов 
        n_gramms_length: Длина n-грамм
        semaphore_threshold: Красный и желтый порог светофора по абсолютной разнице
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: True, если тест информативный
        seed: Начальное значение датчика случайных чисел
        models_names: Список названий сравниваемых языковых моделей 
        main_model: Основная валидируемая модель 
        use_augmentations: Флаг, включаем ли мы в тест проверку с аугментациями или нет
        augmentations_list: Список проводимых аугментаций, доступно: 'word_swap' для случайной перестановки слов,
            'paraphrase' для перефразирования, 'use_special_symbol' для аугментации заменой символов на спец. символы 
            и 'use_translit_symbol' для аугментации заменой символов на транслитерацию
        p_augmented: Вероятность аугментации символа/слова/предложения каждой аугментацией
        paraphrase_model: Языковая модель для перефразирования, имеет методы to_device и generate 
        paraphrase_tokenizer: Соответствующий модели токенизатор, имеет методы encode и decode
        paraphrase_model_config: Параметры языковой модели для перефразирования
        paraphrase_tokenizer_config: Параметры токенизатора
        semantic_model: Языковая модель для оценки смысловой близости, имеет методы to_device, get_similarity_matrix и generate 
        semantic_tokenizer: Соответствующий модели токенизатор, имеет методы encode и decode
        semantic_model_config: Параметры языковой модели для оценки смысловой близости
        semantic_tokenizer_config: Параметры токенизатора
        diversity_values: Опциональный словарь со значениями метрик разнообразия для всех моделей и наборов данных.
            Если не None, то подсчет метрик не производится, и на основе переданного аргумента выставляется светофор.
        aug_diversity_values: Опциональный словарь со значениями метрик разнообразия для всех моделей и наборов данных на аугментированных данных.
            Если не None, то подсчет метрик не производится, и на основе переданного аргумента выставляется светофор.
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if augmentations_list:
        use_augmentations=True
    figs = []
    if models_names is None or main_model is None:
        models_names = [f'model_{i}' for i in range(len(models_list))]
        main_model = 'model_0'
    if aug_diversity_values is None and use_augmentations:
        aug_diversity_values = {}
    if diversity_values is None or use_augmentations:
        if diversity_values is None:
            diversity_values = {}
        semantic_diversity = partial(mean_pairwise_semantic_diversity,
                                     model=semantic_model,
                                     tokenizer=semantic_tokenizer,
                                     model_config=semantic_model_config,
                                     tokenizer_config=semantic_tokenizer_config)
        lingustic_diversity = partial(mean_pairwise_lingustic_diversity,
                                      n=n_gramms_length)
        data = copy.deepcopy(getattr(sampler, data_type)['X'])
        diversity_pred_columns = [column for column in data.columns if column.startswith('pred_valtest_answer_reproducibility')]
        aug_columns = [column for column in data.columns if column.startswith('aug_valtest_answer_reproducibility')]
        aug_preds_columns = [column for column in data.columns if column.startswith('pred_aug_valtest_answer_reproducibility')]
        if use_augmentations:
            if aug_columns:
                augmented_data = data[aug_columns]
            else:
                augmented_data = get_augmented_data(sampler,
                                                    data_type,
                                                    augmentations_list,        
                                                    seed,
                                                    p_augmented,
                                                    n_generated_answers,
                                                    paraphrase_model,
                                                    paraphrase_tokenizer,
                                                    paraphrase_model_config,
                                                    paraphrase_tokenizer_config)

        for model_name, model in zip(models_names, models_list):
            diversity_func = semantic_diversity if is_semantic_diversity else lingustic_diversity
            if model_name not in diversity_values:
                model_reproduction_columns = [column for column in diversity_pred_columns if column.endswith(model_name)]
                if model_reproduction_columns:
                    diversity_values[model_name] = evaluate_mean_metric(data[model_reproduction_columns],
                                                                        diversity_func)
                else:
                    diversity_eval_df = get_predictions_for_df(model,
                                                               data[['question']],
                                                               n_generated_answers)
                    
                    diversity_values[model_name] = evaluate_mean_metric(diversity_eval_df,
                                                                        diversity_func)
            if model_name not in aug_diversity_values and use_augmentations:
                model_aug_reproduction_columns = [column for column in aug_preds_columns if column.endswith(model_name)]
                if model_aug_reproduction_columns:
                    aug_diversity_values[model_name] = evaluate_mean_metric(data[model_aug_reproduction_columns],
                                                                            diversity_func)
                else:
                    aug_eval_df = get_predictions_for_df(model,
                                                         augmented_data,
                                                         n_generated_answers)
                    aug_diversity_values[model_name] = evaluate_mean_metric(aug_eval_df,
                                                                            diversity_func)
    task_type = 'смыслового' if is_semantic_diversity else 'языкового'
    figs = [plot_diversity_metrics(diversity_values,
                                   is_semantic_diversity,
                                   f'Изменение метрики {task_type} разнообразия в зависимости от модели')]
    if use_augmentations:
        figs.append(plot_diversity_metrics(aug_diversity_values,
                                           is_semantic_diversity,
                                           f'Изменение метрики {task_type} разнообразия на аугментированных данных в зависимости от модели'))
    precomputed = {'diversity_values': diversity_values,
                   'aug_diversity_values': aug_diversity_values}
    report = report_valtest_answer_reproducibility(diversity_values,
                                                   main_model,
                                                   is_semantic_diversity,
                                                   data_type,
                                                   figs,
                                                   aug_diversity_values,
                                                   semaphore_threshold,
                                                   greater_is_better,
                                                   is_info)
    return {'report': report,
            'precomputed': precomputed}


def get_augmented(questions: tp.List[str],
                  augmentations_list: tp.List[str],
                  rng: tp.Any,
                  p_augmented: float=0.1,
                  model: tp.Optional[tp.Any]=None,
                  tokenizer: tp.Optional[tp.Any]=None,
                  model_config: tp.Optional[tp.Dict[str, tp.Any]]=None,
                  tokenizer_config: tp.Optional[tp.Dict[str, tp.Any]]=None) -> tp.List[str]:
    '''
    Аугментирует список строк
    Parameters:
        questions: Список аугментируемых строк
        augmentations_list: Список проводимых аугментаций, доступно: 'word_swap' для случайной перестановки слов,
            'paraphrase' для перефразирования, 'use_special_symbol' для аугментации заменой символов на спец. символы 
            и 'use_translit_symbol' для аугментации заменой символов на транслитерацию
        rng: Датчик случайных чисел
        p_augmented: Вероятность аугментации символа/слова/предложения каждой аугментацией
        model: Языковая модель для перефразирования, имеет методы to_device и generate 
        tokenizer: Соответствующий модели токенизатор, имеет методы encode и decode
        model_config: Параметры языковой модели для перефразирования
        tokenizer_config: Параметры токенизатора
    Return: 
        Список аугментированных строк
   '''
    for aug_name in augmentations_list:
        if aug_name == 'word_swap':
            questions = batch_swap_word(batch=questions,
                                        rng=rng,
                                        p=p_augmented)
        elif aug_name == 'paraphrase':
            questions = batch_parapraser(batch=questions,
                                         rng=rng,
                                         model=model,
                                         tokenizer=tokenizer,
                                         model_config=model_config,
                                         tokenizer_config=tokenizer_config,
                                         p=p_augmented)
    char_aug_dict = set(['use_special_symbol', 'use_translit_symbol']) & set(augmentations_list)
    aug_params = {par: True for par in list(char_aug_dict)}
    questions = batch_augmentations(batch=questions,
                                    rng=rng,
                                    p=p_augmented,
                                    **aug_params)
    return questions

def get_predictions_for_df(model: tp.Any,
                           data: pd.DataFrame,
                           n_generated_answers: int=15,
                           batch_size: int=128) -> pd.DataFrame:
    '''
    Получение предсказаний для всех элементов целого DataFrame, если передан DataFrame с одной колонкой,
    то создает DataFrame с n_generated_answers колонок
    Parameters:
        model: Языковая модель, имеет метод predict(список вопросов)
        data: DataFrame с запросами в модель
        n_generated_answers: Число генераций ответов
        batch_size: Размер батча
    Return:
        DataFrame с предиктами модели по каждой ячейке из исходного DataFrame, названия колонок те же
    '''
    if len(data.columns) == 1:
        data = pd.concat([data for i in range(n_generated_answers)], axis=1)
    questions_list = list(data.values.reshape(-1))
    answers = []
    for start in range(0, len(questions_list), batch_size):
        end = start + batch_size if start + batch_size < len(questions_list) else len(questions_list)
        answers.extend(model.predict(questions_list[start:end]))
    answers = pd.DataFrame(data=np.array(answers).reshape(len(data), n_generated_answers),
                           columns=data.columns)
    return answers

def evaluate_mean_metric(answers: pd.DataFrame,
                         metric: tp.Callable[[tp.List[str]], float]):
    '''
    Подсчет среднего значения метрики metric по DataFrame, где один столбец - один прогон модели.
    Parameters:
        answers: DataFrame с ответами модели, формат DataFrame: Строка - ответ модели, столбец - конкретный прогон
        metric: Функция для оценки модели
    Return:
        Значение метрики для ответа модели на заданный набор вопросов
    '''
    return np.mean([metric(row.tolist()) for _, row in answers.iterrows()])

def lines_to_ngrams(lines: tp.List[str],
                    n: int=3) -> tp.List[tp.List[tp.Tuple[str, str]]]:
    '''
    Создает список n-грамм для каждой строки из поданного списка
    Parameters:
        lines: Список строк
        n: Длина n-грамм
    Return:
        Список списков n-грамм для каждой строки из поданного списка
    '''
    ngrams = []
    for s in lines:
        words = [e for e in s.replace('.', '').replace('\n', '').split(' ') if e != '']
        ngrams.append([tuple(words[i:(i + n)]) for i in range(len(words) - n + 1)])
    return ngrams


def mean_pairwise_lingustic_diversity(sentences: tp.List[str],
                                      n: int) -> int:
    '''
    Считает среднее попарное языковое разнообразие
    Parameters:
        sentences: Список строк
        n: Длина n-грамм
    Return:
        Языковое разнообразие ответов
    '''
    ngram_lists = lines_to_ngrams(sentences, n)
    n = len(sentences)
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            unite = set(ngram_lists[i] + ngram_lists[j])
            intersection = set(ngram_lists[i]).intersection(set(ngram_lists[j]))
            sim_value = len(intersection)/len(unite) if len(unite) != 0 else 0
            similarities.append(sim_value)
    return 1 - sum(similarities)/len(similarities)


def mean_pairwise_semantic_diversity(sentences: tp.List[str],
                                     model: tp.Any,
                                     tokenizer: tp.Any,
                                     model_config: tp.Dict[str, tp.Any],
                                     tokenizer_config: tp.Optional[tp.Dict[str, tp.Any]]=None) -> int:
    '''
    Считает среднее попарное смысловое разнообразие
    Parameters:
        sentences: Список строк
        model: Языковая модель для оценки смысловой близости предложений, у которой есть метод get_similarity_matrix для подсчета
            матрицы близостей эмбеддингов
        tokenizer: Соответствующий модели токенизатор, имеет методы encode и decode
        model_config: Параметры языковой модели
        tokenizer_config: Параметры токенизатора
    Return:
        Смысловое разнообразие ответов
    '''
    model.to_device()
    embedding = tokenizer.encode(sentences, tokenizer_config['encode_config'])
    similarity = model.get_similarity_matrix(embedding, model_config)
    n = len(sentences)
    return 1 - 2 * torch.triu(similarity, 1).sum().item() / (n * (n - 1))


def plot_diversity_metrics(diversity_values: tp.Dict[str, float],
                           is_semantic_diversity: tp.List[bool],
                           title: tp.Optional[str]=None) -> tp.Any:
    '''
    Строит сравнительную гистограмму по набору метрик разнообразия LLM-кандидатов  
    Parameters:
        diversity_values: Словарь со значениями метрик разнообразия для всех LLM-кандидатов
        is_semantic_diversity: Список булевых переменных, каждая указывает на то, относится ли соответствующий сэмплер к творческим задачам или нет
        title: Опциональная строка заглавия графика
    Return:
        График гистограммы изменения метрики разнообразия в зависимости от LLM-кандидатов
    '''
    data = pd.DataFrame({key:[value] for key, value in diversity_values.items()})
    fig = plt.figure(figsize=(7, 5))
    sns.barplot(data=data, palette='muted')
    plt.xlabel('Название модели', fontsize=10)
    task_type = 'смыслового' if is_semantic_diversity else 'языкового'
    plt.ylabel(f'Метрика {task_type} разнообразия', fontsize=10)
    if title is None:
        title = f'Изменение метрики {task_type} разнообразия в зависимости от модели'
    plt.title(title, fontsize=12, fontweight='bold')
    return fig

def get_augmented_data(sampler: Sampler,
                       data_type: str,
                       augmentations_list: tp.Optional[tp.List[str]]=None,
                       seed: int=42,
                       p_augmented: float=0.1,
                       n_columns: int=10,
                       paraphrase_model: tp.Optional[tp.Any]=None,
                       paraphrase_tokenizer: tp.Optional[tp.Any]=None,
                       paraphrase_model_config: tp.Optional[tp.Dict[str, tp.Any]]=None,
                       paraphrase_tokenizer_config: tp.Optional[tp.Dict[str, tp.Any]]=None) -> pd.DataFrame:
    '''
    Проводит все аугментации над всеми запросами. Возвращает DataFrame из n_columns столбцов, каждый из которых - копия полученных аугментаций
    Parameters:
        sampler: Объект класса Sampler
        data_types: Тип данных, на которых сравниваются метрики. Hint: название атрибутов в sampler'е
        seed: Начальное значение датчика случайных чисел
        augmentations_list: Список проводимых аугментаций, доступно: 'word_swap' для случайной перестановки слов,
            'paraphrase' для перефразирования, 'use_special_symbol' для аугментации заменой символов на спец. символы 
            и 'use_translit_symbol' для аугментации заменой символов на транслитерацию
        p_augmented: Вероятность аугментации символа/слова/предложения каждой аугментацией
        n_columns: Количество генераций в тесте на воспроизводимость
        paraphrase_model: Языковая модель для перефразирования, имеет методы to_device и generate 
        paraphrase_tokenizer: Соответствующий модели токенизатор, имеет методы encode и decode
        paraphrase_model_config: Параметры языковой модели для перефразирования
        paraphrase_tokenizer_config: Параметры токенизатора
    Return:
        DataFrame, колонками которого являются аугментированные тексты, идущие в том же порядке.
    '''

    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    data = getattr(sampler, data_type)['X'].copy()

    augmented_questions = get_augmented(data['question'].to_list(),
                                        augmentations_list,
                                        rng,
                                        p_augmented,
                                        paraphrase_model,
                                        paraphrase_tokenizer,
                                        paraphrase_model_config,
                                        paraphrase_tokenizer_config)
    
    result_df = pd.DataFrame({f'aug_valtest_answer_reproducibility_{idx}': augmented_questions.copy() for idx in range(n_columns)})
    return result_df