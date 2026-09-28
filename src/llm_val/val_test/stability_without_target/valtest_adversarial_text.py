import numpy as np
import typing as tp
import pandas as pd
from copy import deepcopy
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold


def report_valtest_adversarial_text(gini_value: float,
                                    semaphore_threshold: tp.Tuple[float, float]=(0.4, 0.8),
                                    is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        gini_value: значение Джини
        semaphore_threshold: пороги на жёлтый и красный светофоры соответственно
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    color = 'gray' if is_info else semaphore_by_threshold(gini_value,
                                                          threshold=semaphore_threshold,
                                                          greater_is_better=False,
                                                          left_border_is_bigger=True,
                                                          right_border_is_bigger=True)
    
    result_dataframe = pd.DataFrame({'Значение Джини': [gini_value], 
                                     'Светофор': [color]})
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_dataframes'] = [result_dataframe]
    res_dict['result_plots'] = []
    return res_dict


def valtest_adversarial_text(sampler: Sampler,
                             semaphore_threshold: tp.Tuple[float, float]=(0.4, 0.8),
                             data_types: tp.Tuple[str, str]=('test', 'oot'),
                             resampling_iterations: int=20,
                             test_size: float=0.3,
                             catboost_iterations: tp.Optional[int]=None,
                             catboost_learning_rate: tp.Optional[float]=None,
                             catboost_depth: tp.Optional[int]=None,
                             random_state: int=17,
                             use_context: bool=False,
                             use_tqdm: bool=False,
                             is_info: bool=False,
                             gini_value: tp.Optional[float]=None,
                             **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест для оценки изменения выборки oot по сравнению с oos
    посредством разделения выборок стандартной моделью
    Parameters:
        sampler: объект класса Sampler
        semaphore_threshold: пороги на жёлтый и красный светофоры соответственно
        data_types: Tuple с используемыми типами данных
        resampling_iterations: количество итераций ресемплинга при усреднении метрики
        test_size: доля наблюдений в тестовой выборке
        catboost_iterations: количество деревьев CatBoost
        catboost_learning_rate: learning_rate CatBoost
        catboost_depth: глубина деревьев CatBoost
        random_state: seed для генератора seed'ов :o)
        use_context: использовать ли контекст вопроса в качестве фичи
        use_tqdm: показывать ли прогресс по итерациям ресемплинга
        is_info: True, если тест информативный
        gini_value: опциональное значение Джини
            Если gini_value не None, подсчет Джини не производится
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if gini_value is None:
        cols_to_use = ['question', 'context'] if use_context else ['question']
        base_data = deepcopy(getattr(sampler, data_types[0])['X'][cols_to_use])
        new_data = deepcopy(getattr(sampler, data_types[1])['X'][cols_to_use])
        if use_context:
            base_data['context'] = base_data['context'].apply(lambda x: ' '.join(x))
            new_data['context'] = new_data['context'].apply(lambda x: ' '.join(x))
        
        gini_values = []
        random_states = np.random.default_rng(seed=random_state).integers(low=0, high=10000, size=resampling_iterations)
        iterator = tqdm(range(resampling_iterations)) if use_tqdm else range(resampling_iterations)
        for iteration in iterator:
            adversarial_dataset = make_adversarial_dataset(base_data, 
                                                           new_data,
                                                           test_size=test_size,
                                                           random_state=random_states[iteration])

            gini_values.append(get_adversarial_gini_score(adversarial_dataset, 
                                                          catboost_iterations=catboost_iterations,
                                                          catboost_learning_rate=catboost_learning_rate,
                                                          catboost_depth=catboost_depth,
                                                          random_state=random_states[iteration]))
        gini_value = np.mean(gini_values) 
        
    precomputed = {'gini_value': gini_value}
    report = report_valtest_adversarial_text(gini_value,
                                             semaphore_threshold,
                                             is_info) 
    return {'report': report,
            'precomputed': precomputed}


def get_adversarial_gini_score(adversarial_dataset: tp.Dict[str, pd.DataFrame],
                               catboost_iterations: tp.Optional[int]=None,
                               catboost_learning_rate: tp.Optional[float]=None,
                               catboost_depth: tp.Optional[int]=None,
                               random_state: int=17) -> float:
    '''
    Функция для получения значения Джини,
    означающего разделяемость выборок простой моделью CatBoost
    Parameters:
        adversarial_dataset: Словарь с train и test выборками
        iterations: Количество деревьев CatBoost
        learning_rate: learning_rate CatBoost
        depth: Глубина деревьев CatBoost
        random_state: random_seed для CatBoost :o)
    Returns:
        Значение Джини
    '''
    train_df = adversarial_dataset['train']
    test_df = adversarial_dataset['test']
    text_features = [col for col in train_df.columns if col != 'target']
    train_pool = Pool(train_df.drop('target', axis=1), train_df['target'], text_features=text_features)
    test_pool = Pool(test_df.drop('target', axis=1), test_df['target'], text_features=text_features)
    
    model = CatBoostClassifier(iterations=catboost_iterations,
                               learning_rate=catboost_learning_rate,
                               depth=catboost_depth,
                               random_seed=random_state, 
                               logging_level='Silent',
                               allow_writing_files=False)
    model.fit(train_pool)
    test_preds = model.predict_proba(test_pool)[:,1]
    gini_value = 2 * roc_auc_score(test_df['target'], test_preds) - 1
    return gini_value


def make_adversarial_dataset(base_data: pd.DataFrame,
                             new_data: pd.DataFrame,
                             test_size: float=0.3,
                             random_state: int=17) -> tp.Dict['str', pd.DataFrame]:
    '''
    Функция для создания датасета с таргетом,
    означающим принадлежность к oos/oot выборке.
    Parameters:
        base_data: List с текстами из oos выборки
        new_data: List с текстами из oot выборки
        test_size: размер тестовой выборки
        random_state: random_seed для train_test_split :o)
    Returns:
        Словарь с train и test выборками
    '''
    base_data['target'] = [0] * base_data.shape[0]
    new_data['target'] = [1] * new_data.shape[0]
    full_df = pd.concat([base_data, new_data], axis=0)
    cols_to_use = [col for col in full_df.columns if col != 'target']
    full_df[cols_to_use] = full_df[cols_to_use].fillna('')
    train_df, test_df = train_test_split(full_df, test_size=test_size, random_state=random_state)
    adversarial_dataset = {'train': train_df,
                           'test': test_df}
    return adversarial_dataset

