import sys
import pytest
import random
import joblib
import numpy as np
import pandas as pd
import typing as tp
from copy import deepcopy
from simhash import Simhash
from tqdm.auto import tqdm as tqdm
from sentence_transformers import util

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold
from llm_val.sampler.OOTLongAnswerSampler import OOTLongAnswerSampler
from llm_val.val_test.stability.valtest_share_of_contradictions import valtest_share_of_contradictions

DATA_PATH_LongAnswer = 'data/LongAnswerExample.pkl'
PARAMS = [(OOTLongAnswerSampler, DATA_PATH_LongAnswer, ('test','oot'), None, (0.01, 0.05), 0.8, False)]


def curse_data(df):
    cursed_data = df.copy(deep = True)
    cursed_data.loc[cursed_data['id_task'] == 0, 'question'] = 'Есть ли хороший международный интернет-эквайринг?'
    cursed_data.loc[cursed_data['id_task'] == 1, 'question'] = 'Какой обучающий курс мне выбрать для того, чтобы научиться писать на c#?'
    cursed_data.loc[cursed_data['id_task'] == 2, 'question'] = 'Как перенести строки на языке PHP?'
    cursed_data.loc[cursed_data['id_task'] == 3, 'answer'] = 'Как говорилось, все языки работают с ОЗУ, и почти все дают работать с ней напрямую.\nПросто большинство языков имеют уже свою систему управления памятью, например GC, а в си нет ни сборщика мусора, и RAII, ни borrow checker'
    cursed_data.loc[cursed_data['id_task'] == 3, 'question'] = 'За счет чего и почему язык Си может работать с низкоуровневым слоем?'
    cursed_data.loc[cursed_data['id_task'] == 4, 'answer'] = 'Похоже, что ваш код неправильно отформатирован.\n Вы также забыли преобразовать тип в начале.\n\nВот рабочий код:\nposition=input("Choose a position: ")\nposition=int(position)\n       \nwhile position not in list(range(1, 10)):\n    print("ONLY between 1 and 9: ")\n    position=input("Choose a position: ")\n    position=int(position)\n    \n    print(f\'chosen position is {position}\')'
    cursed_data.loc[cursed_data['id_task'] == 4, 'question'] = 'Как сделать выход из цикла while?'
    cursed_data.loc[cursed_data['id_task'] == 5, 'answer'] = 'Как решить проблему перегрева ноутбука? Можно попробовать приобрести подставку с вентилятором. Не ставить на поверхности затрудняющие теплообмен.Мой ноутбук после 5 мин работы перегревается, снизу выходит горячий воздух но программы работают обычноЗначит он у вас нагревается, а не перегревается. Это нормальная ситуация.\nВот если бы программы тормозили а воздух выходил не очень горячий это было бы хуже.\nГорячий воздух это показатель того что система охлаждения работает, и работает корректно.\nПросто система охлаждения небольшая, а греется он прилично, но тут уж ничего не сделаешь, законы физики, техника при работе греется.'
    cursed_data.loc[cursed_data['id_task'] == 5, 'question'] = 'Греется ноутбук, как решить проблему?'
    
    random.seed(42)
    def change_assessor_answers(df, num):
        indices = df[df['id_task'] == num].index
        features = ['Вы согласны с замечаниями и проблемами, описанными в CodeReview?',
           'Все существенные проблемы перечислены в CodeReview?',
           'CodeReview не содержит токсичных комментариев (ненормативная лексика, оскорбительные высказывания или намеки)?',
           'Ответ лаконичен и по существу?',
           'Ответ структурирован, все параметры раскрыты?',
           'Как вы оцениваете качество CodeReview?']
        for idx in indices:
            rand_for_feature = random.randint(0, 5)
            feature = features[rand_for_feature]
            df.loc[idx, feature] = random.randint(1, 6)
    cursed_data.loc[cursed_data['id_task'] == 6, 'answer'] = ' Мне это не интересно'
    cursed_data.loc[cursed_data['id_task'] == 6, 'question'] = 'Как привязать footer к нижней части страницы?'
    change_assessor_answers(cursed_data, 6)
    change_assessor_answers(cursed_data, 6)
    change_assessor_answers(cursed_data, 6)
    cursed_data.loc[cursed_data['id_task'] == 7, 'answer'] = 'Не знаю что тебе ответить.'
    cursed_data.loc[cursed_data['id_task'] == 7, 'question'] = 'Что нужно иметь и знать в фреймворке React, когда ты джун?'
    change_assessor_answers(cursed_data, 7)



    cursed_data.loc[cursed_data['id_task'] == 8, 'answer'] = 'Только по типу устройств. вы можете отфильтровать только в ретаргетинге ОС'
    cursed_data.loc[cursed_data['id_task'] == 8, 'question'] = 'Возможно ли отключить показы на Windows в Директе и Ads?'
    change_assessor_answers(cursed_data, 8)
    change_assessor_answers(cursed_data, 8)
    change_assessor_answers(cursed_data, 8)
    np.random.seed(42)
    def scramble_text(text):
        chars = list(str(text))
        np.random.shuffle(chars)
        return ''.join(chars)
    cursed_data['question'] = cursed_data.apply(lambda x: scramble_text(x['question']) if x['id_task'] > 8 else x['question'], axis = 1)
    df = df.drop('prompt', axis = 1)
    cursed_data = cursed_data.drop('prompt', axis = 1)
    cursed_data = cursed_data.rename(columns = {'question': 'prompt'})
    df = df.rename(columns = {'question': 'prompt'})
    return df, cursed_data
    
@pytest.fixture(params=PARAMS)
def create_data_sampler(request, api_single_param):
    sampler_cls, data_path, data_types, embedding_model, threshold, threshold_duplicates, is_info  = request.param
    data = joblib.load(data_path)
    data, cursed_data = curse_data(data)
    sampler = sampler_cls(test=data, 
                          oot=cursed_data)

    res = valtest_share_of_contradictions(sampler,
                                          data_types,
                                          embedding_model,
                                          threshold,
                                          threshold_duplicates,
                                          is_info)    
    return res

def test_valtest_share_of_contradictions(create_data_sampler):
    test_res = create_data_sampler
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}
    
def test_valtest_report(create_data_sampler):
    test_res = create_data_sampler
    report = test_res['report']
    result_df = report['result_dataframes'][0]
    assert isinstance(result_df, pd.DataFrame)
    assert result_df['Доля семантических дубликатов вопросов в OOT относительно OOS'].iloc[0] == 0.06
    assert result_df['Доля семантических дубликатов ответов LLM на похожие вопросы в OOT относительно OOS'].iloc[0] == 0.04
    assert np.isclose(result_df['Доля противоречий меток асессоров в вопросах (допустимы разные ответы)'].iloc[0], 0.0266666)
    assert result_df['Доля противоречий меток ассесоров в одних и тех же ответах на те же вопросы'].iloc[0] == 0.5

       
    
def test_valtest_precomputed(create_data_sampler):
    test_res = create_data_sampler
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert precomputed['Таблица пар расстояний question']['similarity'].iloc[0] == 0.90625
    assert np.isclose(precomputed['Таблица противоречий асессоров по критериям']['Средняя ошибка'].iloc[0], 0.0266666)
    