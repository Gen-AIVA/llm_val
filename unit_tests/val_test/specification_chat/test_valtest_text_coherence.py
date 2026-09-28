import os
import torch
import pytest
import joblib
import numpy as np
import typing as tp
import pandas as pd
from transformers import AutoModel, AutoTokenizer

from llm_val.sampler.DialogueSampler import DialogueSampler
from llm_val.utils.parse_dialogues import parse_html_dialogue
from llm_val.val_test.specification_chat.valtest_text_coherence import valtest_text_coherence

DATA_PATH = './data/joy_dataset.pkl'
MODEL_PATH = 'data/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

PARAMS = [(DialogueSampler, DATA_PATH)]

class SentenceEmbModel:
    '''
    Обёртка для модели получения эмбеддингов предложений
    '''
    @staticmethod
    def mean_pooling(model_output, 
                     attention_mask):
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def __init__(self, 
                 model_path: str=MODEL_PATH,
                 device: str='cuda'):
        '''
        model_path: Путь к модели
        device: Устройство для вычислений: CUDA или CPU
        '''
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device
        self.model.to(self.device)


    def encode(self, 
               sentences: tp.List[str]) -> np.array:
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            output = self.model(**encoded_input.to(self.device))
        sentence_embeddings = self.mean_pooling(output, encoded_input['attention_mask'])
        return sentence_embeddings.cpu().numpy()


@pytest.fixture(params=PARAMS)
def create_sampler(request):
    sampler_class, data_path = request.param
    data = joblib.load(data_path)
    sampler = sampler_class(test=data)
    return sampler


def test_valtest_text_coherence_result(create_sampler):
    sampler = create_sampler
    res = valtest_text_coherence(sampler=sampler,
                                 model=SentenceEmbModel(),
                                 data_type='test',
                                 quantile=0.95,
                                 n_samples=3,
                                 seed=42,
                                 use_precomputed_emb=False,
                                 statistics=None)
    assert isinstance(res['report'], dict)
    assert isinstance(res['precomputed'], dict)


def test_valtest_text_coherence_keys(create_sampler):
    sampler = create_sampler
    res = valtest_text_coherence(sampler=sampler,
                                 model=SentenceEmbModel(),
                                 data_type='test',
                                 quantile=0.95,
                                 n_samples=3,
                                 use_precomputed_emb=False,
                                 statistics=None)
    assert set(res.keys()) == {'report', 'precomputed'}
    assert set(res['precomputed'].keys()) == {'llm', 'human'}
    assert set(res['report'].keys()) == {'semaphore', 'result_dataframes', 'result_plots'}


def test_valtest_text_coherence_precomputed(create_sampler):
    sampler = create_sampler
    res = valtest_text_coherence(sampler=sampler,
                                 model=SentenceEmbModel(),
                                 data_type='test',
                                 quantile=0.95,
                                 n_samples=3,
                                 use_precomputed_emb=False,
                                 statistics=None)
    precomputed = res['precomputed'] 
    assert isinstance(precomputed['llm'], dict)
    assert isinstance(precomputed['human'], dict)
    assert set(precomputed['llm'].keys()) == {'real_cosine_list', 
                                              'real_cosine_conf',
                                              'rand_cosine_list',
                                              'rand_cosine_conf'}
    assert set(precomputed['human'].keys()) == {'real_cosine_list', 
                                                'real_cosine_conf',
                                                'rand_cosine_list',
                                                'rand_cosine_conf'}
    assert isinstance(precomputed['llm']['real_cosine_list'], list)
    assert isinstance(precomputed['llm']['rand_cosine_list'], list)
    assert isinstance(precomputed['llm']['real_cosine_conf'], list)
    assert isinstance(precomputed['llm']['rand_cosine_conf'], list)
    assert isinstance(precomputed['human']['real_cosine_list'], list)
    assert isinstance(precomputed['human']['rand_cosine_list'], list)
    assert isinstance(precomputed['human']['real_cosine_conf'], list)
    assert isinstance(precomputed['human']['rand_cosine_conf'], list)


def test_valtest_text_coherence_report(create_sampler):
    sampler = create_sampler
    res = valtest_text_coherence(sampler=sampler,
                                 model=SentenceEmbModel(),
                                 data_type='test',
                                 quantile=0.95,
                                 n_samples=3,
                                 use_precomputed_emb=False,
                                 statistics=None)
    report = res['report']
    assert report['semaphore'] == 'gray'
    assert isinstance(report['result_dataframes'][0], pd.DataFrame)
