import torch
import pytest
import joblib
import numpy as np
import typing as tp
import pandas as pd
from transformers import AutoModel, AutoTokenizer

from llm_val.sampler.DialogueSampler import DialogueSampler
from llm_val.val_test.specification_chat.valtest_nsp_quality import valtest_nsp_quality

DATA_PATH = './data/joy_dataset.pkl'
MODEL_PATH = 'data/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

class SentenceEmbModel:
    '''
    Обёртка для модели получения эмбеддингов предложений
    '''
    @staticmethod
    def mean_pooling(model_output, attention_mask):
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


    def encode(self, sentences: tp.List[str]) -> np.array:
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            output = self.model(**encoded_input.to(self.device))
        sentence_embeddings = self.mean_pooling(output, encoded_input['attention_mask'])
        return sentence_embeddings.cpu().numpy()

PARAMS = [(DialogueSampler, DATA_PATH)]

@pytest.fixture(params=PARAMS)
def create_sampler(request):
    sampler_class, data_path = request.param
    data = joblib.load(data_path)
    sampler = sampler_class(test=data)
    return sampler


def test_valteset_nsp_quality_result(create_sampler):
    sampler = create_sampler
    res = valtest_nsp_quality(sampler=sampler,
                              data_type='test',
                              seed=42,
                              quantile=0.95,
                              use_precomputed_emb=False,
                              toloka_emb=None,
                              model=SentenceEmbModel(),
                              statistics=None) 
    assert set(res.keys()) == {'report', 'precomputed'}
    assert isinstance(res['report'], dict)
    assert isinstance(res['precomputed'], dict)


def test_valtest_nsp_quality_keys(create_sampler):
    sampler = create_sampler
    res = valtest_nsp_quality(sampler=sampler,
                              data_type='test',
                              seed=42,
                              quantile=0.95,
                              use_precomputed_emb=False,
                              toloka_emb=None,
                              model=SentenceEmbModel(),
                              statistics=None) 
    assert set(res['report'].keys()) == {'semaphore', 'result_dataframes', 'result_plots'}
    assert set(res['precomputed'].keys()) == {'correct', 'incorrect'}


def test_valtest_nsp_quality_report(create_sampler):
    sampler = create_sampler
    res = valtest_nsp_quality(sampler=sampler,
                              data_type='test',
                              seed=42,
                              quantile=0.95,
                              use_precomputed_emb=False,
                              toloka_emb=None,
                              model=SentenceEmbModel(),
                              statistics=None) 
    assert res['report']['semaphore'] == 'gray'
    assert isinstance(res['report']['result_dataframes'][0], pd.DataFrame)


def test_valtest_nsp_quality_precomputed(create_sampler):
    sampler = create_sampler
    res = valtest_nsp_quality(sampler=sampler,
                              data_type='test',
                              seed=42,
                              quantile=0.95,
                              use_precomputed_emb=False,
                              toloka_emb=None,
                              model=SentenceEmbModel(),
                              statistics=None) 
    precomputed = res['precomputed']
    assert isinstance(precomputed['correct'], dict)
    assert isinstance(precomputed['incorrect'], dict)
    correct = precomputed['correct']
    incorrect = precomputed['incorrect']
    assert set(correct.keys()) == {'prob_list', 'prob_conf'}
    assert set(incorrect.keys()) == {'prob_list', 'prob_conf'}
    assert isinstance(correct['prob_list'], list)
    assert isinstance(correct['prob_conf'], list)
    assert isinstance(incorrect['prob_list'], list)
    assert isinstance(incorrect['prob_conf'], list)
    