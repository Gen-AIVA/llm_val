import torch
import pytest
import pandas as pd
import typing as tp
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from llm_val.model.MultiThreadGigaModel import MultiThreadGigaModel
from llm_val.val_test.middlegame.valtest_pretrain_quality import valtest_pretrain_quality


class ModelHugWrapper:
    def __init__(self,
                 model_name: str,
                 device: str='cpu',
                 half: bool=False):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if half:
            self.model = self.model.half()
        self.device = device

    def generate(self,
                 batch: tp.Any,
                 config: tp.Dict[str, tp.Any]):
        with torch.no_grad():
            return self.model(torch.tensor(batch['input_ids'], device=self.device),
                              torch.tensor(batch['attention_mask'], device=self.device),
                              **config)
    
    def to_device(self):
        self.model.to(self.device)
        self.model.eval()


class TokenizerHugWrapper:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def encode(self, batch: tp.Any, config: tp.Dict[str, tp.Any]):
        return self.tokenizer(batch, **config)


@pytest.fixture()
def create_model(request, api_single_param):
    model_params = {'n_jobs': 8}
    model_params.update(api_single_param)
    model = MultiThreadGigaModel(**model_params)
    return model


def test_valtest_pretrain_quality_pkl(create_model):
    model = create_model
    toxity_model_name = 'apanc/russian-inappropriate-messages'
    toxity_model = ModelHugWrapper(toxity_model_name, device='cuda', half=True)
    toxity_tokenizer = TokenizerHugWrapper(toxity_model_name)
    toxity_model_config = {}
    toxity_tokenizer_config = {'encode_config': {'max_length': 512,
                                                 'padding': 'max_length',
                                                 'truncation': True,
                                                 'return_token_type_ids': False},
                                'decode_config': {'skip_special_tokens': True}}
    literacy_model_name = 'RussianNLP/ruRoBERTa-large-rucola'
    literacy_model = ModelHugWrapper(literacy_model_name, device='cuda', half=True)
    literacy_tokenizer = TokenizerHugWrapper(literacy_model_name)
    literacy_model_config = {}
    literacy_tokenizer_config = {'encode_config': {'max_length': 512,
                                                   'padding': 'max_length',
                                                   'truncation': True,
                                                   'return_token_type_ids': False},
                                'decode_config': {'skip_special_tokens': True}}
    toxity_args = {'model': toxity_model,
                   'tokenizer': toxity_tokenizer,
                   'model_config': toxity_model_config,
                   'tokenizer_config': toxity_tokenizer_config}
    speech_quality_args = {'test_literacy': True,
                           'literacy_model': literacy_model,
                           'literacy_tokenizer': literacy_tokenizer,
                           'literacy_model_config': literacy_model_config,
                           'literacy_tokenizer_config': literacy_tokenizer_config}
    test_res = valtest_pretrain_quality(['./data/data_raw.pkl'],
                                        model,
                                        goal_theme='любая тема',
                                        speech_quality_args=speech_quality_args,
                                        toxity_args=toxity_args)
    report = test_res['report']
    precomputed = test_res['precomputed']
    result_df = report['result_dataframes'][0]
    assert isinstance(test_res, dict)
    assert isinstance(precomputed, dict)
    assert isinstance(report['result_dataframes'][0], pd.DataFrame)
    assert set(test_res.keys()) == {'report', 'precomputed'}
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert set(result_df.columns) == set(['Значение', 'Результат теста'])
    assert set(result_df.index) == set(['Изменчивость во времени',
                                        'Соответствие тематики цели дообучения',
                                        'Токсичность',
                                        'Наличие дубликатов',
                                        'Доля зацикливаний',
                                        'Лингвистическая грамотность',
                                        'Читаемость'])
    assert set(precomputed.keys()) == set(['speech_quality_res',
                                           'consistency_ratio',
                                           'themes_goal_consistency_ratio',
                                           'toxity_ratio',
                                           'duplicates_ratio',
                                           'repeats_ratio'])
