import pytest
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from llm_val.sampler.MMLUSampler import MMLUSampler
from llm_val.sampler.LongAnswerSampler import LongAnswerSampler
from llm_val.model.MultiThreadGigaModel import MultiThreadGigaModel
from llm_val.val_test.stability_without_target.valtest_answer_reproducibility import valtest_answer_reproducibility

DATA_PATH_MMLU = 'data/data_mmlu.pkl'
DATA_PATH_LA = 'data/LongAnswerExample.pkl'

PARAMS = [(MultiThreadGigaModel, {'n_jobs': 8})]


class SemanticModelHugWrapper:
    def __init__(self,
                 model_name: str,
                 device: str='cpu'):
        self.model = SentenceTransformer(model_name)
        self.device = device
     
    def encode(self,
               batch,
               config):
        return self.model.encode(batch)

    def get_similarity_matrix(self,
                              embedding,
                              config):
        return self.model.similarity(embedding, embedding)
    
    def to_device(self):
        self.model.to(self.device)
        self.model.eval()


class ParaphraseModelHugWrapper:
    def __init__(self,
                 model_name: str,
                 device: str='cpu'):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = device
     
    def generate(self,
                 batch,
                 config):
        return self.model.generate(**batch, **config)
     
    def to_device(self):
        self.model.to(self.device)
        self.model.eval()


class ParaphraseTokenizerHugWrapper:
    def __init__(self,
                 model_name: str,
                 device: str='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.device = device
     
    def decode(self,
               batch,
               config):
        return self.tokenizer.decode(batch.cpu().tolist(), **config)
                                                                                 
    def encode(self,
               batch,
               config):
        return self.tokenizer(batch, **config)


paraphrase_model_config = {'num_beams': 1,
                           'encoder_no_repeat_ngram_size': 3,
                           'do_sample': False}

paraphrase_tokenizer_config = {'decode_config': {'skip_special_tokens': True},
                               'encode_config': {'padding': True,
                                                 'return_tensors': 'pt'}}

semantic_model_config = {'encode_config':{}}

@pytest.fixture(params=PARAMS)
def create_sampler_scorer_model(request, api_single_param):
    model_class, model_params = request.param
    model_params.update(api_single_param)
    model = model_class(model='GigaChat:latest', **model_params)
    paraphrase_model = ParaphraseModelHugWrapper('cointegrated/rut5-base-paraphraser', device='cuda')
    semantic_model = SemanticModelHugWrapper('cointegrated/rubert-tiny2', device='cuda')
    paraphrase_tokenizer = ParaphraseTokenizerHugWrapper('cointegrated/rut5-base-paraphraser')
    return model, paraphrase_model, semantic_model, paraphrase_tokenizer


def test_valtest_answer_reproducibility_mmlu(create_sampler_scorer_model):
    model, paraphrase_model, semantic_model, paraphrase_tokenizer = create_sampler_scorer_model
    data_mmlu = joblib.load(DATA_PATH_MMLU)
    sampler_mmlu = MMLUSampler(data_mmlu)
    test_res = valtest_answer_reproducibility(sampler=sampler_mmlu,
                                              data_type='test',
                                              models_list=[model, model],
                                              is_semantic_diversity=False,
                                              augmentations_list=['paraphrase', 'use_special_symbol', 'use_translit_symbol'],
                                              n_generated_answers=10, 
                                              p_augmented=0.2,
                                              models_names=['gigachat_0', 'gigachat_1'],
                                              main_model='gigachat_0',
                                              paraphrase_model=paraphrase_model,
                                              paraphrase_model_config=paraphrase_model_config,
                                              paraphrase_tokenizer=paraphrase_tokenizer,
                                              paraphrase_tokenizer_config=paraphrase_tokenizer_config)
    precomputed = test_res['precomputed']
    report = test_res['report']
    report_df = report['result_dataframes'][0]
    models_names = ['gigachat_0', 'gigachat_1']
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(report_df, pd.DataFrame)
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'diversity_values', 'aug_diversity_values'}
    assert list(precomputed['diversity_values'].keys()) == models_names
    assert list(precomputed['aug_diversity_values'].keys()) == models_names


def test_valtest_answer_reproducibility_la(create_sampler_scorer_model):
    model, paraphrase_model, semantic_model, paraphrase_tokenizer = create_sampler_scorer_model
    data_la = joblib.load(DATA_PATH_LA)
    sampler_la = LongAnswerSampler(data_la)
    test_res = valtest_answer_reproducibility(sampler=sampler_la,
                                              data_type='test',
                                              models_list=[model, model],
                                              is_semantic_diversity=True,
                                              augmentations_list=['word_swap', 'use_special_symbol', 'use_translit_symbol'],
                                              n_generated_answers=5, 
                                              p_augmented=0.2,
                                              models_names=['gigachat_0', 'gigachat_1'],
                                              main_model='gigachat_1',
                                              semantic_model=semantic_model,
                                              semantic_tokenizer=semantic_model,
                                              semantic_model_config=semantic_model_config,
                                              semantic_tokenizer_config=semantic_model_config)
    precomputed = test_res['precomputed']
    report = test_res['report']
    report_df = report['result_dataframes'][0]
    models_names = ['gigachat_0', 'gigachat_1']
    assert test_res['report']['semaphore'] == 'green'
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(report_df, pd.DataFrame)
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'diversity_values', 'aug_diversity_values'}
    assert list(precomputed['diversity_values'].keys()) == models_names
    assert list(precomputed['aug_diversity_values'].keys()) == models_names
