import numpy as np
import pandas as pd
from typing import Dict, List, Any
from unittest.mock import MagicMock

from llm_val.val_test.stability_without_target.valtest_rag_top_stability import valtest_rag_top_stability

class MockSampler:
    def __init__(self, 
                 test_data):
        self.test = {'X': test_data}

class MockRAGExecutor:
    def __init__(self, 
                 topK):
        self.params = {'topK': topK}
    
    def get_parameters(self):
        return self.params
    
    def get_topK_passages(self, 
                          questions, 
                          topK):
        result = {}
        for question in questions:
            result[question] = {'documents': [f'passage{i+1}' for i in range(topK)],
                                'embeddings': [np.random.rand(768) for _ in range(topK)]}
        return result

def mock_batch_swap_word(batch, 
                         rng, 
                         p):
    return [q[::-1] for q in batch]

def mock_batch_augmentations(batch, 
                             rng, 
                             p, 
                             **kwargs):
    return [q + ' augmented' for q in batch]

test_data = pd.DataFrame({'question': ['What is AI?', 'What is ML?']
})

def test_valtest_rag_top_stability():
    sampler = MockSampler(test_data)
    executor = MockRAGExecutor(topK=3)
    
    augmentations_list = ['word_swap', 'symbol_swap']
    p_augmented = 0.1
    seed = 42
    
    with MagicMock() as mock:
        mock.batch_swap_word.side_effect = mock_batch_swap_word
        mock.batch_augmentations.side_effect = mock_batch_augmentations
        
        result = valtest_rag_top_stability(sampler=sampler,
                                           executor=executor,
                                           augmentations_list=augmentations_list,
                                           p_augmented=p_augmented,
                                           data_type='test',
                                           seed=seed,
                                           paraphrase_model=None,
                                           paraphrase_tokenizer=None,
                                           paraphrase_model_config=None,
                                           paraphrase_tokenizer_config=None
        )
    
    assert 'report' in result
    assert 'precomputed' in result
    
    report = result['report']
    assert 'semaphore' in report
    assert 'result_plots' in report
    assert 'result_dataframes' in report
    
    precomputed = result['precomputed']
    assert 'strict_strategy_results' in precomputed
    assert 'soft_strategy_results' in precomputed
    
    strict_results = precomputed['strict_strategy_results']
    soft_results = precomputed['soft_strategy_results']
    
    topK = executor.get_parameters()['topK']
    for k in range(1, topK + 1):
        assert k in strict_results
        assert k in soft_results
        assert 0 <= strict_results[k] <= 1, f"Strict score for k={k} is out of range: {strict_results[k]}"
        assert 0 <= soft_results[k] <= 1, f"Soft score for k={k} is out of range: {soft_results[k]}"