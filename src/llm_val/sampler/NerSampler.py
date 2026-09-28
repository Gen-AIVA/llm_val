import typing as tp
import pandas as pd
import numpy as np
from llm_val.sampler.Sampler import Sampler


class NerSampler(Sampler):
    '''
    Реализация Sampler для задач извлечения именованных сущностей (NER)
    '''
    def __init__(self, test: pd.DataFrame, entity_types: tp.List[str]):
        '''        
        Parameters:
            test: Тестовый набор данных
            entity_types: Список типов сущностей для извлечения
        '''
        self.entity_types = entity_types        
        test = self._preproc(test)
        super().__init__(test=test)

    def _preproc(self, var: pd.DataFrame) -> tp.Dict[str, pd.DataFrame]:
        X = pd.DataFrame()
        y = pd.DataFrame()
        
        X['id'] = var['id']
        X['text'] = var['text']
        X['extracted'] = var['extracted']
        X['target'] = var['target']
        X['parser_error'] = X['extracted'].apply(lambda x: x == {})
        
        X['flat_extracted_entities'] = X['extracted'].apply(lambda x: [item for sublist in x.values() for item in sublist])
        X['flat_target_entities'] = X['target'].apply(lambda x: [item for sublist in x.values() for item in sublist])
        
        X['extracted_all_types'] = var['extracted_all_types'] if 'extracted_all_types' in var.columns else X['extracted']
        X['extracted_by_type_merged'] = var['extracted_by_type_merged'] if 'extracted_by_type_merged' in var.columns else X['extracted']
        
        X['extracted_description_detalization'] = X['extracted']
        X['extracted_examples_addition'] = X['extracted']
        X['extracted_description_detalization_and_examples_addition'] = X['target']
        
        y['extracted'] = var['extracted']
        y['target'] = var['target']
        return {'X': X, 'y': y}
