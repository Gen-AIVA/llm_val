import typing as tp
import pandas as pd
from llm_val.sampler.Sampler import Sampler

class DialogueSampler(Sampler):
    '''
    Пример реализации sampler'а для диалогов между LLM и пользователем
    '''
    def __init__(self, 
                 test: tp.List[tp.List[str]],
                 embeddings: tp.Optional[tp.Any]=None):
        '''
        test: Тестовый набор данных
        '''
        test = self._preproc(test, embeddings)
        super().__init__(test=test)
        
    def _preproc(self, 
                 var: tp.List[tp.List[tp.Dict[str, str]]],
                 embeddings: tp.Optional[tp.Any]=None) -> tp.List[pd.DataFrame]:
        messages = []
        dialogue_ids = []
        pos_ids = []
        roles = []
        for i, dialogue in enumerate(var):
            for j, message in enumerate(dialogue):
                messages.append(message['content'])
                dialogue_ids.append(i)
                pos_ids.append(j)
                roles.append(message['role'])
        emb_list = []
        if embeddings:
            for dialogue in embeddings:
                emb_list.extend(dialogue)
        else:
            emb_list = [None for _ in range(len(messages))]
        X = pd.DataFrame({'dialogue_id': dialogue_ids, 'content': messages, 'embedding': emb_list, 'pos_id': pos_ids})
        y = pd.DataFrame({'dialogue_id': dialogue_ids, 'role': roles, 'pos_id': pos_ids})
        return {'X': X, 'y': y}
    