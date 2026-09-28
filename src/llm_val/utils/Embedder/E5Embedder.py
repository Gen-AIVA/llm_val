import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Dict, Any

from llm_val.utils.Embedder.Embedder import Embedder


class E5Embedder(Embedder):
    """
    Класс для получения эмбеддингов текста на русском языке
    с использованием модели multilingual-e5-large

    Parameters:
        model_path: Путь до директории с моделью
        config_path: Путь до глобального конфигурационного файла
        use_text_cleaner: Флаг, поднятие которого обеспечит
                          предобработку сессий
        allow_gpu: Флаг, поднятие которого разрешает использование gpu
    """
    def __init__(self,
                 model_path: str,
                 encode_config: Dict[str, Any],
                 use_text_cleaner: bool = False,
                 allow_gpu: bool = True):
        super().__init__()
        self.model_path = model_path
        self.allow_gpu = allow_gpu
        self.model = None
        self.normalize = None
        self.use_text_cleaner = use_text_cleaner
        self.encode_config = encode_config
        self.__init_model()

    def __init_model(self):
        """
        Инициализация токенайзера и самой модели из файлов,
        находящихся в переданной директории
        """
        use_gpu = torch.cuda.is_available() and self.allow_gpu
        self.device = ('cpu', 'cuda:0')[use_gpu]
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        self.model.eval()

    def __average_pool(self,
                       last_hidden_states: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Пулинг эмбеддингов.
        Рекомендован к применению создателями модели.

        Parameters:
            last_hidden_states: Последнее скрытое состояние
            attention_mask: Маска внимания
        Return:
            pooled embedding
        """
        inv_attention_mask = ~attention_mask[..., None].bool()
        last_hidden = last_hidden_states.masked_fill(inv_attention_mask, 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_embedding(self, content: List[str]) -> Tuple[List[float],
                                                         List[List[float]]]:
        """
        Метод, получающий эмбеддинг к каждому переданному тексту из списка.
        Реализация этого метода обязательна для применения
        объектов класса.

        Parameters:
            content: Список текстов, эмбеддинг от которых нужно получить.
                     Первый текст в списке считается постановкой задачи, а иные
                     - идеями.

        Return:
            Кортеж из эмбеддинга задачи и набора
            эмбеддингов для предложенных идей
        """
        if self.use_text_cleaner and self.text_preprocessor is not None:
            content = list(map(lambda x: ' '.join(self.text_preprocessor.clean_text(x)).strip(),
                               content))

        with torch.no_grad():
            batch_dict = self.tokenizer.encode(content,
                                               self.encode_config).to(self.device)
            outputs = self.model(**batch_dict)
            embeddings = self.__average_pool(outputs.last_hidden_state,
                                             batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        listed_embeddings = embeddings.tolist()
        task = listed_embeddings[0]
        ideas = listed_embeddings[1:]
        return task, ideas


    def embed_documents(self, documents: List[str]) -> Tuple[List[float], 
                                                             List[List[float]]]:
        '''
        Метод, получающий эмбеддинг к каждому переданному тексту из списка
        Parameters:
            content: Список текстов, эмбеддинги от которых нужно получить
        Return:
            Список эмбеддингов переданных текстов
        '''
        with torch.no_grad():
            batch_dict = self.tokenizer.encode(documents,
                                                self.encode_config).to(self.device)
            outputs = self.model(**batch_dict)
            embeddings = self.__average_pool(outputs.last_hidden_state,
                                             batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        listed_embeddings = embeddings.tolist()
        return listed_embeddings


    def embed_query(self, query: str) -> List[float]:
        '''
        Метод, получающий эмбеддинг к переданному query
        Parameters:
            query: queriy, эмбеддинги от которого нужно получить
        Return:
            Эмбеддинг переданного query
        '''
        if not isinstance(query, list):
            query = [query]
        with torch.no_grad():
            batch_dict = self.tokenizer.encode(query,
                                                self.encode_config).to(self.device)
            outputs = self.model(**batch_dict)
            embeddings = self.__average_pool(outputs.last_hidden_state,
                                             batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        listed_embeddings = embeddings.tolist()
        return listed_embeddings[0]
