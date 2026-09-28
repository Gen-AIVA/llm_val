import torch
import torch.nn.functional as F
from transformers import AutoModel
from typing import Dict, List, Tuple, Any


from llm_val.utils.Embedder.Embedder import Embedder


class SBERTEmbedder(Embedder):
    """
    Класс для получения эмбеддингов текста на русском языке
    с использованием модели sbert_large_mt_nlu_ru

    Parameters:
        model_path: Путь до директории с моделью
        config_path: Путь до конфигурационного файла
        use_text_cleaner: Флаг, поднятие которого обеспечит предобработку сессий
        normalize: Флаг, поднятие которого обеспечит нормализацию эмбеддингов
        allow_gpu: Флаг, поднятие которого разрешает использование gpu
    """
    def __init__(self,
                 model_path: str,
                 encode_config: Dict[str, Any],
                 use_text_cleaner: bool = False,
                 normalize: bool = False,
                 allow_gpu: bool = True):
        super().__init__()
        self.model_path = model_path
        self.encode_config = encode_config
        self.allow_gpu = allow_gpu

        self.model = None
        self.use_text_cleaner = use_text_cleaner
        self.normalize = normalize
        self.__init_model()

    def __init_model(self):
        """
        Метод для инициализации токенайзера и самой модели
        """
        use_gpu = torch.cuda.is_available() and self.allow_gpu
        self.device = ('cpu', 'cuda:0')[use_gpu]

        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        self.model.eval()

    def __mean_pooling(self,
                       model_output,
                       attention_mask: torch.Tensor):
        """
        Пулинг эмбеддингов.
        Рекомендован к применению создателями модели.

        Parameters:
            model_output: Выходные данные модели
            attention_mask: Маска внимания
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_embedding(self, content: List[str]) -> Tuple[List[float],
                                                         List[List[float]]]:
        """
        Метод, получающий эмбеддинг к переданному тексту.
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

        encoded_input = self.tokenizer.encode(content, self.encode_config)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self.__mean_pooling(model_output,
                                         encoded_input['attention_mask'])
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        listed_embeddings = embeddings.tolist()
        task = listed_embeddings[0]
        ideas = listed_embeddings[1:]
        return task, ideas

    def embed_documents(self, content: List[List[str]]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, content: List[str]) -> List[float]:
        raise NotImplementedError
