import os
import typing as tp
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain.chat_models import GigaChat
from langchain.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from llm_val.utils.Embedder.Embedder import Embedder
from llm_val.utils.Embedder.E5Embedder import E5Embedder
from llm_val.utils.TextPreprocessor import TextPreprocessor
from llm_val.utils.TokenizerWrapper import TokenizerWrapper


class RAGExecutor:
    '''
    Для использования RAGExecutor'а и дальнейшего проведения тестов раздела о качестве RAG-систем необходимо
    наличие локального репозитория с эмбеддером e5-large
    Ссылки для скачивания: 
        https://sberosc.sigma.sbrf.ru/repo/hosted//hosted/ModelsDS/1abdf830-56e5-4b4a-96cf-2d32010abf2f/intfloat_multilingual-e5-large.zip (sigma)
        https://sberosc.ca.sbrf.ru/repo/hosted//hosted/ModelsDS/1abdf830-56e5-4b4a-96cf-2d32010abf2f/intfloat_multilingual-e5-large.zip (alpha)
    '''
    def __init__(self, 
                 embed_model_path: str,
                 encode_config: tp.Dict,
                 text_process_config: tp.Dict,
                 documents_paths: tp.Union[tp.List[str], str],
                 vectorstore_path: str,
                 topK: int,
                 chunk_size: int,
                 chunk_overlap: int,
                 topK_list: tp.List[int],
                 chunk_size_list: tp.List[int],
                 chunk_overlap_list: tp.List[int],
                 chat_model: tp.Optional[GigaChat]=None):
        '''
        Простейший ("базовый") RAG Executor, который хранит переданные в него ссылки и создает векторные базы данных, с которыми можно работать
        Parameters:
            embed_model_path: Путь до локальной директории с эмбеддером
            encode_config: Словарь с параметрами кодирования текстов для токенайзера
            text_process_config: Словарь с параметрами предобработки текста для токенизации и получения эмбеддингов
            documents_paths: Путь до документа или список путей до документов, по которым будет построена RAG-система. 
                Hint: Может быть передан в формате списка с набором путей, если документов несколько, или в виде пути, если документ один
            vectorstore_path: Путь до локальной директории с векторной базой данных
            topK: Число релевантных пассажей, на которых будет строиться RAG-система в бэйзлайн пайплайне
            chunk_size: Размер пассажей для теста с нефиксированной величиной перекрытий между пассажами
            chunk_overlap: Величина перекрытий между пассажами для теста с нефиксированным размером пассажей
            topK_list: Список числа релевантных пассажей, по которым будут проводиться итерации в тесте
            chunk_size_list: Список размеров пассажей, по которым будут проводиться итерации в тесте
            chunk_overlap_list: Список величин перекрытий между пассажами, по которым будут проводиться итерации в тесте
            chat_model: LLM-модель из langchain
        '''
        self.embed_model = E5Embedder(model_path=embed_model_path, 
                                      encode_config=encode_config)
        self.tokenizer = TokenizerWrapper(model_path=embed_model_path)
        self.tune_model(**text_process_config)
        self.chat_model = chat_model
        self.documents = self.load_documents(documents_paths=documents_paths)
        self.vectorstore_path = vectorstore_path
        self.topK = topK
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.topK_list = topK_list
        self.chunk_size_list = chunk_size_list
        self.chunk_overlap_list = chunk_overlap_list
        self.baseline_persist_directory = self.vectorstore_path + f'/baseline'
        self.iter_persist_directory = lambda chunk_size, chunk_overlap: self.vectorstore_path + f'/cs{chunk_size}-co{chunk_overlap}'
        self.persist_directories = []
        self.iter_list = []
        self.create_baseline_vectorstore()
        self.create_vectorstore()
        self.system_add_part = '''
            
        Вопрос:
        {question}

        Контекст:
        {context}
        '''


    def tune_model(self,
                   use_text_cleaner: bool=False,
                   use_lemmatization: bool=False,
                   language: str='russian',
                   stopwords_path: str='data/brainstorm/ru_stop_words.txt') -> Embedder:
        '''
        Функция, подготавливающая модель к работе
        Parameters:
            use_text_cleaner: Флаг, поднятие которого обеспечит предобработку текстов перед токенизацией
            use_lemmatization: Флаг, поднятие которого при очистке текстов приведет к лемматизации, а не стеммингу
            language: Язык, для которого будет применяться токенизатор
            stopwords_path: Путь до документа, включающего в себя стоп-слова
        '''
        self.embed_model.tokenizer = self.tokenizer
        if use_text_cleaner:
            self.embed_model.use_text_cleaner = use_text_cleaner
            self.embed_model.text_preprocessor = TextPreprocessor(stopwords_path, 
                                                                  language, 
                                                                  use_lemmatization)


    def create_baseline_vectorstore(self):
        '''
        Метод для создания векторной базы данных, которая считается бэйзлайном.
        Здесь принято хранить набор нарезанных разработчиками пассажей.
        '''
        self.persist_directories.append(self.baseline_persist_directory)
        for _ in tqdm(range(1), desc='Creating a Baseline Vectorstore'):
            if not os.path.exists(self.baseline_persist_directory):
                passages = self.split_documents(self.chunk_size, 
                                                self.chunk_overlap)
                Chroma.from_documents(passages, 
                                      self.embed_model, 
                                      persist_directory=self.baseline_persist_directory)


    def create_vectorstore(self):
        '''
        Метод для создания векторной базы данных для тестов раздела о качестве RAG-систем.
        Здесь принято хранить наборы нарезанных пассажей.
        '''
        if self.chunk_size not in self.chunk_size_list:
            self.iter_list += [(self.chunk_size, chunk_overlap) for chunk_overlap in self.chunk_overlap_list]
        if self.chunk_overlap not in self.chunk_overlap_list:
            self.iter_list += [(chunk_size, self.chunk_overlap) for chunk_size in self.chunk_size_list]
        self.iter_list += [(chunk_size, chunk_overlap) for chunk_size in self.chunk_size_list for chunk_overlap in self.chunk_overlap_list]
        for chunk_size, chunk_overlap in tqdm(self.iter_list, 
                                              desc=f'Creating an Extended Vectorstore'):
            self.persist_directories.append(self.iter_persist_directory(chunk_size, chunk_overlap))
            if not os.path.exists(self.iter_persist_directory(chunk_size, chunk_overlap)):
                passages = self.split_documents(chunk_size, 
                                                chunk_overlap)
                Chroma.from_documents(passages, 
                                      self.embed_model, 
                                      persist_directory=self.iter_persist_directory(chunk_size, chunk_overlap))


    def get_passages_with_embeddings(self, 
                                     baseline: bool=True,
                                     chunk_size: tp.Optional[int]=None,
                                     chunk_overlap: tp.Optional[int]=None) -> tp.List[tp.Dict[str, tp.List[tp.Any]]]:
        '''
        Метод для получения списка пассажей с их эмбеддингами из векторной базы данных
        Parameters:
            baseline: Если True, то из базы данных получается набор пассажей, соответствующий baseline модели
            chunk_size: Размер пассажей для теста с нефиксированной величиной перекрытий между пассажами
            chunk_overlap: Величина перекрытий между пассажами для теста с нефиксированным размером пассажей
        Return:
            Список пассажей с их эмбеддингами
        '''
        if baseline:
            db = Chroma(embedding_function=self.embed_model, 
                        persist_directory=self.baseline_persist_directory)
        else:
            db = Chroma(embedding_function=self.embed_model, 
                        persist_directory=self.iter_persist_directory(chunk_size, chunk_overlap))
        return db.get(include=['documents', 'embeddings'])


    def get_topK_passages(self,
                          baseline: bool=True,
                          questions: tp.Optional[tp.List[str]]=None,
                          topK: tp.Optional[int]=None,
                          chunk_size: tp.Optional[int]=None,
                          chunk_overlap: tp.Optional[int]=None) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
        '''
        Метод для получения списка top-K пассажей с их эмбеддингами из векторной базы данных
        Parameters:
            baseline: Если True, то из базы данных получается набор пассажей, соответствующий baseline модели
            questions: Список вопросв, по которым находятся top-K наиболее релевантных пассажей из векторной базы данных
            topK: Число наиболее релевантных пассажей
            chunk_size: Размер пассажей для теста с нефиксированной величиной перекрытий между пассажами
            chunk_overlap: Величина перекрытий между пассажами для теста с нефиксированным размером пассажей
        Return:
            Словарь top-K пассажей с их эмбеддингами
        '''
        if baseline:
            db = Chroma(embedding_function=self.embed_model, 
                        persist_directory=self.baseline_persist_directory)
        else:
            db = Chroma(embedding_function=self.embed_model, 
                        persist_directory=self.iter_persist_directory(chunk_size, chunk_overlap))
        passages = db.get(include=['documents', 'embeddings'])
        topK_passages = {}
        if isinstance(questions, str):
            questions = [questions]
        for question in questions:
            topK_passages[question] = {}
            topK_passages[question]['documents'] = [document.page_content for document in db.similarity_search(question, topK)]
            topK_passages[question]['embeddings'] = [embedding for passage, embedding in zip(passages['documents'], passages['embeddings']) 
                                                     if passage in topK_passages[question]['documents']]
        return topK_passages


    def get_llm_answers(self,
                        baseline: bool=True,
                        questions: tp.Optional[tp.List[str]]=None,
                        systems: tp.Optional[tp.List[str]]=None,
                        topK: tp.Optional[int]=None,
                        chunk_size: tp.Optional[int]=None,
                        chunk_overlap: tp.Optional[int]=None) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
        '''
        Метод для получения списка ответов LLM с их эмбеддингами на основе заданных критериев работы с векторной базой данных
        Parameters:
            baseline: Если True, то из базы данных получается набор пассажей, соответствующий baseline модели
            questions: Список вопросв, по которым находятся top-K наиболее релевантных пассажей из векторной базы данных
            system: Список системных промптов для каждого вопроса
            topK: Число наиболее релевантных пассажей
            chunk_size: Размер пассажей для теста с нефиксированной величиной перекрытий между пассажами
            chunk_overlap: Величина перекрытий между пассажами для теста с нефиксированным размером пассажей
        Return:
            Словарь ответов LLM с их эмбеддингами
        '''
        if baseline:
            db = Chroma(embedding_function=self.embed_model, 
                        persist_directory=self.baseline_persist_directory)
        else:
            db = Chroma(embedding_function=self.embed_model, 
                        persist_directory=self.iter_persist_directory(chunk_size, chunk_overlap))
        questions_answers = {}
        if isinstance(questions, str):
            questions = [questions]
        if topK > 0:
            retriever = db.as_retriever(search_type='similarity',
                                        search_kwargs={'k': topK})
        else:
            retriever = {}
        for question, system in zip(questions, systems):
            message = system + self.system_add_part
            prompt = ChatPromptTemplate.from_messages([('human', message)])
            chain = {'context': retriever, 'question': RunnablePassthrough()} | prompt | self.chat_model
            questions_answers[question] = {}
            llm_answer = chain.invoke(question).content
            questions_answers[question]['llm_answer'] = llm_answer
            questions_answers[question]['llm_answer_embedding'] = self.embed_model.embed_query(llm_answer)
        return questions_answers


    def embed_text(self, text: str) -> tp.List[float]:
        '''
        Метод для получения эмбеддинга текста
        Parameters:
            text: Текст, для которого нужно получить эмбеддинг
        Return:
            Эмбеддинг текста
        '''
        return self.embed_model.embed_query(text)


    def get_parameters(self):
        '''
        Метод для использования инициализированных в RAG Executor'е параметров тестов раздела о качестве RAG-систем.
        Return:
            Словарь с параметрами тестов раздела о качестве RAG-систем.
        '''
        return {'topK': self.topK,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'topK_list': self.topK_list,
                'chunk_size_list': self.chunk_size_list,
                'chunk_overlap_list': self.chunk_overlap_list,
                'vectorstore_directory': self.vectorstore_path,
                'persist_directories': self.persist_directories}


    def loader_function(self, path: str) -> BaseLoader:
        '''
        Вспомогательная функция для загрузки документа из списка документов, по которым будет построена RAG-система
        Parameters:
            path: Путь до документа
        Return:
            Текст документа в формате объекта класса BaseLoader из langchain_community
        '''
        doc_loader_dict = {'pdf': PyPDFLoader,
                           'txt': TextLoader}
        postfix = path.split('.')[-1]
        if postfix not in doc_loader_dict:
            raise ValueError(f'Один или несколько документов имеют недопустимый формат. Допустимые на данный момент форматы: {list(doc_loader_dict.keys())}')
        if postfix == 'txt':
            loader = doc_loader_dict[postfix](path, autodetect_encoding=True)
        else:
            loader = doc_loader_dict[postfix](path)
        return loader
    

    def load_documents(self, documents_paths: tp.Union[tp.List[str], str]) -> tp.List[tp.Any]:
        '''
        Вспомогательная функция для загрузки одного или нескольких документов, по которым будет построена RAG-система
        Parameters:
            documents_paths: Пути до документов. 
                Hint: Может быть передан в формате списка с набором путей, если документов несколько, или в виде пути, если документ один
        Return:
            Список текстов документов в формате объекта класса Document из langchain
        '''
        documents = []
        if isinstance(documents_paths, str):
            loader = self.loader_function(documents_paths)
            documents.extend(loader.load())
        else:
            for path in documents_paths:
                loader = self.loader_function(path)
                documents.extend(loader.load())
        return documents


    def split_documents(self,
                        chunk_size: int,
                        chunk_overlap: int) -> tp.List[tp.Any]:
        '''
        Вспомогательная функция для нарезки одного или нескольких документов на пассажи с заданными размером пассажей и величиной перекрытий между пассажами
        Parameters:
            chunk_size: Размер пассажей для теста с нефиксированной величиной перекрытий между пассажами
            chunk_overlap: Величина перекрытий между пассажами для теста с нефиксированным размером пассажей
        Return:
            Тексты документов в формате объекта класса Document из langchain
        '''
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap)
        documents_splitted = text_splitter.split_documents(self.documents)
        return documents_splitted

    