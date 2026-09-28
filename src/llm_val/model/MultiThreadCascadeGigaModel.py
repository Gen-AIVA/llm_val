import typing as tp
from functools import partial
from llm_val.model.GigaModel import GigaModel
from concurrent.futures import ThreadPoolExecutor

class MultiThreadCascadeGigaModel(GigaModel):
    '''
    Пример класса-обертки для API'шки.
    Для инференса принимает список вопросов и на выходе выдает список ответов по каждому вопросу.
    Работает в многопоточном режиме.
    '''
    def __init__(self,
                 system_promt_cascade: tp.Tuple[str],
                 one_shot: tp.Optional[str],
                 probable_answer_cascade: tp.Tuple[str],
                 contour: tp.Optional[str]=None,
                 token: tp.Optional[str]=None,
                 auth_data: tp.Optional[str]=None,
                 api_url: tp.Optional[str]=None,
                 oauth_url: tp.Optional[str]=None,
                 rquid: tp.Optional[str]=None,
                 model: str='GigaChat:latest',
                 function_call: tp.Optional[tp.Union[str, tp.Dict[str, tp.Any]]]=None,
                 functions: tp.Optional[tp.List[tp.Dict[str, tp.Any]]]=None,
                 temperature: tp.Optional[float]=None,
                 top_p: tp.Optional[float]=None,
                 n: tp.Optional[int]=None,
                 stream: tp.Optional[bool]=None,
                 max_tokens: tp.Optional[int]=None,
                 repetition_penalty: tp.Optional[float]=None,
                 update_interval: tp.Optional[int]=None,
                 profanity_check: tp.Optional[bool]=None,
                 n_jobs: int=1):
        '''
        Кастомизация обычного класса MultiThreadGigaModel, чтобы подавать one-shot
        system_promt_cascade: Кортеж системных промтов
        one_shot: Пример вопроса
        probable_answer_cascade: Пример ответов модели на вопрос в соответствии с каждым системным промтом
        contour: Название контура. Принимает одно из трех значениий ['sigma', 'mlspace', 'datalab']
        token: Токен для авторизации. Передается, если уже получен
        auth_data: Авторизационные данные. Client id и Client Secret, закодированные в Base64
        api_url: URL до API GigaChat 
        oauth_url: URL до сервиса авторизации GigaChat
        rquid: Уникальный идентификатор запроса. Соответствует формату uuid4.
                Параметр для журналирования входящих вызовов и разбора инцидентов.
        model: Название модели
        function_call: Поле, которое отвечает за то, как GigaChat будет работать с функциями.
        functions: Массив с описанием пользовательских функций
        temperature: Температура выборки.
                     Значение температуры должно быть не меньше ноля. Чем выше значение, тем более случайным будет ответ модели.
                     Значение по умолчанию зависит от выбранной модели и может изменяться с обновлениями модели.
        top_p: Возможные значения: >= 0 и  <= 1.
               Задает вероятностную массу токенов, которые должна учитывать модель.
               Значение по умолчанию зависит от выбранной модели и может изменяться с обновлениями модели.
        n: Количество вариантов ответов, которые нужно сгенерировать для каждого входного сообщения.
        stream: Указывает, что сообщения надо передавать по частям в потоке.
        max_tokens: Максимальное количество токенов, которые будут использованы для создания ответов.
        repetition_penalty: Количество повторений слов. Значение 1 — нейтральное значение.
                            При значении больше 1 модель будет стараться не повторять слова.
                            Значение по умолчанию зависит от выбранной модели и может изменяться с обновлениями модели.
        update_interval: Задает минимальный интервал в секундах, который проходит между отправкой токенов.
        profanity_check: Переключатель цензора. Если False, то цензор выключен
        n_jobs: Количество потоков
        '''
        super().__init__(contour=contour,
                         token=token,
                         auth_data=auth_data,
                         api_url=api_url,
                         oauth_url=oauth_url,
                         rquid=rquid,
                         model=model,
                         sys_promt=None)
        self.function_call = function_call
        self.functions = functions
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.stream = stream
        self.max_tokens = max_tokens
        self.repetition_penalty = repetition_penalty
        self.update_interval = update_interval
        self.profanity_check = profanity_check
        self.n_jobs = n_jobs
        self.system_promt_cascade = system_promt_cascade
        self.one_shot = one_shot
        self.probable_answer_cascade = probable_answer_cascade

    
    def fit(self, messages):
        return self


    def predict(self,
                messages: tp.List[str]):
        question_dict = {}
        answers_dict = {}
        for j, sys_promt in enumerate(self.system_promt_cascade):
            question_dict[j] = []
            for i in range(len(messages)):
                user_msg = {'role': 'user', 
                            'content': messages[i]}
                question_dict[j].append([])
                question_dict[j][i].append({'role': 'system',
                                            'content': sys_promt})
                question_dict[j][i].append({'role': 'user',
                                            'content': self.one_shot})
                question_dict[j][i].append({'role': 'assistant',
                                            'content': self.probable_answer_cascade[j]})
                question_dict[j][i].append(user_msg)
            inference = partial(self.api.get_answer_gigachat,
                                model=self.model,
                                function_call=self.function_call,
                                functions=self.functions,
                                temperature=self.temperature,
                                top_p=self.top_p,
                                n=self.n,
                                stream=self.stream,
                                max_tokens=self.max_tokens,
                                repetition_penalty=self.repetition_penalty,
                                update_interval=self.update_interval,
                                profanity_check=self.profanity_check)
            with ThreadPoolExecutor(self.n_jobs) as executor:
                answ = executor.map(inference, question_dict[j])
            answers_dict[j] = [i[0]['message']['content'] for i in answ]
        return answers_dict
     