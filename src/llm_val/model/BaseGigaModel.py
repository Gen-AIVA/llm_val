import typing as tp
from llm_val.model.GigaModel import GigaModel

class BaseGigaModel(GigaModel):
    '''
    Пример класса-обертки для API'шки.
    Для инференса принимает список вопросов и на выходе выдает список ответов по каждому вопросу.
    '''
    def __init__(self,
                 contour: tp.Optional[str]=None,
                 token: tp.Optional[str]=None,
                 auth_data: tp.Optional[str]=None,
                 api_url: tp.Optional[str]=None,
                 oauth_url: tp.Optional[str]=None,
                 rquid: tp.Optional[str]=None,
                 model: str='GigaChat:latest',
                 sys_promt: tp.Optional[str]=None,
                 function_call: tp.Optional[tp.Union[str, tp.Dict[str, tp.Any]]]=None,
                 functions: tp.Optional[tp.List[tp.Dict[str, tp.Any]]]=None,
                 temperature: tp.Optional[float]=None,
                 top_p: tp.Optional[float]=None,
                 n: tp.Optional[int]=None,
                 stream: tp.Optional[bool]=None,
                 max_tokens: tp.Optional[int]=None,
                 repetition_penalty: tp.Optional[float]=None,
                 update_interval: tp.Optional[int]=None,
                 profanity_check: tp.Optional[bool]=None):
        '''
        contour: Название контура. Принимает одно из трех значениий ['sigma', 'mlspace', 'datalab']
        token: Токен для авторизации. Передается, если уже получен
        auth_data: Авторизационные данные. Client id и Client Secret, закодированные в Base64
        api_url: URL до API GigaChat 
        oauth_url: URL до сервиса авторизации GigaChat
        rquid: Уникальный идентификатор запроса. Соответствует формату uuid4.
                Параметр для журналирования входящих вызовов и разбора инцидентов.
        model: Название модели
        sys_promt: Системный промт
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
        '''
        super().__init__(contour=contour,
                         token=token,
                         auth_data=auth_data,
                         api_url=api_url,
                         oauth_url=oauth_url,
                         rquid=rquid,
                         model=model,
                         sys_promt=sys_promt)
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

    
    def fit(self, messages):
        return self


    def get_models(self):
        return self.api.get_list_models()


    def predict(self,
                messages: tp.Union[str, tp.List[str]],
                chat_history: tp.Optional[tp.Iterable[tp.Iterable[tp.Dict[str, str]]]]=None):
        '''
        Получить ответ GC
        Parameters:
            messages: список сообщений, на которые необходимо получить ответы
            chat_history: опциональный список, содержащий диалоги вида
                          [{'role': 'user',
                            'content': 'Вопрос'},
                           {'role': 'assistant',
                            'content': 'Ответ'}, ...]
        '''
        if isinstance(messages, str):
            msg_list = [messages]
        else:
            msg_list = messages
        answers = []
        question = []
        for i in range(len(msg_list)):
            question.append([])
            user_msg = {'role': 'user', 
                        'content': msg_list[i]}
            if self.sys_promt is not None:
                question[i].append(self.sys_promt)
            if chat_history is not None:
                question[i].extend(list(chat_history[i]))
            question[i].append(user_msg)
            answ = self.api.get_answer_gigachat(question[i], 
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
            answers.append(answ[0]['message']['content'])
        return answers
