import time
import numpy as np
import vowpalwabbit
import typing as tp
from jiwer import wer
from copy import deepcopy
from urllib.error import HTTPError

class NoAugment:
    '''
    Обёртка для того случая, когда не используется аугментация
    '''
    def __init__(self):
        self.name = 'Without_Aug'


    def augment(self, 
                text: str) -> str:
        return text


class IntegerActBandit:
    '''
    Контекстный бандит для adversarial атак
    '''
    def __init__(self, 
                 bounds: tp.Tuple[int, int], 
                 str_config: tp.Optional[str]=None, 
                 seed: int=42):
        '''
        bounds: Левая и правая границы поиска
        str_config: Строка с JSON конфигом модели
        seed: Начальное значение датчика случайной величины
        '''        
        low_bound, high_bound = bounds
        num_actions = high_bound - low_bound + 1
        if str_config is None:
            str_config = f""" --cb_explore {num_actions} --ngram t1 --epsilon 0.1 --quiet"""    
        self.model = vowpalwabbit.Workspace(str_config)
        self.shift_act = low_bound - 1
        self.generator = np.random.RandomState(seed=seed)
      

    def _to_vw_example_format(self, 
                              context: str, 
                              cats_label=None) -> str:
        '''
        Функция, преобразующая (context, action, cost, probability) в JSON формат для VW 
        Parameters:
            context: Контекст
            cats_label: Tuple с действием, наградой за выбор действия, вероятностью выбрать действие
        Return:
            Строка для обучения модели в vowpalwabbit
        '''
        example = ''
        if cats_label is not None:
            action, cost, pdf_value = cats_label
            example = str(action) + ":" + str(cost) + ":" + str(pdf_value)
        learn_example = example + " |t " + context
        return learn_example
    

    def predict(self, 
                context: str) -> tp.Tuple[tp.Any]:
        '''
        Предсказание модели
        Parameters:
            context: Контекст среды
        Return: 
            Действие, вероятность выборать действие
        '''
        vw_text_example = self._to_vw_example_format(context)
        vw_format = self.model.example(vw_text_example)
        actions_probs = np.array(self.model.predict(vw_format))
        actions_probs /= actions_probs.sum()
        action = int(self.generator.choice(range(len(actions_probs)), p=actions_probs))
        proba = actions_probs[action]
        action += self.shift_act + 1
        return action, proba, actions_probs
    

    def fit(self, 
            context: str, 
            action: int, 
            proba: float, 
            reward: int) -> None:
        '''
        Обучение модели
        Parameters:
            context: Контекст среды
            action: Действие
            proba: Вероятность выбрать случайную стратегию
            reward: Награда
        '''
        action -= self.shift_act
        txt_ex = self._to_vw_example_format(context, 
                                            cats_label=(action, -reward, proba))
        vw_format = self.model.example(txt_ex)
        self.model.learn(vw_format)
        self.model.finish_example(vw_format)


class Agent:
    '''
    Агент. Взаимодействует со средой.  Сремится подобрать такие параметры аугментации,
    при которых валидируемый сервис будет выдавать неправомочные результаты.
    '''
    def __init__(self, 
                 model: tp. Any,
                 bounds: tp.Tuple[int, int],  
                 seed: int=42,
                 time_step: tp.Optional[int]=None):
        '''
        model: Обёртка для диалоговой модели
        bounds: Левая и правая границы поиска
        seed: Начальное значение датчика случаной величины
        time_step: Интервал между запросами к модели 
        '''
        self.bandit = IntegerActBandit(bounds, seed=seed)
        self.model = model
        self.time_step = time_step

    def fit(self, 
            context: str, 
            action: int, 
            proba: float, 
            reward: int) -> None:
        '''
        Обучение бандита
        Parameters:
            context: Контекст среды
            action: Действие
            proba: Вероятность выбрать случайную стратегию
            reward: Награда
        '''
        self.bandit.fit('; '.join(context), 
                        action, 
                        proba, 
                        reward)
        

    def predict(self, 
                context: str) -> tp.Tuple[tp.Tuple[tp.Any], str]:
        '''
        Предсказание агента
        Parameters:
            context: Фраза на вход LLM
        Return:
            Tuple с действием, вероятностями и ответом LLM
        '''
        action, proba_action, probs = self.bandit.predict('; '.join(context))
        try: 
            reply = self.model.predict(context)
        except HTTPError as err:
            if not self.time_step is None:
                time.sleep(self.time_step)
            reply = self.model.predict(context)
        return (action, proba_action, probs), reply[0]


class AgentOpponent:
    '''
    Агент, который с помощью аугментаций пытается заставить модель 
    выдать неправомочный ответ
    '''
    def __init__(self,
                 model: tp.Any, 
                 augments: tp.List[tp.Any],
                 bounds: tp.Tuple[int, int],
                 seed: int=42,
                 time_step: tp.Optional[int]=None):
        '''
        model: Обёртка для диалоговой модели
        augments: Список с используемыми аугментациями
        bounds: Левая и правая границы поиска
        seed: Начальное значение датчика случаной величины
        time_step: Интервал между запросами к модели 
        '''
        self.bandit = IntegerActBandit(bounds, 
                                       seed=seed)
        self.band_aug = IntegerActBandit((0, len(augments)),
                                         seed=seed)
        self.model = model
        self.augments = [NoAugment()] + augments
        self.time_step = time_step
        
        
    def fit(self, 
            context: str, 
            action: int, 
            proba: float, 
            reward: int) -> None:
        '''
        Обучение контекстных бандитов
        Parameters:
            context: Контекст среды
            action: Действие 
            proba: Вероятность выбора действия
            reward: Награда, получаемая за действие
        '''
        act, augm = action
        act_proba, augm_proba = proba
        self.bandit.fit('; '.join(context), 
                        act, 
                        act_proba, 
                        reward)
        self.band_aug.fit('; '.join(context), 
                          augm, 
                          augm_proba, 
                          reward)
        

    def predict(self, 
                context: str) -> tp.Tuple[tp.Any]:
        '''
        Предсказание модели
        Parameters:
            context: Строка с контекстом диалога
        Return:
            
        '''
        action, proba_action, probs = self.bandit.predict('; '.join(context))
        augm, proba_augm, probs_aug = self.band_aug.predict('; '.join(context))
        try:
            reply = self.model.predict(context)
        except HTTPError as err:
            if not self.time_step is None:
                time.sleep(self.time_step)
            reply = self.model.predict(context)
        new_repl = self.augments[augm].augment(reply[0])
        return ((action, augm), (proba_action, proba_augm), (probs, probs_aug)), new_repl[0]


class Environment:
    '''
    Среда, в которой две генеративные модели общаются друг с другом.
    Первая реплика выбирается случайно из предоставленного датасета,
    которая с вероятностью epsilon изменяется с помощью paraphraser
    '''
    def __init__(self, 
                 dialogue_dataset: tp.List[str],
                 stop_phrases: tp.List[str],
                 paraphraser: tp.Any, 
                 epsilon: float,
                 max_iter: int,
                 threshold: float=0.08,
                 seed: int=42):
        
        '''
        dialog_dataset: датасет с запросами пользователей
        stop_phrases: фразы, которые отражают непонимание моделью пользователя
        paraphraser: модель, которая перефразирует запрос с помощью контекстных синонимов
        epsilon: вероятность перефраза входного запроса
        max_iter: максимальное число итераций в эпизоде
        seed: номер псевдослучайной подпоследовательности
        '''
        self.dialogue_dataset = dialogue_dataset
        self.stop_phrases = stop_phrases
        self.paraphraser = paraphraser
        self.epsilon = epsilon
        self.max_iter = max_iter
        self._len = len(dialogue_dataset)
        self._generator = np.random.RandomState(seed)
        self.state = None
        self.player = None
        self.statistic_discusses = {1: {'Победа': 0,
                                        'Зацикливание': 0,
                                        'Непонимание': 0},
                                    2: {'Победа': 0,
                                        'Зацикливание': 0,
                                        'Непонимание': 0}}
        self.threshold=threshold
    

    def get_statistic_discusses(self) -> None:    
        '''
        Метод, который возвращает статистику проигрышей
        для обоих собеседников
        Return:
            Словарь со статистикой проигрышей
        '''
        return self.stat_lose
    
    
    def reset_statistic_discusses(self) -> None:
        '''
        Сброс статистики выигрышей и проигрышей
        '''
        self.statistic_discusses = {1: {'Победа': 0,
                                        'Зацикливание': 0,
                                        'Непонимание': 0},
                                    2: {'Победа': 0,
                                        'Зацикливание': 0,
                                        'Непонимание': 0}}    
        

    def get_opponent(self) -> int:
        '''
        Метод, возвращающий номер собеседника-оппонента
        Return: 
            Номер собеседника опонента
        '''
        if self.player == 1:
            ans = 2
        else:
            ans = 1
        return ans
    
    
    def reset(self) -> tp.Tuple[tp.Any]:
        '''
        Сброс среды. Возвращает входной запрос пользователя,
        измененный с вероятностью epsilon
        Return: 
            Cтартовое состояние среды
        '''
        rand_num_sample = self._generator.randint(0, self._len)
        rand_sample = self.dialogue_dataset[rand_num_sample]
        r = self._generator.uniform()
        if r < self.epsilon:
            rand_sample = self.paraphraser.paraphrase(rand_sample)
        self.state = [rand_sample]
        self.player = 1
        self.iter = -1
        return deepcopy(self.state), self.player
        
    
    def step(self, 
             reply: str) -> tp.Tuple[tp.Any]:
        '''
        Метод, меняющий состояние среды
        Return: 
            Состояние среды, вознаграждения для обоих собеседников, флаг завершения эпизода, id следующего собеседника
        '''
        self.iter += 1
        done = False
        rewards = {1: -1.0,
                   2: -1.0}
        player = deepcopy(self.player)
        opponent = self.get_opponent()
        error = wer(self.state[-1], reply)
        if error < self.threshold:
            rewards[player] = -100.
            rewards[opponent] = 100.
            done = True
            self.statistic_discusses[player]['Зацикливание'] += 1
            self.statistic_discusses[opponent]['Победа'] += 1
        elif reply in self.stop_phrases:
            rewards[player] = -100.
            rewards[opponent] = 100.
            done = True
            self.statistic_discusses[player]['Непонимание'] += 1
            self.statistic_discusses[opponent]['Победа'] += 1            
        self.state.append(reply)
        self.player = deepcopy(opponent)
        if self.iter >= self.max_iter:
            done = True
        return deepcopy(self.state), rewards, done, opponent
