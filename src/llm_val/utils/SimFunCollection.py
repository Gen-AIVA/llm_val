import numpy as np
import typing as tp
from sklearn.metrics.pairwise import cosine_similarity


class SimFunCollection:
    """
    Класс, представляющий собой набор мер
    близости для эмбеддингов одинаковой длины
    """
    def one_minus_ned(self,
                      emb_0: tp.List[float],
                      emb_1: tp.List[float]) -> float:
        """
        Вычисляет величину, обратную нормализированному
        евклидову расстоянию между эмбеддингами

        Parameters:
            emb_0: Эмбеддинг №0
            emb_1: Эмбеддинг №1

        Return: 
            Мера близости переданных эмбеддингов
        """
        mag_emb_0 = np.linalg.norm(emb_0)
        mag_emb_1 = np.linalg.norm(emb_1)
        ned = sum([(emb_0[i] / mag_emb_0 - emb_1[i] / mag_emb_1)
                   ** 2 for i in range(len(emb_0))])
        return 1 - ned

    def dot_sim(self, emb_0: tp.List[float], emb_1: tp.List[float]) -> float:
        """
        Вычисляет dot-произведение для оценки
        расстояния между эмбеддингами

        Parameters:
            emb_0: Эмбеддинг №0
            emb_1: Эмбеддинг №1
        Return:
            Мера близости переданных эмбеддингов
        """
        sim_val = 0
        for i in range(len(emb_0)):
            sim_val += emb_0[i]*emb_1[i]
        return sim_val / len(emb_0)

    def __norm_vector(self, vector: tp.List[float]) -> np.ndarray:
        return vector / np.linalg.norm(vector)

    def cos_sim(self, emb_0: tp.List[float], emb_1: tp.List[float]) -> float:
        """
        Вычисляет косинусную меру близости между двумя эмбеддингами

        Parameters:
            emb_0: Эмбеддинг №0
            emb_1: Эмбеддинг №1

        Return:
            Мера близости переданных эмбеддингов
        """
        return cosine_similarity([emb_0], [emb_1])[0][0]

    def craft_sim_idea_idea(self,
                            idea_0_embed: tp.List[float],
                            idea_1_embed: tp.List[float]) -> float:
        """
        Подсчитывает метрику, созданную для оценки среднего
        попарного различия идей, предложенных моделью

        Parameters:
            idea_0_embed: Эмбеддинг идеи №0
            idea_1_embed: Эмбеддинг идеи №1

        Return:
            Мера расхождения переданных эмбеддингов
        """
        total_sum = 0
        norm_idea_0_embed = self.__norm_vector(idea_0_embed)
        norm_idea_1_embed = self.__norm_vector(idea_1_embed)
        for i in range(len(idea_0_embed)):
            delta = 1 - np.abs(norm_idea_0_embed[i] - norm_idea_1_embed[i])
            total_sum += delta
        return total_sum/len(idea_0_embed)

    def craft_sim_task_idea(self,
                            task_embed: tp.List[float],
                            idea_embed: tp.List[float]) -> float:
        """
        Подсчитывает метрику, созданную для оценки сходства постановки
        задачи мозгового штурма и идей, предложенных моделью

        Parameters:
            task_embed: Эмбеддинг постановки задачи
            idea_1_embed: Эмбеддинг предложенной идеи

        Return:
            Мера близости переданных эмбеддингов
        """
        total_sum = 0
        norm_task_embed = self.__norm_vector(task_embed)
        norm_idea_embed = self.__norm_vector(idea_embed)
        for i in range(len(idea_embed)):
            delta = np.abs(norm_task_embed[i] - norm_idea_embed[i])
            total_sum += delta
        return total_sum/len(idea_embed)

    def inverted_craft_sim_task_idea(self,
                                     task_embed: tp.List[float],
                                     idea_embed: tp.List[float]) -> float:
        """
        Подсчитывает метрику, обратную к craft_sim_task_idea

        Parameters:
            task_embed: Эмбеддинг постановки задачи
            idea_1_embed: Эмбеддинг предложенной идеи

        Return:
            Мера близости переданных эмбеддингов
        """
        return 1 - self.craft_sim_task_idea(task_embed, idea_embed)

    def peak_distance_task_idea(self,
                                emb_0: tp.List[float],
                                emb_1: tp.List[float]) -> float:
        """
        Подсчитывает пиковое расстояние между
        двумя эмбеддингами одинаковой длины

        Parameters:
        emb_0: Эмбеддинг №0
        emb_1: Эмбеддинг №1

        Return:
            Мера близости переданных эмбеддингов
        """
        total_sum = 0
        for idx in range(len(emb_0)):
            numerator = np.abs(emb_0[idx] - emb_1[idx])
            denominator = np.abs(emb_0[idx] + emb_1[idx])
            if denominator >= 1:
                total_sum += numerator / denominator
            else:
                total_sum += numerator * denominator
        return total_sum/len(emb_0)

    def peak_distance_idea_idea(self,
                                emb_0: tp.List[float],
                                emb_1: tp.List[float]) -> float:
        """
        Подсчитывает метрику, созданную для оценки среднего
        попарного различия идей, предложенных моделью

        Parameters:
            emb_0: Эмбеддинг идеи №0
            emb_1: Эмбеддинг идеи №1

        Return:
            Мера расхождения переданных эмбеддингов
        """
        return 1 - self.peak_distance_task_idea(emb_0, emb_1)
