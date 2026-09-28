import typing as tp
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from llm_val.sampler.DialogueSampler import DialogueSampler

def report_valtest_empathy_assessment(plot_list: tp.List[tp.Any]) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        plot_list: Список с построенными температурным графиком и распределениями эмоциональных окрасок LLM для каждой эмоциональной окраски пользователей
    Return:
        report в соответствии с концепцией llm_val
    '''
    res_dict = {}
    res_dict['semaphore'] = 'gray'
    res_dict['result_plots'] = plot_list
    res_dict['result_dataframes'] = []
    return res_dict


def valtest_empathy_assessment(sampler: DialogueSampler,
                               model: tp.Any,
                               labels: tp.List[str],
                               emotion_pairs: tp.Optional[pd.DataFrame],
                               **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест, который строит графики распределения эмоциональных окрасок ответов LLM в зависимости от эмоциональных окрасок промптов пользователей
    Parameters:
        sampler: Объект класса DialogueSampler
        model: Модель, которая возвращает спиcок эмоциональных окрасок для текстов из списка labels с помощью метода predict
        labels: Список возможных эмоциональных окрасок, которые может выдать model
        emotion_pairs: Опциональный предпосчитанный датафрейм с эмоциональными окрасками вопросов и соответствующими им ответами
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if emotion_pairs is None:  
        questions = sampler.test['X'][sampler.test['y']['role'] == 'user'].rename(columns={'content': 'questions'})
        answers = sampler.test['X'][sampler.test['y']['role'] == 'assistant'].rename(columns={'content': 'answers'})
        questions = questions.reset_index(drop=True)
        answers = answers.reset_index(drop=True)
        df_questions_answers = pd.concat([questions, answers], axis=1).dropna(subset=['questions', 'answers'])
        question_emotions = model.predict(list(df_questions_answers['questions']))
        answer_emotions = model.predict(list(df_questions_answers['answers']))
        emotion_pairs = pd.DataFrame({'question_emotions': question_emotions,
                                      'answer_emotions': answer_emotions})
    fig_cm = plot_emotion_confusion_matrix(emotion_pairs,
                                           labels=labels)
    fig_distr = plot_emotion_distributions(emotion_pairs,
                                           labels=labels)
    report = report_valtest_empathy_assessment([fig_cm, fig_distr])
    precomputed = {'emotion_pairs': emotion_pairs}
    return {'report': report, 
            'precomputed': precomputed}


def plot_emotion_confusion_matrix(emotion_pairs: pd.DataFrame,
                                  labels: tp.List[str]) -> tp.Any:
    '''
    Функция для построения температурного графика confusion matrix между эмоциональными окрасками промтов пользователей и соответствующими им ответами LLM
    Parameters:
        emotion_pairs: Предпосчитанный датафрейм с эмоциональными окрасками вопросов и соответствующими им ответами
        labels: Список возможных эмоциональных окрасок
    Return:
        figure с температурным графиком
    '''
    plt.rc('font', family='DejaVu Sans', size=8)
    cm = confusion_matrix(emotion_pairs.answer_emotions,
                          emotion_pairs.question_emotions,
                          labels=labels)
    fig_cm, ax_cm = plt.subplots(figsize=(9,7))
    sns.heatmap(cm,
                xticklabels=labels,
                yticklabels=labels,
                annot=True,
                fmt="d",
                linewidths=.5,
                cmap="YlGnBu",)
    ax_cm.set_xlabel('User emotions')
    ax_cm.set_ylabel('LLM emotions')
    ax_cm.set_title('Распределение эмоциональной окраски ответа LLM \n в зависимости от эмоции пользователя')
    plt.close(fig_cm)
    return fig_cm


def plot_emotion_distributions(emotion_pairs: pd.DataFrame,
                               labels: tp.List[str]) -> tp.Any:
    '''
    Функция для построения графиков распределения эмоциональных окрасок ответов LLM для каждой эмоциональной окраски промтов пользователей
    Parameters:
        emotion_pairs: Предпосчитанный датафрейм с эмоциональными окрасками вопросов и соответствующими им ответами
        labels: Список возможных эмоциональных окрасок
    Return:
        figure с графиками распределения
    '''
    plt.rc('font',
           family='DejaVu Sans',
           size=8)
    fig_strings = (len(labels)//2) + (len(labels)%2)
    fig_distr, ax_distr = plt.subplots(fig_strings,
                                       2,
                                       figsize=(3*fig_strings, 10))
    for i, emotion in enumerate(labels):
        emo_df = emotion_pairs[emotion_pairs.question_emotions == emotion]
        sns.histplot(emo_df.answer_emotions,
                     ax=ax_distr[i//2, i%2])
        ax_distr[i//2, i%2].set_xlabel('LLM Emotions')
        ax_distr[i//2, i%2].set_title(f'User emotion - {emotion}')
    plt.tight_layout()
    plt.close(fig_distr)
    return fig_distr
