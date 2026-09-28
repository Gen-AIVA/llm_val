import numpy as np
import typing as tp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_literacy_confusion_matrix(prompt_literacy: tp.List[float],
                                   answer_literacy: tp.List[float]) -> tp.Any:
    '''
    Строит матрицу зависимости лингвистической приемлемости ответа модели от значений запроса
    Parameters:
        prompt_literacy: Подсчитанная метрика лингвистической приемлемости для запросов
        answer_literacy: Подсчитанная метрика лингвистической приемлемости ответа модели 
    Return:
        График зависимости лингвистической приемлемости ответа модели от значений запроса
    '''
    def round_col(x):
        if x < 0.2:
            out = '< 0.2'
        elif x < 0.4:
            out = '0.2-0.4'
        elif x < 0.6:
            out = '0.4-0.6'
        elif x <= 0.8:
            out = '0.6-0.8'
        else:
            out = '> 0.8'
        return out

    prompt_literacy = np.array([round_col(x) for x in prompt_literacy])
    answer_literacy = np.array([round_col(x) for x in answer_literacy])
    labels = ['< 0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '> 0.8']
    cm = confusion_matrix(answer_literacy, prompt_literacy, labels=labels)
    cm = np.multiply(cm, (1 / (cm + 1e-16).sum(axis=0))) * 100
    cm = cm[::-1, :]
    fig = plt.figure(figsize=(9, 7))
    lables_x = [f'{l}\n N = {sum(prompt_literacy == str(l))}' for l in labels]
    literacy_heatmap = sns.heatmap(data=cm,
                                   xticklabels=lables_x,
                                   yticklabels=labels[::-1],
                                   vmin=0,
                                   vmax=100,
                                   annot=True,
                                   fmt='.2f',
                                   linewidths=0.5,
                                   cmap=sns.color_palette('rocket_r', as_cmap=True))
    literacy_heatmap.set(xlabel='Prompt', ylabel='Answer')
    plt.title('Linguistic Acceptability score')
    return fig
