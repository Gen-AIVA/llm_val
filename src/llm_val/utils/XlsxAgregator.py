import math
import itertools
import xlsxwriter
import numpy as np
import typing as tp
import pandas as pd
from dataclasses import dataclass

from llm_val.utils.image import plt2PIL, PIL2IOBytes


class XlsxAgregator:
    '''
    Агрегатор для сохранения результата выполнения тестов в .xlsx файл
    '''
    def __init__(self,
                 xlsx_path: str,
                 is_semaphore_desc_info: bool=False):
        '''
        Конструктор агрегатора. 
        Создает объект для работы с .xlsx файлами
        Parameters:
            xlsx_path: Путь до .xlsx файла
            is_semaphore_desc_info: Если False, то информация о критериях светофоров не выводится
        '''
        self.bot = XlsxBot(xlsx_path)
        self.is_semaphore_desc_info = is_semaphore_desc_info


    def save_xlsx(self,
                  data: tp.Dict[str, tp.Any],
                  agg_table: tp.Dict[str, tp.Any],
                  desc_config: tp.Dict[str, tp.Any]):
        '''
        Метод для сохранения .xlsx файла
        Parameters:
            data: Словарь с результатами проведения тестов. 
                Ключ - название теста, значение - report по тестам
            agg_table: Словарь с агрегированными светофорами по блокам
                Ключ - название блока, значение - информация о агрегации
            desc_config: Словарь с описанием тестов.
                Ключ - название теста, значение - информация о тесте
        '''
        self._insert_title_page(agg_table)
        for test_name, test_value in data.items():
            self.bot.set_active_sheet(test_name[:31])
            self.bot.cursor = Cursor(2, 0)
            test_desc = desc_config[test_name]
            self._insert_tables(test_value['result_dataframes'])
            self._insert_image(test_value['result_plots'])
            self.bot.cursor += Cursor(1, 1)
            self._insert_info_text(test_desc['excel'])
            self.bot.set_tab_color(test_value['semaphore'])
            self.bot.cursor = Cursor(0, 0)
            self._insert_title(test_desc)
        self.bot.save_wb()
        return 


    def _insert_title(self,
                      test_desc: tp.Dict[str, tp.Any]):
        '''
        Метод, который добавляет шапку для .xlsx листа
        Parameters:
            test_desc: Словарь с описанием теста
        '''
        title_test = f'Тест {test_desc["number"]} {test_desc["name"]}'
        self.bot.insert_text(title_test, Cursor(2, 0), 'title')


    def _insert_tables(self,
                       table_list: tp.List[pd.DataFrame]):
        '''
        Метод, который добавляет таблицы на лист
        Parameters:
            table_list: Список pd.DataFrame, которые необходимо добавить на лист
        '''
        for table in table_list:
            self.bot.insert_table(table, Cursor(1, 0))


    def _insert_image(self,
                      image_list: tp.List[tp.Any]):
        '''
        Метод, который добавляет картинки на лист
        Parameters:
            image_list: Список plt-изображений, которые необходимо добавить на лист
        '''
        for image in image_list:
            image = plt2PIL(image)
            self.bot.insert_pic(image, Cursor(1, 0))


    def _insert_title_page(self,
                           agg_table: tp.Dict[str, tp.Dict]):
        '''
        Метод, который добавляет начальный лист с информацией о агрегации по блокам
        Parameters:
            agg_table: Словарь с агрегированными светофорами по блокам
        '''
        self.bot.set_active_sheet('Aggregation')
        self.bot.cursor = Cursor(2, 0)
        agg_table = pd.DataFrame(agg_table).T
        self.bot.insert_table(agg_table, Cursor(1, 0))
        self.bot.cursor = Cursor(0, 0)
        self.bot.insert_text('Результаты тестов', Cursor(2, 0), 'title')
    

    def _insert_info_text(self,
                          excel_desc: tp.Dict[str, tp.Any]):
        '''
        Метод, который добавляет дополнительную информацию о тесте
        Parameters:
            excel_desc: Словарь с описанием, которое надо добавить
        '''
        if self.is_semaphore_desc_info:
            self.bot.insert_text('Границы светофора', Cursor(1, 0), 'header')
            for color, text in excel_desc['semaphore_desc'].items():
                color = color if color in ['green', 'red', 'yellow', 'gray'] else 'text'
                self.bot.insert_text(text, Cursor(1, 0), color)
        self.bot.cursor += Cursor(1, 0)
        for i in ['Цель', 'Интерпретация']:
            self.bot.insert_text(i, Cursor(1, 0), 'header')
            self.bot.insert_text_box(excel_desc[i], Cursor(1, 0))
        

@dataclass
class Cursor:
    row: int = 0
    col: int = 0

    def __add__(self, val):
        if not isinstance(val, Cursor):
            raise ArithmeticError('Вы можете сложить только Cursor c Cursor')
        new_row = self.row + val.row
        new_col = self.col + val.col
        new_row = new_row if new_row > 0 else 0
        new_col = new_col if new_col > 0 else 0
        return Cursor(new_row, new_col)


class XlsxBot:
    '''
    Класс, для удобной работы с xlsx файлами
    '''
    def __init__(self,
                 xlsx_path: str):
        '''
        Конструктор 
        Parameters:
            xlsx_path: Путь до .xlsx файла
        '''
        self.CELL_HEIGHT = 20
        self.CELL_WIDTH = 64
        self.TEXTBOX_CELL = 10
        self.wb = xlsxwriter.Workbook(xlsx_path)
        self.cursor = Cursor()
        self.active_sheet = None
        self.formats = self._create_formats()


    def insert_text(self,
                    text: str,
                    shift: Cursor, 
                    format_name: str='text'):
        '''
        Метод, который добавляет текст на текущий лист
        Parameters:
            text: Текст, который необходимо добавить
            shift: Сдвиг, на который надо сдвинуться после добавления текста
            format_name: Название форматирования текста
        '''
        if self.active_sheet is None:
            raise RuntimeError('Не выбран активный лист')
        self.active_sheet.write(self.cursor.row, 
                                self.cursor.col, 
                                text, 
                                self.formats[format_name])
        self.cursor += shift


    def insert_table(self, 
                     table: pd.DataFrame, 
                     shift: Cursor):
        '''
        Метод, который добавляет таблицу на текущий лист
        Parameters:
            table: Таблица, которую необходимо добавить
            shift: Сдвиг, на который надо сдвинуться после добавления таблицы
        '''
        table_size = table.shape
        start_cursor = Cursor(self.cursor.row,
                              self.cursor.col)
        cols_format = self._get_col_formats(table)
        rows_color = self._get_rows_color(table)
        self.insert_text('', Cursor(0, 1), 'column')
        
        for i, col_name in enumerate(table.columns):
            self.insert_text(col_name, Cursor(0, 1), 'column')
        self.cursor = Cursor(self.cursor.row + 1, start_cursor.col)
        
        for i, (idx, row) in enumerate(table.iterrows()):
            self.insert_text(idx, Cursor(0, 1), 'column')
            for j, cell_value in enumerate(row):
                border = f'text_border'
                if j >= table_size[1] - 1 and i >= table_size[0] - 1:
                    border = f'text_fat_right_bottom_border'
                elif j >= table_size[1] - 1:
                    border = f'text_fat_right_border'
                elif i >= table_size[0] - 1:
                    border = f'text_fat_bottom_border'
                format_name = '_'.join([border, str(cols_format[j]), f'back_{rows_color[i]}'])
                self.insert_text(cell_value, Cursor(0, 1), format_name)
            self.cursor = Cursor(self.cursor.row + 1, start_cursor.col)
        num_cols_to_autofit = table_size[1] + 1
        for col_idx in range(start_cursor.col, start_cursor.col + num_cols_to_autofit):
            self.active_sheet.set_column(col_idx, col_idx, None)
        self.cursor = self.cursor + shift


    def insert_pic(self,
                   image: tp.Any,
                   shift: Cursor):
        '''
        Метод, который добавляет картинку на текущий лист
        Parameters:
            image: Картинка, которую необходимо добавить
            shift: Сдвиг, на который надо сдвинуться после добавления таблицы
        '''
        w, h = image.size
        row_occ = math.ceil(h / self.CELL_HEIGHT)
        col_occ = math.ceil(w / self.CELL_WIDTH)
        self.active_sheet.insert_image(self.cursor.row, 
                                       self.cursor.col, 
                                       'image.png',
                                       {'image_data': PIL2IOBytes(image),
                                        'x_scale': (col_occ * self.CELL_WIDTH) / w,
                                        'y_scale': (row_occ * self.CELL_HEIGHT) / h})
        shift = Cursor(row_occ * int(bool(shift.row)) + shift.row - (1 * int(bool(shift.row))), 
                       col_occ * int(bool(shift.col)) + shift.col - (1 * int(bool(shift.col))))
        self.cursor += shift


    def insert_text_box(self,
                        text: str,
                        shift: Cursor):
        '''
        Метод, который добавляет текстбокс на текущий лист
        Parameters:
            text: Текст, который необходимо добавить в текстбокс
            shift: Сдвиг, на который надо сдвинуться после добавления текста
        '''
        text_height = math.ceil(len(text) * 7.5 / (self.TEXTBOX_CELL * self.CELL_WIDTH)) + 1
        self.active_sheet.insert_textbox(self.cursor.row, 
                                         self.cursor.col, 
                                         text,
                                         {'width': self.TEXTBOX_CELL * self.CELL_WIDTH,
                                          'height': text_height * self.CELL_HEIGHT})
        self.cursor += shift + Cursor(text_height, 0)


    def set_active_sheet(self,
                         name: str):
        '''
        Метод, который добавляет лист в файл
        Parameters:
            name: Название листа
        '''
        if name not in self.wb.sheetnames:
            self.active_sheet = self.wb.add_worksheet(name)
        self.active_sheet = self.wb.sheetnames[name]
        self.cursor.row, self.cursor.col = 0, 0


    def set_tab_color(self,
                      color: str):
        '''
        Метод, который устанавливает цвет листу
        Parameters:
            color: Цвет листа
        '''
        if not color in ['green', 'yellow', 'red', 'gray']:
            color = 'gray'
        self.active_sheet.set_tab_color(color)


    def save_wb(self):
        '''
        Метод, который закрывает и сохраняет таблицу
        '''
        self.wb.close()


    def _get_col_formats(self, 
                         table: pd.DataFrame) -> tp.List[str]:
        '''
        Метод, который возвращает информацию по типу данных в колонках
        Parameters:
            table: pd.DataFrame таблица
        Return:
            Список с типами данных в разрезе колонок
        '''
        col_formats = []
        for col in table.columns:
            if table[col].dtype == np.float64:
                format_name = 'float'
            elif table[col].dtype == np.int64:
                format_name = 'int'
            else:
                format_name = None
            col_formats.append(format_name)
        return col_formats

    
    def _get_rows_color(self, 
                        table: pd.DataFrame):
        '''
        Метод, который возвращает информацию о цвете строки
        Parameters:
            table: pd.DataFrame таблица
        Return:
            Список с названиями цветов в разрезе строк
        '''
        rows_color = [None for _ in range(len(table))]
        color_set = set(['red', 'green', 'gray', 'yellow'])
        color_col = None
        for col in table.columns:
            if len(set(table[col].unique()) - color_set) == 0:
                color_col = col
                break
        if not color_col is None:
            rows_color = table[color_col].values
        return rows_color


    def _create_formats(self):
        '''
        Метод, который создает словарь с форматами данных
        Return:
            Словарь с форматрированием строк
        '''
        formats = {}
        formats['title'] = {'bold': True,
                            'font_size': 18}
        formats['text'] = {}
        formats['column'] = {'bold': True,
                             'border': 2,
                             'align': 'center',
                             'valign': 'vcenter',
                             'text_wrap': 1}
        formats['header'] = {'bold': True,
                             'italic': True}
        formats['red'] = {'bold': True,
                          'font_color': '#BB0000'}
        formats['yellow'] = {'bold': True,
                             'font_color': '#BBBB00'}
        formats['green'] = {'bold': True,
                            'font_color': '#00BB00'}
        formats['gray'] = {'bold': True,
                           'font_color': '#424242'}
        formats['back_red'] = {'bg_color': '#FFC7CE',
                               'font_color': '#9C0006'}
        formats['back_yellow'] = {'bg_color': '#FFEB9C',
                                  'font_color': '#9C5700'}
        formats['back_green'] = {'bg_color': '#C6EFCE',
                                 'font_color': '#006100'}
        formats['back_gray'] = {'bg_color': '#B8B2B1',
                                'font_color': '#424242'}
        formats[None] = {}
        formats['back_None'] = {}
        formats['float'] = {'num_format': '0.000'}
        formats['int'] = {'num_format': '0'}
        formats['text_border'] = {'border': 1}
        formats['text_fat_right_border'] = {'border': 1,
                                            'right': 2}
        formats['text_fat_bottom_border'] = {'border': 1,
                                             'bottom': 2}
        formats['text_fat_right_bottom_border'] = {**formats['text_fat_right_border'],
                                                   **formats['text_fat_bottom_border']}
        
        border_name = ['text_border',
                       'text_fat_right_border',
                       'text_fat_bottom_border',
                       'text_fat_right_bottom_border']
        back_name = ['back_red',
                     'back_yellow',
                     'back_green',
                     'back_gray',
                     'back_None']
        num_name = ['float', 'int', None]
        tmp = itertools.product(border_name, back_name, num_name)
        for border, back, num in tmp:
            formats[f'{border}_{num}_{back}'] = {**formats[border],
                                                 **formats[num],
                                                 **formats[back]}
        res_formats = {}
        for name, props in formats.items():
            res_formats[name] = self.wb.add_format(props)
        return res_formats