import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QSplitter
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor
import re
from main import (
    ResolutionEngine, 
    to_cnf, 
    cnf_to_string,
    parse_formula,
    And,
    Pred
)

from LLM import get_answer_from_LLM


# ИСПРАВЛЕННАЯ функция парсинга вывода LLM
def parse_llm_output(text: str):
    """Парсит вывод LLM в формулу - исправленная версия"""
    # Очищаем и нормализуем текст
    text = text.strip()
    
    # Удаляем лишние пробелы вокруг запятых
    text = re.sub(r'\s*,\s*', ',', text)
    
    clauses = []
    current_clause = []
    depth = 0
    i = 0
    
    while i < len(text):
        char = text[i]
        
        if char == '(':
            depth += 1
            current_clause.append(char)
        elif char == ')':
            depth -= 1
            current_clause.append(char)
        elif char == ',' and depth == 0:
            # Находим запятую на верхнем уровне - конец клаузы
            clause_str = ''.join(current_clause).strip()
            if clause_str:
                clauses.append(clause_str)
            current_clause = []
        else:
            current_clause.append(char)
        
        i += 1
    
    # Добавляем последнюю клаузу
    if current_clause:
        clause_str = ''.join(current_clause).strip()
        if clause_str:
            clauses.append(clause_str)
    
    print(f"Распарсенные клаузы: {clauses}")
    
    if not clauses:
        raise ValueError("Пустой вывод от LLM")

    # Парсим каждую клаузу отдельно
    parsed_clauses = []
    for clause in clauses:
        try:
            parsed = parse_formula(clause)
            parsed_clauses.append(parsed)
            print(f"Успешно распарсено: {clause} -> {parsed}")
        except Exception as e:
            print(f"Ошибка парсинга клаузы '{clause}': {e}")
            raise
    
    # Объединяем через AND
    result = parsed_clauses[0]
    for clause in parsed_clauses[1:]:
        result = And(result, clause)
    
    return result


# ДОПОЛНИТЕЛЬНО: исправленная функция parse_predicate для правильного парсинга аргументов
def parse_predicate_fixed(s: str):
    """Исправленный парсинг предикатов"""
    s = s.strip()
    if '(' not in s:
        return Pred(s, [])

    open_pos = s.find('(')
    name = s[:open_pos].strip()
    args_str = s[open_pos + 1: -1].strip()  # всё внутри скобок

    # Разбиваем аргументы с учетом вложенных скобок
    args = []
    current_arg = []
    depth = 0
    
    for char in args_str:
        if char == '(':
            depth += 1
            current_arg.append(char)
        elif char == ')':
            depth -= 1
            current_arg.append(char)
        elif char == ',' and depth == 0:
            # Запятая на верхнем уровне - конец аргумента
            arg_str = ''.join(current_arg).strip()
            if arg_str:
                args.append(arg_str)
            current_arg = []
        else:
            current_arg.append(char)
    
    # Добавляем последний аргумент
    if current_arg:
        arg_str = ''.join(current_arg).strip()
        if arg_str:
            args.append(arg_str)

    return Pred(name, args)


class StreamRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.original_stdout = sys.stdout

    def write(self, text):
        self.original_stdout.write(text)
        self.text_widget.moveCursor(QTextCursor.MoveOperation.End)
        self.text_widget.insertPlainText(text)
        QApplication.processEvents()

    def flush(self):
        self.original_stdout.flush()


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAT Solver")
        self.resize(1000, 600)
        self.init_ui()

        sys.stdout = StreamRedirector(self.log_field)

    def init_ui(self):
        # ЛЕВАЯ ПАНЕЛЬ (Основной интерфейс)
        self.label_input = QLabel("Введите текст задачи:")
        self.input_field = QTextEdit()

        self.label_output = QLabel("Результат (объяснение):")
        self.output_field = QTextEdit()
        self.output_field.setReadOnly(True)
        self.output_field.setStyleSheet("font-size: 14px;")

        self.button = QPushButton("Получить ответ")
        self.button.clicked.connect(self.process_text)
        self.button.setStyleSheet("padding: 10px; font-weight: bold;")

        # Компоновка левой части
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.label_input)
        left_layout.addWidget(self.input_field)
        left_layout.addWidget(self.label_output)
        left_layout.addWidget(self.output_field)
        left_layout.addWidget(self.button)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # ПРАВАЯ ПАНЕЛЬ (Логи)
        self.label_logs = QLabel("Логи системы:")
        self.log_field = QTextEdit()
        self.log_field.setReadOnly(True)
        self.log_field.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 12px;
            }
        """)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.label_logs)
        right_layout.addWidget(self.log_field)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        # ОБЩАЯ КОМПОНОВКА
        main_layout = QHBoxLayout()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 400])

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def process_text(self):
        self.log_field.clear()
        print(">>> НАЧАЛО ОБРАБОТКИ ЗАПРОСА <<<")

        input_text = self.input_field.toPlainText()

        if not input_text.strip():
            print("ОШИБКА: Поле ввода пустое!")
            return

        try:
            # 1. Читаем промпт
            print("Чтение файла prompt.txt...")
            with open("prompt.txt", mode='r', encoding="UTF-8") as file:
                prompt = file.read()

            LLM_prompt = prompt + "\nТекст задачи: " + input_text

            # 2. Запрос к LLM
            print("-" * 30)
            print("ШАГ 1: Отправка запроса к LLM (понимание задачи)...")
            llm_raw = get_answer_from_LLM(name_openrouter_model="x-ai/grok-4.1-fast:free", prompt=LLM_prompt)

            print(f"LLM выдала (сырые формулы):\n{llm_raw}")
            if not llm_raw.strip():
                print("ОШИБКА: LLM вернул пустой ответ")
                return

            # 3. Сколемизация и КНФ делаем САМИ!
            try:
                print("Парсинг формул от LLM...")
                big_formula = parse_llm_output(llm_raw)
                print(f"Успешно распарсена общая формула: {big_formula}")
                
                print("Преобразование в КНФ...")
                cnf_formula = to_cnf(big_formula)
                print(f"Формула в КНФ: {cnf_formula}")
                
                cnf_string = cnf_to_string(cnf_formula)
                print(f"После сколемизации и приведения к КНФ:\n{cnf_string}")
            except Exception as e:
                print(f"ОШИБКА при символическом преобразовании: {e}")
                import traceback
                print(f"Трассировка: {traceback.format_exc()}")
                self.output_field.setMarkdown("Не удалось преобразовать формулы в КНФ.\nПроверьте ответ LLM.")
                return

            # 4. Запуск резолюций
            print("-" * 30)
            print("ШАГ 2: Запуск метода резолюций...")
            engine = ResolutionEngine()
            proven, steps = engine.prove(cnf_string)

            processed_text = f"Исходные формулы от LLM:\n{llm_raw}\n\n" \
                            f"После сколемизации и КНФ:\n{cnf_string}\n\n" \
                            f"Доказано: {'ДА' if proven else 'НЕТ'}\n"

            steps_text = "\nШаги доказательства:\n"
            for i, step in enumerate(steps, 1):
                line = f"{i}) {step}"
                print(line)
                steps_text += line + "\n"

            full_logic_text = processed_text + steps_text

            # 5. Запрос объяснения
            print("-" * 30)
            print("ШАГ 3: Генерация объяснения...")

            explain_prompt_text = '''Ты — учитель логики. Объясни доказательство,
            представленное в виде последовательности логических
            шагов, как если бы ты объяснял его студенту. Будь
            последовательным и ясным. Используй естественный
            русский язык. Не пиши лишнего, не предлагай проверки на других алгоритмах. Не используй слово клауза, используй дизъюнкт или конъюнкт'''

            LLM_explain_prompt = explain_prompt_text + "\nТекст доказательства:\n" + full_logic_text

            LLM_explain = get_answer_from_LLM(name_openrouter_model="x-ai/grok-4.1-fast:free", prompt=LLM_explain_prompt)

            print("Объяснение получено.")
            print(">>> ЗАВЕРШЕНО <<<")

            self.output_field.setMarkdown(LLM_explain)

        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА: {e}")
            import traceback
            print(f"Трассировка: {traceback.format_exc()}")

    def closeEvent(self, event):
        sys.stdout = sys.__stdout__
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())