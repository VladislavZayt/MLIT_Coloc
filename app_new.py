import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QSplitter
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor

from main import ResolutionEngine
from LLM import get_answer_from_LLM

class StreamRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.original_stdout = sys.stdout # Сохраняем стандартный вывод

    def write(self, text):
        self.original_stdout.write(text) 
        
        # Добавляем текст в виджет
        self.text_widget.moveCursor(QTextCursor.MoveOperation.End)
        self.text_widget.insertPlainText(text)
        
        # Обновляем интерфейс немедленно, иначе текст появится 
        # только после завершения всех вычислений
        QApplication.processEvents()

    def flush(self):
        self.original_stdout.flush()

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAT Solver")
        self.resize(1000, 600) 
        self.init_ui()

        # Настраиваем перехват принтов
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
        
        # виджет-контейнер для левой части
        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # ПРАВАЯ ПАНЕЛЬ (Логи)
        self.label_logs = QLabel("Логи системы:")
        self.log_field = QTextEdit()
        self.log_field.setReadOnly(True)
        # консоль: черный фон, зеленый шрифт
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

        # ОБЩАЯ КОМПОНОВКА (Горизонтальная)
        main_layout = QHBoxLayout()
        
        # QSplitter, чтобы можно было двигать границу мышкой
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 400]) # Начальное соотношение ширины

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
            
            # 2. Запрос к LLM для формализации
            print("-" * 30)
            print("ШАГ 1: Отправка запроса к LLM (Формализация)...")
            LLM_transcription = get_answer_from_LLM(name_openrouter_model="x-ai/grok-4.1-fast:free", prompt=LLM_prompt)
            
            print(f"Получен ответ от LLM (сырой):")
            print(LLM_transcription)
            
            if not LLM_transcription:
                print("ОШИБКА: Пустой ответ от LLM")
                return

            # 3. Работа ResolutionEngine
            print("-" * 30)
            print("ШАГ 2: Запуск ResolutionEngine...")
            engine = ResolutionEngine()
            proven, steps = engine.prove(LLM_transcription)
            
            processed_text = f"Формулы: {LLM_transcription}\nДоказано: {proven}\n"
            print(f"Статус доказательства: {proven}")
            
            steps_text = ""
            for i, step in enumerate(steps, 1):
                line = f"{i}) {step}"
                print(line) 
                steps_text += line + "\n"
                
            full_logic_text = processed_text + steps_text

            # Запрос объяснения
            print("-" * 30)
            print("ШАГ 3: Генерация объяснения...")
            
            explain_prompt_text = '''Ты — учитель логики. Объясни доказательство,
            представленное в виде последовательности логических
            шагов, как если бы ты объяснял его студенту. Будь
            последовательным и ясным. Используй естественный
            русский язык.'''
            
            LLM_explain_prompt = explain_prompt_text + "\nТекст доказательства:\n" + full_logic_text
            
            LLM_explain = get_answer_from_LLM(name_openrouter_model="x-ai/grok-4.1-fast:free", prompt=LLM_explain_prompt)
            
            print("Объяснение получено.")
            print(">>> ЗАВЕРШЕНО <<<")
            
            self.output_field.setMarkdown(LLM_explain)
            
        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА: {e}")
            

    # вернуть stdout на место при закрытии, чтобы не сломать терминал
    def closeEvent(self, event):
        sys.stdout = sys.__stdout__
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())
