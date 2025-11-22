import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, 
    QTextEdit, QPushButton, QLabel
)
from main import ResolutionEngine

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAT solver")
        self.resize(400, 300)
        self.init_ui()

    def init_ui(self):
        #Элементы интерфейса
        self.label_input = QLabel("Введите текст здесь:")
        self.input_field = QTextEdit()
        

        self.label_output = QLabel("Результат:")
        self.output_field = QTextEdit()
        self.output_field.setReadOnly(True)  # Запрет редактирования вывода
        self.output_field.setStyleSheet("background-color: #ffffff;")

        self.button = QPushButton("Получить ответ")
        self.button.clicked.connect(self.process_text)

        layout = QVBoxLayout()
        layout.addWidget(self.label_input)
        layout.addWidget(self.input_field)
        layout.addWidget(self.label_output)
        layout.addWidget(self.output_field)
        layout.addWidget(self.button)

        self.setLayout(layout)

    def process_text(self):
        input_text = self.input_field.toPlainText()

        if not input_text.strip():
             answer = "Поле ввода пустое!"
        else:
            LLM_transcription = input_text # вот сюда добавить ответ LLM
            engine = ResolutionEngine()
            proven, steps = engine.prove(LLM_transcription)
            processed_text = f"Формулы: {LLM_transcription}\nДоказано: {proven}\n"
            for i, step in enumerate(steps, 1):
                processed_text += f"{i}) {step}\n"

        self.output_field.setPlainText(processed_text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())
