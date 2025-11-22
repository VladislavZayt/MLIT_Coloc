#1
Создание виртуального окружения
```python -m venv venv```
#2
Активация виртуального окружения
Для линукс
```source venv\bin\activare```
Для windows
```venv\Scripts\activate```
#3
Установка зависимостей
```pip install -r req.txt```
#4
Создаете ключ для LLM в OpenRouter
далее создаете файл .env
туда нужно поместить переменную + ключ
```OPENROUTER_API_KEY=\*ключ\*```
#5
запуск
```python3 app_new.py```


