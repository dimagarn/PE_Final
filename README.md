[![flake8 Lint](https://github.com/dimagarn/PE_Final/actions/workflows/flake8-Lint.yml/badge.svg)](https://github.com/dimagarn/PE_Final/actions/workflows/flake8-Lint.yml)
[![Python application](https://github.com/dimagarn/PE_Final/actions/workflows/python-app.yml/badge.svg)](https://github.com/dimagarn/PE_Final/actions/workflows/python-app.yml)
# UrFU_SE_Final
Итоговый проект по предмету "Программная инженерия" (осенний семеcтр, 2023г)
## Участники команды
- Гарнышев Дмитрий Александрович, РИМ-130907;
- Коренев Иван Александрович, РИМ-130908;
- Репьева Марина Владимировна, РИМ-130906;
- Юрин Михаил Евгеньевич, РИМ-130907
## Описание модели
[Модель](https://huggingface.co/SamLowe/roberta-base-go_emotions) для классификации настроения текста на английском языке. Данная модель основана на [RoBERTa base model](https://huggingface.co/roberta-base), обученной на датасете [go_emotions](https://huggingface.co/datasets/go_emotions) 
(подробнее ознакомиться с моделью можно по [ссылке](https://huggingface.co/SamLowe/roberta-base-go_emotions)). В качестве входных данных принимается текст в виде строки, в качестве выходных данных выводится список словарей вида  
```{'label': 'настроение', 'score': число с плавающей точкой, выражающее степень уверенности модели в том, что текст соответствует данному настроению}```.  
Пример выходных данных:  
```[{'label': 'admiration', 'score': 0.25540509819984436}, {'label': 'excitement', 'score': 0.23905347287654877}, ...]```
## Применение
Модель может пригодиться для решения проблемы написания текста для презентации на иностранном языке, когда нужно понять, какое настроение передает текст.
## Использование модели
```python
from transformers import pipeline

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

sentences = ["I am not having a great day"]

model_outputs = classifier(sentences)
print(model_outputs[0])
# produces a list of dicts for each of the labels
```
## Web-приложение модели
Данное web-приложение анализирует и выводит настроение текста на английском языке. Используются библиотеки:  
- streamlit
- transformers[torch]
- torch

Для запуска веб интерфейса необходимо в каталоге с файлами проекта ввести в консоли следующую команду:
``` 
streamlit run streamlit.py
```  
В браузере откроется окно с полем для ввода текста: 
![image](https://github.com/dimagarn/PE_Final/assets/136446022/24907d24-030c-47ad-b70d-1680d776deaa)
При вводе текста и нажатии на кнопку "Отправить" (или при нажатии на клавиатуре на клавишу "Enter") будет выведен список возможных настроений текста с их показателями уверенности модели в том, что текст соответствует данному настроению:
![image](https://github.com/dimagarn/PE_Final/assets/136446022/228c1c35-902b-4683-8b5b-671d87247845)
## API модели
Данный API позволяет анализировать и выводить настроение текста на английском языке. Используются библиотеки:
- fastapi
- uvicorn
- transformers[torch]
- pydantic
- torch

Для запуска сервера необходимо в каталоге с файлами проекта ввести в консоли следующую команду:
```
uvicorn main:app
```  
После запуска сервера можно посылать POST-запросы к модели по локальному адресу [http://127.0.0.1:8000/predict/](http://127.0.0.1:8000/predict/) с помощью командной строки (терминала), POSTMAN или через интерфейс документации FastAPI:
![image](https://github.com/dimagarn/PE_Final/assets/136446022/4b37bfae-bce4-449b-bec3-7e24c1c31c46)
![image](https://github.com/dimagarn/PE_Final/assets/136446022/3970625e-1324-4bc7-b857-543a9d45c5b8)
![image](https://github.com/dimagarn/PE_Final/assets/136446022/7ce65ed6-cd5f-4bfb-b248-ba40bfff0688)
## Тестирование
Реализованы тесты, проверяющие корректность работы API. Используются библиотеки:
- pytest
- fastapi
- httpx
- uvicorn
- transformers[torch]
- pydantic
- torch

Для запуска тестирования необходимо в каталоге с файлами проекта ввести в консоли следующую команду:
```
pytest
```
Далее достаточно дождаться окончания выполнения тестирования и узнать об успешности прохождения тестов:
```
=============================================================== 4 passed, 1 warning in 17.87s ===============================================================
```
В данном репозитории также настроена система Continuous Integration: при выполнении push в репозиторий GitHub выполняется автоматический запуск тестов.
## Web-приложение и API модели в облаке
Данные Web-приложение и API модели в облаке анализируют и выводят настроение текста на английском языке.  
- Для того, чтобы воспользоваться Web-приложением модели в облаке достаточно перейти по [ссылке](https://pefinalgit-ak4fqhbnmwfjm2hjtgb7vr.streamlit.app/) на данное web-приложение, ввести текст в поле для ввода текста, нажать на кнопку "Отправить"
  (или на клавишу "Enter" на клавиатуре) и получить ответ от модели в виде списка возможных настроений текста с их показателями уверенности модели в том, что текст соответствует данному настроению:

![image](https://github.com/dimagarn/PE_Final/assets/136446022/6f2b0f88-7b81-49d2-8e9e-da88016bf10d)

- Для того, чтобы воспользоваться API модели в облаке достаточно посылать POST-запросы к модели по публичному адресу [http://158.160.137.174:8000/predict/](http://158.160.137.174:8000/predict/) с помощью командной строки (терминала), POSTMAN или через интерфейс документации FastAPI.

![image](https://github.com/dimagarn/PE_Final/assets/136446022/f0f4db35-418f-4ef0-b906-8fdeaf451f3c)
![image](https://github.com/dimagarn/PE_Final/assets/136446022/cacc079d-e53a-4540-8372-5ae4238cb938)
![image](https://github.com/dimagarn/PE_Final/assets/136446022/a0627a03-6eb5-430e-9066-ca905f6441d7)
