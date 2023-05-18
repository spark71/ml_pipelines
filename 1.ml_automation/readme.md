## 1. Установка окружения и библиотек

В папке проекта устанавливаем окружение. 
```bash
pip3.9 install virtualenv
python -m venv env
source env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Описание конвеерных скриптов

| Скрипт                  | описание                                                    |
| ----------------------- | ----------------------------------------------------------- |
| `data_creation.py`      | создание набора данных                                      |
| `data_preprocessing.py` | предобработка данных (устранение выбросов, кодирование признаков) |
| `model_preparation.py`  | создание и обучение модели на train-данных                  |
| `model_testing.py`      | проверка работы модели на test-данных                       |
| `pipeline.sh`                        |   последовательный запуск скриптов                                                            |

## 3. Последовательный запуск конвеерных скриптов

```bash
sh pieline.sh
```





