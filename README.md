## Описание алгоритма ы

Для деноизинга используется UNet-архитектура + в сети есть дополнительный выход, выполняющий классификацию. Схематично это можно изобразить так.

![UNet](/img/unet_with_class_head.jpg)

Получается, что модель тренируется одновременно и классифицировать чистые/грязные аудиодорожки, представленные в виде mel-спектрограмм, и восстанавливать чистую запись из грязной.

## Как запускать

Докер-репозиторий лежит тут: [egusev/audio-denoising](https://hub.docker.com/r/egusev/audio-denoising)

Чтобы запустить тренировку нужно запустить файл train.py из текущей директории

```bash
docker run --rm -it -v {YOUR_DIR}:/app -w /app denoising python train.py
```

Предсказания получить можно следующим образом:

```bash
docker run --rm -it -v {YOUR_DIR}:/app -w /app denoising python predict.py --mel-path {YOUR_PATH}
```

Ответом будет словарь:

```python
{
    'denoised': denoised,
    'is_clean': is_clean
}
```

В denoised будет лежать очищенная звуковая дорожка оригинальной размерности, в class будет лежать 1, если дорожка считается чистой и 0, если шумы присутствуют.

## Возможные улучшения

- Потренировать подольше (classic)

- Ускорить инференс, разбив обученную на обе задачи одновременно сеть на две подсети - класссифицирующую и убирающую шум

- Добавить признаки, зависящие от голоса человека, как сделали ребята [в этой статье](https://arxiv.org/abs/1810.04826), но это зависит от условий использования модели в дальнейшем
