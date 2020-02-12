## Описание алгоритма

Для деноизинга используется UNet-архитектура + в сети есть дополнительный выход, выполняющий классификацию. Подробнее на картинке ниже.

![UNet](/img/unet_with_class_head.jpg)

Докер-репозиторий лежит тут: [egusev/audio-denoising](https://hub.docker.com/repository/docker/egusev/audio-denoising)

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
