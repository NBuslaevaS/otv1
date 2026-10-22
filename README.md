# OTV1 — локальный синхронный переводчик речи (RU → EN/DE/ZH)

Микрофон лектора → WebSocket → Whisper (ASR) → NLLB-200 (перевод) → 4 окна на страницу проектора.
Сервер: `server.py` (FastAPI + HTTPS). Страница: `https://localhost:8008`.

## Быстрый запуск

1. Двойной клик **`run.bat`** (или запуск из консоли: `C:\OTV1\run.bat`).
2. Дождаться строки `Uvicorn running on https://0.0.0.0:8008` (первый старт грузит модели из кэша ~10–30 с).
3. Открыть в браузере **https://localhost:8008** → при предупреждении сертификата («Небезопасно») → «Перейти на сайт» (сертификат самоподписанный, генерируется в `certs\`).
4. Нажать «Включить трансляцию» и разрешить доступ к микрофону.
5. Для работы с другого компьютера в сети: открыть `https://<IP-этой-машины>:8008` (порт 8008 должен быть разрешён в брандмауэре).

Остановка: Ctrl+C в окне консоли (или закрыть окно run.bat).

## Проверка (e2e-тест)

При запущенном сервере, в отдельном окне консоли:

```bat
C:\OTV1\.venv\Scripts\python.exe C:\OTV1\test_e2e.py
```

- На CPU тест идёт ~1,5–2,5 минуты (Whisper medium int8 + NLLB 1.3B на CPU — это нормально).
- Успех = `rc 0`, `TOTAL: N translation batches` и осмысленные тексты в `[ru] [en] [de] [zh]`.
- Аудио для теста тест берёт сам: `edge-tts` (интернет) → `test_speech.mp3`/`test_speech.wav` → генерация голосом Windows (Microsoft Irina, офлайн). На этой машине edge-tts отдаёт 403 (несовпадение системных часов с реальным временем — токен Microsoft просрочивается), поэтому используется офлайн-генерация — это штатный путь, тест всё равно проходит.
- Другая машина/аудио: `set OTV_URL=wss://192.168.1.10:8008/ws`, `set OTV_AUDIO=C:\path\audio.wav`.

## Что установлено (окружение `.venv`)

Окружение пересобрано на этой машине (2026-10-21): Python 3.14, `torch 2.14.0+cpu`,
`transformers 5.17`, `faster-whisper 1.2.1`, `ctranslate2 4.8.2`, `av`, `websockets`, `edge-tts`, `fastapi`, `uvicorn`, `cryptography`.
Полный список без torch — `reqs_no_torch.txt` (это фактический рабочий набор; `requirements.txt` — старый расширенный список, он не требуется).

Модели закэшированы в `C:\Users\<user>\.cache\huggingface` (`Systran/faster-whisper-medium`, `facebook/nllb-200-distilled-1.3B`) — при наличии интернета качаются сами при первом старте.

## Перенос на другую машину

`.venv` привязан к этой машине. На новой машине:

```bat
cd C:\OTV1
python -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r reqs_no_torch.txt
```

(для GPU: `pip install torch --index-url https://download.pytorch.org/whl/cu126` — сервер сам определит CUDA и включит fp16)

## Неполадки

| Симптом | Решение |
|---|---|
| `Only one usage of each socket address` / порт занят | Другой процесс держит 8008: `netstat -ano \| findstr :8008` → `taskkill /PID <pid> /F`, либо закрыть старое окно сервера |
| Браузер не даёт открыть сайт | «Дополнительно» → «Перейти на localhost (небезопасно)»; сертификат самоподписанный |
| Микрофон молчит / не спрашивает доступ | HTTPS-страница требует разрешения микрофона в замке в адресной строке; на `http://` браузер микрофон не даст |
| `[ERROR] Виртуальное окружение не найдено` в run.bat | Пересоздать `.venv` (см. «Перенос на другую машину») |
| Тест `NO RESPONSE FROM SERVER` | Сервер не запущен / неверный `OTV_URL`; на слабом CPU увеличить `timeout` в `test_e2e.py` |
| Медленная работа | Это CPU-режим. Для ускорения — CUDA-GPU (torch cu126) или модель whisper `small` в `server.py` |

## Файлы

- `server.py` — сервер (ASR+перевод+HTML), HTTPS :8008
- `run.bat` — запуск сервера
- `test_e2e.py` — сквозной тест (TTS → WS → Whisper → NLLB)
- `test_speech.mp3` / `test_speech.wav` — тестовое аудио (создаются автоматически)
- `certs/` — самоподписанный сертификат (регенерируется при смене IP)
- `.venv_old/` — старое сломанное окружение с другой машины (можно удалить)
