import asyncio
import json
import logging
import os
import queue
import threading
import numpy as np
import torch
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("online-translate")

app = FastAPI()

# ===== Конфигурация live-конвейера (задержки), переопределяется переменными окружения =====
# На CPU задержка трансляции = скорость моделей + скорость обработки. Всё настраивается:
OTV_WHISPER_MODEL = os.environ.get("OTV_WHISPER_MODEL", "medium")  # "small" — заметно быстрее, качество чуть ниже
OTV_NLLB_MODEL = os.environ.get("OTV_NLLB_MODEL", "facebook/nllb-200-distilled-1.3B")
# int8-квантизация линейных слоёв: 1.3B становится ~3x быстрее fp32 на CPU и быстрее 600M fp32
# (замер: 0.63 с/фразу против 0.96 у 600M и 2.0 у 1.3B fp32), качество перевода как у полной 1.3B.
OTV_NLLB_QUANT = os.environ.get("OTV_NLLB_QUANT", "1")  # "0" — fp32 (медленнее, смысла на CPU нет)
OTV_CHUNK_SEC = float(os.environ.get("OTV_CHUNK_SEC", "3.0"))      # длина распознаваемого фрагмента (меньше — быстрее, но режет фразы)
WHISPER_THREADS = int(os.environ.get("OTV_WHISPER_THREADS", "8"))  # потоки ctranslate2 (Whisper)
TORCH_THREADS = int(os.environ.get("OTV_TORCH_THREADS", "6"))      # потоков torch на ОДНУ генерацию (три перевода идут параллельно)
torch.set_num_threads(TORCH_THREADS)

# Проверяем доступность GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Используется устройство: {device}")

# 1. Загружаем Whisper (ASR). cpu_threads ограничен, чтобы ядра оставались трём потокам-переводчикам.
whisper_model = WhisperModel(OTV_WHISPER_MODEL, device=device,
                             compute_type="float16" if device == "cuda" else "int8",
                             cpu_threads=WHISPER_THREADS)
# 2. Мультиязычный переводчик Meta NLLB-200: модель ОДНА, но у каждого потока-переводчика
#    свой экземпляр токенизатора (set_src/set_tgt мутируют его состояние — общий нельзя).
try:
    nllb_model_name = OTV_NLLB_MODEL
    translation_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name).to(device).eval()
    if OTV_NLLB_QUANT == "1" and device == "cpu":
        import torch.ao.quantization
        translation_model = torch.ao.quantization.quantize_dynamic(
            translation_model, {torch.nn.Linear}, dtype=torch.qint8)
        logger.info("NLLB: int8 dynamic quantization enabled")
except Exception:
    logger.exception(f"Не удалось загрузить {OTV_NLLB_MODEL}, откат на facebook/nllb-200-distilled-600M")
    nllb_model_name = "facebook/nllb-200-distilled-600M"
    translation_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name).to(device).eval()
if device == "cuda":
    translation_model.half()  # fp16: ~2x быстрее и меньше VRAM

# Коды языков для NLLB-200
NLLB_LANG_CODES = {
    "ru": "rus_Cyrl",
    "en": "eng_Latn",
    "de": "deu_Latn",
    "zh": "zho_Hans",
    "es": "spa_Latn"
}

TRANSLATE_LANGS = ("en", "de", "zh")  # ru -> эти языки, каждый в своём отдельном потоке


def make_tokenizer():
    """Отдельный экземпляр токенизатора для потока-переводчика."""
    return AutoTokenizer.from_pretrained(nllb_model_name)


def translate_with(tokenizer, text: str, src_lang: str, tgt_lang: str) -> str:
    """Перевод текста NLLB-200 своим токенизатором (не разделяется между потоками) — лок не нужен."""
    src_code = NLLB_LANG_CODES.get(src_lang)
    tgt_code = NLLB_LANG_CODES.get(tgt_lang)
    if src_code is None or tgt_code is None:
        raise ValueError(f"Неподдерживаемый язык: src={src_lang}, tgt={tgt_lang}")
    if src_lang == tgt_lang:
        return text
    tokenizer.set_src_lang_special_tokens(src_code)
    tokenizer.set_tgt_lang_special_tokens(tgt_code)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.inference_mode():
        translated_tokens = translation_model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code),
            max_length=256
        )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


# Хранилище HTML-интерфейса (для простоты запуска в одном файле)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Локальный Синхронный Переводчик</title>
    <style>
        body { font-family: sans-serif; background: #1a1a1a; color: #fff; margin: 20px; }
        .controls { margin-bottom: 20px; padding: 15px; background: #2a2a2a; border-radius: 8px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .lang-box { background: #222; border: 2px solid #444; border-radius: 8px; padding: 15px; height: 60vh; overflow-y: auto; }
        .lang-box h2 { margin-top: 0; color: #00adb5; border-bottom: 1px solid #444; padding-bottom: 5px; }
        .text-content { font-size: 24px; line-height: 1.4; white-space: pre-wrap; }
        button { background: #00adb5; border: none; color: white; padding: 10px 20px; font-size: 16px; border-radius: 5px; cursor: pointer; }
        button:disabled { background: #555; }
    </style>
</head>
<body>
    <div class="controls">
        <button id="startBtn">Включить трансляцию на проектор</button>
        <button id="stopBtn" style="margin-left:10px; display:none;">Остановить</button>
        <span id="status" style="margin-left: 15px; color: #aaa;">Ожидание запуска...</span>
    </div>
    
    <div class="grid">
        <div class="lang-box"><h2>Оригинал (Русский)</h2><div id="box-ru" class="text-content"></div></div>
        <div class="lang-box"><h2>English</h2><div id="box-en" class="text-content"></div></div>
        <div class="lang-box"><h2>Deutsch</h2><div id="box-de" class="text-content"></div></div>
        <div class="lang-box"><h2>中文 (Китайский)</h2><div id="box-zh" class="text-content"></div></div>
    </div>

    <script>
        let ws;
        let audioContext = null;
        let userStopped = false;
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const status = document.getElementById('status');

        function setStatus(text, color) {
            status.innerText = text;
            status.style.color = color || "#aaa";
        }

        function stop() {
            try { if (audioContext) audioContext.close(); } catch (e) {}
            audioContext = null;
            try { if (ws) ws.close(); } catch (e) {}
            startBtn.disabled = false;
            stopBtn.style.display = "none";
        }

        async function start() {
            userStopped = false;
            startBtn.disabled = true;
            stopBtn.style.display = "inline-block";
            setStatus("Подключение к серверу AI...");

            // Открываем WebSocket соединение
            const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${wsProto}//${window.location.host}/ws`);

            ws.onopen = async () => {
                try {
                    setStatus("Микрофон активен. Говорите...", "#4e9f3d");

                    // Захват аудио с микрофона лектора (16kHz, моно — стандарт для Whisper)
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    audioContext = new AudioContext({ sampleRate: 16000 });
                    const source = audioContext.createMediaStreamSource(stream);
                    const processor = audioContext.createScriptProcessor(4096, 1, 1);

                    // Gain=0: ScriptProcessor обязан быть подключён к destination,
                    // но звук не проигрывается обратно (нет эхо микрофона)
                    const silentGain = audioContext.createGain();
                    silentGain.gain.value = 0;
                    source.connect(processor);
                    processor.connect(silentGain);
                    silentGain.connect(audioContext.destination);

                    processor.onaudioprocess = (e) => {
                        if (!ws || ws.readyState !== WebSocket.OPEN) return;
                        const inputData = e.inputBuffer.getChannelData(0);
                        // Отправляем сырые аудио-байты на сервер
                        ws.send(inputData.buffer);
                    };
                } catch (err) {
                    setStatus("Ошибка микрофона: " + err.message, "#d9534f");
                    stop();
                }
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                // Распределяем полученные переводы по окнам
                for (const [lang, text] of Object.entries(data)) {
                    const box = document.getElementById(`box-${lang}`);
                    if (box && text) {
                        box.innerText += " " + text;
                        // Автопрокрутка вниз, чтобы лектор и студенты видели новый текст
                        box.parentElement.scrollTop = box.parentElement.scrollHeight;
                    }
                }
            };

            ws.onerror = () => setStatus("Ошибка соединения с сервером", "#d9534f");
            ws.onclose = () => {
                if (userStopped) return;
                stop();
                setStatus("Соединение закрыто. Нажмите «Включить» для перезапуска.", "#d9534f");
            };
        }

        startBtn.onclick = start;
        stopBtn.onclick = () => {
            userStopped = true;
            stop();
            setStatus("Остановлено. Нажмите «Включить» для перезапуска.");
        };
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(HTML_TEMPLATE)

def _put_latest(q, item) -> None:
    """Ставит элемент в очередь; если она полна — выбрасывает самый старый фрагмент.
    Лучше потерять звук давней паузы, чем копить всё растущую задержку трансляции."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Конвейер с малой задержкой вместо последовательной обработки.

    Старая версия обрабатывала фрагмент ПОСЛЕДОВАТЕЛЬНО прямо в цикле приёма
    (ASR + 3 перевода через общий лок ≈ 10 с): пока идет обработка, новые байты
    микрофона копятся в сокете, и задержка растёт всю лекцию.
    Теперь цикл приёма только кладёт аудио в очередь, а обработка параллельная:
    ASR-поток (сразу шлёт оригинал) -> 3 потока-переводчика (en/de/zh),
    каждый со своим токенизатором; результаты складываются в исходящую очередь.
    """
    await websocket.accept()
    loop = asyncio.get_running_loop()
    chunk_samples = int(16000 * OTV_CHUNK_SEC)

    audio_q = queue.Queue(maxsize=40)                                     # ~100 с аудио максимум
    text_qs = {lang: queue.Queue(maxsize=32) for lang in TRANSLATE_LANGS}
    out_q = asyncio.Queue()

    def asr_worker():
        while True:
            buf = audio_q.get()
            if buf is None:
                break
            try:
                # VAD отсекает тишину — меньше галлюцинаций Whisper на паузах; beam_size=1 для live-задержки
                segments, _info = whisper_model.transcribe(buf, beam_size=1, language="ru", vad_filter=True)
                text_orig = "".join(s.text for s in segments).strip()
            except Exception:
                logger.exception("Ошибка распознавания аудио-фрагмента")
                continue
            if not text_orig:
                continue
            # Оригинал уходит в браузер сразу, не дожидаясь переводов
            loop.call_soon_threadsafe(out_q.put_nowait, {"ru": text_orig})
            for tq in text_qs.values():
                _put_latest(tq, text_orig)

    def trans_worker(lang):
        tokenizer = make_tokenizer()  # собственный токенизатор — без мутаций между потоками
        tq = text_qs[lang]
        while True:
            text = tq.get()
            if text is None:
                break
            try:
                translated = translate_with(tokenizer, text, "ru", lang)
            except Exception:
                logger.exception(f"Ошибка перевода фрагмента ({lang})")
                continue
            if translated:
                loop.call_soon_threadsafe(out_q.put_nowait, {lang: translated})

    threads = [threading.Thread(target=asr_worker, daemon=True, name="asr-worker")]
    threads += [threading.Thread(target=trans_worker, args=(lang,), daemon=True, name=f"tr-{lang}")
                for lang in TRANSLATE_LANGS]
    for t in threads:
        t.start()

    async def sender():
        try:
            while True:
                data = await out_q.get()
                await websocket.send_text(json.dumps(data, ensure_ascii=False))
        except Exception:
            pass  # соединение закрыто — завершаемся тихо

    sender_task = asyncio.create_task(sender())
    audio_buffer = []
    try:
        while True:
            # Получаем бинарные данные аудио от браузера; обработка НЕ блокирует приём
            bytes_data = await websocket.receive_bytes()
            audio_buffer.extend(np.frombuffer(bytes_data, dtype=np.float32))
            while len(audio_buffer) >= chunk_samples:
                audio_np = np.array(audio_buffer[:chunk_samples], dtype=np.float32)
                del audio_buffer[:chunk_samples]
                _put_latest(audio_q, audio_np)
    except Exception as e:
        logger.info(f"Соединение закрыто или ошибка: {e}")
    finally:
        # Хвост речи (< полного фрагмента, но >=0.5 с) тоже отправляем, чтобы не терялся
        if len(audio_buffer) >= int(16000 * 0.5):
            _put_latest(audio_q, np.array(audio_buffer, dtype=np.float32))
        audio_q.put(None)
        try:
            # ждём окончания распознавания хвоста, затем закрываем переводчиков
            await loop.run_in_executor(None, lambda: threads[0].join(timeout=60))
            for tq in text_qs.values():
                tq.put(None)
            await loop.run_in_executor(None, lambda: [t.join(timeout=30) for t in threads[1:]])
        except Exception:
            pass
        sender_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass  # клиент уже отключился — traceback не нужен



def ensure_ssl_certificates(cert_dir):
    """Генерирует самоподписанный сертификат (если нет или сменились IP машины).

    Покрывает localhost и все локальные IPv4, чтобы микрофон работал по HTTPS
    при открытии страницы с другого компьютера в локальной сети.
    """
    import ipaddress
    import json as _json
    import socket
    from datetime import datetime, timedelta, timezone

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    os.makedirs(cert_dir, exist_ok=True)
    key_path = os.path.join(cert_dir, "key.pem")
    crt_path = os.path.join(cert_dir, "cert.pem")
    meta_path = os.path.join(cert_dir, "ips.json")

    # Собираем IP этой машины (localhost + все интерфейсы)
    ips = {"127.0.0.1"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass

    # Сертификат переиспользуется, пока список IP не изменился
    if os.path.exists(key_path) and os.path.exists(crt_path):
        try:
            if set(_json.load(open(meta_path))) == ips:
                return key_path, crt_path
        except Exception:
            pass

    logger.info("HTTPS: генерирую самоподписанный сертификат для %s", sorted(ips))
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    san = [x509.DNSName("localhost")] + [x509.IPAddress(ipaddress.ip_address(ip)) for ip in sorted(ips)]
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=824))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .sign(key, hashes.SHA256())
    )
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))
    with open(crt_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(meta_path, "w") as f:
        _json.dump(sorted(ips), f)
    return key_path, crt_path


if __name__ == "__main__":
    import uvicorn
    # HTTPS на порту 8008: микрофон в браузере работает только в secure
    # context (localhost или HTTPS), поэтому страница раздаётся по TLS
    cert_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs")
    key_file, cert_file = ensure_ssl_certificates(cert_dir)
    uvicorn.run(app, host="0.0.0.0", port=8008, ssl_certfile=cert_file, ssl_keyfile=key_file)
