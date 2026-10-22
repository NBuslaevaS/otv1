import asyncio
import json
import logging
import os
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

# Проверяем доступность GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Используется устройство: {device}")

# 1. Загружаем Whisper (модель medium сбалансирована по скорости/качеству)
whisper_model = WhisperModel("medium", device=device, compute_type="float16" if device == "cuda" else "int8")
# 2. Загружаем мультиязычный переводчик Meta NLLB-200 (1.3B параметров)
nllb_model_name = "facebook/nllb-200-distilled-1.3B"
tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name).to(device)
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

# Предрассчитанные ID токенов целевых языков (transformers 5.x: lang_code_to_id удалён)
NLLB_TGT_TOKEN_IDS = {code: tokenizer.convert_tokens_to_ids(code) for code in NLLB_LANG_CODES.values()}

_translate_lock = threading.Lock()  # set_src/set_tgt мутируют состояние токенизатора — защищаем от параллельных вызовов

def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """Функция перевода текста с помощью NLLB-200 на GPU"""
    if src_lang == tgt_lang:
        return text
    
    src_code = NLLB_LANG_CODES.get(src_lang)
    tgt_code = NLLB_LANG_CODES.get(tgt_lang)
    if src_code is None or tgt_code is None:
        raise ValueError(f"Неподдерживаемый язык: src={src_lang}, tgt={tgt_lang}")
    
    with _translate_lock:
        tokenizer.set_src_lang_special_tokens(src_code)
        tokenizer.set_tgt_lang_special_tokens(tgt_code)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.inference_mode():
            translated_tokens = translation_model.generate(
                **inputs, 
                forced_bos_token_id=NLLB_TGT_TOKEN_IDS[tgt_code], 
                max_length=256
            )
    result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return result

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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Буфер накопления звука (накапливаем фрагменты по ~3 секунды для качественного контекста перевода)
    audio_buffer = []
    chunk_size_threshold = 16000 * 3  
    
    async def process_buffer(audio_np):
        """Распознавание + перевод одного фрагмента. Ошибки логируем, не роняем соединение."""
        loop = asyncio.get_running_loop()
        try:
            # VAD отсекает тишину — меньше галлюцинаций Whisper на паузах; beam_size=1 для live-задержки
            segments, info = await loop.run_in_executor(
                None, lambda: whisper_model.transcribe(audio_np, beam_size=1, language="ru", vad_filter=True)
            )
            text_orig = "".join([segment.text for segment in segments]).strip()
            if not text_orig:
                return

            # Переводы запускаем параллельно (лок NLLB всё равно сериализует генерацию,
            # но event loop не блокируется)
            translations = await asyncio.gather(
                loop.run_in_executor(None, translate_text, text_orig, "ru", "en"),
                loop.run_in_executor(None, translate_text, text_orig, "ru", "de"),
                loop.run_in_executor(None, translate_text, text_orig, "ru", "zh"),
            )
            response_data = {
                "ru": text_orig,
                "en": translations[0],
                "de": translations[1],
                "zh": translations[2],
            }
        except Exception:
            logger.exception("Ошибка обработки аудио-фрагмента")
            return

        try:
            await websocket.send_text(json.dumps(response_data))
        except Exception:
            logger.warning("Не удалось отправить ответ (соединение закрыто)")

    async def flush_buffer():
        """Обработка остатка буфера (<3 с) — чтобы хвост речи не терялся."""
        if len(audio_buffer) >= 16000 * 0.5:  # минимум ~полсекунды звука, иначе тишина
            audio_np = np.array(audio_buffer, dtype=np.float32)
            audio_buffer.clear()
            await process_buffer(audio_np)

    try:
        while True:
            # Получаем бинарные данные аудио от браузера
            bytes_data = await websocket.receive_bytes()
            audio_chunk = np.frombuffer(bytes_data, dtype=np.float32)
            audio_buffer.extend(audio_chunk)

            if len(audio_buffer) >= chunk_size_threshold:
                audio_np = np.array(audio_buffer, dtype=np.float32)
                audio_buffer.clear()  # сброс буфера
                await process_buffer(audio_np)

    except Exception as e:
        logger.info(f"Соединение закрыто или ошибка: {e}")
    finally:
        # Финальный flush — хвост речи доходит до распознавания
        try:
            await flush_buffer()
        except Exception:
            pass
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
