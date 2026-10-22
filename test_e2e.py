r"""End-to-end test for OTV1 server: Russian speech -> WS -> Whisper + NLLB.

Запуск: .venv\Scripts\python.exe test_e2e.py   (сервер должен быть запущен)
Переадресация на другую машину: set OTV_URL=wss://192.168.1.10:8008/ws
Аудио берётся (по порядку): $OTV_AUDIO -> test_speech.mp3 -> test_speech.wav ->
генерируется Windows-SAPI (офлайн) или edge-tts (нужен интернет и точные часы).
"""
import asyncio, json, os, ssl, subprocess, time
import av
import numpy as np
import websockets

TEXT = "Здравствуйте, уважаемые студенты. Сегодня мы будем изучать нейронные сети и их применение в переводе."
HERE = os.path.dirname(os.path.abspath(__file__))


def sapi_speak_wav(text, wav_path):
    """Синтез речи голосом Windows (русский) в wav — работает без интернета."""
    tmp = wav_path + ".tmp.wav"
    import base64
    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$t = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('" + b64 + "')); "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$v = $s.GetInstalledVoices() | Where-Object { $_.Enabled -and "
        "$_.VoiceInfo.Culture.Name -like 'ru*' } | Select-Object -First 1; "
        "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }; "
        f"$s.SetOutputToWaveFile('{tmp}'); "
        "$s.Speak($t); "
        "$s.Dispose();"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True)
    if r.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) < 1000:
        raise RuntimeError("SAPI TTS failed: " + r.stderr.decode("utf-8", "replace")[:300])
    os.replace(tmp, wav_path)


async def ensure_audio():
    """Возвращает путь к аудиофайлу с русской речью (mp3 или wav)."""
    # 1) явное переопределение
    p = os.environ.get("OTV_AUDIO")
    if p and os.path.exists(p) and os.path.getsize(p) > 0:
        print(f"[audio] using OTV_AUDIO={p}")
        return p
    # 2) edge-tts (интернет + корректные часы); пишем во временный файл,
    #    чтобы при 403 не затереть существующий test_speech.mp3
    mp3 = os.path.join(HERE, "test_speech.mp3")
    tmp = mp3 + ".tmp"
    try:
        import edge_tts
        await edge_tts.Communicate(TEXT, voice="ru-RU-DmitryNeural").save(tmp)
        if os.path.getsize(tmp) > 1000:
            os.replace(tmp, mp3)
            print(f"[audio] edge-tts -> {mp3}")
            return mp3
    except Exception as e:
        print(f"[audio] edge-tts недоступен ({type(e).__name__}: {e})")
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass
    # 3) готовые файлы
    for name in ("test_speech.mp3", "test_speech.wav"):
        p = os.path.join(HERE, name)
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            print(f"[audio] using existing {p}")
            return p
    # 4) офлайн-генерация Windows-SAPI
    wav = os.path.join(HERE, "test_speech.wav")
    sapi_speak_wav(TEXT, wav)
    print(f"[audio] Windows SAPI -> {wav}")
    return wav


async def main():
    # 1) аудио с русской речью
    audio_path = await ensure_audio()

    # 2) декодирование в 16kHz mono float32 через PyAV
    container = av.open(audio_path)
    frames = []
    for frame in container.decode(audio=0):
        arr = frame.to_ndarray()
        if arr.ndim > 1:
            arr = arr[0]
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / 32768.0  # s16 -> [-1,1]
        else:
            arr = arr.astype(np.float32)
        frames.append(arr)
    audio = np.concatenate(frames).astype(np.float32)
    sr_in = int(container.streams.audio[0].rate or 24000)
    container.close()
    if sr_in != 16000:
        n_out = int(len(audio) * 16000 / sr_in)
        x_old = np.linspace(0, 1, len(audio), endpoint=False)
        x_new = np.linspace(0, 1, n_out, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    # pad with 3s of silence so the final partial buffer gets flushed by the server
    audio = np.concatenate([audio, np.zeros(16000 * 3, dtype=np.float32)])
    print(f"[audio] {len(audio)/16000:.2f}s @16kHz mono (incl. silence pad)")

    # 3) Stream over WebSocket in browser-like chunks (4096 samples each)
    results = []
    t0 = time.time()
    url = os.environ.get("OTV_URL", "wss://localhost:8008/ws")
    ssl_ctx = ssl._create_unverified_context() if url.startswith("wss://") else None
    print(f"[ws] connecting {url}")
    async with websockets.connect(url, max_size=None, ssl=ssl_ctx) as ws:
        for i in range(0, len(audio), 4096):
            chunk = audio[i:i+4096]
            await ws.send(chunk.tobytes())
            await asyncio.sleep(0.02)
        # wait for responses (server transcribes every >=3s buffer;
        # CPU-обработка чанка может занимать 20-40 с, поэтому таймауты большие)
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=90)
            except asyncio.TimeoutError:
                break
            data = json.loads(msg)
            results.append(data)
            print(f"[recv @+{time.time()-t0:.1f}s] {json.dumps(data, ensure_ascii=False)}")

    print(f"\n=== TOTAL: {len(results)} translation batches in {time.time()-t0:.1f}s ===")
    assert len(results) > 0, "NO RESPONSE FROM SERVER"
    for lang in ("ru", "en", "de", "zh"):
        full = " ".join(r.get(lang, "") for r in results).strip()
        print(f"\n[{lang}] {full}")

asyncio.run(main())
