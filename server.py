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
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Локальный Синхронный Переводчик</title>
<style>
    :root {
        --accent1: #22d3ee;
        --accent2: #818cf8;
        --accent3: #e879f9;
        --glass: rgba(255, 255, 255, 0.055);
        --glass-border: rgba(255, 255, 255, 0.13);
        --text-soft: rgba(235, 240, 255, 0.72);
    }
    * { box-sizing: border-box; }
    html, body { min-height: 100%; }
    body {
        margin: 0; color: #eef2ff;
        font-family: "Segoe UI", system-ui, -apple-system, Roboto, sans-serif;
        background: #0b1020;
        background-image:
            radial-gradient(1100px 620px at 15% -10%, rgba(56, 189, 248, 0.16), transparent 60%),
            radial-gradient(1000px 700px at 110% 10%, rgba(167, 139, 250, 0.17), transparent 60%),
            radial-gradient(900px 620px at 50% 115%, rgba(232, 121, 249, 0.12), transparent 60%);
        background-attachment: fixed;
        overflow-x: hidden;
    }
    /* «Живой» фон: медленно плывущие световые пятна */
    .blob { position: fixed; border-radius: 50%; filter: blur(90px); opacity: .5; z-index: -1; animation: drift 26s ease-in-out infinite alternate; }
    .blob.b1 { width: 520px; height: 520px; left: -140px; top: -160px; background: rgba(34, 211, 238, 0.22); }
    .blob.b2 { width: 560px; height: 560px; right: -160px; top: 4%; background: rgba(129, 140, 248, 0.20); animation-delay: -9s; }
    .blob.b3 { width: 620px; height: 620px; left: 28%; bottom: -260px; background: rgba(232, 121, 249, 0.14); animation-delay: -17s; }
    @keyframes drift {
        0%   { transform: translate(0, 0) scale(1); }
        50%  { transform: translate(60px, 40px) scale(1.12); }
        100% { transform: translate(-40px, 70px) scale(0.95); }
    }

    /* ===== Приветственный экран ===== */
    .welcome {
        min-height: 100vh; display: flex; flex-direction: column;
        align-items: center; justify-content: center; text-align: center;
        padding: 40px 20px; gap: 0;
    }
    h1.title {
        margin: 0 0 18px; font-weight: 800; line-height: 1.12;
        font-size: clamp(34px, 6vw, 68px); letter-spacing: -1px;
        background: linear-gradient(100deg, #67e8f9 0%, #a5b4fc 45%, #f0abfc 80%);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent; color: transparent;
        text-shadow: 0 0 90px rgba(129, 140, 248, 0.35);
        animation: fadeUp .7s .1s ease both;
    }
    .subtitle {
        margin: 0 auto 44px; max-width: 720px;
        font-size: clamp(17px, 2.2vw, 22px); color: var(--text-soft); line-height: 1.55;
        animation: fadeUp .7s .2s ease both;
    }
    .subtitle b { color: #dbeafe; font-weight: 600; }
    @keyframes fadeUp { from { opacity: 0; transform: translateY(22px); } to { opacity: 1; transform: none; } }

    /* ===== Стеклянные карточки ===== */
    .card {
        width: min(640px, 94vw); padding: 30px 34px; border-radius: 22px; text-align: left;
        background: var(--glass); border: 1px solid var(--glass-border);
        backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.38), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        animation: fadeUp .55s ease both; margin-bottom: 18px;
    }
    .step-question { font-size: 21px; font-weight: 600; margin: 0 0 22px; }

    /* Счётчик языков с кнопками - / + */
    .counter { display: flex; align-items: center; justify-content: center; gap: 16px; margin: 6px 0 24px; }
    .counter button {
        width: 54px; height: 54px; border-radius: 50%; font-size: 26px; line-height: 1;
        border: 1px solid var(--glass-border); background: rgba(255, 255, 255, 0.07);
        color: #eef2ff; cursor: pointer; transition: .18s;
    }
    .counter button:hover { background: rgba(103, 232, 249, 0.18); transform: translateY(-2px); }
    .counter input {
        width: 92px; height: 74px; text-align: center; font-size: 40px; font-weight: 700;
        color: #fff; background: rgba(0, 0, 0, 0.28);
        border: 1px solid var(--glass-border); border-radius: 16px; outline: none;
        transition: .2s; -moz-appearance: textfield;
    }
    .counter input:focus { border-color: var(--accent1); box-shadow: 0 0 0 4px rgba(34, 211, 238, 0.18); }
    .counter input::-webkit-outer-spin-button, .counter input::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
    .counter-note { text-align: center; color: var(--text-soft); font-size: 14px; margin-bottom: 22px; }

    .btn {
        display: block; width: 100%; padding: 16px 26px; border: none; border-radius: 14px;
        font-size: 17px; font-weight: 700; letter-spacing: .3px; color: #06121f; cursor: pointer;
        background: linear-gradient(100deg, var(--accent1), var(--accent2));
        box-shadow: 0 12px 34px rgba(56, 189, 248, 0.35); transition: .18s;
    }
    .btn:hover { transform: translateY(-2px); box-shadow: 0 16px 42px rgba(129, 140, 248, 0.45); }
    .btn:active { transform: translateY(0); }
    .btn:disabled { background: #3a4152; color: #98a2b8; box-shadow: none; cursor: default; transform: none; }
    .btn.ghost {
        background: transparent; color: var(--text-soft); border: 1px solid var(--glass-border);
        box-shadow: none; font-weight: 500;
    }
    .btn.ghost:hover { color: #fff; background: rgba(255, 255, 255, 0.06); }
    .btn.danger { background: linear-gradient(100deg, #fb7185, #f43f5e); color: #fff; box-shadow: 0 12px 34px rgba(244, 63, 94, 0.32); }

    .error { color: #fda4af; font-size: 14px; min-height: 20px; margin: 12px 0 0; text-align: center; }

    /* Списки языков */
    .lang-row {
        display: flex; align-items: center; gap: 14px; padding: 11px 14px; margin-bottom: 10px;
        background: rgba(0, 0, 0, 0.22); border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px; transition: .2s;
    }
    .lang-row:hover { border-color: rgba(103, 232, 249, 0.4); }
    .lang-row .num {
        width: 34px; height: 34px; flex: 0 0 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 15px; color: #06121f;
        background: linear-gradient(135deg, var(--accent1), var(--accent2));
    }
    .lang-row select {
        flex: 1; padding: 12px 14px; font-size: 16px; color: #eef2ff; outline: none; cursor: pointer;
        background: rgba(255, 255, 255, 0.07); border: 1px solid var(--glass-border); border-radius: 10px;
        transition: .2s;
    }
    .lang-row select:focus, .lang-row select:hover { border-color: var(--accent1); }
    .lang-row option { background: #151b2e; color: #eef2ff; }
    .row-actions { display: flex; gap: 12px; margin-top: 18px; }
    .row-actions .btn { width: auto; flex: 1; padding: 14px 22px; }

    /* ===== Экран трансляции ===== */
    .live-wrap { display: none; min-height: 100vh; padding: 26px 28px; flex-direction: column; }
    .live-head {
        display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
        padding: 16px 22px; margin-bottom: 22px; border-radius: 18px;
        background: var(--glass); border: 1px solid var(--glass-border); backdrop-filter: blur(18px);
    }
    .live-head .live-title {
        font-size: 19px; font-weight: 700; margin-right: auto;
        background: linear-gradient(100deg, #67e8f9, #c4b5fd);
        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    }
    .mic-dot { width: 11px; height: 11px; border-radius: 50%; background: #64748b; margin-right: 8px; display: inline-block; }
    .mic-dot.on { background: #f43f5e; animation: pulse 1.4s infinite; }
    @keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(244, 63, 94, 0.55); } 50% { box-shadow: 0 0 0 9px rgba(244, 63, 94, 0); } }
    .status { color: var(--text-soft); font-size: 14.5px; }
    .live-head .btn { width: auto; padding: 11px 24px; font-size: 15px; }

    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; flex: 1; align-content: start; }
    .lang-box {
        background: var(--glass); border: 1px solid var(--glass-border); border-radius: 20px;
        padding: 20px 22px; height: 62vh; overflow-y: auto; backdrop-filter: blur(14px);
        box-shadow: 0 18px 44px rgba(0, 0, 0, 0.30); animation: fadeUp .5s ease both;
        transition: border-color .3s;
    }
    .lang-box:hover { border-color: rgba(103, 232, 249, 0.35); }
    .lang-box h2 {
        margin: 0 0 14px; font-size: 17px; letter-spacing: .4px; color: #c7d2fe;
        border-bottom: 1px solid rgba(255, 255, 255, 0.10); padding-bottom: 10px;
        position: sticky; top: -20px; background: #141a2e;
    }
    .lang-box .text-content { font-size: 23px; line-height: 1.5; white-space: pre-wrap; color: #f1f5ff; }

    ::-webkit-scrollbar { width: 9px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.16); border-radius: 8px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.28); }

    @media (max-width: 640px) {
        .card { padding: 24px 20px; }
        .live-wrap { padding: 14px; }
        .lang-box { height: 40vh; }
        .lang-box .text-content { font-size: 19px; }
    }
</style>
</head>
<body>
    <div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div>

    <!-- ===== Приветственный экран: заголовок по центру ===== -->
    <div class="welcome" id="welcomeScreen">
        <h1 class="title">Локальный синхронный<br>переводчик</h1>
        <p class="subtitle"><b>Добро пожаловать на сервис!</b><br>Выберите количество языков для перевода лекций</p>

        <!-- Шаг 1 -->
        <div class="card" id="step1Card">
            <p class="step-question" style="text-align:center;">На каком языке вы читаете лекцию?</p>
            <div class="lang-row" style="max-width:360px;margin:0 auto 30px;">
                <select id="srcLang"></select>
                <style>#srcLang{width:100%;}</style>
            </div>
            <p class="step-question" style="text-align:center;">Сколько языков перевода нужно?</p>
            <div class="counter">
                <button type="button" id="minusBtn" title="Меньше">&minus;</button>
                <input type="number" id="numLangs" min="1" max="4" value="3">
                <button type="button" id="plusBtn" title="Больше">+</button>
            </div>
            <div class="counter-note">от 1 до 4 языков</div>
            <button class="btn" id="buildBtn">Выбрать языки &#8594;</button>
            <div class="error" id="setupError"></div>
        </div>

        <!-- Шаг 2 -->
        <div class="card" id="langSelects" style="display:none;">
            <p class="step-question" style="text-align:center;">Какие языки переводить?</p>
            <div id="selectRows"></div>
            <div class="row-actions">
                <button class="btn ghost" id="backBtn">&#8592; Назад</button>
                <button class="btn" id="applyBtn">Продолжить &#8594;</button>
            </div>
            <div class="error" id="applyError"></div>
        </div>
    </div>

    <!-- ===== Экран трансляции ===== -->
    <div class="live-wrap" id="liveScreen">
        <div class="live-head">
            <span class="mic-dot" id="micDot"></span>
            <span class="live-title">Локальный синхронный переводчик</span>
            <span class="status" id="status"></span>
            <button class="btn" id="startBtn">&#127908;&nbsp; Начать запись и трансляцию</button>
            <button class="btn danger" id="stopBtn" style="display:none;">&#9632;&nbsp; Стоп</button>
            <button class="btn ghost" id="changeBtn" style="display:none;">&#8634;&nbsp; Изменить языки</button>
        </div>
        <div class="grid" id="grid"></div>
    </div>

    <script>
        let ws;
        let audioContext = null;
        let userStopped = false;
        let targets = [];

        // Поддерживаемые языки (NLLB + Whisper)
        const LANGS = {
            ru: { name: "Русский",   flag: "🇷🇺" },
            en: { name: "English",   flag: "🇬🇧" },
            de: { name: "Deutsch",   flag: "🇩🇪" },
            zh: { name: "中文",       flag: "🇨🇳" },
            es: { name: "Español",   flag: "🇪🇸" }
        };
        const TRANSLATE_ORDER = ["en", "de", "zh", "es"];   // языки перевода (без ru)
        let sourceLang = "ru";                              // язык лекции

        const $ = id => document.getElementById(id);
        const numLangs = $("numLangs"), setupError = $("setupError"), applyError = $("applyError");
        const step1Card = $("step1Card"), langSelects = $("langSelects"), selectRows = $("selectRows");
        const welcomeScreen = $("welcomeScreen"), liveScreen = $("liveScreen");
        const startBtn = $("startBtn"), stopBtn = $("stopBtn"), changeBtn = $("changeBtn");
        const status = $("status"), grid = $("grid"), micDot = $("micDot");

        function clampN() {
            let n = parseInt(numLangs.value, 10);
            if (isNaN(n)) n = 1;
            numLangs.value = Math.min(4, Math.max(1, n));
        }
        $("minusBtn").onclick = () => { numLangs.value = Math.max(1, (parseInt(numLangs.value,10)||1) - 1); };
        $("plusBtn").onclick  = () => { numLangs.value = Math.min(4, (parseInt(numLangs.value,10)||1) + 1); };
        numLangs.onchange = clampN;

        // Язык лекции: любой из пяти; языки перевода — все, кроме выбранного
        const srcLangSel = $("srcLang");
        for (const [code, l] of Object.entries(LANGS)) {
            const opt = document.createElement("option");
            opt.value = code;
            opt.innerText = l.flag + "  " + l.name;
            if (code === "ru") opt.selected = true;
            srcLangSel.appendChild(opt);
        }
        srcLangSel.onchange = () => { sourceLang = srcLangSel.value; };

        function setStatus(text, color) { status.innerText = text; status.style.color = color || ""; }

        function stopAudio() {
            try { if (audioContext) audioContext.close(); } catch (e) {}
            audioContext = null;
            try { if (ws) ws.close(); } catch (e) {}
            micDot.classList.remove("on");
        }

        // Шаг 1: ввели количество N -> показываем N выпадающих списков
        $("buildBtn").onclick = () => {
            setupError.innerText = "";
            sourceLang = srcLangSel.value;
            const n = parseInt(numLangs.value, 10);
            const maxN = Object.keys(LANGS).length - 1; // все языки, кроме языка лекции
            if (isNaN(n) || n < 1 || n > maxN) {
                setupError.innerText = "Укажите количество от 1 до " + maxN;
                return;
            }
            // Для перевода доступны ВСЕ языки базы, кроме самого языка лекции
            // (при иностранном языке лекции в списке оказывается и русский)
            const options = Object.keys(LANGS).filter(c => c !== sourceLang);
            if (n > options.length) {
                setupError.innerText = "Для языка лекции доступно " + options.length + " язык(ов) перевода";
                return;
            }
            selectRows.innerHTML = "";
            for (let i = 0; i < n; i++) {
                const row = document.createElement("div");
                row.className = "lang-row";
                const num = document.createElement("span");
                num.className = "num"; num.innerText = i + 1;
                const sel = document.createElement("select");
                for (const code of options) {
                    const opt = document.createElement("option");
                    opt.value = code;
                    opt.innerText = LANGS[code].flag + "  " + LANGS[code].name;
                    if (code === options[i]) opt.selected = true;
                    sel.appendChild(opt);
                }
                row.appendChild(num); row.appendChild(sel);
                selectRows.appendChild(row);
            }
            step1Card.style.display = "none";
            langSelects.style.display = "block";
        };
        $("backBtn").onclick = () => {
            langSelects.style.display = "none";
            step1Card.style.display = "block";
        };

        // Шаг 2: подтвердили языки -> окна с языками, как в первой версии
        $("applyBtn").onclick = () => {
            applyError.innerText = "";
            const chosen = Array.from(selectRows.querySelectorAll("select")).map(s => s.value);
            if (new Set(chosen).size !== chosen.length) {
                applyError.innerText = "Языки не должны повторяться — выберите разные.";
                return;
            }
            targets = chosen;
            grid.innerHTML = "";
            addBox(sourceLang, LANGS[sourceLang].flag + "  Оригинал · " + LANGS[sourceLang].name);
            targets.forEach(l => addBox(l, LANGS[l].flag + "  " + LANGS[l].name));
            welcomeScreen.style.display = "none";
            liveScreen.style.display = "flex";
            startBtn.style.display = "inline-block";
            stopBtn.style.display = "none";
            changeBtn.style.display = "inline-block";
            setStatus("Нажмите «Начать запись и трансляцию» и разрешите доступ к микрофону.");
        };

        $("changeBtn").onclick = () => {
            stopAudio();
            liveScreen.style.display = "none";
            welcomeScreen.style.display = "flex";
            langSelects.style.display = "none";
            step1Card.style.display = "block";
        };

        function addBox(code, title) {
            const box = document.createElement("div");
            box.className = "lang-box";
            const h = document.createElement("h2");
            h.innerText = title;
            const content = document.createElement("div");
            content.id = "box-" + code;
            content.className = "text-content";
            box.appendChild(h); box.appendChild(content);
            grid.appendChild(box);
        }

        async function start() {
            userStopped = false;
            startBtn.style.display = "none";
            changeBtn.style.display = "none";
            stopBtn.style.display = "inline-block";
            setStatus("Подключение к серверу AI...");

            const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
            ws = new WebSocket(`${wsProto}//${window.location.host}/ws`);

            ws.onopen = async () => {
                try {
                    // Первым сообщением сообщаем серверу язык лекции и языки перевода
                    ws.send(JSON.stringify({ source: sourceLang, targets: targets }));
                    setStatus("Микрофон активен. Говорите...", "#86efac");
                    micDot.classList.add("on");

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
                    setStatus("Ошибка микрофона: " + err.message, "#fda4af");
                    stopAudio(); resetButtons();
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

            ws.onerror = () => setStatus("Ошибка соединения с сервером", "#fda4af");
            ws.onclose = () => {
                if (userStopped) return;
                stopAudio(); resetButtons();
                setStatus("Соединение закрыто. Нажмите «Начать запись» для перезапуска.", "#fda4af");
            };
        }

        function resetButtons() {
            startBtn.style.display = "inline-block";
            stopBtn.style.display = "none";
            changeBtn.style.display = "inline-block";
        }

        startBtn.onclick = start;
        stopBtn.onclick = () => {
            userStopped = true;
            stopAudio(); resetButtons();
            setStatus("Остановлено. Можно перезапустить запись или изменить языки.");
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

    # Браузер первым сообщением (JSON) присылает язык лекции и языки перевода:
    # {"source": "de", "targets": ["en", "zh"]}. Без сообщения — ru -> TRANSLATE_LANGS.
    src_lang = "ru"
    target_langs = list(TRANSLATE_LANGS)
    try:
        first = await asyncio.wait_for(websocket.receive_text(), timeout=5)
        handshake = json.loads(first) if first.strip().startswith("{") else {}
        req_src = handshake.get("source")
        if req_src in NLLB_LANG_CODES:
            src_lang = req_src
        requested = handshake.get("targets")
        if isinstance(requested, list) and requested:
            valid = [l for l in dict.fromkeys(requested)
                     if l in NLLB_LANG_CODES and l != src_lang]
            if valid:
                target_langs = valid
            else:
                target_langs = [l for l in TRANSLATE_LANGS if l != src_lang]
        else:
            target_langs = [l for l in TRANSLATE_LANGS if l != src_lang]
        logger.info("Трансляция: %s -> %s", src_lang, ", ".join(target_langs))
    except Exception:
        target_langs = [l for l in target_langs if l != src_lang] or list(TRANSLATE_LANGS)
        # без JSON-рукопожатия — ru -> языки по умолчанию

    whisper_lang = src_lang  # у faster-whisper коды ru/en/de/zh/es совпадают с NLLB-кодами проекта
    audio_q = queue.Queue(maxsize=40)                                     # ~100 с аудио максимум
    text_qs = {lang: queue.Queue(maxsize=32) for lang in target_langs}
    out_q = asyncio.Queue()

    def asr_worker():
        while True:
            buf = audio_q.get()
            if buf is None:
                break
            try:
                # VAD отсекает тишину — меньше галлюцинаций Whisper на паузах; beam_size=1 для live-задержки
                # язык лекции задаёт браузер (ru/en/de/zh/es), иначе Whisper определяет сам
                segments, _info = whisper_model.transcribe(buf, beam_size=1, language=whisper_lang, vad_filter=True)
                text_orig = "".join(s.text for s in segments).strip()
            except Exception:
                logger.exception("Ошибка распознавания аудио-фрагмента")
                continue
            if not text_orig:
                continue
            # Оригинал уходит в браузер сразу, не дожидаясь переводов
            loop.call_soon_threadsafe(out_q.put_nowait, {src_lang: text_orig})
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
                translated = translate_with(tokenizer, text, src_lang, lang)
            except Exception:
                logger.exception(f"Ошибка перевода фрагмента ({lang})")
                continue
            if translated:
                loop.call_soon_threadsafe(out_q.put_nowait, {lang: translated})

    threads = [threading.Thread(target=asr_worker, daemon=True, name="asr-worker")]
    threads += [threading.Thread(target=trans_worker, args=(lang,), daemon=True, name=f"tr-{lang}")
                for lang in target_langs]
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
