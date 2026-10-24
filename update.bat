@echo off
chcp 65001 >nul
cd /d %~dp0
echo === Обновление OTV1 из GitHub ===
git pull --ff-only || (echo. & echo НЕ УДАЛОСЬ: есть локальные правки или конфликт. Сделайте commit на этом ПК или откатите правки. & pause & exit /b 1)
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -m pip install -q -r requirements.txt
  echo Зависимости обновлены.
) else (
  echo ВНИМАНИЕ: .venv не найден. Сначала установите окружение: см. README, раздел "Перенос на другую машину".
)
echo Готово. Модели Whisper/NLLB при первом запуске докачаются автоматически.
pause
