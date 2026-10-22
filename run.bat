@echo off
REM === ЗАПУСК OTV1 (один сервер: server.py, Python из .venv проекта) ===
REM Работает из любой папки: берёт путь к .venv рядом с этим файлом.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Виртуальное окружение не найдено: %~dp0.venv
    echo Создайте его: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo [INFO] Запуск server.py через .venv...
echo [INFO] Откройте в браузере: https://localhost:8008  (самоподписанный сертификат - "Перейти на сайт")
call ".venv\Scripts\python.exe" server.py

pause
