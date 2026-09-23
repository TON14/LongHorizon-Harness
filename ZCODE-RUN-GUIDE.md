# Запуск lhht с бэкендом ZCode — рабочая инструкция

Составлено 21.09.2026 по боевой конфигурации прогонов 20–21.09 (7 успешных задач).
Всё проверено на этой машине; пути и версии актуальны.

---

## 0. Что где стоит (эталон)

| Компонент | Путь / версия |
|---|---|
| Форк lhht (источник правды) | ваш локальный чекаут форка, версия **0.3.4** |
| Установка | `pip install --user .` (из корня форка) |
| Исполняемый файл | `%USERPROFILE%\AppData\Roaming\Python\Python314\Scripts\lhht.exe` |
| ZCode CLI (headless runtime) | `%USERPROFILE%\AppData\Local\Programs\ZCode\resources\glm\zcode.cjs`, версия **0.16.9** |
| Обёртка (альтернатива) | `%USERPROFILE%\.local\bin\zcode.cmd` = `node …zcode.cjs %*` |
| Модель | `glm-5.3-flash`, усилия: **low / high / max** |
| Python | 3.14 (`C:\Python314\python.exe`) |
| Node | ≥ 20 (сейчас 24.14.1) |
| Логин/API-ключ ZCode | `~\.zcode\v2\config.json` (подхватывается адаптером сам) |
| Провайдер direct-ключа | `~\.zcode\v2\provider_config.json` (пишет lhht при `--api-key`, провайдер `zai-direct`, base `https://api.z.ai/api/anthropic`) |

---

## 1. Быстрая диагностика: `lhht doctor`

```cmd
%USERPROFILE%\AppData\Roaming\Python\Python314\Scripts\lhht.exe doctor
```

Должно быть:
- `[OK ] zcode: 0.16.x (…\ZCode\resources\glm\zcode.cjs); effort: low, high, max`
- `[OK ] Python: 3.14…`, `[OK ] Node.js`

Если `zcode` не найден — задайте переменную (см. §3) и повторите. Строка
`[SKIP] Project config: .lhht\config.toml does not exist` означает, что вы не в
каталоге проекта — конфиг ищется в текущем каталоге (§2).

---

## 2. Главные правила (из-за них обычно «не работает»)

1. **Дефолтный агент — `codex`; с 23.09.2026 он установлен** (`npm i -g
   @openai/codex`, 0.156.1) и делит логин с десктопным приложением OpenAI.Codex
   через общий `~\.codex\auth.json` — отдельно логиниться не нужно. Для
   GLM-схемы всё равно нужен явный `agent = "zcode"` (конфиг) или `--agent
   zcode` (CLI). Модели codex: gpt-5.6-sol (дефолт lhht) / terra / luna /
   gpt-6-astra / gpt-5.5 — каталог подтягивает их сам.
2. **`lhht` ищет конфиг `.lhht/config.toml` в текущем каталоге.** Запускайте из
   каталога проекта (`cd /d <проект>`), тогда workspace = cwd и подхватываются
   `[run]`-дефолты. Конфиг в другом каталоге на прогон не влияет.
3. **Дефолтные таймауты слишком малы** (manager/auditor 300 с, executor 1800 с).
   Эпизоды на реальных задачах идут 10–40+ минут — ставьте 10800 с (3 ч).
4. **ZCode-десктопный лаунчер не годится** (открывает GUI, флаги игнорирует).
   Нужен именно headless-рантайм `zcode.cjs` из установки (или обёртка
   `~\.local\bin\zcode.cmd`). Адаптер сам общается с ним по протоколу
   `app-server --stdio` (у ZCode ≥0.16 флаг `-p` не работает — это нормально).
5. **Тишина `events.jsonl` ≠ зависание.** События пишутся на границах эпизодов.
   Живость проверять по `chat.jsonl`/роликам в `runs\<id>\lhht\*_episodes\`,
   rollout-файлам `~\.zcode\cli\rollout\model-io-sess_*.jsonl` и росту CPU.

---

## 3. Готовый проектный конфиг (эталон, работает как есть)

`<проект>\.lhht\config.toml` (создать вручную или `lhht init` и привести к виду):

```toml
# Backend: ZCode headless on the Z.ai coding plan.
# Valid reasoning efforts for zai models: low, high, max.

[run]
agent = "zcode"                # ОБЯЗАТЕЛЬНО: дефолт codex не установлен
model = "glm-5.3-flash"

env = "local"
runs_root = "./.lhht/runs"
prompt_language = "en"
mcp_add_dirs = []
guard_exclude_paths = []       # сюда — волатильные каталоги сборок, если есть

max_rounds = 60                # дефолт харнесса 25 — мало
dashboard = false

[run.timeouts]                 # дефолты 300/1800 с — слишком мало
manager = 10800
gui_executor = 10800
cli_executor = 10800
auditor = 10800

[run.roles.manager]            # проверенная схема ролей
reasoning_effort = "low"       # планировщик — быстро и дёшево
[run.roles.executor]
reasoning_effort = "high"      # исполнитель — рабочий уровень
[run.roles.auditor]
reasoning_effort = "max"       # аудитор — максимум
```

Если бинарник zcode не находится сам (нестандартный путь установки), перед
запуском:

```bat
set "LHHT_ZCODE_BINARY=%USERPROFILE%\AppData\Local\Programs\ZCode\resources\glm\zcode.cjs"
rem либо на обёртку: set "LHHT_ZCODE_BINARY=%USERPROFILE%\.local\bin\zcode.cmd"
```

---

## 4. Запуск одной задачи

```cmd
cd /d D:\путь\к\проекту
%USERPROFILE%\AppData\Roaming\Python\Python314\Scripts\lhht.exe run --task "@TASK-my.md" --max-rounds 40
```

- `--task` принимает текст или `@путь-к-файлу` (относительно workspace).
- Всё из конфига можно переопределить флагами: `--agent/--model/--reasoning-effort`
  (на все роли) или полегче — `--manager-model`, `--executor-model`,
  `--auditor-model`, `--*-reasoning-effort`.
- Полезные флаги: `--run-id <имя>` (именованный прогон), `--workspace <dir>`
  (рабочая папка агентов), `--guard-exclude-path <rel>` (волатильные пути),
  `--mcp-add-dir <dir>` (доп. каталог), `--api-key`/`--base-url` (свой ключ Z.ai
  вместо логина десктопа; lhht сам пропишет провайдера `zai-direct` в
  `~\.zcode\v2\provider_config.json`).
- Результат: `<runs_root>\<run-id>\` — `FINAL_RESPONSE.md` (итог для вас),
  `chat.jsonl`, `lhht\events.jsonl`, подкаталоги эпизодов ролей.

---

## 5. Драйвер для нескольких задач подряд (шаблон из боёв)

`run-driver.bat` — задачи строго последовательно, лог с EXIT-кодами:

```bat
@echo off
setlocal
set "LHHT_ZCODE_BINARY=%USERPROFILE%\AppData\Local\Programs\ZCode\resources\glm\zcode.cjs"
set "LHHT=%USERPROFILE%\AppData\Roaming\Python\Python314\Scripts\lhht.exe"
cd /d D:\путь\к\проекту
set "LOG=_work\LHHT-DRIVER.log"

echo === DRIVER START %date% %time% === >> "%LOG%"

echo === T1 START %date% %time% === >> "%LOG%"
"%LHHT%" run --task "@TASK-one.md" --max-rounds 60 >> "%LOG%" 2>&1
echo === T1 EXIT %errorlevel% %date% %time% === >> "%LOG%"

echo === T2 START %date% %time% === >> "%LOG%"
"%LHHT%" run --task "@TASK-two.md" --max-rounds 60 >> "%LOG%" 2>&1
echo === T2 EXIT %errorlevel% %date% %time% === >> "%LOG%"

echo === DRIVER DONE %date% %time% === >> "%LOG%"
endlocal
```

Запуск: `start /b cmd /c run-driver.bat` или просто двойным щелчком. EXIT-код
эпизода 0 — задача завершена штатно; nonzero — смотреть хвост лога.

---

## 5a. Запуск через планировщик задач (schtasks)

Прогон, запущенный из терминала, умирает вместе с сессией. Планировщик делает
родителем `svchost.exe` — прогон переживает закрытие терминала.

```cmd
schtasks /create /tn "lh_night_run" /tr "D:\путь\run_lh.cmd" /sc once /st 23:59 /f
schtasks /run /tn "lh_night_run"
```

`/sc once /st 23:59` создаёт разовую задачу «сегодня в 23:59», `/run` запускает
её немедленно. **После старта задачу удалить**
(`schtasks /delete /tn "lh_night_run" /f`) — иначе в 23:59 она запустит
второй прогон поверх живого. Удаление задачи НЕ убивает уже запущенный процесс.
Родителя запущенного прогона проверить: должен быть `svchost.exe`, а не
сессия терминала.

### Две ловушки контекста планировщика (нет env десктопа)

Интерактивный shell, открытый внутри ZCode-десктопа, наследует его переменные
окружения. Планировщик — нет. Обе проблемы воспроизводятся только там;
интерактивно всё работает, что и делает их коварными.

1. **`zcode app-server exited with 1` через ~0.7 с после старта.** Без
   `ZCODE_PERSONAL_PROVIDER_CONFIG_FILE` и `ZCODE_BUILTIN_PROVIDER_CONFIG_FILE`
   app-server ищет бандл `zcode-builtin.json` рядом с `zcode.cjs` (в
   desktop-инсталляции его нет) и падает. Настоящий текст ошибки
   (`无法定位 CLI ZCode Built-in Provider Config: …`) теряется — раннер глотает
   не-JSON stderr; диагностировать запуском `node …\zcode.cjs app-server
   --stdio < nul 2> err.txt` в том же контексте. Лечится экспортом обеих
   переменных; путь к builtin-конфигу версионный (`…\runtime\provider\
   windows-x86_64\<версия>\endpoint-<hash>\zcode-builtin.json`) и меняется при
   обновлении десктопа — вычислять динамически, не хардкодить.
2. **`UnicodeEncodeError: 'charmap' codec can't encode …` в раннере.** Консоль
   планировщика даёт stdout в cp1252; любой не-ASCII символ в ответе модели
   (например `→`) роняет `zcode_runner._emit_result`. Лечится
   `set PYTHONUTF8=1` и `set PYTHONIOENCODING=utf-8` до запуска lhht.

Боевая обёртка (проверена 21.09.2026, прогон `night-run-5`):

```bat
@echo off
cd /d D:\путь\к\проекту
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "LHHT_ZCODE_BINARY=%USERPROFILE%\AppData\Local\Programs\ZCode\resources\glm\zcode.cjs"
set "ZCODE_PERSONAL_PROVIDER_CONFIG_FILE=%USERPROFILE%\.zcode\v2\provider_config.json"
for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-ChildItem -Path \"$env:USERPROFILE\.zcode\v2\runtime\provider\windows-x86_64\" -Recurse -Filter zcode-builtin.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName"') do set "ZCODE_BUILTIN_PROVIDER_CONFIG_FILE=%%i"
%USERPROFILE%\AppData\Roaming\Python\Python314\Scripts\lhht.exe run --task "@TASK-my.md" --run-id my-run --max-rounds 60 >> lh_run_my-run.log 2>&1
```

---

## 6. Мониторинг живого прогона

```cmd
rem события (пишутся на границах эпизодов):
type .lhht\runs\<run-id>\lhht\events.jsonl
rem последний эпизод роли:
dir /od .lhht\runs\<run-id>\lhht\manager_episodes
rem потоковые логи модели (живость ZCode):
dir /od %USERPROFILE%\.zcode\cli\rollout\model-io-sess_*.jsonl
```

Правило: эпизод 10–40+ минут — норма; «тишина» 15 минут при живом CPU и растущем
rollout — НЕ зависание. Паниковать и убивать процесс не нужно.

---

## 7. Типовые проблемы → решения

| Симптом | Причина | Решение |
|---|---|---|
| `codex was not found` при старте | npm-CLI удалён/не в PATH | `npm i -g @openai/codex` (логин уже в `~\.codex`) |
| `zcode was not found` в doctor | нестандартный путь/нет PATH | `set LHHT_ZCODE_BINARY=…zcode.cjs` (или `zcode.cmd`) |
| Эпизоды падают по таймауту | дефолты 300/1800 с | `[run.timeouts] *= 10800` |
| Прогон «висит», events молчат | норма: события на границах эпизодов | проверить rollout/CPU (§6), ждать |
| Аудит циклится `blocked/suspect/unknown` | 0.3.2 не понимает RU-заголовки | lhht **0.3.3** уже чинит; если повторилось — чинить парсер, бампать 0.3.4 |
| Изменения в форке не действуют | не переустановлен пакет | из корня форка: остановить процессы → `pip install --user .` |
| Конфиг не применился | запуск не из каталога проекта | `cd /d <проект>` (конфиг ищется в cwd) |
| Аудитор ругается на чужие изменения в git | волатильные пути в снапшотах | `--guard-exclude-path <отн.путь>` (осознанно) |
| `zcode app-server exited with 1` сразу (~0.7 с) | запуск без env ZCode-десктопа (планировщик, сервис) | экспортировать `ZCODE_*_PROVIDER_CONFIG_FILE`, см. §5a |
| Прогон умер в середине: `Provider Registry 中不存在 Model: zai-direct/<модель>` | **параллельный lhht-прогон с другой моделью**: `ensure_provider_config` на каждом эпизоде перезаписывал `~\.zcode\v2\provider_config.json` (`personalModelIds` = одна модель текущего прогона), последний писатель побеждал | **исправлено в форке 21.09.2026**: списки моделей объединяются (`personalModelIds` — union). Требует переустановки `pip install --user .` из корня форка (при живых прогонах — сначала остановить). Прогон со старым кодом всё ещё затирает список — не смешивать старые и новые до полной переустановки; упавший перезапускать после завершения чужого |
| `UnicodeEncodeError: 'charmap' codec` в раннере | stdout в cp1252 в консоли планировщика | `set PYTHONUTF8=1` + `PYTHONIOENCODING=utf-8`, см. §5a |
| Ночью сам запустился второй прогон | забыта разовая задача `/sc once /st 23:59` | `schtasks /delete /tn <имя> /f` сразу после `/run` |
| Нужно продолжить прерванный | — | `lhht run --task … ` заново (каждый run изолирован) |

---

## 8. Smoke-тест перед боевой задачей (2 минуты)

```cmd
cd /d D:\путь\к\проекту
lhht.exe doctor
lhht.exe run --task "Посчитай файлы в текущем каталоге и напиши число. Ничего не изменяй." --max-rounds 1
```

Если smoke прошёл (EXIT 0, FINAL_RESPONSE.md содержит ответ) — окружение живо,
можно запускать боевую задачу.

---

## 9. Скорер SemIf, salvage и auditor-fast (добавлено 23.09.2026, ветка experiment/semif-integration)

Форк умеет использовать локальную модель-классификатор (SemIf, Qwen3.5-4B)
для быстрых семантических решений. Три части:

1. **Резидентный GPU-сервер** — модель грузится один раз, все параллельные
   прогоны делят её через шим. Запуск: `lhht server start` (GPU/torch по
   умолчанию; `lhht server start --cpu` для llamacpp, `--model` для 2B —
   детали: `lhht server start --help`). Диагностика всей цепочки:
   `lhht server doctor`; проверка живого: `lhht server status`.
   Сервер не поднят → все фичи ниже молча отключаются, прогон идёт как раньше.
2. **Salvage контрольных строк** (`[run.semif]` в config.toml): незнакомое
   написание ключевого слова (маршрут менеджера, заголовки аудитора) не
   сжигает раунд — строку судит скорер. Регекс всегда прав, salvage только
   добирает промахи. См. docs/semif-salvage.md.
3. **auditor-fast** (`auditor_fast = true` в `[run.semif]`): после исполнителя
   и до запуска медленного аудита гейт за секунды проверяет раунд по живому
   svn/git-статусу, выкладке исполнителя и acceptance-ограничениям. Уверенный
   провал (порог 0.95) → медленный аудит не запускается. Гейт fail-only:
   никогда не certifyицирует завершение, `done` по-прежнему требует чистого
   медленного аудита. См. docs/auditor-fast.md.

Пример секции конфига проекта:

```toml
[run.semif]
enabled = true
# Резидентный сервер (lhht server start); модель живёт на сервере:
server = "http://127.0.0.1:8790"
timeout_seconds = 60
auditor_fast = true
auditor_fast_threshold = 0.95
```
