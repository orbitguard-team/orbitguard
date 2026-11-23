# Проверка MVP — OG-10

Таблица сверена на `4f1f3753889bfce6ea264f4aa348884c28450164`
(ветка PR #31 поверх `main` `54e00f82a216ee284ab1cab11671a183ca5db8e2`).
Первый снимок был на `c04f55c2e5c5d160a173af78f85e19be97d5abb6`;
между ним и `54e00f8` расчётные модули не менялись. Сценарии
синтетические, эпоха и состояния заданы в их JSON-файлах.
Система ECI-J2000, единицы СИ. Включено только центральное поле Земли;
J2, атмосфера, Луна и Солнце выключены. Импульс мгновенный. Дистанция —
детерминированный минимум, не вероятность столкновения.

Параметры: `rtol=1e-10`, `position_atol_m=0.001`,
`velocity_atol_mps=1e-6`, `closest_time_tol_s=0.001`,
`max_step_s=10`, `distance_margin_m=1`. Безопасность требует
`d_min >= safe_distance_m + distance_margin_m`, то есть 1001 м.

| Сценарий | Горизонт | Статус | До: t_min, d_min | После: t_min, d_min | Выбранный Δv |
|---|---:|---|---:|---:|---:|
| `safe_flyby` | 4 ч | `already_safe` | 12600.000305 с; 4000.002311 м | то же | 0 |
| `dangerous_feasible` | 4 ч | `feasible` | 10800.000305 с; 120.065788 м | 10800.220642 с; 1900.362321 м | 0.100000 м/с в t=0 |
| `late_warning_no_solution` | 45 мин | `no_feasible_candidate` | 899.999695 с; 120.065791 м | нет | нет |

Для `feasible` проверен записанный в сценарии пример импульса. Он даёт
безопасный результат, но не заявлен как минимум Δv по всем возможным
манёврам. Для `no_feasible_candidate` в табличном прогоне список кандидатов пуст.
Дополнительный интеграционный тест рассчитывает 12 импульсов: в начале
и конце окна, по ±R/±T/±N, каждый с максимальным Δv. Все отклонены.
Статус относится к проверенной сетке. Аналитическая
оценка невозможности для заданного окна и бюджета описана отдельно в
[scenarios.md](scenarios.md), но не является результатом этого перебора.

В `tests/integration/test_end_to_end.py` дистанция после манёвра проверена
на сетке 1 с по всему горизонту. Каждый внутренний локальный минимум
независимо уточнён `scipy.optimize.minimize_scalar`, затем времена и
дистанции сопоставлены с `all_minima`, включая границы горизонта. Найдено
7 минимумов, как и в результате поиска; наименьший — 1900.362 м ≥ 1001 м.
Это численная проверка данной траектории, но не доказательство для
произвольно узких сближений между узлами. Ужесточение
`rtol` и абсолютных допусков в 10 раз, `max_step_s` в 2 раза сохранило
`feasible`: t_min не изменился в показанной точности, d_min изменился
на −0.000000966 м на Windows и −0.000001278 м на Linux (независимое
ревью; отличие обусловлено платформой). Проверки нулевого бюджета и окна, исключающего
импульс, вернули `no_feasible_candidate` без ложной метрики «после».
Импульс ровно на границе бюджета и окно ровно в момент импульса проверены
отдельно: оба варианта дают `feasible` и допустимого кандидата.

Обнаруженный дефект публичного API: до слияния
[PR #33](https://github.com/orbitguard-team/orbitguard/pull/33) функции
`orbitguard.api.find_closest_approach`, `search_maneuver` и `load_scenario`
были заглушками и вызывали `NotImplementedError`, хотя реализации уже
работали в соответствующих модулях. Это нарушало контракт из
[contracts.md](contracts.md) и блокировало релиз. Исправление слито в
`main` коммитом `6dcd7bb`; ветка OG-10 объединена с ним для интеграционной
проверки.

Проверки на Windows, Python 3.13.12, NumPy 2.5.5, SciPy 1.18.1:

```text
python -m pytest tests/integration/test_end_to_end.py -q
7 passed in 24.69s

python -m pytest -q
221 passed, 16 skipped, 7 subtests passed in 60.50s

# после слияния OG-08 в main, без локальных UI-зависимостей:
python -m pytest tests/integration/test_end_to_end.py -q
8 passed in 15.35s
python -m pytest -q
222 passed, 20 skipped, 7 subtests passed in 34.34s
python -m pylint --disable=duplicate-code,redefined-outer-name tests/integration/test_end_to_end.py
10.00/10

# после исправлений по ревью (этот PR), без локальных UI-зависимостей:
python -m pytest tests/integration/test_end_to_end.py -q
9 passed in 19.06s
python -m pytest -q
223 passed, 20 skipped, 7 subtests passed in 34.71s
python -m pylint --disable=duplicate-code,redefined-outer-name tests/integration/test_end_to_end.py
10.00/10

# после слияния PR #33 в main и объединения main с веткой OG-10:
python -m pytest -q
223 passed, 20 skipped, 7 subtests passed in 138.23s
python -m pylint --disable=duplicate-code,redefined-outer-name tests/integration/test_end_to_end.py
10.00/10
```

Независимая проверка `fvdzg` (Linux, Python 3.13.18, новый клон и новый
venv): `pip install -e ".[dev]"`, затем `pytest -q` — 221 passed, 20 skipped
до установки UI. После `pip install -e ".[ui]"` и объединения с OG-08 —
225 passed, 16 skipped. `streamlit run app.py --server.headless true`
ответил HTTP 200, health endpoint — `ok`. Три статуса интерфейса,
сброс результата и ошибки ввода проверены через Streamlit AppTest в
`tests/integration/test_app.py` из PR #32. Числа таблицы совпали до
последнего показанного знака. Это независимый участник и чистая установка,
но не подтверждён отдельный физический компьютер.

В локальном прогоне после OG-08 четыре UI-теста пропущены из-за отсутствия
Plotly и Streamlit в этом venv; независимый прогон выше выполнил их с UI
зависимостями. CI на обновлённой ветке фиксируется в PR. Тесты
J2/атмосферы/третьих тел сейчас пропущены до OG-13…15. Основные ограничения
поиска конечных кандидатов и шагов описаны в
[contracts.md](contracts.md) и [maneuver.md](maneuver.md).
