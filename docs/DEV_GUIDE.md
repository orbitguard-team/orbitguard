# Руководство разработчика OrbitGuard

Правила работы, стиль кода, контракт данных и API — единые для всех участников.

> **Статус:** действует для всех участников.
> Правила работы (раздел 1) и стиль кода (раздел 3) обязательны.
> Контракт и API (разделы 4–7) — основа, принятая в #1. Точные структуры и сигнатуры
> фиксирует `docs/contracts.md` (OG-01); при расхождении верен он.

## 1. Правила работы

| Что | Правило |
|---|---|
| Задача | Один issue — один исполнитель. Перед началом: назначить себя, написать в issue «Беру задачу», перевести в In process |
| Ветка | От актуальной `main`: `codex/issue-<N>-<кратко>`, например `codex/issue-3-conjunction` |
| PR | Один issue — один небольшой PR в `main` с `Closes #N`. Шаблон PR заполняется полностью |
| Ревью | Ревьюер — другой участник. Merge — после Approve и зелёного CI, только squash |
| Файлы | Правим только файлы своей задачи. Чужой модуль — через issue его владельца |
| Контракт | Меняется только через issue OG-01, с упоминанием затронутых участников |
| Коммиты | `<тип>: <что сделано>`, типы: `feat`, `fix`, `test`, `docs`, `refactor`. Пример: `feat: поиск сближения по нулю скорости сближения` |
| Данные | Синтетические данные помечены. Демо-числа не выдаются за расчёт |
| Секреты | Токены, пароли, личные данные в репозиторий не попадают |

## 2. Структура проекта

```
orbitguard/
├── app.py                      # интерфейс Streamlit (OG-07, OG-08)
├── pyproject.toml              # зависимости и настройки (OG-01)
├── scenarios/                  # сценарии в JSON (OG-06)
├── src/orbitguard/
│   ├── constants.py            # физические константы (OG-01)
│   ├── models.py               # структуры данных (OG-01)
│   ├── elements.py             # перевод элементов орбиты в положение и скорость (OG-01)
│   ├── propagation.py          # прогноз движения (OG-02)
│   ├── forces/
│   │   ├── gravity.py          # Земля и J2 (OG-02, OG-13)
│   │   ├── drag.py             # атмосфера (OG-14)
│   │   └── third_body.py       # Луна и Солнце (OG-15)
│   ├── atmosphere.py           # плотность по высоте (OG-14)
│   ├── ephemeris.py            # положения Луны и Солнца (OG-15)
│   ├── conjunction.py          # поиск сближения (OG-03)
│   ├── maneuver.py             # подбор манёвра (OG-04)
│   ├── metrics.py              # метрики и экспорт (OG-05)
│   └── scenarios.py            # загрузка сценариев (OG-06)
└── tests/
    ├── test_<модуль>.py        # модульные тесты
    └── integration/            # сквозные тесты (OG-10)
```

## 3. Стиль кода

Основа — [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
Ниже — главное из него и дополнения проекта (единицы в именах, массивы, статусы).

### Общее

| Правило | Пример |
|---|---|
| Python 3.11+, отступ 4 пробела, строка до 80 символов | |
| Импортируем модули, а не отдельные имена; стандартные сокращения разрешены | `import numpy as np`, `from orbitguard import models` |
| Порядок импортов: стандартная библиотека → сторонние → проект, группы через пустую строку | |
| Аннотации типов у всех публичных функций и методов | `def propagate(obj: models.SpaceObject, ...) -> Trajectory:` |
| Изменяемые значения не используются как аргументы по умолчанию | `impulses: Sequence[Impulse] = ()`, а не `= []` |
| Строки форматируются через f-строки | `f"Высота {altitude_m:.0f} м"` |
| Без магических чисел: константы — в `constants.py` | `MU_EARTH_M3PS2`, а не `3.986e14` в коде |
| Без `print`, глобального состояния и скрытых значений по умолчанию | Для диагностики — `logging` |
| Входные данные не изменяются | Структуры данных `frozen=True` |
| Исключения ловим конкретные, не голый `except:` | `except ValueError:` |
| Проверка стиля — `pylint` с конфигурацией Google, если команда подключит его в OG-09 | |

### Docstring

Формат Google: краткая строка, пустая строка, разделы `Args:`, `Returns:`, `Raises:`.
Текст — по-русски, у каждой величины единица и система координат.

```python
def find_closest_approach(
    a: Trajectory,
    b: Trajectory,
    safe_distance_m: float,
    t_span_s: tuple[float, float],
) -> ConjunctionResult:
    """Находит момент и дистанцию минимального сближения двух объектов.

    Минимумы ищутся как нули скорости сближения r_rel · v_rel и уточняются
    между точками сетки.

    Args:
      a: Траектория первого объекта, ECI.
      b: Траектория второго объекта, ECI.
      safe_distance_m: Безопасная дистанция, м.
      t_span_s: Интервал поиска, с от даты начала.

    Returns:
      Ближайшее сближение, все локальные минимумы и признак нарушения
      безопасной дистанции.

    Raises:
      ValueError: Если интервал выходит за пределы траекторий.
    """
```

Комментарии в коде — тоже по-русски; объясняют «зачем», а не пересказывают код.

### Имена

| Что | Стиль | Пример |
|---|---|---|
| Модуль, пакет, файл | `lower_with_under` | `third_body.py` |
| Класс, исключение | `CapWords` | `ConjunctionResult`, `PropagationError` |
| Функция, метод | `lower_with_under`, глагол | `find_closest_approach` |
| Переменная, аргумент | `lower_with_under` | `miss_distance_m` |
| Константа модуля | `CAPS_WITH_UNDER` с единицей | `R_EARTH_M` |
| Внутреннее | префикс `_` | `_refine_minimum` |
| Тест | `test_<что>_<ожидание>` | `test_circular_orbit_period_matches_kepler` |

Имена описательные: без однобуквенных переменных, кроме счётчиков (`i`, `j`)
и общепринятых обозначений в формулах (`r`, `v`, `mu`), если рядом есть комментарий с единицами.

### Единицы и системы координат в имени

Каждая физическая величина несёт единицу в суффиксе имени. Внутри кода — только СИ и радианы.

| Суффикс | Единица | Пример |
|---|---|---|
| `_m` | метр | `altitude_m`, `r_eci_m` |
| `_mps` | м/с | `v_eci_mps`, `dv_mps` |
| `_mps2` | м/с² | `accel_mps2` |
| `_s` | секунда от даты начала | `t_s`, `tca_s` |
| `_kg` | килограмм | `mass_kg` |
| `_m2` | м² | `area_m2` |
| `_rad` | радиан | `inclination_rad` |
| `_deg` | градус — **только** ввод и вывод | `inclination_deg` |
| `_kgpm3` | кг/м³ | `density_kgpm3` |
| `_radps` | рад/с | `OMEGA_EARTH_RADPS` |
| `_m3ps2` | м³/с² | `MU_EARTH_M3PS2` |

| Суффикс | Система координат |
|---|---|
| `_eci` | инерциальная, ось z — ось вращения Земли |
| `_rtn` | локальная: радиально, вдоль орбиты, по нормали к плоскости орбиты |

Без единицы — только безразмерные величины: `cd`, `density_scale`, `J2`.

### Массивы

| Величина | Форма | Имя |
|---|---|---|
| Положение | `(3,)` | `r_eci_m` |
| Скорость | `(3,)` | `v_eci_mps` |
| Состояние | `(6,)` = `[x, y, z, vx, vy, vz]` | `state_eci` |
| Сетка времени | `(N,)` | `t_s` |
| Состояния на сетке | `(N, 6)` | `states_eci` |

Форма массива указывается в комментарии при объявлении. Тип — `float64`.

### Ошибки и статусы

| Ситуация | Как сообщать |
|---|---|
| Неверный ввод (отрицательная дистанция, нет даты) | `ValueError` с понятным текстом на русском при загрузке сценария |
| Нормальный исход расчёта (манёвр не найден) | Статус в результате |
| Сбой расчёта (интегратор не сошёлся, сход с орбиты) | Статус `PROPAGATION_FAILED` + текст причины |
| Сила не посчитана из-за нехватки данных | Предупреждение в результате, а не молчаливый пропуск |

### Тесты

| Правило | Пример |
|---|---|
| Файл на модуль: `tests/test_<модуль>.py` | `tests/test_conjunction.py` |
| Эталон — аналитика или опубликованные значения, а не вывод своего же кода | Период круговой орбиты `2π·√(a³/μ)` |
| Допуск — именованная константа с обоснованием | `PERIOD_TOL_S = 1e-3  # точность интегратора` |
| Без сети и случайности; если нужна случайность — фиксированный seed | `np.random.default_rng(0)` |
| Запуск: `pytest` из корня репозитория | |

## 4. Контракт данных

| Что | Решение |
|---|---|
| Единицы | СИ; углы внутри — радианы |
| Система координат | ECI, ось z — ось вращения Земли, без прецессии и нутации |
| Время | `epoch` — дата и время начала в UTC (`datetime` с часовым поясом); всё остальное — секунды от неё |
| Состояние объекта | `[x, y, z, vx, vy, vz]` в ECI |
| Ввод орбит | Положение и скорость или элементы орбиты; элементы переводятся при загрузке |
| Структуры | `dataclass(frozen=True)` + NumPy |
| Сценарии | JSON в папке `scenarios/` |
| Допуски интегратора (стартовые) | `rtol = 1e-10`, `atol` = 1e-3 м по положению и 1e-6 м/с по скорости; подтверждаются тестом сходимости |

### Структуры (`models.py`)

```python
@dataclass(frozen=True)
class SpaceObject:
    name: str
    state0_eci: np.ndarray            # (6,) м и м/с в момент epoch
    # Без массы, площади и cd атмосфера для объекта не считается.
    mass_kg: float | None = None
    area_m2: float | None = None
    cd: float | None = None


@dataclass(frozen=True)
class ForceModel:
    j2: bool = True
    drag: bool = False
    moon: bool = False
    sun: bool = False
    density_scale: float = 1.0        # поправка на солнечную активность


@dataclass(frozen=True)
class ManeuverLimits:
    dv_max_mps: float
    t_earliest_s: float
    t_latest_s: float                 # строго раньше момента сближения


@dataclass(frozen=True)
class Scenario:
    name: str
    epoch: datetime                   # UTC
    horizon_s: float
    satellite: SpaceObject
    debris: SpaceObject
    forces: ForceModel
    safe_distance_m: float
    limits: ManeuverLimits
    synthetic: bool = True
    source: str = ""                  # откуда данные или как построены


@dataclass(frozen=True)
class Impulse:
    t_s: float
    dv_eci_mps: np.ndarray            # (3,)
    dv_rtn_mps: np.ndarray            # (3,)


class Trajectory(Protocol):
    t_s: np.ndarray                   # (N,)
    states_eci: np.ndarray            # (N, 6)

    def state_at(self, t_s: float | np.ndarray) -> np.ndarray:
        """Состояние в любой момент: (6,) или (M, 6).

        Учитывает разрыв скорости в момент импульса.
        """


@dataclass(frozen=True)
class Approach:
    tca_s: float                      # момент сближения
    miss_distance_m: float
    rel_speed_mps: float


@dataclass(frozen=True)
class ConjunctionResult:
    closest: Approach
    all_minima: list[Approach]        # все локальные минимумы
    violated: bool                    # ближе безопасной дистанции


class Status(Enum):
    ALREADY_SAFE = "already_safe"
    FEASIBLE = "feasible"
    NO_FEASIBLE_CANDIDATE = "no_feasible_candidate"
    PROPAGATION_FAILED = "propagation_failed"
    NOT_IMPLEMENTED = "not_implemented"   # только для заглушек


@dataclass(frozen=True)
class Candidate:
    burn: Impulse
    miss_distance_m: float
    feasible: bool


@dataclass(frozen=True)
class ManeuverResult:
    status: Status
    best: Impulse | None
    before: ConjunctionResult
    after: ConjunctionResult | None
    candidates: list[Candidate]
    forces: ForceModel                # какие силы были включены
    warnings: list[str]
    compute_time_s: float
```

## 5. API

| Модуль | Функция | Вход | Выход | Задача |
|---|---|---|---|---|
| `scenarios` | `load_scenario` | путь к JSON | `Scenario` | OG-06 |
| `elements` | `elements_to_state` | элементы орбиты | `state_eci` `(6,)` | OG-01 |
| `propagation` | `propagate` | объект, дата, интервал, силы, импульсы | `Trajectory` | OG-02 |
| `forces.*` | `accel_*` | `t_s`, `r_eci_m`, `v_eci_mps`, `ctx` | ускорение `(3,)` м/с² | OG-02, 13–15 |
| `conjunction` | `find_closest_approach` | две траектории, дистанция, интервал | `ConjunctionResult` | OG-03 |
| `maneuver` | `search_maneuver` | `Scenario` | `ManeuverResult` | OG-04 |
| `metrics` | `summarize`, `export_json` | `ManeuverResult` | таблица / файл | OG-05 |

```python
# scenarios.py
def load_scenario(path: str | Path) -> Scenario: ...

# elements.py
def elements_to_state(
    sma_m: float, ecc: float, inclination_rad: float,
    raan_rad: float, argp_rad: float, true_anomaly_rad: float,
) -> np.ndarray: ...                  # (6,) ECI

# propagation.py
def propagate(
    obj: SpaceObject, epoch: datetime, t_span_s: tuple[float, float],
    forces: ForceModel, impulses: Sequence[Impulse] = (),
) -> Trajectory: ...

# forces/*.py — одна сигнатура для всех сил
@dataclass(frozen=True)
class ForceContext:
    epoch: datetime
    obj: SpaceObject
    forces: ForceModel

def accel_central(
    t_s: float,
    r_eci_m: np.ndarray,
    v_eci_mps: np.ndarray,
    ctx: ForceContext,
) -> np.ndarray: ...                  # (3,) м/с²

# Остальные силы — с той же сигнатурой:
def accel_j2(t_s, r_eci_m, v_eci_mps, ctx) -> np.ndarray: ...
def accel_drag(t_s, r_eci_m, v_eci_mps, ctx) -> np.ndarray: ...
def accel_moon(t_s, r_eci_m, v_eci_mps, ctx) -> np.ndarray: ...
def accel_sun(t_s, r_eci_m, v_eci_mps, ctx) -> np.ndarray: ...

# conjunction.py
def find_closest_approach(
    a: Trajectory,
    b: Trajectory,
    safe_distance_m: float,
    t_span_s: tuple[float, float],
) -> ConjunctionResult: ...

# maneuver.py
def search_maneuver(scenario: Scenario) -> ManeuverResult: ...

# metrics.py
def summarize(result: ManeuverResult) -> dict[str, float | str]: ...
def export_json(
    scenario: Scenario, result: ManeuverResult, path: str | Path
) -> None: ...
```

### Статусы

| Статус | Когда | Что показывает интерфейс |
|---|---|---|
| `already_safe` | Без манёвра дистанция больше безопасной | «Манёвр не нужен» |
| `feasible` | Найден безопасный манёвр | Манёвр, Δv, до/после |
| `no_feasible_candidate` | Среди проверенных вариантов безопасного нет | «Не найден в заданных пределах» — без отметки «безопасно» |
| `propagation_failed` | Сбой расчёта | Причина ошибки |
| `not_implemented` | Модуль ещё заглушка | «Расчёт не подключён» |

## 6. Формат сценария (JSON)

| Поле | Тип | Обязательно | Пример |
|---|---|---|---|
| `name` | строка | да | `"dangerous_iss_like"` |
| `synthetic` | bool | да | `true` |
| `source` | строка | да | `"МКС-подобная орбита, мусор задан обратным расчётом"` |
| `epoch_utc` | ISO 8601 | да | `"2026-01-01T00:00:00Z"` |
| `horizon_s` | число | да | `86400` |
| `safe_distance_m` | число | да | `1000` |
| `forces` | объект | да | `{"j2": true, "drag": false, "moon": false, "sun": false, "density_scale": 1.0}` |
| `satellite`, `debris` | объект | да | `state0_eci` **или** `elements`; `mass_kg`, `area_m2`, `cd` — по желанию |
| `limits` | объект | да | `{"dv_max_mps": 1.0, "t_earliest_s": 0, "t_latest_s": 39600}` |

В JSON углы элементов орбиты задаются в градусах с суффиксом `_deg`; при загрузке переводятся в радианы.

## 7. Константы (`constants.py`)

| Имя | Значение | Источник |
|---|---|---|
| `MU_EARTH_M3PS2` | 3.986004418e14 | WGS 84 |
| `R_EARTH_M` | 6378137.0 | WGS 84 |
| `FLATTENING_EARTH` | 1 / 298.257223563 | WGS 84 |
| `J2` | 1.08262668e-3 | EGM2008 |
| `R_EARTH_GRAV_M` | 6378136.3 | EGM2008 — радиус для формулы J2, в паре с `J2` |
| `OMEGA_EARTH_RADPS` | 7.292115e-5 | WGS 84 |
| `MU_MOON_M3PS2` | 4.902800066e12 | JPL DE430 |
| `MU_SUN_M3PS2` | 1.32712440018e20 | IAU |
| `REENTRY_ALTITUDE_M` | 100e3 | Условная граница остановки расчёта |
