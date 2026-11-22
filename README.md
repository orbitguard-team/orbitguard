# OrbitGuard

Исследовательский симулятор безопасных орбитальных манёвров команды Chuykinteam.
Кейс: «Моделирование полета и расчет траекторий для космических миссий».

Подготовлены контракт данных и каркас Python-пакета. Численные алгоритмы ещё не реализованы.

**Участникам и их агентам: начните с [START_HERE.md](START_HERE.md).**

Репозиторий: https://github.com/orbitguard-team/orbitguard

Доска: https://github.com/orgs/orbitguard-team/projects/1/views/1

- [Настройка GitHub и работа участников](docs/GITHUB_SETUP.md)
- [Требования, роли и план до защиты](docs/PLAN.md)
- [Заготовки задач для GitHub Issues](docs/BACKLOG.md)
- [Правила для агентов](AGENTS.md)
- [Контракт данных и команды установки/проверки](docs/contracts.md)

Первый запуск (Python 3.11–3.13, в виртуальном окружении):

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python examples/contract_example.py
```

Пример проверяет только входные данные, не вычисляет траекторию или безопасность.

Предлагаемый MVP: Python, NumPy, SciPy (RK45), Plotly, Streamlit, pytest.
Два объекта, модель сил (центральное поле, J2, атмосфера, Луна и Солнце — [docs/PHYSICS.md](docs/PHYSICS.md)), прогноз сближения, поиск импульсного манёвра,
сравнение до/после. Это предложение реализации по портфолио, а не уже готовый продукт.
