"""Проверки репозитория по правилам AGENTS.md и docs/DEV_GUIDE.md.

- В репозитории нет PDF-рендеров и секретов.
- В коде пакета нет физических констант вне constants.py и нет print.
"""

import pathlib
import re
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src" / "orbitguard"

SECRET_PATTERNS = (
    re.compile(r"ghp_[A-Za-z0-9]{36}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)

# Начала значений физических констант (docs/DEV_GUIDE.md, раздел 7).
CONSTANT_LITERALS = re.compile(
    r"\b(3\.986|6\.378|6378|1\.0826|7\.292|4\.9028|1\.3271)")


def _tracked_files() -> list[pathlib.Path]:
    """Файлы под контролем git.

    Returns:
        Пути к отслеживаемым файлам.
    """
    try:
        output = subprocess.run(
            ["git", "ls-files"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git недоступен")
    return [REPO_ROOT / name for name in output.splitlines()]


def _package_files() -> list[pathlib.Path]:
    """Модули пакета orbitguard.

    Returns:
        Пути к .py файлам пакета.
    """
    if not SRC_DIR.is_dir():
        pytest.skip("Пакета src/orbitguard ещё нет: ждёт OG-01 (#1)")
    return sorted(SRC_DIR.rglob("*.py"))


def test_no_pdf_files_tracked():
    """PDF-рендеры не попадают в репозиторий."""
    pdfs = [p.name for p in _tracked_files() if p.suffix.lower() == ".pdf"]
    assert not pdfs, f"PDF в репозитории: {pdfs}"


def test_no_secrets_in_tracked_files():
    """В отслеживаемых файлах нет токенов и ключей."""
    found = []
    for path in _tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            found.append(str(path.relative_to(REPO_ROOT)))
    assert not found, f"Похоже на секрет: {found}"


def test_physical_constants_only_in_constants_module():
    """Значения констант записаны только в constants.py."""
    violations = []
    for path in _package_files():
        if path.name == "constants.py":
            continue
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            code = line.split("#", 1)[0]
            if CONSTANT_LITERALS.search(code):
                violations.append(f"{path.name}:{number}")
    assert not violations, f"Константы вне constants.py: {violations}"


def test_no_print_in_package():
    """В коде пакета нет print: для диагностики — logging."""
    violations = [
        path.name for path in _package_files()
        if re.search(r"^\s*print\(", path.read_text(encoding="utf-8"),
                     flags=re.MULTILINE)
    ]
    assert not violations, f"print в модулях: {violations}"
