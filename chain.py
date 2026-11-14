"""Цепочка генерации модуля из текстового описания.

Звенья: анализ → инструменты → реализация → машинная проверка.
Модель вызывается через LangChain LCEL: prompt | model | parser.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

BASE = Path(__file__).parent
PROMPTS = BASE / "prompts"
RUNS = BASE / "runs"

MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

STEP_SETTINGS = {
    1: {"prompt": "analysis.md", "prompt_id": "chain.analysis", "version": 1, "temperature": 0.2, "max_tokens": 800},
    2: {"prompt": "tools.md", "prompt_id": "chain.tools", "version": 1, "temperature": 0.2, "max_tokens": 800},
    3: {"prompt": "code.md", "prompt_id": "chain.code", "version": 1, "temperature": 0.1, "max_tokens": 4000},
}

RUN_TIMEOUT = 60
COMPILE_TIMEOUT = 30
MAX_REPAIR = 1
ENVELOPE_KEYS = ("run_id", "module", "version", "status", "output", "stats")

logger = logging.getLogger("chain")


# ---------------------------------------------------------------- служебное

def setup_logging(run_dir: Path) -> None:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(run_dir / "chain.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def extract_json(text: str) -> dict:
    """Достать JSON из ответа модели, отбросив возможную markdown-обёртку."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"в ответе нет JSON: {text[:300]}")
    return json.loads(cleaned[start:end + 1])


def parse_message(message: AIMessage) -> dict:
    """Стадия parser в цепочке: текст модели → словарь + статистика."""
    return {
        "data": extract_json(message.content),
        "usage": message.usage_metadata or {},
        "raw": message.content,
    }


def escape_braces(text: str) -> str:
    """Экранировать { } в тексте промпта: LangChain считает их переменными шаблона."""
    return text.replace("{", "{{").replace("}", "}}")


def build_chain(settings: dict, payload: str):
    system_text = escape_braces((PROMPTS / settings["prompt"]).read_text(encoding="utf-8"))
    prompt = ChatPromptTemplate.from_messages([("system", system_text), ("human", "{payload}")])
    llm = ChatOpenAI(
        model=MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=BASE_URL,
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
        timeout=120,
        max_retries=2,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return prompt | llm | RunnableLambda(parse_message)


# ---------------------------------------------------------------- звенья

def run_step(number: int, payload: dict, repair_note: str = "") -> dict:
    settings = STEP_SETTINGS[number]
    user_payload = json.dumps(payload, ensure_ascii=False, indent=2)
    if repair_note:
        user_payload += "\n\n## Ошибка предыдущего запуска\n" + repair_note

    started = time.perf_counter()
    for attempt in (1, 2):
        chain = build_chain(settings, payload)
        try:
            result = chain.invoke({"payload": user_payload})
            break
        except (ValueError, json.JSONDecodeError) as exc:
            if attempt == 2:
                raise
            logger.warning("шаг %s: JSON не разобрался (%s) — повтор с уточнением", number, exc)
            user_payload += "\n\nВерни ТОЛЬКО JSON, без markdown и без пояснений."
    elapsed = time.perf_counter() - started

    usage = result["usage"]
    logger.info(
        "шаг %s %s v%s | model=%s temperature=%s max_tokens=%s | tokens_in=%s tokens_out=%s | %.1f с",
        number,
        settings["prompt_id"],
        settings["version"],
        MODEL,
        settings["temperature"],
        settings["max_tokens"],
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        elapsed,
    )
    return result


# ---------------------------------------------------------------- проверка

def run_command(name: str, command: list[str], cwd: Path, timeout: int) -> dict:
    try:
        completed = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace"
        )
    except subprocess.TimeoutExpired:
        return {"name": name, "command": " ".join(command), "status": "fail", "output": f"таймаут {timeout} с"}
    output = ((completed.stdout or "") + (completed.stderr or "")).strip()
    return {
        "name": name,
        "command": " ".join(command),
        "status": "pass" if completed.returncode == 0 else "fail",
        "output": output[-2000:],
    }


def check_envelope(artifact: Path) -> dict:
    try:
        data = json.loads(artifact.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"name": "envelope_schema", "command": "проверка ключей", "status": "fail", "output": str(exc)}
    missing = [key for key in ENVELOPE_KEYS if key not in data]
    return {
        "name": "envelope_schema",
        "command": "проверка ключей " + ", ".join(ENVELOPE_KEYS),
        "status": "pass" if not missing else "fail",
        "output": "ok" if not missing else f"нет ключей: {missing}",
    }


def run_checks(code_dir: Path) -> dict:
    python_files = sorted(p.name for p in code_dir.glob("*.py"))
    checks = [run_command("py_compile", [sys.executable, "-m", "py_compile", *python_files], code_dir, COMPILE_TIMEOUT)]

    if checks[-1]["status"] == "pass":
        checks.append(run_command("run_test", [sys.executable, "main_test.py"], code_dir, RUN_TIMEOUT))
    else:
        checks.append({"name": "run_test", "command": "main_test.py", "status": "skipped", "output": "компиляция не прошла"})

    artifact = code_dir / "outputdata" / "output.json"
    checks.append(
        {
            "name": "artifact_exists",
            "command": "outputdata/output.json",
            "status": "pass" if artifact.exists() else "fail",
            "output": "найден" if artifact.exists() else "файл не создан",
        }
    )
    if artifact.exists():
        checks.append(check_envelope(artifact))

    failed = [c for c in checks if c["status"] != "pass"]
    verdict = "pass" if not failed else "fail"
    errors_text = "\n\n".join(f"[{c['name']}] {c['output']}" for c in failed)
    return {"checks": checks, "verdict": verdict, "errors_text": errors_text}


# ---------------------------------------------------------------- сценарий

def write_files(payload: dict, code_dir: Path) -> list[str]:
    files = payload.get("files") or {}
    if not files:
        raise ValueError("шаг 3 не вернул файлы")
    written = []
    for name, content in files.items():
        target = code_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(name)
    return written


def make_report(run_dir: Path, analysis: dict, tools: dict, checks: dict, repair_attempts: int) -> None:
    lines = [
        f"# Прогон {run_dir.name}",
        "",
        f"- модуль: `{analysis.get('module_name', '?')}`",
        f"- функций: {len(analysis.get('functions', []))}",
        f"- библиотеки: {', '.join(tools.get('libraries', [])) or '—'}",
        f"- проверка: **{checks['verdict']}**",
        f"- ремонтов: {repair_attempts}",
        "",
        "## Проверки",
        "",
        "| Проверка | Статус | Команда |",
        "|---|---|---|",
    ]
    for check in checks["checks"]:
        lines.append(f"| {check['name']} | {check['status']} | `{check['command']}` |")
    if checks["errors_text"]:
        lines += ["", "## Ошибки", "", "```", checks["errors_text"][:3000], "```"]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_chain(description: str, run_dir: Path) -> dict:
    logger.info("run_id=%s | model=%s | base_url=%s", run_dir.name, MODEL, BASE_URL)

    analysis = run_step(1, {"description": description})["data"]
    (run_dir / "01-analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("шаг 1: модуль=%s, функций=%s", analysis.get("module_name"), len(analysis.get("functions", [])))

    tools = run_step(2, {"analysis": analysis})["data"]
    (run_dir / "02-tools.json").write_text(json.dumps(tools, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("шаг 2: библиотеки=%s", tools.get("libraries"))

    code_dir = run_dir / "03-code"
    code_dir.mkdir()
    repair_attempts = 0
    repair_note = ""
    while True:
        files = run_step(3, {"analysis": analysis, "tools": tools}, repair_note)["data"]
        written = write_files(files, code_dir)
        logger.info("шаг 3: файлы=%s", written)

        checks = run_checks(code_dir)
        logger.info("проверка: %s", checks["verdict"])
        if checks["verdict"] == "pass" or repair_attempts >= MAX_REPAIR:
            break
        repair_attempts += 1
        repair_note = checks["errors_text"]
        logger.warning("ремонт %s из %s", repair_attempts, MAX_REPAIR)

    result = {**checks, "repair_attempts": repair_attempts}
    (run_dir / "04-check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    make_report(run_dir, analysis, tools, checks, repair_attempts)
    return result


def check_only(run_name: str) -> int:
    """Повторить машинную проверку готового прогона, без обращения к модели."""
    run_dir = Path(run_name)
    if not run_dir.is_absolute():
        run_dir = RUNS / run_name
    code_dir = run_dir / "03-code"
    if not code_dir.exists():
        print(f"Нет папки с кодом: {code_dir}", file=sys.stderr)
        return 2


    setup_logging(run_dir)
    checks = run_checks(code_dir)

    check_path = run_dir / "04-check.json"
    previous = {}
    if check_path.exists():
        try:
            previous = json.loads(check_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
    result = {**checks, "repair_attempts": previous.get("repair_attempts", 0)}
    check_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("повторная проверка %s: %s", run_dir.name, checks["verdict"])
    print(f"Проверка: {checks['verdict']}")
    if checks["errors_text"]:
        print(checks["errors_text"])
    return 0 if checks["verdict"] == "pass" else 1


def main() -> int:
    load_dotenv(BASE / ".env")
    parser = argparse.ArgumentParser(description="Цепочка генерации модуля из описания")
    parser.add_argument("--input", help="файл с описанием задачи")
    parser.add_argument("--run-id", default=None, help="имя прогона (по умолчанию дата-время)")
    parser.add_argument("--check-only", metavar="RUN", help="повторить проверку существующего прогона")
    args = parser.parse_args()

    if args.check_only:
        return check_only(args.check_only)

    if not args.input:
        parser.error("нужен --input (новый прогон) или --check-only (повторная проверка)")

    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Нет DEEPSEEK_API_KEY: создайте .env по образцу .env.example", file=sys.stderr)
        return 2

    source = Path(args.input)
    if not source.exists():
        print(f"Нет файла описания: {source}", file=sys.stderr)
        return 2
    description = source.read_text(encoding="utf-8").strip()
    if not description:
        print("Описание пустое", file=sys.stderr)
        return 2

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS / run_id
    if run_dir.exists():
        run_dir = RUNS / f"{run_id}-{int(time.time())}"
    run_dir.mkdir(parents=True)
    setup_logging(run_dir)

    try:
        result = run_chain(description, run_dir)
    except Exception as exc:  # noqa: BLE001
        logger.error("прогон остановлен: %s", exc)
        return 1

    print(f"\nИтог: {result['verdict']} | артефакты: {run_dir}")
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())