#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基于 session_logs 的意图路由评估脚本。"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rag_intent_router import IntentType, RAGIntentRouter
from core.vector_memory import VectorMemory


def infer_expected_intent(message: str) -> Optional[str]:
    text = (message or "").strip()
    if not text:
        return None
    if any(key in text for key in ("我之前", "历史", "回顾", "上次")):
        return IntentType.MEMORY_QUERY.value
    if any(key in text for key in ("查找", "读取", "阅读", "解析", "查看", "打开", "扫描", "编辑", "写入", "运行")):
        return IntentType.TOOLS.value
    if any(key in text for key in ("重构", "设计", "方案", "多步骤", "计划", "实现")):
        return IntentType.PLAN.value
    if any(key in text.lower() for key in ("pdf", "代码审查", "知识库", "skill", "技能")):
        return IntentType.SKILLS.value
    return IntentType.CHAT.value


def iter_session_messages(session_dir: Path, limit: int) -> Iterable[Tuple[str, str]]:
    files = sorted(session_dir.glob("*.json"), reverse=True)
    count = 0
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in data.get("messages", []):
            user_message = item.get("user_message", "")
            if not user_message or user_message.startswith("["):
                continue
            yield path.name, user_message
            count += 1
            if count >= limit:
                return


def load_labeled_samples(labels_file: Path, limit: int) -> List[Tuple[str, str, str]]:
    samples = []
    if not labels_file.exists():
        return samples
    for line in labels_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        user_message = item.get("user_message", "")
        expected = item.get("expected_intent", "")
        source = item.get("source_file", labels_file.name)
        if not user_message or not expected:
            continue
        samples.append((source, user_message, expected))
        if len(samples) >= limit:
            break
    return samples


def build_router(memory_dir: Path) -> RAGIntentRouter:
    vm = VectorMemory(memory_dir=str(memory_dir))
    return RAGIntentRouter(vector_memory=vm, llm_forward_fn=None, confidence_threshold=0.7)


def evaluate(session_dir: Path, limit: int, memory_dir: Path, labels_file: Optional[Path] = None) -> Dict[str, object]:
    router = build_router(memory_dir)
    confusion = defaultdict(Counter)
    by_expected = Counter()
    by_predicted = Counter()
    mismatches: List[Dict[str, str]] = []
    correct = 0
    total = 0

    labeled_samples = load_labeled_samples(labels_file, limit) if labels_file else []
    sample_iter = (
        ((source, user_message, expected) for source, user_message, expected in labeled_samples)
        if labeled_samples else
        ((source, user_message, infer_expected_intent(user_message)) for source, user_message in iter_session_messages(session_dir, limit))
    )

    for source_file, user_message, expected in sample_iter:
        if expected is None:
            continue
        result = router.route(user_message, context={})
        predicted = result.intent.value
        total += 1
        by_expected[expected] += 1
        by_predicted[predicted] += 1
        confusion[expected][predicted] += 1
        if predicted == expected:
            correct += 1
        else:
            mismatches.append({
                "file": source_file,
                "user_message": user_message,
                "expected": expected,
                "predicted": predicted,
                "reasoning": result.reasoning,
            })

    accuracy = (correct / total) if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "by_expected": dict(by_expected),
        "by_predicted": dict(by_predicted),
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "top_mismatches": mismatches[:20],
        "used_labels_file": str(labels_file) if labels_file else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate intent router with session logs.")
    parser.add_argument("--session-dir", default="session_logs", help="Path to session log directory")
    parser.add_argument("--limit", type=int, default=200, help="Maximum number of messages to evaluate")
    parser.add_argument(
        "--memory-dir",
        default=".agent_memory_eval",
        help="Temporary vector memory directory used during evaluation",
    )
    parser.add_argument("--labels-file", default="", help="Optional JSONL file with user_message/expected_intent labels")
    parser.add_argument("--min-accuracy", type=float, default=0.0, help="Fail if accuracy is below this threshold")
    args = parser.parse_args()

    session_dir = Path(args.session_dir).resolve()
    memory_dir = Path(args.memory_dir).resolve()
    memory_dir.mkdir(parents=True, exist_ok=True)

    labels_file = Path(args.labels_file).resolve() if args.labels_file else None
    report = evaluate(session_dir=session_dir, limit=args.limit, memory_dir=memory_dir, labels_file=labels_file)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["accuracy"] < args.min_accuracy:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
