"""Score the sample knowledge base. Run: python -m ragbot.evaluate"""

from __future__ import annotations

from ragbot.gold import FOLLOWUPS, GOLD, SEMANTIC


def problems(answer, item: dict) -> list[str]:
    if item.get("refuse"):
        return [] if answer.refused else ["expected a refusal"]
    issues: list[str] = []
    if answer.refused:
        issues.append("refused")
    text = answer.text.lower()
    for fragment in item.get("answer_contains", []):
        if fragment.lower() not in text:
            issues.append(f"missing {fragment!r}")
    source_name = item.get("source_contains", "")
    if source_name and not any(source_name in source.source for source in answer.sources):
        issues.append(f"missing source {source_name}")
    return issues


def main() -> int:
    from ragbot.config import Settings
    from ragbot.pipeline import RagPipeline

    settings = Settings.from_env()
    pipeline = RagPipeline(settings)
    sample = settings.sample_dir
    if not pipeline.load_cached([sample]):
        print("Indexing the sample knowledge base...")
        stats = pipeline.index_paths([sample])
    else:
        stats = pipeline.stats
    pipeline.warm()
    print(f"Indexed {stats.files} files into {stats.chunks} passages. Engine: {pipeline.backend_label()}")
    print()
    cases = [("gold", item, None) for item in GOLD]
    cases += [("semantic", item, None) for item in SEMANTIC]
    cases += [("follow-up", item, item.get("history")) for item in FOLLOWUPS]
    failed = 0
    for kind, item, history in cases:
        answer = pipeline.ask(item["question"], history=history)
        issues = problems(answer, item)
        status = "PASS" if not issues else "FAIL"
        if issues:
            failed += 1
        top = answer.sources[0] if answer.sources else None
        scores = ""
        if top is not None:
            scores = f" dense={top.dense_score:.3f} bm25={top.sparse_score:.2f}"
            if top.rerank_score is not None:
                scores += f" rerank={top.rerank_score:.3f}"
            scores += f" source={top.source}"
        print(f"[{status}] {kind}: {item['question']}")
        print(f"       backend={answer.backend} refused={answer.refused}{scores}")
        if issues:
            print(f"       issues: {', '.join(issues)}")
            print(f"       answer: {answer.text}")
        else:
            preview = answer.text.replace("\n", " ")
            if len(preview) > 180:
                preview = preview[:180] + "..."
            print(f"       {preview}")
    print()
    passed = len(cases) - failed
    print(f"{passed} passed, {failed} failed, out of {len(cases)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
