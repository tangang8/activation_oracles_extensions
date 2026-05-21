from pathlib import Path
from typing import Any
import html

from transformers import AutoTokenizer


def _resolve_output_path(output_path: str) -> Path:
    out_path = Path(output_path)
    if not out_path.is_absolute():
        website_dir = Path(__file__).resolve().parent / "website"
        website_dir.mkdir(parents=True, exist_ok=True)
        out_path = website_dir / out_path
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def _as_blocks(items: Any) -> str:
    if not isinstance(items, list):
        items = [items]
    if not items:
        return "<em>None</em>"
    return "".join(
        f"<details><summary>Sample {j}</summary><pre>{html.escape(str(v))}</pre></details>"
        for j, v in enumerate(items)
    )


def _build_token_visualization_html(
    input_text: str,
    tokenizer: AutoTokenizer,
    token_points: dict[str, int],
) -> str:
    input_ids = tokenizer(
        input_text,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    )["input_ids"][0].tolist()
    points_by_idx: dict[int, list[str]] = {}
    for name, idx in token_points.items():
        points_by_idx.setdefault(idx, []).append(name)

    rows = []
    for i, token_id in enumerate(input_ids):
        token_str = tokenizer.decode([token_id]).replace("\n", "\\n").replace("\r", "\\r")
        labels = ", ".join(points_by_idx.get(i, []))
        marker = "selected" if labels else ""
        rows.append(
            f"<tr class='{marker}'><td>{i}</td><td class='mono'>{html.escape(token_str)}</td><td>{html.escape(labels)}</td></tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>Index</th><th>Token</th><th>Labels</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _named_token_point_responses(result: dict[str, Any]) -> dict[str, list[Any]]:
    """
    Return name->responses for token-point probes.

    Works with both:
      - precomputed `named_token_points`, and
      - cache/native format using `points.token_points` + `token_points`.
    """
    named = result.get("named_token_points")
    if isinstance(named, dict):
        normalized: dict[str, list[Any]] = {}
        for name, values in named.items():
            if isinstance(values, list):
                normalized[str(name)] = values
            elif values is None:
                normalized[str(name)] = []
            else:
                normalized[str(name)] = [values]
        return normalized

    points = result.get("points", {}).get("token_points", {})
    token_point_outputs = result.get("token_points", {})
    resolved: dict[str, list[Any]] = {}
    if not isinstance(points, dict) or not isinstance(token_point_outputs, dict):
        return resolved
    for name, idx in points.items():
        responses = token_point_outputs.get(idx, token_point_outputs.get(str(idx), []))
        if isinstance(responses, list):
            resolved[str(name)] = responses
        elif responses is None:
            resolved[str(name)] = []
        else:
            resolved[str(name)] = [responses]
    return resolved


def _compliance_stats(entries: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}

    def add(probe_kind: str, leaf: Any) -> None:
        bucket = stats.setdefault(
            probe_kind,
            {"count": 0.0, "scored": 0.0, "sum_score": 0.0, "skipped": 0.0, "valid_format": 0.0},
        )
        if not isinstance(leaf, dict):
            return
        bucket["count"] += 1.0
        score = leaf.get("score")
        if isinstance(score, int):
            bucket["scored"] += 1.0
            bucket["sum_score"] += float(score)
        if bool(leaf.get("judge_skipped")):
            bucket["skipped"] += 1.0
        if leaf.get("valid_judge_format") is True:
            bucket["valid_format"] += 1.0

    for entry in entries:
        compliance = entry.get("compliance", {})
        if not isinstance(compliance, dict):
            continue
        for probe_kind in ("full_seq", "segment", "prompt_segment", "rollout_segment"):
            add(probe_kind, compliance.get(probe_kind))
        for probe_kind in ("tokens", "token_points"):
            container = compliance.get(probe_kind, {})
            if not isinstance(container, dict):
                continue
            for leaf in container.values():
                add(probe_kind, leaf)
    return stats


def _leaf_cell(leaf: Any, field: str, default: str = "") -> str:
    if not isinstance(leaf, dict):
        return default
    value = leaf.get(field, default)
    return html.escape(str(value))


def save_rollouts_html(
    rollout_entries: list[dict[str, Any]],
    compliance_results: dict[str, Any],
    output_path: str = "rollouts_report.html",
) -> Path:
    out_path = _resolve_output_path(output_path)

    cards = []
    for entry in rollout_entries:
        idx = entry["rollout_index"]
        compliance = entry["compliance"]
        target_format = entry.get("target_format", {})
        cards.append(
            f"""
            <section class="card" data-rollout="{idx}">
              <h2>Rollout {idx}</h2>
              <p><strong>Judge Instruction:</strong> {html.escape(str(compliance.get("judge_instruction_file", "N/A")))}</p>
              <p><strong>Score:</strong> {html.escape(str(compliance.get("score")))} |
                 <strong>Reason:</strong> {html.escape(str(compliance.get("reason", "")))}</p>
              <details open><summary>Target Parsed Response</summary><pre>{html.escape(str(target_format.get("response_only", "")))}</pre></details>
              <details><summary>Target Thinking</summary><pre>{html.escape(str(target_format.get("thinking", "")))}</pre></details>
              <details><summary>Judge Parsed Response</summary><pre>{html.escape(str(compliance.get("response_only", "")))}</pre></details>
              <details><summary>Judge Thinking</summary><pre>{html.escape(str(compliance.get("thinking", "")))}</pre></details>
            </section>
            """
        )

    page_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Rollouts Report</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #fafafa; }}
    .card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 14px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f6f8fa; padding: 10px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Rollouts Report</h1>
  <p><strong>Compliance rate:</strong> {compliance_results.get("compliance_rate", 0.0):.2%}</p>
  <p><strong>Partial compliance rate:</strong> {compliance_results.get("partial_compliance_rate", 0.0):.2%}</p>
  <p><strong>Total rollouts:</strong> {compliance_results.get("total", len(rollout_entries))}</p>
  {''.join(cards)}
</body>
</html>"""

    out_path.write_text(page_html, encoding="utf-8")
    return out_path


def save_oracle_rollouts_html(
    oracle_results: list[dict[str, Any]] | dict[str, Any],
    oracle_prompt: str,
    tokenizer: AutoTokenizer,
    output_path: str = "oracle_rollouts_report.html",
) -> Path:
    results = oracle_results if isinstance(oracle_results, list) else [oracle_results]
    out_path = _resolve_output_path(output_path)

    deterministic_schema = bool(results) and isinstance(results[0], dict) and "oracle_response" in results[0]
    if deterministic_schema:
        token_viz_html = "<em>No oracle results available.</em>"
        if results:
            first_points = results[0].get("oracle_points", {}).get("token_points", {})
            first_text = results[0].get("oracle_points", {}).get("combined_text", "")
            if isinstance(first_points, dict) and first_text:
                token_viz_html = _build_token_visualization_html(
                    input_text=first_text,
                    tokenizer=tokenizer,
                    token_points=first_points,
                )

        stats = _compliance_stats(results)
        summary_rows = []
        for probe_kind in sorted(stats.keys()):
            bucket = stats[probe_kind]
            scored = bucket["scored"]
            avg_score = (bucket["sum_score"] / scored) if scored > 0 else 0.0
            summary_rows.append(
                "<tr>"
                f"<td>{html.escape(probe_kind)}</td>"
                f"<td>{int(bucket['count'])}</td>"
                f"<td>{int(scored)}</td>"
                f"<td>{avg_score:.3f}</td>"
                f"<td>{int(bucket['skipped'])}</td>"
                f"<td>{int(bucket['valid_format'])}</td>"
                "</tr>"
            )
        summary_table = (
            "<table><thead><tr>"
            "<th>Probe Kind</th><th>Total</th><th>Scored</th><th>Avg Score</th><th>Skipped</th><th>Valid Format</th>"
            f"</tr></thead><tbody>{''.join(summary_rows) or '<tr><td colspan=\"6\"><em>No compliance data</em></td></tr>'}</tbody></table>"
        )

        cards = []
        for i, entry in enumerate(results):
            rollout_index = entry.get("rollout_index", i)
            target_format = entry.get("target_format", {})
            oracle_response = entry.get("oracle_response", {})
            oracle_format = entry.get("oracle_format", {})
            compliance = entry.get("compliance", {})

            scalar_probe_sections = []
            for probe_kind in ("full_seq", "segment", "prompt_segment", "rollout_segment"):
                response_text = oracle_response.get(probe_kind, "")
                format_leaf = oracle_format.get(probe_kind, {})
                compliance_leaf = compliance.get(probe_kind, {})
                scalar_probe_sections.append(
                    "<details>"
                    f"<summary>{html.escape(probe_kind)}</summary>"
                    f"<p><strong>Score:</strong> {_leaf_cell(compliance_leaf, 'score', 'None')} | "
                    f"<strong>Reason:</strong> {_leaf_cell(compliance_leaf, 'reason')}</p>"
                    f"<p><strong>Judge Response:</strong> {_leaf_cell(compliance_leaf, 'response_only')}</p>"
                    f"<p><strong>Judge Thinking:</strong> {_leaf_cell(compliance_leaf, 'thinking')}</p>"
                    f"<p><strong>Oracle Response (parsed):</strong> {_leaf_cell(format_leaf, 'response_only', str(response_text))}</p>"
                    f"<details><summary>Oracle Response Raw</summary><pre>{html.escape(str(response_text))}</pre></details>"
                    "</details>"
                )

            token_rows = []
            token_responses = oracle_response.get("token_points", {})
            token_formats = oracle_format.get("token_points", {})
            token_compliance = compliance.get("token_points", {})
            if isinstance(token_responses, dict):
                for name, response_text in token_responses.items():
                    key = str(name)
                    format_leaf = token_formats.get(key, {}) if isinstance(token_formats, dict) else {}
                    compliance_leaf = token_compliance.get(key, {}) if isinstance(token_compliance, dict) else {}
                    token_rows.append(
                        "<tr>"
                        f"<td>{html.escape(key)}</td>"
                        f"<td>{html.escape(str(response_text))}</td>"
                        f"<td>{_leaf_cell(format_leaf, 'response_only')}</td>"
                        f"<td>{_leaf_cell(compliance_leaf, 'score', 'None')}</td>"
                        f"<td>{_leaf_cell(compliance_leaf, 'reason')}</td>"
                        "</tr>"
                    )
            token_table = "".join(token_rows) or "<tr><td colspan='5'><em>No token-point responses</em></td></tr>"

            cards.append(
                f"""
                <section class="card" data-rollout="{rollout_index}">
                  <h2>Rollout {html.escape(str(rollout_index))}</h2>
                  <details open><summary>Target Parsed Response</summary><pre>{html.escape(str(target_format.get("response_only", "")))}</pre></details>
                  <details><summary>Target Thinking</summary><pre>{html.escape(str(target_format.get("thinking", "")))}</pre></details>
                  {''.join(scalar_probe_sections)}
                  <details><summary>Token-Point Compliance</summary>
                    <table>
                      <thead><tr><th>Name</th><th>Oracle Response</th><th>Oracle Parsed</th><th>Score</th><th>Reason</th></tr></thead>
                      <tbody>{token_table}</tbody>
                    </table>
                  </details>
                </section>
                """
            )

        page_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Oracle Rollouts Report</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #fafafa; }}
    .controls {{ margin-bottom: 16px; }}
    .card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 14px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f6f8fa; padding: 10px; border-radius: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
  </style>
</head>
<body>
  <h1>Oracle Rollouts Report</h1>
  <p><strong>Oracle prompt:</strong> {html.escape(oracle_prompt)}</p>
  <h2>Compliance Summary</h2>
  {summary_table}
  <h2>Token Visualization (Rollout 0)</h2>
  {token_viz_html}
  <div class="controls">
    <label>Show rollout index:
      <select id="rolloutSelect">
        <option value="all">All</option>
        {"".join(f"<option value='{html.escape(str(entry.get('rollout_index', i)))}'>{html.escape(str(entry.get('rollout_index', i)))}</option>" for i, entry in enumerate(results))}
      </select>
    </label>
  </div>
  {''.join(cards)}
  <script>
    const select = document.getElementById('rolloutSelect');
    select.addEventListener('change', () => {{
      const value = select.value;
      for (const card of document.querySelectorAll('.card')) {{
        const match = value === 'all' || card.dataset.rollout === value;
        card.style.display = match ? 'block' : 'none';
      }}
    }});
  </script>
</body>
</html>"""

        out_path.write_text(page_html, encoding="utf-8")
        return out_path

    token_viz_html = "<em>No oracle results available.</em>"
    if results:
        token_viz_html = _build_token_visualization_html(
            input_text=results[0]["combined_text"],
            tokenizer=tokenizer,
            token_points=results[0]["points"]["token_points"],
        )

    cards = []
    for i, result in enumerate(results):
        points = result.get("points", {}).get("token_points", {})
        named_points = _named_token_point_responses(result)
        full_seq = result.get("full_seq", [])
        prompt_segment = result.get("prompt_segment", [])
        rollout_segment = result.get("rollout_segment", [])
        repeat_count = max(len(full_seq), len(prompt_segment), len(rollout_segment), 1)

        token_rows = []
        for name, idx in points.items():
            responses = named_points.get(name, [])
            if not isinstance(responses, list):
                responses = [responses]
            response_html = "<br>".join(f"<div class='mono'>{html.escape(str(r))}</div>" for r in responses)
            token_rows.append(
                f"<tr><td>{html.escape(str(name))}</td><td>{idx}</td><td>{response_html}</td></tr>"
            )
        token_table = "".join(token_rows) or "<tr><td colspan='3'><em>No token-point responses</em></td></tr>"

        cards.append(
            f"""
            <section class="card" data-rollout="{i}">
              <h2>Rollout {i}</h2>
              <p><strong>Oracle repeats observed:</strong> {repeat_count}</p>
              <details open><summary>Full Sequence Responses</summary>{_as_blocks(full_seq)}</details>
              <details><summary>Prompt Segment Responses</summary>{_as_blocks(prompt_segment)}</details>
              <details><summary>Rollout Segment Responses</summary>{_as_blocks(rollout_segment)}</details>
              <details><summary>Token-Point Responses</summary>
                <table>
                  <thead><tr><th>Name</th><th>Index</th><th>Responses</th></tr></thead>
                  <tbody>{token_table}</tbody>
                </table>
              </details>
            </section>
            """
        )

    page_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Oracle Rollouts Report</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #fafafa; }}
    .controls {{ margin-bottom: 16px; }}
    .card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 14px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f6f8fa; padding: 10px; border-radius: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; margin-bottom: 6px; }}
    tr.selected td {{ background: #fff9db; }}
  </style>
</head>
<body>
  <h1>Oracle Rollouts Report</h1>
  <p><strong>Oracle prompt:</strong> {html.escape(oracle_prompt)}</p>
  <h2>Token Visualization (Rollout 0)</h2>
  {token_viz_html}
  <div class="controls">
    <label>Show rollout index:
      <select id="rolloutSelect">
        <option value="all">All</option>
        {"".join(f"<option value='{i}'>{i}</option>" for i in range(len(results)))}
      </select>
    </label>
  </div>
  {''.join(cards)}
  <script>
    const select = document.getElementById('rolloutSelect');
    select.addEventListener('change', () => {{
      const value = select.value;
      for (const card of document.querySelectorAll('.card')) {{
        const match = value === 'all' || card.dataset.rollout === value;
        card.style.display = match ? 'block' : 'none';
      }}
    }});
  </script>
</body>
</html>"""

    out_path.write_text(page_html, encoding="utf-8")
    return out_path
