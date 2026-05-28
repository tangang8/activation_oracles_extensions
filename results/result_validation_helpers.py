"""Result validation helpers for inspecting and peeking into StrongReject cache files."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from IPython.display import display

from viz_helpers import (
    PathAliaser,
    apply_display_transforms,
    apply_path_segment_aliases,
    clip_text,
    condition_rank,
    rename_display_columns,
)
from prompt_utils import load_target_prompts_from_dataset


def longest_common_path_prefix(paths) -> str:
    """Longest component-wise common path prefix of an iterable of '/'-separated paths."""
    paths = [p for p in paths if isinstance(p, str)]
    if not paths:
        return ''
    split = [[x for x in p.split('/') if x] for p in paths]
    common: list[str] = []
    for components in zip(*split):
        first = components[0]
        if all(c == first for c in components):
            common.append(first)
        else:
            break
    leading = '/' if paths[0].startswith('/') else ''
    return leading + '/'.join(common)


def build_reliability_with_shared_prefix(
    reliability: pd.DataFrame,
    details: pd.DataFrame,
    cache_root,
    group_keys=('condition', 'probe_kind', 'probe_name', 'oracle_prompt_file'),
) -> pd.DataFrame:
    """Add a `shared_cache_prefix` column to the reliability table.

    For each (condition, probe_kind, probe_name, oracle_prompt_file) group, finds the
    longest common parent-directory prefix of the contributing cache files, strips the
    common cache root, and applies known path-segment aliases for readability.
    """
    group_keys = list(group_keys)
    prefix_map = (
        details.groupby(group_keys)['cache_path']
        .apply(longest_common_path_prefix)
        .reset_index(name='shared_cache_prefix')
    )
    cache_root_str = str(Path(cache_root).resolve()).rstrip('/') + '/'
    prefix_map['shared_cache_prefix'] = (
        prefix_map['shared_cache_prefix']
        .map(lambda p: p[len(cache_root_str):] if isinstance(p, str) and p.startswith(cache_root_str) else p)
        .map(apply_path_segment_aliases)
    )
    rel = reliability.drop(columns=['oracle_prompt', 'oracle_prompt_index'], errors='ignore')
    return rel.merge(prefix_map, on=group_keys, how='left')


def build_oracle_output_examples(
    details: pd.DataFrame,
    probe_order: dict,
    path_aliaser: PathAliaser,
) -> pd.DataFrame:
    """One example oracle output per unique score for every (condition, probe, oracle_prompt_file).

    Picks the first cache row in each (condition, probe_kind, probe_name,
    oracle_prompt_file, score) group, then orders rows by token-stream probe
    rank (then score). Returns the peek DataFrame ready for display.
    """
    group_keys = ['condition', 'probe_kind', 'probe_name', 'oracle_prompt_file', 'score']
    examples = (
        details.dropna(subset=['score', 'cache_path'])
        .sort_values(group_keys + ['target_prompt_index', 'rollout_index'])
        .groupby(group_keys, as_index=False, dropna=False)
        .first()
    )
    examples['_condition_rank'] = examples['condition'].map(condition_rank)
    examples['_probe_rank'] = examples.apply(
        lambda r: probe_order.get((r.get('probe_kind'), r.get('probe_name')), 1e9), axis=1,
    )
    examples = examples.sort_values(
        ['_condition_rank', 'oracle_prompt_file', 'probe_kind', '_probe_rank', 'probe_name', 'score']
    ).drop(columns=['_condition_rank', '_probe_rank']).reset_index(drop=True)
    # Full oracle (and target, when present) rollouts — the whole point of the
    # examples table is to see what text actually got the given judge score.
    return build_peek_table(
        examples, path_aliaser,
        oracle_rollout_clip=None, target_rollout_clip=None,
    )


def build_coverage_df(manifest: dict) -> pd.DataFrame:
    """Return a per-condition coverage summary DataFrame from a compiled manifest."""
    all_conditions = sorted(
        set(manifest.get('expected_files', {})) | set(manifest.get('loaded_files', {}))
    )
    df = pd.DataFrame([
        {
            'condition': condition,
            'expected_files': manifest.get('expected_files', {}).get(condition, 0),
            'loaded_files': manifest.get('loaded_files', {}).get(condition, 0),
        }
        for condition in all_conditions
    ])
    df['missing_files'] = df['expected_files'] - df['loaded_files']
    df['coverage_pct'] = df.apply(
        lambda row: (row['loaded_files'] / row['expected_files']) if row['expected_files'] else 1.0,
        axis=1,
    )
    return df


def _get_compliance_leaf(entry: dict, probe_kind: str | None, probe_name: str | None):
    compliance = entry.get('compliance')
    if not isinstance(compliance, dict):
        return None
    if probe_kind in ('tokens', 'token_points'):
        container = compliance.get(probe_kind, {})
        return container.get(probe_name) if isinstance(container, dict) else None
    if probe_kind:
        return compliance.get(probe_kind)
    return compliance


def _leaf_skip_reason(leaf, judge_instruction_stem: str | None) -> str:
    if not isinstance(leaf, dict):
        return 'compliance leaf missing or not a dict'
    score = leaf.get('score')
    if score is None:
        return leaf.get('reason') or 'score is null'
    try:
        v = float(score)
    except (TypeError, ValueError):
        return f'score not numeric: {score!r}'
    scale = leaf.get('score_scale')
    if scale is not None and scale != 'strongreject_0_1':
        return f'unexpected score_scale={scale!r}'
    if v < 0.0 or v > 1.0:
        return f'score outside [0, 1]: {v}'
    instr = str(leaf.get('judge_instruction_file', ''))
    if instr and judge_instruction_stem and Path(instr).stem != judge_instruction_stem:
        return f'wrong judge_instruction_file={Path(instr).name!r}'
    return 'score valid (unexpected)'


def _reasons_for_missing_indices(
    path: str,
    missing_indices: list,
    probe_kind: str | None,
    probe_name: str | None,
    judge_instruction_stem: str | None,
) -> str:
    try:
        entries = load_cache_entries(path)
    except Exception as exc:
        return f'error reading cache: {exc}'
    index_to_entry = {}
    for e in entries:
        idx = e.get('rollout_index') if e.get('rollout_index') is not None else e.get('oracle_rollout_index')
        if idx is not None:
            index_to_entry[idx] = e
    parts = []
    for idx in sorted(missing_indices):
        if idx not in index_to_entry:
            parts.append(f'idx {idx}: entry absent')
        else:
            leaf = _get_compliance_leaf(index_to_entry[idx], probe_kind, probe_name)
            reason = _leaf_skip_reason(leaf, judge_instruction_stem)
            if 'Judge output format invalid:' in reason:
                reason = 'invalid judge output'
            parts.append(f'idx {idx}: {reason}')
    return '; '.join(parts) if parts else 'unknown'


def _coverage_warning_reason(
    row: dict,
    missing_set: set,
    skipped_by_path: dict,
    judge_instruction_stem: str | None = None,
) -> str:
    """Derive a human-readable reason for a coverage warning row."""
    if row.get('reason'):
        return str(row['reason'])
    path = str(row.get('path', ''))
    if path and path in missing_set:
        return 'file missing'
    missing_indices = row.get('missing_rollout_indices') or []
    if path and missing_indices:
        return _reasons_for_missing_indices(
            path, missing_indices,
            row.get('probe_kind'), row.get('probe_name'),
            judge_instruction_stem,
        )
    if path:
        leaves = skipped_by_path.get(path, [])
        if leaves:
            unique = list(dict.fromkeys(leaf['reason'] for leaf in leaves))
            return 'skipped score leaves: ' + '; '.join(unique)
    return 'rollouts not generated or not scored'


@dataclass
class CoverageReport:
    aliaser: PathAliaser
    warnings: pd.DataFrame
    n_missing_files: int
    n_malformed_files: int
    n_skipped_leaves: int

    def summary_line(self) -> str:
        return (
            f"Missing files: {self.n_missing_files} | "
            f"Malformed files: {self.n_malformed_files} | "
            f"Skipped score leaves: {self.n_skipped_leaves} | "
            f"Coverage warning rows: {len(self.warnings)}"
        )


def build_coverage_report(manifest: dict, cfg) -> CoverageReport:
    """Build the coverage validation artifacts for a compiled manifest.

    Returns a CoverageReport with the path alias legend (aliaser) and the
    coverage warnings table; the caller is responsible for displaying them.
    """
    target_prompts = load_target_prompts_from_dataset(
        limit=cfg.expected_target_prompts, offset=cfg.target_prompt_offset
    )
    prompt_by_index = {cfg.target_prompt_offset + i: p for i, p in enumerate(target_prompts)}

    missing_set = set(manifest.get('missing_files', []))
    skipped_by_path: dict = {}
    for leaf in manifest.get('skipped_score_leaves', []):
        skipped_by_path.setdefault(leaf['path'], []).append(leaf)

    all_paths: list[str] = []
    all_paths.extend(str(x) for x in manifest.get('missing_files', []))
    all_paths.extend(str(leaf['path']) for leaf in manifest.get('skipped_score_leaves', []))
    for w in manifest.get('coverage_warnings', []):
        if w.get('path'):
            all_paths.append(str(w['path']))
    aliaser = PathAliaser(cfg.target_model_name, cfg.cache_root, cfg.output_dir, all_paths)

    warnings_df = pd.DataFrame(manifest.get('coverage_warnings', []))
    if not warnings_df.empty:
        judge_stem = getattr(cfg, 'judge_instruction_stem', None)
        warnings_df['skip_reason'] = warnings_df.apply(
            lambda row: _coverage_warning_reason(row.to_dict(), missing_set, skipped_by_path, judge_stem),
            axis=1,
        )
        if 'target_prompt_index' in warnings_df.columns:
            warnings_df.insert(
                warnings_df.columns.get_loc('target_prompt_index') + 1,
                'target_prompt_preview',
                warnings_df['target_prompt_index'].map(
                    lambda idx: clip_text(prompt_by_index.get(idx, ''), 120)
                ),
            )
            warnings_df = warnings_df.drop(columns=['target_prompt_index'])
        if 'path' in warnings_df.columns:
            warnings_df['path'] = warnings_df['path'].map(
                lambda p: aliaser.alias(p) if isinstance(p, str) and p else p
            )
        warnings_df = apply_display_transforms(warnings_df)
        if 'probe_kind' in warnings_df.columns and 'probe_name' in warnings_df.columns:
            warnings_df = warnings_df.drop(columns=['probe_kind'])

    return CoverageReport(
        aliaser=aliaser,
        warnings=warnings_df,
        n_missing_files=len(manifest.get('missing_files', [])),
        n_malformed_files=len(manifest.get('malformed_files', [])),
        n_skipped_leaves=len(manifest.get('skipped_score_leaves', [])),
    )


def apply_filter(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    out = df.copy()
    for key in [
        'condition', 'oracle_prompt_file', 'probe_kind', 'probe_name',
        'target_prompt_index', 'target_rollout_index', 'oracle_rollout_index',
    ]:
        value = spec.get(key)
        if value is None:
            continue
        out = out[out[key] == value]
    return out


def load_cache_entries(cache_path: str) -> list:
    with open(cache_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    return payload.get('entries', []) if isinstance(payload, dict) else payload


def extract_leaf(container, probe_kind: str, probe_name: str | None):
    if not isinstance(container, dict):
        return None
    node = container.get(probe_kind)
    if probe_name is None:
        return node
    if isinstance(node, dict):
        return node.get(probe_name)
    # Scalar probe kinds (e.g. 'rollout_segment', 'full_seq') store the leaf
    # value (string for oracle_response, dict for compliance) directly at
    # container[probe_kind] rather than nesting it under probe_name. In compile
    # we emit these as (probe_kind, probe_kind), so probe_name == probe_kind.
    if probe_name == probe_kind:
        return node
    return None


def match_entry(entries: list, row: dict) -> dict | None:
    # 1) Strongest match: locate the entry whose compliance leaf for this
    #    (probe_kind, probe_name) actually produced this row's score. This is
    #    required for the examples table where multiple rollouts/scores share a
    #    cache file and index-based matching would pick the wrong rollout.
    row_score = row.get('score')
    probe_kind = row.get('probe_kind')
    probe_name = row.get('probe_name')
    if row_score is not None and probe_kind is not None:
        try:
            target = float(row_score)
        except (TypeError, ValueError):
            target = None
        if target is not None:
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                leaf = extract_leaf(entry.get('compliance', {}), probe_kind, probe_name)
                if not isinstance(leaf, dict):
                    continue
                try:
                    leaf_score = float(leaf.get('score'))
                except (TypeError, ValueError):
                    continue
                if abs(leaf_score - target) < 1e-9:
                    return entry
    # 2) Index-based fallback.
    row_rollout = row.get('rollout_index')
    if row_rollout is not None:
        for entry in entries:
            if isinstance(entry, dict) and entry.get('rollout_index') == row_rollout:
                return entry
    row_oracle = row.get('oracle_rollout_index')
    if row_oracle is not None:
        for entry in entries:
            if isinstance(entry, dict) and entry.get('oracle_rollout_index') == row_oracle:
                return entry
    return None


def build_peek_table(
    filtered: pd.DataFrame,
    path_aliaser,
    *,
    oracle_rollout_clip: int | None = 220,
    target_rollout_clip: int | None = 220,
    compliance_leaf_clip: int | None = 220,
    prompt_clip: int | None = 160,
) -> pd.DataFrame:
    """Load cache entries for each row in *filtered* and return a preview DataFrame.

    Args:
        filtered: Subset of the details DataFrame to inspect.
        path_aliaser: A PathAliaser instance used to compact cache paths for display.
        oracle_rollout_clip: Max chars for the oracle rollout text; None = unclipped.
        target_rollout_clip: Max chars for the target rollout text (when the cache
            entry includes a target_response that was judged alongside the oracle);
            None = unclipped.
        compliance_leaf_clip: Max chars for the compliance leaf preview; None = unclipped.
        prompt_clip: Max chars for the target/oracle prompt previews; None = unclipped.
    """
    rows = []
    for _, row in filtered.iterrows():
        cache_path = row['cache_path']
        try:
            entries = load_cache_entries(cache_path)
            entry = match_entry(entries, row)
        except Exception as exc:
            rows.append({
                'cache_path': cache_path,
                'probe_name': row.get('probe_name'),
                'score': row.get('score'),
                'error': str(exc),
            })
            continue

        oracle_leaf = (
            extract_leaf(entry.get('oracle_response', {}), row.get('probe_kind'), row.get('probe_name'))
            if isinstance(entry, dict) else None
        )
        compliance_leaf = (
            extract_leaf(entry.get('compliance', {}), row.get('probe_kind'), row.get('probe_name'))
            if isinstance(entry, dict) else None
        )
        # Use the post-thinking 'response_only' portion when present, so the
        # displayed target rollout matches what the judge actually scored.
        target_response_only = None
        if isinstance(entry, dict):
            target_format = entry.get('target_format')
            if isinstance(target_format, dict):
                target_response_only = target_format.get('response_only')
            if target_response_only is None:
                target_response_only = entry.get('target_response')

        rows.append({
            'condition': row.get('condition'),
            'oracle_prompt_file': row.get('oracle_prompt_file'),
            'target_prompt_index': row.get('target_prompt_index'),
            'target_rollout_index': row.get('target_rollout_index'),
            'oracle_rollout_index': row.get('oracle_rollout_index'),
            'probe_kind': row.get('probe_kind'),
            'probe_name': row.get('probe_name'),
            'score': row.get('score'),
            'target_prompt': clip_text(row.get('target_prompt'), prompt_clip),
            'oracle_prompt': clip_text(row.get('oracle_prompt'), prompt_clip),
            'target_rollout_response': clip_text(target_response_only, target_rollout_clip),
            'oracle_rollout': clip_text(oracle_leaf, oracle_rollout_clip),
            'compliance_leaf_preview': clip_text(compliance_leaf, compliance_leaf_clip),
            'cache_path': cache_path,
            'cache_path_alias': path_aliaser.alias(cache_path),
        })

    out = pd.DataFrame(rows)
    # Use pandas' nullable integer dtype so indices render as 'N' (not 'N.0') and NaN -> '<NA>'.
    for col in ('target_prompt_index', 'target_rollout_index', 'oracle_rollout_index'):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce').astype('Int64')
    out = apply_display_transforms(out)
    return out
