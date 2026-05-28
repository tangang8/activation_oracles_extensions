"""Display and visualization helpers for StrongReject result notebooks."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_LABELS = {
    'target_baseline': 'Target Baseline',
    'oracle_rollout_control': 'Oracle Control Baseline',
    'user_prompt_oracle': 'User Prompt Oracle',
    'target_rollout_oracle': 'Target Rollout Oracle',
}

# Display ordering for the condition column. Used by apply_probe_sort and any
# table that pivots/sorts by condition so the experiment progression is shown
# left-to-right / top-to-bottom: baselines first, then the two oracle modes.
CONDITION_ORDER: dict[str, int] = {name: i for i, name in enumerate(CONDITION_LABELS.keys())}


def condition_rank(value) -> float:
    """Return the configured display rank for a condition name (1e9 if unknown)."""
    return CONDITION_ORDER.get(value, 1e9)

SCORE_COLS = {
    'mean_score', 'se_score', 'asr_0_2', 'asr_0_2_se', 'asr_0_5', 'asr_0_5_se',
    'asr_0_8', 'asr_0_8_se', 'asr_1', 'asr_1_se',
    'sd_within_prompt_oracle_rollouts', 'sd_within_prompt_target_rollouts',
    'mean_within_prompt_sd_oracle_rollouts', 'mean_within_prompt_sd_target_rollouts',
    'score',
}


def display_condition(value: str) -> str:
    return CONDITION_LABELS.get(value, value)


ORACLE_PROMPT_FILE_LABELS = {
    'default_oracle_prompts': 'Oracle Prompt A',
    'model_answer_min_200_words': 'Oracle Prompt B',
}


def display_oracle_prompt_file(value) -> str | None:
    if pd.isna(value):
        return None
    name = Path(str(value)).name
    stem = name[:-5] if name.endswith('.json') else name
    return ORACLE_PROMPT_FILE_LABELS.get(stem, stem)


PATH_SEGMENT_ALIASES: dict[str, str] = {
    'Decode_these_activations_into_the_most_detailed_67e237049b5359ee': 'Decode',
    'What_is_the_model_s_answer_Provide_specific_det_ab6f30fe97edfb33': 'What',
}


def apply_path_segment_aliases(text):
    """Replace known long path segments with their short display labels."""
    if not isinstance(text, str):
        return text
    for raw, short in PATH_SEGMENT_ALIASES.items():
        text = text.replace(raw, short)
    return text


class PathAliaser:
    """Compacts long cache paths for display by aliasing shared subdirectory prefixes."""

    def __init__(
        self,
        target_model_name: str,
        cache_root: Path | str,
        output_dir: Path | str,
        paths_for_aliasing: list[str] | None = None,
        max_aliases: int = 52,
    ):
        self.target_model_dir = f"target_{target_model_name.replace('/', '_')}"
        self.target_marker = f"/{self.target_model_dir}/"
        self.cache_root = Path(cache_root)
        self.output_dir = Path(output_dir)
        self._aliases: dict[str, str] = {}
        if paths_for_aliasing:
            self._aliases = self._build_aliases(paths_for_aliasing, max_aliases)

    def _path_tail(self, path_text: str) -> str:
        if self.target_marker in path_text:
            return path_text.split(self.target_marker, 1)[1]
        cache_prefix = str(self.cache_root.resolve()).rstrip('/') + '/'
        if path_text.startswith(cache_prefix):
            return path_text[len(cache_prefix):]
        out_prefix = str(self.output_dir.resolve()).rstrip('/') + '/'
        if path_text.startswith(out_prefix):
            return path_text[len(out_prefix):]
        return path_text

    @staticmethod
    def _alias_labels(n: int) -> list[str]:
        """Generate up to n letter labels: A..Z, then AA..AZ, BA..BZ, ..."""
        letters = [chr(ord('A') + i) for i in range(26)]
        out = list(letters)
        for first in letters:
            for second in letters:
                if len(out) >= n:
                    return out[:n]
                out.append(first + second)
        return out[:n]

    def _build_aliases(self, paths: list[str], max_aliases: int = 52) -> dict[str, str]:
        # Use the longest prefix per path (its parent directory) so the alias
        # collapses as much of the path as possible — only the filename remains visible.
        counts: dict[str, int] = {}
        for p in paths:
            tail = self._path_tail(str(p))
            parts = [x for x in tail.split('/') if x]
            if len(parts) >= 2:
                prefix = '/'.join(parts[:-1])
                counts[prefix] = counts.get(prefix, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        labels = self._alias_labels(min(len(ranked), max_aliases))
        return {labels[i]: prefix for i, (prefix, _) in enumerate(ranked[:max_aliases])}

    def alias(self, path) -> str:
        if pd.isna(path):
            return path
        tail = self._path_tail(str(path))
        result = tail
        for alias, prefix in self._aliases.items():
            marker = prefix + '/'
            if tail.startswith(marker):
                result = f"{alias}/{tail[len(marker):]}"
                break
            if tail == prefix:
                result = alias
                break
        return apply_path_segment_aliases(result)

    def add_alias_column(self, df: pd.DataFrame, source_col: str, alias_col: str) -> pd.DataFrame:
        out = df.copy()
        out[alias_col] = out[source_col].map(self.alias)
        return out

    @staticmethod
    def _common_root(prefixes: list[str]) -> str:
        """Longest component-wise common prefix of the given path-like strings."""
        if not prefixes:
            return ''
        split = [[x for x in p.split('/') if x] for p in prefixes]
        common: list[str] = []
        for components in zip(*split):
            first = components[0]
            if all(c == first for c in components):
                common.append(first)
            else:
                break
        return '/'.join(common)

    def legend_df(self) -> pd.DataFrame:
        common = self._common_root(list(self._aliases.values()))
        prefix_strip = common + '/' if common else ''
        rows = []
        for alias, prefix in self._aliases.items():
            shown = prefix[len(prefix_strip):] if prefix_strip and prefix.startswith(prefix_strip) else (
                '' if prefix == common else prefix
            )
            rows.append({'alias': alias, 'shared_subdir_prefix': apply_path_segment_aliases(shown)})
        return pd.DataFrame(rows)

    def _repr_html_(self) -> str:
        # When Jupyter auto-renders the aliaser (e.g. as a cell's return value),
        # show the legend table instead of the default <PathAliaser at 0x...> repr.
        if not self._aliases:
            return (
                '<div><b>Path alias legend (letter -> shared parent directory):</b></div>'
                '<div><i>(no path aliases — all paths shown in full)</i></div>'
            )
        common = self._common_root(list(self._aliases.values()))
        header = '<div><b>Path alias legend (letter -> shared parent directory)'
        if common:
            header += f' &mdash; common root: <code>{apply_path_segment_aliases(common)}/</code>'
        header += ':</b></div>'
        df = self.legend_df()
        with pd.option_context('display.max_colwidth', None):
            table_html = df.to_html(index=False)
        table_html = table_html.replace(
            '<table',
            '<table style="white-space: nowrap; text-align: left;"',
            1,
        )
        return header + table_html


_SCORE_LIKE_COLS = {
    'mean_score', 'score', 'asr_0_2', 'asr_0_5', 'asr_0_8', 'asr_1',
    'Mean Score', 'Score', 'ASR >= 0.2', 'ASR >= 0.5', 'ASR >= 0.8', 'ASR = 1.0',
}
_UNCERTAINTY_LIKE_COLS = {
    'se_score', 'asr_0_2_se', 'asr_0_5_se', 'asr_0_8_se', 'asr_1_se',
    'sd_within_prompt_oracle_rollouts', 'sd_within_prompt_target_rollouts',
    'mean_within_prompt_sd_oracle_rollouts', 'mean_within_prompt_sd_target_rollouts',
    'SE Across Prompts',
    'Within-Prompt Std across Oracle Rollouts', 'Within-Prompt Std across Target Rollouts',
    'ASR >= 0.2 SE Across Prompts', 'ASR >= 0.5 SE Across Prompts',
    'ASR >= 0.8 SE Across Prompts', 'ASR = 1.0 SE Across Prompts',
}

_HEATMAP_ALPHA = 0.88


def _contrasting_text(r8: int, g8: int, b8: int, alpha: float = _HEATMAP_ALPHA) -> str:
    """Return #ffffff or #1a1a1a based on WCAG relative luminance of the alpha-blended cell."""
    re = alpha * r8 + (1.0 - alpha) * 255.0
    ge = alpha * g8 + (1.0 - alpha) * 255.0
    be = alpha * b8 + (1.0 - alpha) * 255.0

    def _lin(c: float) -> float:
        c = min(max(c / 255.0, 0.0), 1.0)
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * _lin(re) + 0.7152 * _lin(ge) + 0.0722 * _lin(be)
    return '#ffffff' if lum < 0.45 else '#1a1a1a'


def _heatmap_cell_styles(
    df: pd.DataFrame,
    score_cols: list[str],
    uncertainty_cols: list[str],
    alpha: float = _HEATMAP_ALPHA,
    relative_score_norm: bool = False,
) -> pd.DataFrame:
    """Return a same-shape DataFrame of CSS strings with per-cell background + contrasting text."""
    score_cmap = plt.get_cmap('YlGn')
    unc_cmap = plt.get_cmap('YlOrRd')
    if relative_score_norm:
        all_vals = np.concatenate([
            df[c].dropna().to_numpy(dtype=float) for c in score_cols if c in df.columns
        ]) if score_cols else np.array([])
        finite = all_vals[np.isfinite(all_vals)] if all_vals.size else np.array([])
        score_norm = mcolors.Normalize(
            vmin=float(finite.min()) if finite.size else 0.0,
            vmax=float(finite.max()) if finite.size else 1.0,
        )
    else:
        score_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    unc_vals = np.concatenate([
        df[c].dropna().to_numpy(dtype=float) for c in uncertainty_cols if c in df.columns
    ]) if uncertainty_cols else np.array([])
    finite_unc = unc_vals[np.isfinite(unc_vals)] if unc_vals.size else np.array([])
    if finite_unc.size >= 2:
        unc_norm = mcolors.Normalize(
            vmin=float(np.percentile(finite_unc, 5)),
            vmax=float(np.percentile(finite_unc, 95)),
        )
    else:
        unc_norm = mcolors.Normalize(vmin=0.0, vmax=0.3)

    base = 'text-align: right; font-variant-numeric: tabular-nums; padding: 5px 8px; vertical-align: middle;'
    out = pd.DataFrame(base, index=df.index, columns=df.columns)

    def _color_cell(val, cmap, norm) -> str:
        try:
            v = float(val)
        except (TypeError, ValueError):
            return base
        if not np.isfinite(v):
            return base
        rgba = cmap(norm(v))
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        fg = _contrasting_text(r, g, b, alpha)
        return f'background-color: rgba({r},{g},{b},{alpha}); color: {fg}; {base}'

    for col in score_cols:
        if col in df.columns:
            out[col] = df[col].map(lambda v: _color_cell(v, score_cmap, score_norm))
    for col in uncertainty_cols:
        if col in df.columns:
            out[col] = df[col].map(lambda v: _color_cell(v, unc_cmap, unc_norm))

    return out


def percent_style(df: pd.DataFrame, extra_pct_cols=None, relative_score_norm: bool = False):
    pct_cols = [c for c in df.columns if c in SCORE_COLS or c.startswith('asr_')]
    pretty_pct_cols = [
        c for c in df.columns
        if c in _SCORE_LIKE_COLS or c in _UNCERTAINTY_LIKE_COLS or c.startswith('ASR')
    ]
    if extra_pct_cols:
        pct_cols = sorted(set(pct_cols) | set(extra_pct_cols))
    pct_cols = sorted(set(pct_cols) | set(pretty_pct_cols))

    fmt = {c: '{:.1%}' for c in pct_cols if c in df.columns}
    styler = df.style.format(fmt, na_rep='—')

    score_cols = [c for c in _SCORE_LIKE_COLS if c in df.columns]
    uncertainty_cols = [c for c in _UNCERTAINTY_LIKE_COLS if c in df.columns]
    if score_cols or uncertainty_cols:
        styler = styler.apply(
            lambda d: _heatmap_cell_styles(d, score_cols, uncertainty_cols, relative_score_norm=relative_score_norm),
            axis=None,
        )

    styler = styler.set_table_styles([
        {
            'selector': 'table',
            'props': [('border-collapse', 'collapse'), ('margin', '0 auto')],
        },
        {
            'selector': 'thead th',
            'props': [
                ('background-color', '#1e293b'), ('color', 'white'),
                ('font-weight', '600'), ('text-align', 'center'),
                ('vertical-align', 'middle'), ('padding', '5px 8px'),
                ('border-bottom', '2px solid #475569'),
            ],
        },
        {
            'selector': 'tbody td',
            'props': [
                ('border-bottom', '1px solid #e2e8f0'),
                ('padding', '5px 8px'),
            ],
        },
        {
            'selector': 'tbody tr:hover td',
            'props': [('filter', 'brightness(0.93)')],
        },
        {
            'selector': 'caption',
            'props': [('caption-side', 'top'), ('font-weight', '600'), ('color', '#1e293b')],
        },
    ], overwrite=True)

    try:
        styler = styler.hide(axis='index')
    except Exception:
        pass
    return styler


def _fmt_pct(value, na_rep: str = '—') -> str:
    if pd.isna(value):
        return na_rep
    return f'{float(value) * 100:.1f}%'


def _fmt_mean_pm_se(mean, se, na_rep: str = '—') -> str:
    if pd.isna(mean):
        return na_rep
    if pd.isna(se):
        return f'{float(mean) * 100:.1f}%'
    return f'{float(mean) * 100:.1f}% ± {float(se) * 100:.1f}%'


def _gradient_styles_for_columns(
    out: pd.DataFrame,
    column_means: dict[str, pd.Series],
    cmap_name: str,
    shared: bool,
    alpha: float = _HEATMAP_ALPHA,
) -> None:
    """Mutate `out` (a CSS-string DataFrame) to add a heatmap gradient.

    column_means maps display column name -> numeric Series (aligned to out's rows).
    If shared=True, vmin/vmax come from pooling all column values; else per-column.
    """
    cmap = plt.get_cmap(cmap_name)
    base = 'text-align: right; font-variant-numeric: tabular-nums; padding: 5px 8px; vertical-align: middle;'

    def _color(val, norm) -> str:
        if pd.isna(val):
            return base
        try:
            v = float(val)
        except (TypeError, ValueError):
            return base
        if not np.isfinite(v):
            return base
        rgba = cmap(norm(v))
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        fg = _contrasting_text(r, g, b, alpha)
        return f'background-color: rgba({r},{g},{b},{alpha}); color: {fg}; {base}'

    if shared:
        pooled = pd.concat([s for s in column_means.values()]).dropna()
        if pooled.empty:
            return
        norm = mcolors.Normalize(vmin=float(pooled.min()), vmax=float(pooled.max()))
        for col, series in column_means.items():
            if col in out.columns:
                out[col] = [_color(v, norm) for v in series.to_list()]
    else:
        for col, series in column_means.items():
            vals = series.dropna()
            if vals.empty or col not in out.columns:
                continue
            norm = mcolors.Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
            out[col] = [_color(v, norm) for v in series.to_list()]


def render_score_std_table(
    df: pd.DataFrame,
    *,
    include_within_prompt_std_oracle: bool = True,
    include_within_prompt_std_target: bool = True,
):
    """Build a Score + uncertainty table from summary rows.

    Columns: condition, probe_name, oracle_prompt_file,
        Mean Score                                        (green gradient on its own scale)
        SE Across Prompts                                 (one shared orange-red gradient)
        Within-Prompt Std across Oracle Rollouts          (shared orange-red gradient)
        Within-Prompt Std across Target Rollouts          (shared orange-red gradient)

    SE and Within-Prompt Std measure *different* uncertainties (precision of the
    across-prompts mean vs typical within-prompt rollout variability), so they
    are kept as separate columns — only the visual gradient is shared.
    """
    src = df.reset_index(drop=True)
    keep = ['condition', 'probe_name', 'oracle_prompt_file']
    se_col = 'se_score'
    sd_oracle = 'mean_within_prompt_sd_oracle_rollouts'
    sd_target = 'mean_within_prompt_sd_target_rollouts'

    mean_label = 'Mean Score'
    se_label = 'SE Across Prompts'
    sd_oracle_label = 'Within-Prompt Std across Oracle Rollouts'
    sd_target_label = 'Within-Prompt Std across Target Rollouts'

    display_df = src[keep].copy()
    display_df[mean_label] = [_fmt_pct(v) for v in src['mean_score']]
    display_df[se_label] = [_fmt_pct(v) for v in src[se_col]]
    if include_within_prompt_std_oracle:
        display_df[sd_oracle_label] = [_fmt_pct(v) for v in src[sd_oracle]]
    if include_within_prompt_std_target:
        display_df[sd_target_label] = [_fmt_pct(v) for v in src[sd_target]]
    display_df = apply_display_transforms(display_df)
    display_df = rename_display_columns(display_df)

    score_means = {mean_label: src['mean_score']}
    uncertainty_means: dict[str, pd.Series] = {se_label: src[se_col]}
    if include_within_prompt_std_oracle:
        uncertainty_means[sd_oracle_label] = src[sd_oracle]
    if include_within_prompt_std_target:
        uncertainty_means[sd_target_label] = src[sd_target]

    def _styler(d: pd.DataFrame) -> pd.DataFrame:
        base = 'text-align: right; font-variant-numeric: tabular-nums; padding: 5px 8px; vertical-align: middle;'
        out = pd.DataFrame(base, index=d.index, columns=d.columns)
        _gradient_styles_for_columns(out, score_means, 'YlGn', shared=False)
        _gradient_styles_for_columns(out, uncertainty_means, 'YlOrRd', shared=True)
        return out

    styler = display_df.style.apply(_styler, axis=None)
    styler = _apply_table_styles(styler)
    try:
        styler = styler.hide(axis='index')
    except Exception:
        pass
    return styler


def render_asr_table(df: pd.DataFrame):
    """Build an ASR styled table (condition / probe / oracle_prompt_file / ASR thresholds).

    Each ASR column shows 'mean% ± SE%'. All ASR cells share a single green
    gradient computed over every ASR mean in the table.
    """
    src = df.reset_index(drop=True)
    keep = ['condition', 'probe_name', 'oracle_prompt_file']
    asr_specs = [
        ('asr_0_2', 'asr_0_2_se', 'ASR >= 0.2'),
        ('asr_0_5', 'asr_0_5_se', 'ASR >= 0.5'),
        ('asr_0_8', 'asr_0_8_se', 'ASR >= 0.8'),
        ('asr_1',   'asr_1_se',   'ASR = 1.0'),
    ]

    display_df = src[keep].copy()
    column_means: dict[str, pd.Series] = {}
    for m, s, label in asr_specs:
        display_df[label] = [_fmt_mean_pm_se(mv, sv) for mv, sv in zip(src[m], src[s])]
        column_means[label] = src[m]
    display_df = apply_display_transforms(display_df)
    display_df = rename_display_columns(display_df)

    def _styler(d: pd.DataFrame) -> pd.DataFrame:
        base = 'text-align: right; font-variant-numeric: tabular-nums; padding: 5px 8px; vertical-align: middle;'
        out = pd.DataFrame(base, index=d.index, columns=d.columns)
        _gradient_styles_for_columns(out, column_means, 'YlGn', shared=True)
        return out

    styler = display_df.style.apply(_styler, axis=None)
    styler = _apply_table_styles(styler)
    try:
        styler = styler.hide(axis='index')
    except Exception:
        pass
    return styler


def render_baseline_table(df: pd.DataFrame):
    """Baseline rows shown as one combined table: Mean Score, SE, and all ASR columns.

    Color gradients:
      - Mean Score: green gradient on its own scale
      - SE Across Prompts: orange-red gradient on its own scale
      - All ASR cells: single green gradient pooled across every ASR mean.
    """
    src = df.reset_index(drop=True)
    keep = ['condition', 'probe_name', 'oracle_prompt_file']
    asr_specs = [
        ('asr_0_2', 'asr_0_2_se', 'ASR >= 0.2'),
        ('asr_0_5', 'asr_0_5_se', 'ASR >= 0.5'),
        ('asr_0_8', 'asr_0_8_se', 'ASR >= 0.8'),
        ('asr_1',   'asr_1_se',   'ASR = 1.0'),
    ]
    mean_label = 'Mean Score'
    se_label = 'SE Across Prompts'

    display_df = src[keep].copy()
    display_df[mean_label] = [_fmt_pct(v) for v in src['mean_score']]
    display_df[se_label] = [_fmt_pct(v) for v in src['se_score']]
    asr_means: dict[str, pd.Series] = {}
    for m, s, label in asr_specs:
        display_df[label] = [_fmt_mean_pm_se(mv, sv) for mv, sv in zip(src[m], src[s])]
        asr_means[label] = src[m]

    display_df = apply_display_transforms(display_df)
    display_df = rename_display_columns(display_df)

    score_means = {mean_label: src['mean_score']}
    se_means = {se_label: src['se_score']}

    def _styler(d: pd.DataFrame) -> pd.DataFrame:
        base = 'text-align: right; font-variant-numeric: tabular-nums; padding: 5px 8px; vertical-align: middle;'
        out = pd.DataFrame(base, index=d.index, columns=d.columns)
        _gradient_styles_for_columns(out, score_means, 'YlGn', shared=False)
        _gradient_styles_for_columns(out, se_means, 'YlOrRd', shared=False)
        _gradient_styles_for_columns(out, asr_means, 'YlGn', shared=True)
        return out

    styler = display_df.style.apply(_styler, axis=None)
    styler = _apply_table_styles(styler)
    try:
        styler = styler.hide(axis='index')
    except Exception:
        pass
    return styler


def print_oracle_prompts_legend(cfg) -> None:
    """Print 'Oracle Prompt A/B/...: <first prompt text>' for each prompts path in cfg."""
    from prompt_utils import load_oracle_prompts_from_file  # local: avoid top-level dep
    print('Oracle prompt legend:')
    for path in cfg.oracle_prompts_paths:
        label = display_oracle_prompt_file(path)
        try:
            prompts = load_oracle_prompts_from_file(path)
            first = prompts[0] if prompts else '<no prompts>'
        except Exception as exc:
            first = f'<error: {exc}>'
        print(f'  {label}: {first}')


def render_oracle_prompt_comparison_table(df: pd.DataFrame, probe_order: dict | None = None):
    """Compare 'mean ± SE' across Oracle Prompt A vs B for one condition.

    Expects rows already filtered to a single condition. Pivots so each
    oracle_prompt_file becomes one column (labeled 'Oracle Prompt A' / 'B').
    Rows are identified by (condition, probe_name) and ordered by `probe_order`
    (token-stream rank) instead of alphabetical name.
    A single green gradient is applied across all mean values in the table.
    """
    cols = ['condition', 'probe_kind', 'probe_name', 'oracle_prompt_file', 'mean_score', 'se_score']
    src = df[[c for c in cols if c in df.columns]].copy()

    # Sort source rows by (condition order, token-stream rank, probe_name) so
    # the pivot preserves this ordering when sort=False.
    if probe_order is not None and 'probe_kind' in src.columns:
        src['_condition_rank'] = src['condition'].map(condition_rank)
        src['_probe_rank'] = src.apply(
            lambda r: probe_order.get((r.get('probe_kind'), r.get('probe_name')), 1e9), axis=1,
        )
        src = src.sort_values(['_condition_rank', '_probe_rank', 'probe_name'])
        src = src.drop(columns=['_condition_rank', '_probe_rank'])

    index_cols = ['condition', 'probe_name']
    mean_pivot = src.pivot_table(
        index=index_cols, columns='oracle_prompt_file', values='mean_score', aggfunc='first', sort=False,
    )
    se_pivot = src.pivot_table(
        index=index_cols, columns='oracle_prompt_file', values='se_score', aggfunc='first', sort=False,
    )

    display_df = mean_pivot.reset_index()[index_cols].copy()
    column_means: dict[str, pd.Series] = {}
    for raw_col in mean_pivot.columns:
        label = display_oracle_prompt_file(raw_col) or str(raw_col)
        means_series = mean_pivot[raw_col].reset_index(drop=True)
        se_series = se_pivot[raw_col].reset_index(drop=True)
        display_df[label] = [_fmt_mean_pm_se(m, s) for m, s in zip(means_series, se_series)]
        column_means[label] = means_series

    display_df = apply_display_transforms(display_df)
    display_df = rename_display_columns(display_df)
    display_df = display_df.reset_index(drop=True)

    def _styler(d: pd.DataFrame) -> pd.DataFrame:
        base = 'text-align: right; font-variant-numeric: tabular-nums; padding: 5px 8px; vertical-align: middle;'
        out = pd.DataFrame(base, index=d.index, columns=d.columns)
        _gradient_styles_for_columns(out, column_means, 'YlGn', shared=True)
        return out

    styler = display_df.style.apply(_styler, axis=None)
    styler = _apply_table_styles(styler)
    try:
        styler = styler.hide(axis='index')
    except Exception:
        pass
    return styler


def _apply_table_styles(styler):
    return styler.set_table_styles([
        {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('margin', '0 auto')]},
        {'selector': 'thead th', 'props': [
            ('background-color', '#1e293b'), ('color', 'white'),
            ('font-weight', '600'), ('text-align', 'center'),
            ('vertical-align', 'middle'), ('padding', '5px 8px'),
            ('border-bottom', '2px solid #475569'),
        ]},
        {'selector': 'tbody td', 'props': [
            ('border-bottom', '1px solid #e2e8f0'),
            ('padding', '5px 8px'),
        ]},
        {'selector': 'tbody tr:hover td', 'props': [('filter', 'brightness(0.93)')]},
        {'selector': 'caption', 'props': [
            ('caption-side', 'top'), ('font-weight', '600'), ('color', '#1e293b'),
        ]},
    ], overwrite=True)


def clip_text(value, n: int | None = 180):
    if pd.isna(value):
        return value
    text = str(value)
    if n is None:
        return text
    return text if len(text) <= n else text[:n] + '...'


def apply_display_transforms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'condition' in out.columns:
        out['condition'] = out['condition'].map(display_condition)
    if 'oracle_prompt_file' in out.columns:
        out['oracle_prompt_file'] = out['oracle_prompt_file'].map(display_oracle_prompt_file)
    return out


def rename_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    if 'probe_kind' in df.columns and 'probe_name' in df.columns:
        df = df.drop(columns=['probe_kind'])
    pretty = {
        'condition': 'Condition',
        'preset_source': 'Preset Source',
        'probe_kind': 'Probe Kind',
        'probe_name': 'Probe Name',
        'oracle_prompt_file': 'Oracle Prompt File',
        'target_prompt_index': 'Target Prompt Index',
        'rollout_index': 'Rollout Index',
        'target_rollout_index': 'Target Rollout Index',
        'oracle_rollout_index': 'Oracle Rollout Index',
        'n_prompts': 'Prompt Count',
        'n_rows': 'Scored Rows',
        'n_target_prompts': 'Unique Target Prompts',
        'n_cache_files': 'Cache Files',
        'mean_score': 'Mean Score',
        'se_score': 'SE Across Prompts',
        'score': 'Score',
        'asr_0_2': 'ASR >= 0.2',
        'asr_0_2_se': 'ASR >= 0.2 SE Across Prompts',
        'asr_0_5': 'ASR >= 0.5',
        'asr_0_5_se': 'ASR >= 0.5 SE Across Prompts',
        'asr_0_8': 'ASR >= 0.8',
        'asr_0_8_se': 'ASR >= 0.8 SE Across Prompts',
        'asr_1': 'ASR = 1.0',
        'asr_1_se': 'ASR = 1.0 SE Across Prompts',
        'n_prompts_with_sd': 'Prompts With Within-Prompt SD',
        'sd_within_prompt_oracle_rollouts': 'Within-Prompt Std across Oracle Rollouts',
        'sd_within_prompt_target_rollouts': 'Within-Prompt Std across Target Rollouts',
        'mean_within_prompt_sd_oracle_rollouts': 'Within-Prompt Std across Oracle Rollouts',
        'mean_within_prompt_sd_target_rollouts': 'Within-Prompt Std across Target Rollouts',
        'mean_within_prompt_n': 'Mean Scored Rollouts Per Prompt',
        'cache_path_alias': 'Cache Path',
        'shared_cache_prefix': 'Shared Cache Prefix',
        'missing_cache_path_alias': 'Missing Cache Path',
        'compliance_leaf_preview': 'Compliance Leaf Preview',
        'oracle_rollout': 'Oracle Rollout',
        'target_rollout': 'Target Rollout',
        'target_rollout_response': 'Target Rollout Response',
        'oracle_response': 'Oracle Response',
        'oracle_response_preview': 'Oracle Response Preview',
        'target_prompt': 'Target Prompt Preview',
        'oracle_prompt': 'Oracle Prompt Preview',
    }
    return df.rename(columns={k: v for k, v in pretty.items() if k in df.columns})


def probe_order_map(details_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    order_map: dict[tuple[str, str], float] = {}
    sample = (
        details_df[details_df['probe_kind'].isin(['token_points', 'tokens'])]
        [['cache_path', 'probe_kind', 'probe_name']].dropna().drop_duplicates()
    )
    for _, row in sample.iterrows():
        probe_kind = row['probe_kind']
        probe_name = row['probe_name']
        key = (probe_kind, probe_name)
        if key in order_map:
            continue
        try:
            with open(row['cache_path'], 'r', encoding='utf-8') as f:
                payload = json.load(f)
            entries = payload.get('entries', []) if isinstance(payload, dict) else payload
            if not entries:
                continue
            first = entries[0]
            points = first.get('oracle_points', {}).get('token_points', {})
            if isinstance(points, dict) and probe_name in points:
                order_map[key] = float(points[probe_name])
        except Exception:
            continue
    return order_map


def apply_probe_sort(df: pd.DataFrame, probe_order: dict | None = None) -> pd.DataFrame:
    out = df.copy()
    if 'probe_kind' not in out.columns or 'probe_name' not in out.columns:
        return out
    if probe_order is None:
        probe_order = {}
    out['_probe_rank'] = out.apply(
        lambda r: probe_order.get((r.get('probe_kind'), r.get('probe_name')), 1e9), axis=1
    )
    out['_probe_name_sort'] = out['probe_name'].astype(str)
    sort_cols: list[str] = []
    if 'condition' in out.columns:
        out['_condition_rank'] = out['condition'].map(condition_rank)
        sort_cols.append('_condition_rank')
    if 'oracle_prompt_file' in out.columns:
        sort_cols.append('oracle_prompt_file')
    if 'probe_kind' in out.columns:
        sort_cols.append('probe_kind')
    sort_cols.extend(['_probe_rank', '_probe_name_sort'])
    out = out.sort_values(sort_cols)
    return out.drop(columns=[c for c in ['_probe_rank', '_probe_name_sort', '_condition_rank'] if c in out.columns])


def build_provenance(details: pd.DataFrame, path_aliaser: PathAliaser, probe_order: dict) -> pd.DataFrame:
    """Aggregate details into a per-condition/probe provenance table, ready for display."""
    prov = (
        details
        .groupby(['condition', 'preset_source', 'oracle_prompt_file', 'probe_kind', 'probe_name'], dropna=False)
        .agg(
            n_rows=('score', 'size'),
            n_target_prompts=('target_prompt_key', 'nunique'),
            n_cache_files=('cache_path', 'nunique'),
            mean_score=('score', 'mean'),
        )
        .reset_index()
    )
    prov = path_aliaser.add_alias_column(prov, 'oracle_prompt_file', 'oracle_prompt_file_alias')
    prov = apply_probe_sort(prov, probe_order)
    return apply_display_transforms(prov)


def plot_summary(summary: pd.DataFrame, manifest: dict, top_n: int = 30) -> None:
    plot_df = summary.dropna(subset=['mean_score']).copy()
    plot_df['label'] = (
        plot_df['condition'] + ' | '
        + plot_df['probe_name'].astype(str) + ' | '
        + plot_df['oracle_prompt_file'].fillna('')
    )
    plot_df = plot_df.sort_values('mean_score', ascending=False).head(top_n)

    ax = plot_df.plot.barh(x='label', y='mean_score', xerr='se_score', figsize=(11, 10), legend=False)
    sample_size = f"actual prompts={manifest.get('actual_target_prompts')} / expected={manifest.get('expected_target_prompts')}"
    ax.set_title(f'StrongReject summary ({sample_size})')
    ax.set_xlabel('StrongReject score (%)')
    ax.set_ylabel('')
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, pos: f"{v * 100:.0f}%"))
    ax.invert_yaxis()
    plt.tight_layout()
