"""HTML component builders for backtest reports."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..core import detect_core


class MetricsTableBuilder:
    """Build performance metrics HTML tables."""

    def __init__(
        self,
        summary_df: Any,
        fees: List[float],
        get_display_name_func,
        format_metric_func
    ):
        """
        Initialize the metrics table builder.

        Args:
            summary_df: DataFrame with metrics (index = fee labels, columns = metric names)
            fees: List of fee values
            get_display_name_func: Function to get display name for metric
            format_metric_func: Function to format metric value
        """
        self.summary_df = summary_df
        self.fees = fees
        self.get_display_name = get_display_name_func
        self.format_metric = format_metric_func

        core = detect_core(summary_df)
        if core == "pandas":
            self._rows_by_cost = {
                str(k): v
                for k, v in summary_df.to_dict(orient="index").items()
            }
            self._metric_columns = [str(c) for c in summary_df.columns]
        else:
            try:
                import polars as pl  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("polars is required for polars summary_df") from exc

            if not isinstance(summary_df, pl.DataFrame):
                raise TypeError(f"Unsupported summary_df type: {type(summary_df)}")
            if "cost" not in summary_df.columns:
                raise ValueError("polars summary_df must include a 'cost' column")

            self._rows_by_cost = {
                str(row["cost"]): row for row in summary_df.to_dicts()
            }
            self._metric_columns = [c for c in summary_df.columns if c != "cost"]

    def build_table(
        self,
        metrics_list: List[str],
        include_cost_column: bool = True,
        single_row: bool = False
    ) -> str:
        """
        Build a single metrics table HTML for a given list of metrics.

        Args:
            metrics_list: List of metric names to include
            include_cost_column: Whether to include cost column
            single_row: Whether to show only first fee row

        Returns:
            HTML string for the metrics table
        """
        available_metrics = [m for m in metrics_list if m in self._metric_columns]
        if not available_metrics:
            return ""

        table_html = """<table class="metrics-table-consolidated">
        <thead>
            <tr>
"""
        if include_cost_column:
            table_html += '                <th class="cost-column">Cost</th>\n'

        for metric_name in available_metrics:
            display_name = self.get_display_name(metric_name)
            table_html += f"                <th>{display_name}</th>\n"

        table_html += """            </tr>
        </thead>
        <tbody>
"""

        fees_to_use = [self.fees[0]] if single_row else self.fees
        for fee in fees_to_use:
            fee_bp = round(fee * 1e4, 1)
            fee_col = f"{fee_bp} bp"

            metrics = self._rows_by_cost.get(fee_col)
            if metrics is not None:
                table_html += "            <tr>\n"
                if include_cost_column:
                    table_html += f'                <td class="cost-column">{fee_bp} bp</td>\n'

                for metric_name in available_metrics:
                    metric_value = metrics.get(metric_name)
                    formatted_value = self.format_metric(metric_name, metric_value)
                    table_html += f"                <td>{formatted_value}</td>\n"

                table_html += "            </tr>\n"

        table_html += """        </tbody>
    </table>"""
        return table_html


class FactorInfoBuilder:
    """Build factor information HTML section."""

    def __init__(self, factor_info_dict: Optional[Dict]):
        """
        Initialize the factor info builder.

        Args:
            factor_info_dict: Dictionary with factor metadata
        """
        self.factor_info_dict = factor_info_dict

    @staticmethod
    def _sanitize_latex(latex_str: str) -> str:
        latex = latex_str.strip()
        latex = latex.replace(r"\ ?", "?")
        latex = latex.replace(r"\ :", ":")
        # Remove literal \n (LaTeX newline commands)
        latex = latex.replace(r"\n", "")
        # Remove LaTeX bracket delimiters if they exist in the string
        latex = latex.replace(r"\[", "").replace(r"\]", "")
        return latex

    def _latex_block(self, latex_str: str) -> str:
        latex = self._sanitize_latex(latex_str)
        # Use proper LaTeX display math delimiters for MathJax
        return f"\\[{latex}\\]"

    def _latex_inline(self, latex_str: str) -> str:
        latex = self._sanitize_latex(latex_str)
        return f"\\({latex}\\)"

    def _extract_formulas_from_latex(self, latex_str: str) -> Dict[str, str]:
        """Extract individual formula equations from the main LaTeX block.
        
        Returns:
            Dictionary mapping variable names to their complete LaTeX formulas
        """
        formulas = {}
        
        # The LaTeX format: r_{1,t} &= ...,\\\nREV_{20,t} &= ...,\\\n...
        # Split by pattern: ,\\\n (comma followed by newline in LaTeX)
        equations = re.split(r',\\\\\\n', latex_str)
        
        for equation in equations:
            equation = equation.strip()
            if not equation:
                continue
                
            # Match: var_{...} &= formula OR var_x &= formula
            # Handles: r_{1,t}, REV_{20,t}, alpha_t, etc.
            match = re.search(r'([a-zA-Z_\\]+(?:_{[^}]+}|_[a-z]))\s*&?=\s*(.+?)$', equation, re.DOTALL)
            
            if match:
                var_name = match.group(1).strip()
                formula_expr = match.group(2).strip()
                
                # Clean variable name: remove LaTeX commands, keep subscripts
                var_clean = re.sub(r'\\operatorname\{([^}]+)\}', r'\1', var_name)
                var_clean = var_clean.replace('\\', '')
                # Remove leading 'n' artifact from \n
                var_clean = re.sub(r'^n([a-z]_)', r'\1', var_clean)
                
                # Store formula
                formulas[var_clean] = f'${formula_expr}$'
        
        return formulas
    
    def _inject_formulas_into_description(self, description: str, formulas_dict: Dict[str, str]) -> str:
        """Replace plain-text formulas with clean LaTeX and convert remaining math expressions."""
        text = description
        
        # Use placeholders to protect injected LaTeX from Step 2
        placeholders = {}
        placeholder_idx = [0]
        
        def make_placeholder(content):
            ph = f'__LATEX_FORMULA_{placeholder_idx[0]}__'
            placeholders[ph] = content
            placeholder_idx[0] += 1
            return ph
        
        # Step 1: Replace formulas we extracted from the main LaTeX block
        for var_name, latex_formula in formulas_dict.items():
            var_escaped = re.escape(var_name)
            
            # Match the entire formula including any plain-text representation
            # Capture everything from VAR = until the next sentence (starts with capital letter after period)
            
            # Pattern 1: With colon prefix (e.g., "reversal: VAR = formula.")
            pattern1 = rf':\s*{var_escaped}\s*=\s*.+?(?=\.\s+[A-Z])'
            # Pattern 2: Without colon
            pattern2 = rf'\b{var_escaped}\s*=\s*.+?(?=\.\s+[A-Z]|\s+\(included)'
            
            def replacer(m):
                # Check if match starts with colon
                matched_text = m.group(0)
                if matched_text.startswith(':'):
                    replacement = f': ${var_name}$={latex_formula}'
                else:
                    replacement = f'${var_name}$={latex_formula}'
                return make_placeholder(replacement)
            
            # Try pattern 1 first (with colon), then pattern 2
            text = re.sub(pattern1, replacer, text)
            text = re.sub(pattern2, replacer, text)
        
        # Step 2: Convert remaining mathematical notation to LaTeX
        # (Placeholders won't be touched)
        
        # Convert variables with subscripts
        text = re.sub(r'\b([A-Za-z]+)_{([^}]+)}', r'$\1_{\2}$', text)  # With braces: r_{1,t}
        text = re.sub(r'\b([A-Z]+)_([a-z0-9t-]+)\b', r'$\1_{\2}$', text)  # Uppercase: S_t, REV_20
        text = re.sub(r'\b([a-z]{2,})_([a-z0-9t-]+)\b', r'$\1_{\2}$', text)  # Lowercase multi-char
        text = re.sub(r'\b([a-z])_([a-z0-9t])\b', r'$\1_{\2}$', text)  # Single letter
        
        # Convert function calls
        text = re.sub(r'\b(z|rank|ln)\(([^)]+)\)', r'$\1(\2)$', text)
        
        # Clean up nested dollars
        text = re.sub(r'\$([a-zA-Z]+)\(\$([^$]+)\$\)\$', r'$\1(\2)$', text)
        text = re.sub(r'\(\$([^$)]+)\$\)', r'($\1$)', text)
        text = re.sub(r'\$\s*\$', '', text)
        text = re.sub(r'\$+', '$', text)
        
        # Step 3: Restore placeholders
        for ph, content in placeholders.items():
            text = text.replace(ph, content)
        
        return text
    

    def build(self) -> str:
        """
        Build HTML for factor information section.

        Returns:
            HTML string for factor info section, empty if no factor_info_dict
        """
        if not self.factor_info_dict:
            return ""

        factor_name = self.factor_info_dict.get("factor_name", "")
        description = self.factor_info_dict.get("description", "")
        formulas = self.factor_info_dict.get("formulas", [])

        html = '<div id="factor-info" class="factor-info-section">\n'
        html += '    <h2>Factor Information</h2>\n'
        html += '    <div class="factor-info-content">\n'

        if factor_name:
            html += f'        <h3 class="factor-name">{factor_name}</h3>\n'

        if description:
            desc_html = description.replace('\n', '<br>\n')
            html += f'        <p class="factor-description">{desc_html}</p>\n'

        if formulas:
            for i, formula in enumerate(formulas):
                formula_latex = formula.get("latex", "")
                formula_desc = formula.get("description", "")
                formula_vars = formula.get("variables", {})

                html += '        <div class="factor-formula-block">\n'

                if len(formulas) > 1:
                    html += f'            <h4>Formula {i + 1}</h4>\n'
                else:
                    html += '            <h4>Formula</h4>\n'

                if formula_latex:
                    html += '            <div class="formula-latex">\n'
                    html += f'                {self._latex_block(formula_latex)}\n'
                    html += '            </div>\n'

                if formula_desc:
                    html += '            <div class="formula-description">\n'
                    html += '                <h5>Calculation Steps</h5>\n'
                    
                    # Extract formulas from the LaTeX block if available
                    formulas_dict = {}
                    if formula_latex:
                        formulas_dict = self._extract_formulas_from_latex(formula_latex)
                    
                    # Inject extracted formulas and convert remaining math to LaTeX
                    formula_desc = self._inject_formulas_into_description(formula_desc, formulas_dict)
                    
                    desc_lines = formula_desc.split('\n')
                    html += '                <ol class="calculation-steps">\n'
                    for line in desc_lines:
                        line = line.strip()
                        if line:
                            line = re.sub(r'^\d+\.\s*', '', line)
                            if line:
                                html += f'                    <li>{line}</li>\n'
                    html += '                </ol>\n'
                    html += '            </div>\n'

                if formula_vars:
                    html += '            <div class="formula-variables">\n'
                    html += '                <h5>Variables</h5>\n'
                    html += '                <table class="overview-table">\n'
                    html += '                    <thead>\n'
                    html += '                        <tr>'
                    html += '<th>Variable</th><th>Description</th>'
                    html += '</tr>\n'
                    html += '                    </thead>\n'
                    html += '                    <tbody>\n'
                    for var_name, var_desc in formula_vars.items():
                        html += '                        <tr>'
                        html += f'<td><span class="math-var">{self._latex_inline(var_name)}</span></td>'
                        html += f'<td>{var_desc}</td>'
                        html += '</tr>\n'
                    html += '                    </tbody>\n'
                    html += '                </table>\n'
                    html += '            </div>\n'

                html += '        </div>\n'

        html += '    </div>\n'
        html += '</div>\n'

        return html


class ParamsTableBuilder:
    """Build backtest parameters HTML table."""

    @staticmethod
    def build_cs_params(backtest_mode: str, backtest_params: Dict) -> str:
        """
        Build HTML for CS backtest parameters section.

        Args:
            backtest_mode: Backtest mode string
            backtest_params: Dictionary of backtest parameters

        Returns:
            HTML string for parameters table
        """
        params_html = '<table class="overview-table">\n'
        params_html += '    <thead>\n'
        params_html += '        <tr><th>Parameter</th><th>Value</th></tr>\n'
        params_html += '    </thead>\n'
        params_html += '    <tbody>\n'

        if backtest_mode == "long/short_layers":
            long_indices = backtest_params.get("long_layer_index", [])
            short_indices = backtest_params.get("short_layer_index", [])
            long_layers_str = ', '.join(map(str, long_indices))
            short_layers_str = ', '.join(map(str, short_indices))

            params_html += f'        <tr><td><strong>Long Layers Index</strong></td>'
            params_html += f'<td>{long_layers_str}</td></tr>\n'
            params_html += f'        <tr><td><strong>Short Layers Index</strong></td>'
            params_html += f'<td>{short_layers_str}</td></tr>\n'

        elif backtest_mode == "long/short_normalization":
            norm_method = backtest_params.get('normalization_method', '')
            params_html += f'        <tr><td><strong>Normalization Method</strong></td>'
            params_html += f'<td>{norm_method}</td></tr>\n'

            if 'winsorize_method' in backtest_params:
                winsorize_method = backtest_params.get('winsorize_method', '')
                params_html += f'        <tr><td><strong>Winsorize Method</strong></td>'
                params_html += f'<td>{winsorize_method}</td></tr>\n'

            if 'winsorize_n' in backtest_params:
                winsorize_n = backtest_params.get('winsorize_n', '')
                params_html += f'        <tr><td><strong>Winsorize N</strong></td>'
                params_html += f'<td>{winsorize_n}</td></tr>\n'

        params_html += '    </tbody>\n'
        params_html += '</table>\n'

        return params_html

    @staticmethod
    def build_ts_params(
        backtest_mode: str,
        backtest_params: Dict,
        thresholds: Dict
    ) -> str:
        """
        Build HTML for TS backtest parameters section.

        Args:
            backtest_mode: Backtest mode string
            backtest_params: Dictionary of backtest parameters
            thresholds: Dictionary of threshold values

        Returns:
            HTML string for parameters table
        """
        params_html = '<table class="overview-table">\n'
        params_html += '    <thead>\n'
        params_html += '        <tr><th>Parameter</th><th>Value</th></tr>\n'
        params_html += '    </thead>\n'
        params_html += '    <tbody>\n'

        if backtest_mode in ["long/short_threshold", "gradual_long/short_threshold"]:
            if 'long_threshold' in thresholds:
                params_html += f'        <tr><td><strong>Long Threshold</strong></td>'
                params_html += f'<td>{thresholds["long_threshold"]:.4f}</td></tr>\n'
            if 'short_threshold' in thresholds:
                params_html += f'        <tr><td><strong>Short Threshold</strong></td>'
                params_html += f'<td>{thresholds["short_threshold"]:.4f}</td></tr>\n'

            long_quantile = backtest_params.get("long_quantile", 0.7)
            short_quantile = backtest_params.get("short_quantile", 0.3)
            params_html += f'        <tr><td><strong>Long Quantile</strong></td>'
            params_html += f'<td>{long_quantile}</td></tr>\n'
            params_html += f'        <tr><td><strong>Short Quantile</strong></td>'
            params_html += f'<td>{short_quantile}</td></tr>\n'

        elif backtest_mode == "long/short_normalization":
            norm_method = backtest_params.get('normalization_method', '')
            params_html += f'        <tr><td><strong>Normalization Method</strong></td>'
            params_html += f'<td>{norm_method}</td></tr>\n'

            window = backtest_params.get('window', 'N/A')
            params_html += f'        <tr><td><strong>Window Size</strong></td>'
            params_html += f'<td>{window}</td></tr>\n'

            if 'winsorize_method' in backtest_params:
                winsorize_method = backtest_params.get('winsorize_method', '')
                params_html += f'        <tr><td><strong>Winsorize Method</strong></td>'
                params_html += f'<td>{winsorize_method}</td></tr>\n'

            if 'winsorize_n' in backtest_params:
                winsorize_n = backtest_params.get('winsorize_n', '')
                params_html += f'        <tr><td><strong>Winsorize N</strong></td>'
                params_html += f'<td>{winsorize_n}</td></tr>\n'

        elif backtest_mode == "weights":
            params_html += '        <tr><td><strong>Mode</strong></td>'
            params_html += '<td>Direct position weights from factor</td></tr>\n'

        params_html += '    </tbody>\n'
        params_html += '</table>\n'

        return params_html
