"""Report generators for backtesting HTML reports."""

from datetime import datetime
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .styles import CSS_STYLESHEET
from .templates import (
    HTML_DOCUMENT_TEMPLATE,
    SIDEBAR_TEMPLATE,
    TOGGLE_SCRIPT,
    CS_METRICS_NAV,
    TS_METRICS_NAV,
)
from .builders import MetricsTableBuilder, FactorInfoBuilder, ParamsTableBuilder
from .plot_embedder import embed_plots_in_html

def _load_mathjax_script_tag() -> str:
    """
    Return an inline MathJax script tag (offline-safe).

    The report must be self-contained when opened without Internet access.
    """
    mathjax_path = (
        Path(__file__).resolve().parent.parent
        / "resources"
        / "mathjax"
        / "tex-chtml.js"
    )
    if not mathjax_path.exists():
        raise FileNotFoundError(f"MathJax bundle not found: {mathjax_path}")

    js = mathjax_path.read_text(encoding="utf-8")
    return f"<script>{js}</script>"

class BaseReportGenerator:
    """Base class for HTML report generation."""

    # Metric display names (shared between CS and TS)
    METRIC_DISPLAY_NAMES = {
        # L/S metrics
        'long_short_ret_annual': 'L/S Annualized Return',
        'long_short_ret_sharpe': 'L/S Sharpe',
        'long_short_ret_max_dd': 'L/S Max Drawdown',
        'long_short_ret_calmar': 'L/S Calmar',
        'long_short_ret_sortino': 'L/S Sortino',
        'long_short_ret_sharpe_per_turnover': 'L/S Sharpe/Turnover',
        'long_short_turnover_ratio': 'L/S Turnover',
        'long_short_turnover': 'L/S Turnover',
        'long_short_win_rate': 'L/S Win Rate',
        # Long metrics
        'long_ret_annual': 'Long Annualized Return',
        'long_excess_ret_annual': 'Long Annualized Excess Return',
        'long_ret_sharpe': 'Long Sharpe',
        'long_ret_max_dd': 'Long Max Drawdown',
        'long_ret_calmar': 'Long Calmar',
        'long_ret_sortino': 'Long Sortino',
        'long_ret_sharpe_per_turnover': 'Long Sharpe/Turnover',
        'long_turnover_ratio': 'Long Turnover',
        'long_turnover': 'Long Turnover',
        'long_win_rate': 'Long Win Rate',
        # Short metrics
        'short_ret_annual': 'Short Annualized Return',
        'short_excess_ret_annual': 'Short Annualized Excess Return',
        'short_ret_sharpe': 'Short Sharpe',
        'short_ret_max_dd': 'Short Max Drawdown',
        'short_ret_calmar': 'Short Calmar',
        'short_ret_sortino': 'Short Sortino',
        'short_ret_sharpe_per_turnover': 'Short Sharpe/Turnover',
        'short_turnover_ratio': 'Short Turnover',
        'short_turnover': 'Short Turnover',
        'short_win_rate': 'Short Win Rate',
        # Long passive metrics
        'passive_ret_annual': 'Long Passive Annualized Return',
        'passive_ret_sharpe': 'Long Passive Sharpe',
        'passive_ret_max_dd': 'Long Passive Max Drawdown',
        'passive_ret_calmar': 'Long Passive Calmar',
        'passive_ret_sortino': 'Long Passive Sortino',
        'passive_ret_sharpe_per_turnover': 'Long Passive Sharpe/Turnover',
        'passive_turnover_ratio': 'Long Passive Turnover',
        'passive_turnover': 'Long Passive Turnover',
        'passive_win_rate': 'Long Passive Win Rate',
        # Short passive metrics
        'short_passive_ret_annual': 'Short Passive Annualized Return',
        'short_passive_ret_sharpe': 'Short Passive Sharpe',
        'short_passive_ret_max_dd': 'Short Passive Max Drawdown',
        'short_passive_ret_calmar': 'Short Passive Calmar',
        'short_passive_ret_sortino': 'Short Passive Sortino',
        'short_passive_ret_sharpe_per_turnover': 'Short Passive Sharpe/Turnover',
        'short_passive_turnover_ratio': 'Short Passive Turnover',
        'short_passive_turnover': 'Short Passive Turnover',
        'short_passive_win_rate': 'Short Passive Win Rate',
        # Rank IC metrics
        'rank_ic': 'Rank IC',
        'rank_icir': 'Rank ICIR',
        'rank_icir_annual': 'Rank ICIR Annualized',
        'rank_ic_p_value': 'Rank IC p-value',
        'rank_ic_winratio': 'Rank IC Win Ratio',
        # IC metrics
        'ic': 'IC',
        'icir': 'ICIR',
        'icir_annual': 'ICIR Annualized',
        'ic_p_value': 'IC p-value',
        'ic_winratio': 'IC Win Ratio',
        # ACF metrics
        'acf_1': 'ACF 1',
        'acf_halflife': 'ACF Half-life',
    }

    # Percentage-formatted metrics
    PERCENTAGE_METRICS = {
        'long_short_ret_annual',
        'long_short_ret_max_dd', 'long_short_win_rate',
        'long_short_turnover_ratio', 'long_short_turnover',
        'long_ret_annual', 'long_excess_ret_annual', 'long_ret_max_dd', 'long_win_rate',
        'long_turnover_ratio', 'long_turnover',
        'short_ret_annual', 'short_excess_ret_annual', 'short_ret_max_dd', 'short_win_rate',
        'short_turnover_ratio', 'short_turnover',
        'passive_ret_annual', 'passive_ret_max_dd', 'passive_win_rate',
        'passive_turnover_ratio', 'passive_turnover',
        'short_passive_ret_annual', 'short_passive_ret_max_dd', 'short_passive_win_rate',
        'short_passive_turnover_ratio', 'short_passive_turnover',
        'rank_ic_winratio',
    }

    # Ratio-formatted metrics
    RATIO_METRICS = {
        'long_short_ret_sharpe', 'long_ret_sharpe', 'short_ret_sharpe', 'passive_ret_sharpe',
        'short_passive_ret_sharpe',
        'long_short_ret_calmar', 'long_ret_calmar', 'short_ret_calmar', 'passive_ret_calmar',
        'short_passive_ret_calmar',
        'long_short_ret_sortino', 'long_ret_sortino', 'short_ret_sortino', 'passive_ret_sortino',
        'short_passive_ret_sortino',
        'long_short_ret_sharpe_per_turnover', 'long_ret_sharpe_per_turnover',
        'short_ret_sharpe_per_turnover', 'passive_ret_sharpe_per_turnover',
        'short_passive_ret_sharpe_per_turnover',
        'rank_ic', 'rank_icir', 'rank_icir_annual',
        'ic', 'icir', 'icir_annual',
        'acf_1',
    }

    def __init__(
        self,
        result_dict: Dict,
        summary_df: Any,
        config: Dict
    ):
        """
        Initialize report generator.

        Args:
            result_dict: Backtest results dictionary
            summary_df: Summary DataFrame with metrics
            config: Configuration dictionary
        """
        self.result_dict = result_dict
        self.summary_df = summary_df
        self.config = config

    def _get_metric_display_name(self, name: str) -> str:
        """Get display name for a metric."""
        return self.METRIC_DISPLAY_NAMES.get(name, name.replace('_', ' ').title())

    def _format_metric(self, name: str, value: float) -> str:
        """Format a metric value for display."""
        try:
            v = float(value)
        except Exception:
            return "NaN"

        if not math.isfinite(v):
            return "NaN"

        if name in self.PERCENTAGE_METRICS:
            return f"{v * 100:.2f}%"
        elif name == 'acf_halflife':
            return f"{int(v)}"
        elif name in self.RATIO_METRICS:
            return f"{v:.4f}"
        elif name == 'rank_ic_p_value':
            return f"{v:.4f}"
        else:
            return f"{v:.4f}"

    def _build_metrics_table_builder(self) -> MetricsTableBuilder:
        """Create a MetricsTableBuilder instance."""
        return MetricsTableBuilder(
            self.summary_df,
            self.config['fees'],
            self._get_metric_display_name,
            self._format_metric
        )

    def _build_factor_info_html(self) -> str:
        """Build factor info HTML section."""
        builder = FactorInfoBuilder(self.config.get('factor_info_dict'))
        return builder.build()

    def _build_plot_nav_links(self, plot_info: List[Tuple[str, str, Any]]) -> str:
        """Build sidebar navigation links for plots."""
        plot_nav_links = ""
        for i, (plot_id, plot_title, _) in enumerate(plot_info):
            tree_char = "└─" if i == len(plot_info) - 1 else "├─"
            plot_nav_links += f'            <a href="#{plot_id}" class="sub-nav">'
            plot_nav_links += f'{tree_char} {plot_title}</a>\n'
        return plot_nav_links

    def _build_sidebar(
        self,
        plot_info: List[Tuple[str, str, Any]],
        metrics_nav: str
    ) -> str:
        """Build the sidebar HTML."""
        factor_info_nav = ('    <a href="#factor-info">Factor Information</a>\n'
                          if self.config.get('factor_info_dict') else "")
        plot_nav_links = self._build_plot_nav_links(plot_info)

        return SIDEBAR_TEMPLATE.format(
            factor_info_nav=factor_info_nav,
            metrics_nav_links=metrics_nav,
            plot_nav_links=plot_nav_links
        )

    def generate(self, output_dir: Path) -> str:
        """
        Generate the complete HTML report.

        Args:
            output_dir: Output directory for the report

        Returns:
            Path to generated HTML file
        """
        raise NotImplementedError("Subclasses must implement generate()")


class CSReportGenerator(BaseReportGenerator):
    """Cross-sectional backtest report generator."""

    # CS-specific metric groups
    RANK_IC_METRICS = [
        'rank_ic',
        'rank_icir',
        'rank_icir_annual',
        'rank_ic_p_value',
        'rank_ic_winratio',
    ]
    IC_METRICS = [
        'ic',
        'icir',
        'icir_annual',
        'ic_p_value',
        'ic_winratio',
    ]
    ACF_METRICS = ['acf_1', 'acf_halflife']
    LONGSHORT_METRICS = [
        'long_short_ret_annual', 'long_short_ret_sharpe',
        'long_short_ret_max_dd',
        'long_short_ret_calmar', 'long_short_ret_sortino', 'long_short_ret_sharpe_per_turnover',
        'long_short_turnover_ratio', 'long_short_win_rate'
    ]
    LONG_METRICS = [
        'long_ret_annual', 'long_excess_ret_annual', 'long_ret_sharpe', 'long_ret_max_dd',
        'long_ret_calmar', 'long_ret_sortino', 'long_ret_sharpe_per_turnover',
        'long_turnover_ratio', 'long_win_rate'
    ]
    SHORT_METRICS = [
        'short_ret_annual', 'short_excess_ret_annual', 'short_ret_sharpe', 'short_ret_max_dd',
        'short_ret_calmar', 'short_ret_sortino', 'short_ret_sharpe_per_turnover',
        'short_turnover_ratio', 'short_win_rate'
    ]
    LONG_PASSIVE_METRICS = [
        'passive_ret_annual', 'passive_ret_sharpe', 'passive_ret_max_dd',
        'passive_ret_calmar', 'passive_ret_sortino', 'passive_ret_sharpe_per_turnover',
        'passive_turnover_ratio', 'passive_turnover', 'passive_win_rate'
    ]
    SHORT_PASSIVE_METRICS = [
        'short_passive_ret_annual', 'short_passive_ret_sharpe', 'short_passive_ret_max_dd',
        'short_passive_ret_calmar', 'short_passive_ret_sortino',
        'short_passive_ret_sharpe_per_turnover',
        'short_passive_turnover_ratio', 'short_passive_turnover', 'short_passive_win_rate'
    ]

    def _build_performance_metrics_html(self) -> str:
        """Build HTML for performance metrics section."""
        builder = self._build_metrics_table_builder()

        metrics_html = '<div id="performance-metrics" class="metrics-section">\n'

        metrics_html += f'''
    <div id="alpha-metrics">
        <h3>Alpha Metrics</h3>
        {builder.build_table(self.RANK_IC_METRICS, include_cost_column=False, single_row=True)}
        {builder.build_table(self.IC_METRICS, include_cost_column=False, single_row=True)}
        {builder.build_table(self.ACF_METRICS, include_cost_column=False, single_row=True)}
    </div>

    <h3 id="by-cost-metrics">By Cost Metrics</h3>

    <div id="longshort-metrics">
        <h3>Long/Short Metrics</h3>
        {builder.build_table(self.LONGSHORT_METRICS)}
    </div>

    <div class="metrics-row-50-50">
        <div id="long-metrics">
            <h3>Long Only Metrics</h3>
            {builder.build_table(self.LONG_METRICS)}
        </div>
        <div id="short-metrics">
            <h3>Short Only Metrics</h3>
            {builder.build_table(self.SHORT_METRICS)}
        </div>
    </div>

    <div id="passive-metrics">
        <h3>Long Passive Investment Metrics</h3>
        {builder.build_table(self.LONG_PASSIVE_METRICS, include_cost_column=False, single_row=True)}
    </div>

    <div id="short-passive-metrics">
        <h3>Short Passive Investment Metrics</h3>
        {builder.build_table(self.SHORT_PASSIVE_METRICS, include_cost_column=False, single_row=True)}
    </div>
</div>
'''
        return metrics_html

    def _build_overview_html(self) -> str:
        """Build overview table HTML."""
        cfg = self.config
        factor_display_name = (cfg.get('factor_info_dict', {}).get('factor_name', cfg['alpha_name'])
                              if cfg.get('factor_info_dict') else cfg['alpha_name'])
        backtest_mode_display = {
            "long/short_layers": "Long/Short Layers",
            "long/short_normalization": "Long/Short Normalization"
        }.get(cfg['backtest_mode'], cfg['backtest_mode'])
        annual_days_display = str(cfg.get('annual_days')) if cfg.get('annual_days') else "Auto"
        fees_str = ', '.join([f"{fee * 1e4:.1f}" for fee in cfg['fees']])

        return f"""<table class="overview-table">
                <thead>
                    <tr><th>Parameter</th><th>Value</th></tr>
                </thead>
                <tbody>
                    <tr><td><strong>Factor</strong></td><td>{factor_display_name}</td></tr>
                    <tr><td><strong>Symbols Count</strong></td><td>{cfg['n_symbols']}</td></tr>
                    <tr><td><strong>Date Range</strong></td><td>{cfg['start_dt']} to {cfg['end_dt']}</td></tr>
                    <tr><td><strong>Datetime Points</strong></td><td>{cfg['n_datetimes']}</td></tr>
                    <tr><td><strong>Frequency</strong></td><td>{cfg['freq']}</td></tr>
                    <tr><td><strong>Layers</strong></td><td>{cfg['layers_use']}</td></tr>
                    <tr><td><strong>Backtest Mode</strong></td><td>{backtest_mode_display}</td></tr>
                    <tr><td><strong>Lag</strong></td><td>{cfg['lag']}</td></tr>
                    <tr><td><strong>Annual Days</strong></td><td>{annual_days_display}</td></tr>
                    <tr><td><strong>Fees</strong></td><td>{fees_str} bp</td></tr>
                </tbody>
            </table>"""

    def generate(
        self,
        plot_info: List[Tuple[str, str, Any]],
        output_dir: Path,
        documentation_html: str = ""
    ) -> str:
        """
        Generate the CS backtest HTML report.

        Args:
            plot_info: List of (plot_id, plot_title, plot_object) tuples
            output_dir: Output directory
            documentation_html: Optional documentation HTML

        Returns:
            Path to generated HTML file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        html_path = output_dir / "backtesting_report.html"

        plot_html_blocks = embed_plots_in_html(plot_info, plots_dir)
        sidebar_html = self._build_sidebar(plot_info, CS_METRICS_NAV)
        overview_html = self._build_overview_html()
        params_html = ParamsTableBuilder.build_cs_params(
            self.config['backtest_mode'],
            self.config['backtest_params']
        )
        factor_info_html = self._build_factor_info_html()
        performance_html = self._build_performance_metrics_html()

        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        content_html = f"""
    <div class="overview-params-row">
        <div>
            <h2 id="overview">Overview</h2>
            {overview_html}
        </div>
        <div>
            <h2 id="parameters">Backtest Parameters</h2>
            {params_html}
        </div>
    </div>

    {factor_info_html}

    <h2 id="performance">Performance Metrics</h2>
    {performance_html}

    <h2 id="plot">Performance Plots</h2>
    <div class="plot-grid">
{''.join(plot_html_blocks)}
    </div>

    <h2 id="documentation">Documentation</h2>
    {documentation_html if documentation_html else '<div class="documentation-content"><p>Documentation not available.</p></div>'}
"""

        html = HTML_DOCUMENT_TEMPLATE.format(
            title="Cross-Sectional Backtest Report",
            css_stylesheet=CSS_STYLESHEET,
            mathjax_script=_load_mathjax_script_tag(),
            sidebar_html=sidebar_html,
            gen_time=gen_time,
            content_html=content_html,
            toggle_script=TOGGLE_SCRIPT
        )

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        return str(html_path)


class TSReportGenerator(BaseReportGenerator):
    """Time-series backtest report generator."""

    # TS-specific metric groups
    RANK_IC_METRICS = [
        'rank_ic',
        'rank_icir',
        'rank_icir_annual',
        'rank_ic_p_value',
        'rank_ic_winratio',
    ]
    IC_METRICS = [
        'ic',
        'icir',
        'icir_annual',
        'ic_p_value',
        'ic_winratio',
    ]
    ACF_METRICS = ['acf_1', 'acf_halflife']
    LONGSHORT_METRICS = [
        'long_short_ret_annual', 'long_short_ret_sharpe',
        'long_short_ret_max_dd',
        'long_short_ret_calmar', 'long_short_ret_sortino', 'long_short_ret_sharpe_per_turnover',
        'long_short_turnover', 'long_short_win_rate'
    ]
    LONG_METRICS = [
        'long_ret_annual', 'long_excess_ret_annual', 'long_ret_sharpe', 'long_ret_max_dd',
        'long_ret_calmar', 'long_ret_sortino', 'long_ret_sharpe_per_turnover',
        'long_turnover', 'long_win_rate'
    ]
    SHORT_METRICS = [
        'short_ret_annual', 'short_excess_ret_annual', 'short_ret_sharpe', 'short_ret_max_dd',
        'short_ret_calmar', 'short_ret_sortino', 'short_ret_sharpe_per_turnover',
        'short_turnover', 'short_win_rate'
    ]
    LONG_PASSIVE_METRICS = [
        'passive_ret_annual', 'passive_ret_sharpe', 'passive_ret_max_dd',
        'passive_ret_calmar', 'passive_ret_sortino', 'passive_ret_sharpe_per_turnover',
        'passive_turnover_ratio', 'passive_turnover', 'passive_win_rate'
    ]
    SHORT_PASSIVE_METRICS = [
        'short_passive_ret_annual', 'short_passive_ret_sharpe', 'short_passive_ret_max_dd',
        'short_passive_ret_calmar', 'short_passive_ret_sortino',
        'short_passive_ret_sharpe_per_turnover',
        'short_passive_turnover_ratio', 'short_passive_turnover', 'short_passive_win_rate'
    ]

    def _build_performance_metrics_html(self) -> str:
        """Build HTML for performance metrics section."""
        builder = self._build_metrics_table_builder()

        metrics_html = '<div id="performance-metrics" class="metrics-section">\n'

        metrics_html += f'''
    <div id="alpha-metrics">
        <h3>Alpha Metrics</h3>
        {builder.build_table(self.RANK_IC_METRICS, include_cost_column=False, single_row=True)}
        {builder.build_table(self.IC_METRICS, include_cost_column=False, single_row=True)}
        {builder.build_table(self.ACF_METRICS, include_cost_column=False, single_row=True)}
    </div>

    <h3 id="by-cost-metrics">By Cost Metrics</h3>

    <div id="longshort-metrics">
        <h3>Long/Short Metrics</h3>
        {builder.build_table(self.LONGSHORT_METRICS)}
    </div>

    <div class="metrics-row-50-50">
        <div id="long-metrics">
            <h3>Long Only Metrics</h3>
            {builder.build_table(self.LONG_METRICS)}
        </div>
        <div id="short-metrics">
            <h3>Short Only Metrics</h3>
            {builder.build_table(self.SHORT_METRICS)}
        </div>
    </div>

    <div class="metrics-row-50-50">
        <div id="passive-metrics">
            <h3>Long Passive Investment Metrics</h3>
            {builder.build_table(self.LONG_PASSIVE_METRICS, include_cost_column=False, single_row=True)}
        </div>
        <div id="short-passive-metrics">
            <h3>Short Passive Investment Metrics</h3>
            {builder.build_table(self.SHORT_PASSIVE_METRICS, include_cost_column=False, single_row=True)}
        </div>
    </div>
</div>
'''
        return metrics_html

    def _build_overview_html(self) -> str:
        """Build overview table HTML."""
        cfg = self.config
        factor_display_name = (cfg.get('factor_info_dict', {}).get('factor_name', cfg['alpha_name'])
                              if cfg.get('factor_info_dict') else cfg['alpha_name'])
        backtest_mode_display = {
            "long/short_threshold": "Long/Short Threshold",
            "gradual_long/short_threshold": "Gradual Long/Short Threshold",
            "long/short_normalization": "Long/Short Normalization",
            "weights": "Direct Weights"
        }.get(cfg['backtest_mode'], cfg['backtest_mode'])
        annual_days_display = str(cfg.get('annual_days')) if cfg.get('annual_days') else "Auto"
        fees_str = ', '.join([f"{fee * 1e4:.1f}" for fee in cfg['fees']])

        return f"""<table class="overview-table">
                <thead>
                    <tr><th>Parameter</th><th>Value</th></tr>
                </thead>
                <tbody>
                    <tr><td><strong>Factor</strong></td><td>{factor_display_name}</td></tr>
                    <tr><td><strong>Symbol</strong></td><td>{cfg.get('symbol', 'N/A')}</td></tr>
                    <tr><td><strong>Date Range</strong></td><td>{cfg['start_dt']} to {cfg['end_dt']}</td></tr>
                    <tr><td><strong>Datetime Points</strong></td><td>{cfg['n_datetimes']}</td></tr>
                    <tr><td><strong>Frequency</strong></td><td>{cfg['freq']}</td></tr>
                    <tr><td><strong>Backtest Mode</strong></td><td>{backtest_mode_display}</td></tr>
                    <tr><td><strong>Lag</strong></td><td>{cfg['lag']}</td></tr>
                    <tr><td><strong>Annual Days</strong></td><td>{annual_days_display}</td></tr>
                    <tr><td><strong>Fees</strong></td><td>{fees_str} bp</td></tr>
                </tbody>
            </table>"""

    def generate(
        self,
        plot_info: List[Tuple[str, str, Any]],
        output_dir: Path,
        thresholds: Dict = None,
        documentation_html: str = ""
    ) -> str:
        """
        Generate the TS backtest HTML report.

        Args:
            plot_info: List of (plot_id, plot_title, plot_object) tuples
            output_dir: Output directory
            thresholds: Dictionary of threshold values for display
            documentation_html: Optional documentation HTML

        Returns:
            Path to generated HTML file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        html_path = output_dir / "backtesting_report.html"

        plot_html_blocks = embed_plots_in_html(plot_info, plots_dir)
        sidebar_html = self._build_sidebar(plot_info, TS_METRICS_NAV)
        overview_html = self._build_overview_html()
        params_html = ParamsTableBuilder.build_ts_params(
            self.config['backtest_mode'],
            self.config['backtest_params'],
            thresholds or {}
        )
        factor_info_html = self._build_factor_info_html()
        performance_html = self._build_performance_metrics_html()

        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        content_html = f"""
    <div class="overview-params-row">
        <div>
            <h2 id="overview">Overview</h2>
            {overview_html}
        </div>
        <div>
            <h2 id="parameters">Backtest Parameters</h2>
            {params_html}
        </div>
    </div>

    {factor_info_html}

    <h2 id="performance">Performance Metrics</h2>
    {performance_html}

    <h2 id="plot">Performance Plots</h2>
    <div class="plot-grid">
{''.join(plot_html_blocks)}
    </div>

    <h2 id="documentation">Documentation</h2>
    {documentation_html if documentation_html else '<div class="documentation-content"><p>Documentation not available.</p></div>'}
"""

        html = HTML_DOCUMENT_TEMPLATE.format(
            title="Time-Series Backtest Report",
            css_stylesheet=CSS_STYLESHEET,
            mathjax_script=_load_mathjax_script_tag(),
            sidebar_html=sidebar_html,
            gen_time=gen_time,
            content_html=content_html,
            toggle_script=TOGGLE_SCRIPT
        )

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        return str(html_path)
