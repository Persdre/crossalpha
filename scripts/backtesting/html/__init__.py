"""HTML report generation modules for backtesting."""

from .styles import CSS_STYLESHEET
from .templates import (
    HTML_DOCUMENT_TEMPLATE,
    SIDEBAR_TEMPLATE,
    TOGGLE_SCRIPT,
)
from .builders import (
    MetricsTableBuilder,
    FactorInfoBuilder,
    ParamsTableBuilder,
)
from .plot_embedder import embed_plots_in_html
from .report_generator import CSReportGenerator, TSReportGenerator

__all__ = [
    'CSS_STYLESHEET',
    'HTML_DOCUMENT_TEMPLATE',
    'SIDEBAR_TEMPLATE',
    'TOGGLE_SCRIPT',
    'MetricsTableBuilder',
    'FactorInfoBuilder',
    'ParamsTableBuilder',
    'embed_plots_in_html',
    'CSReportGenerator',
    'TSReportGenerator',
]
