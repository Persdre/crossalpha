"""Plot embedding utilities for HTML reports."""

import os
from pathlib import Path
from typing import List, Tuple, Any

import holoviews as hv


def embed_plots_in_html(
    plot_info: List[Tuple[str, str, Any]],
    plots_dir: Path
) -> List[str]:
    """
    Embed HoloViews plots as inline HTML blocks.

    Args:
        plot_info: List of tuples (plot_id, plot_title, plot_object)
        plots_dir: Directory for temporary plot files

    Returns:
        List of HTML strings, one per plot
    """
    from bokeh.resources import INLINE

    plot_html_blocks = []

    for plot_id, plot_title, plot_obj in plot_info:
        temp_plot_path = plots_dir / f"{plot_id}_temp.html"
        hv.save(plot_obj, filename=str(temp_plot_path), backend='bokeh', resources=INLINE)

        with open(temp_plot_path, 'r', encoding='utf-8') as f:
            plot_full_html = f.read()

        temp_plot_path.unlink()

        plot_html = f'''
<div id="{plot_id}" class="plot-container">
    <h3 class="plot-title">{plot_title}</h3>
    <div class="bokeh-plot-wrapper">
        {plot_full_html}
    </div>
</div>'''

        plot_html_blocks.append(plot_html)

    try:
        os.rmdir(plots_dir)
    except OSError:
        pass

    return plot_html_blocks
