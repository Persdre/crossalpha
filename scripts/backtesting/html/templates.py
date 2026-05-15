"""HTML templates for backtest reports."""

HTML_DOCUMENT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{title}</title>

<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true
  }},
  options: {{
    enableMenu: false,
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  }}
}};
</script>
{mathjax_script}

<style>
{css_stylesheet}
</style>
</head>
<body>

{sidebar_html}

<span id="toggle-btn" onclick="toggleNav()">≡</span>

<div class="main">
    <h1>{title}</h1>
    <p style="color: #7f8c8d; margin-bottom: 30px;"><strong>Generated:</strong> {gen_time}</p>

    {content_html}
</div>

{toggle_script}
</body>
</html>"""

SIDEBAR_TEMPLATE = """<div class="sidebar" id="sidebar">
    <a href="#overview">Overview</a>
{factor_info_nav}    <a href="#parameters">Backtest Parameters</a>
    <a href="#performance">Performance Metrics</a>
{metrics_nav_links}    <a href="#plot">Performance Plots</a>
{plot_nav_links}    <a href="#documentation">Documentation</a>
</div>"""

TOGGLE_SCRIPT = """
<script>
function toggleNav() {
    var sb = document.getElementById('sidebar');
    var main = document.querySelector('.main');
    var btn = document.getElementById('toggle-btn');

    if (sb.style.width === "0px" || sb.style.width === '') {
        sb.style.width = "240px";
        main.style.marginLeft = "250px";
        btn.style.left = "250px";
    } else {
        sb.style.width = "0px";
        main.style.marginLeft = "0px";
        btn.style.left = "15px";
    }
    
    // Trigger Bokeh plot resize after sidebar toggle
    setTimeout(function() {
        resizeBokehPlots();
    }, 350);
}

// Make Bokeh plots responsive by scaling them to fit containers
function resizeBokehPlots() {
    try {
        // Find all plot containers
        var containers = document.querySelectorAll('.plot-container');
        
        containers.forEach(function(container) {
            var wrapper = container.querySelector('.bokeh-plot-wrapper');
            var bkRoot = wrapper ? wrapper.querySelector('.bk-root') : null;
            
            if (bkRoot) {
                // Get the natural dimensions of the Bokeh plot
                var naturalWidth = 380;  // Width set in backtest_utils.py
                var naturalHeight = 320; // Height set in backtest_utils.py
                var toolbarWidth = 40;   // Approximate toolbar width
                
                // Get the available container width (minus padding)
                var containerWidth = container.offsetWidth - 40; // 20px padding on each side
                
                // Calculate scale factor to fit plot + toolbar in container
                var totalNaturalWidth = naturalWidth + toolbarWidth;
                var scale = Math.min(containerWidth / totalNaturalWidth, 1);
                
                // Apply transform with proper centering
                bkRoot.style.transform = 'scale(' + scale + ')';
                bkRoot.style.transformOrigin = 'top center';
                
                // Adjust wrapper height to match scaled plot height
                wrapper.style.height = (naturalHeight * scale) + 'px';
                
                // Adjust container min-height to accommodate plot
                container.style.minHeight = (naturalHeight * scale + 80) + 'px'; // +80 for title and padding
            }
        });
    } catch(e) {
        console.error('Error resizing plots:', e);
    }
}

// Resize plots on window resize
window.addEventListener('resize', function() {
    clearTimeout(window.resizeTimer);
    window.resizeTimer = setTimeout(function() {
        resizeBokehPlots();
    }, 250);
});

// Initialize responsive behavior when page loads
window.addEventListener('load', function() {
    resizeBokehPlots();
});
</script>"""

OVERVIEW_TABLE_TEMPLATE = """<table class="overview-table">
    <thead>
        <tr><th>Parameter</th><th>Value</th></tr>
    </thead>
    <tbody>
{rows}
    </tbody>
</table>"""

METRICS_TABLE_TEMPLATE = """<table class="metrics-table-consolidated">
    <thead>
        <tr>
{header_cells}
        </tr>
    </thead>
    <tbody>
{body_rows}
    </tbody>
</table>"""

PLOT_CONTAINER_TEMPLATE = """
<div id="{plot_id}" class="plot-container">
    <h3 class="plot-title">{plot_title}</h3>
    <div class="bokeh-plot-wrapper">
        {plot_html}
    </div>
</div>"""


# CS-specific metrics sidebar nav
CS_METRICS_NAV = """    <a href="#alpha-metrics" class="sub-nav">├─ Alpha Metrics</a>
    <a href="#by-cost-metrics" class="sub-nav">├─ By Cost Metrics</a>
    <a href="#passive-metrics" class="sub-nav">├─ Long Passive Investment</a>
    <a href="#short-passive-metrics" class="sub-nav">└─ Short Passive Investment</a>
"""

# TS-specific metrics sidebar nav
TS_METRICS_NAV = """    <a href="#alpha-metrics" class="sub-nav">├─ Alpha Metrics</a>
    <a href="#by-cost-metrics" class="sub-nav">├─ By Cost Metrics</a>
    <a href="#passive-metrics" class="sub-nav">├─ Long Passive Investment</a>
    <a href="#short-passive-metrics" class="sub-nav">└─ Short Passive Investment</a>
"""
