"""CSS stylesheet for backtest HTML reports."""

CSS_STYLESHEET = """
/* Reset and base styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    font-size: 14px;
    line-height: 1.5;
    color: #333;
    background-color: #f8f9fa;
}

/* Sidebar */
.sidebar {
    position: fixed;
    left: 0;
    top: 0;
    height: 100vh;
    width: 240px;
    background: #2c3e50;
    padding: 20px 0;
    transition: width 0.3s ease;
    overflow-x: hidden;
    z-index: 1000;
}

.sidebar a {
    display: block;
    color: #ecf0f1;
    padding: 12px 20px;
    text-decoration: none;
    font-size: 14px;
    font-weight: 500;
    transition: all 0.25s ease;
    border-left: 3px solid transparent;
}

.sidebar a:hover {
    background-color: #34495e;
    border-left-color: #3498db;
    color: #fff;
}

.sidebar a.sub-nav {
    padding-left: 35px;
    font-size: 13px;
    color: #bdc3c7;
}

#toggle-btn {
    position: fixed;
    left: 250px;
    top: 15px;
    font-size: 20px;
    cursor: pointer;
    color: #2c3e50;
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 8px 12px;
    transition: all 0.3s ease;
    z-index: 1001;
}

#toggle-btn:hover {
    background-color: #f8f9fa;
}

/* Main content */
.main {
    margin-left: 250px;
    padding: 30px;
    transition: margin-left 0.3s ease;
    background-color: #fff;
    min-height: 100vh;
}

/* Typography */
h1 {
    font-size: 28px;
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 8px;
}

h2 {
    font-size: 20px;
    font-weight: 600;
    color: #34495e;
    margin: 30px 0 20px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #3498db;
}

h3 {
    font-size: 18px;
    font-weight: 600;
    color: #2c3e50;
    margin: 25px 0 15px 0;
}

.cost-title {
    font-size: 16px;
    font-weight: 600;
    color: #2c3e50;
    margin: 20px 0 12px 0;
    padding: 8px 12px;
    background-color: #ecf0f1;
    border-left: 4px solid #3498db;
    border-radius: 4px;
}

/* Tables */
.overview-table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
    background-color: #fff;
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.overview-table th, .overview-table td {
    padding: 14px 16px;
    text-align: left;
    border-bottom: 1px solid #ecf0f1;
    font-size: 16px;
}

.overview-table th {
    background-color: #2c3e50;
    color: #fff;
    font-weight: 600;
    font-size: 16px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.overview-table tr:hover {
    background-color: #f8f9fa;
}

/* Performance Metrics Tables */
.metrics-section {
    margin-bottom: 30px;
    overflow-x: auto;
    overflow-y: visible;
    -webkit-overflow-scrolling: touch;
}

.metrics-table-consolidated {
    width: auto;
    min-width: 100%;
    max-width: max-content;
    table-layout: fixed;
    border-collapse: collapse;
    margin: 15px 0;
    background-color: #fff;
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 2px 6px rgba(0,0,0,0.12);
}

.metrics-table-consolidated th {
    background-color: #2c3e50;
    color: #fff;
    padding: 12px 10px;
    text-align: center;
    font-weight: 600;
    font-size: 14px;
    min-width: 90px;
    max-width: 120px;
    white-space: normal;
    line-height: 1.3;
    vertical-align: middle;
    border-right: 1px solid #34495e;
}

.metrics-table-consolidated th:last-child {
    border-right: none;
}

.metrics-table-consolidated td {
    padding: 10px 8px;
    text-align: center;
    border-bottom: 1px solid #ecf0f1;
    border-right: 1px solid #ecf0f1;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 13px;
    font-weight: 500;
    min-width: 90px;
    white-space: nowrap;
}

.metrics-table-consolidated td:last-child {
    border-right: none;
}

.metrics-table-consolidated .cost-column {
    background-color: #34495e;
    color: #fff;
    font-weight: 700;
    text-align: center;
    font-size: 15px;
    white-space: nowrap;
    position: sticky;
    left: 0;
    z-index: 10;
    box-shadow: 2px 0 5px rgba(0,0,0,0.15);
}

.metrics-table-consolidated tbody tr:nth-child(even) {
    background-color: #f8f9fa;
}

.metrics-table-consolidated tbody tr:hover {
    background-color: #e8f4fd;
}

/* Two-column layout for Overview and Backtest Parameters */
.overview-params-row {
    display: flex;
    gap: 30px;
    margin-bottom: 30px;
}

.overview-params-row > div {
    flex: 1;
    min-width: 0;
}

/* Two-column layout for metrics tables (Long Only + Short Only) */
.metrics-row-50-50 {
    display: flex;
    gap: 30px;
    margin-bottom: 30px;
}

.metrics-row-50-50 > div {
    flex: 1;
    min-width: 0;
    overflow-x: auto;
}

/* Plot grid layout - responsive */
.plot-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 30px;
    margin: 25px 0;
}

/* Adjust grid for smaller screens */
@media (max-width: 1400px) {
    .plot-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 900px) {
    .plot-grid {
        grid-template-columns: 1fr;
    }
}

.plot-container {
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    padding: 20px;
    overflow: visible; /* Allow toolbar to be visible */
    min-width: 0;
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%; /* Take full width of grid cell */
    height: auto; /* Auto height to fit content */
}

.plot-title {
    font-size: 18px;
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 15px;
    text-align: center;
}

.bokeh-plot-wrapper {
    width: 100%;
    position: relative;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    overflow: visible;
    transition: height 0.3s ease;
}

/* Bokeh root will be scaled by JavaScript */
.bokeh-plot-wrapper .bk-root {
    transform-origin: top center;
    transition: transform 0.3s ease;
}

/* Make sure toolbar is always visible */
.bk-toolbar {
    z-index: 1000 !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* Ensure proper layout for toolbar */
.bk-toolbar-right {
    position: absolute !important;
    right: 0 !important;
}

.bk-canvas-events,
.bk-canvas-overlays {
    overflow: visible !important;
}

/* Chart container */
.chart-container {
    text-align: center;
    margin: 25px 0;
    padding: 20px;
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.chart-container img {
    max-width: 100%;
    max-height: 1000px;
    width: auto;
    height: auto;
    object-fit: contain;
}

/* Print styles */
.web-only { display: block; }
.print-only { display: none; width: 100%; }

@media print {
    .web-only { display: none; }
    .print-only { display: block; }
    .sidebar { display: none; }
    .main { margin-left: 0px; }
    #toggle-btn { display: none; }
    body { background-color: #fff; }
}

/* Responsive design */
@media (max-width: 768px) {
    .sidebar {
        width: 0;
    }

    .main {
        margin-left: 0;
        padding: 15px;
    }

    #toggle-btn {
        left: 15px;
    }
}

/* Documentation content styles */
.documentation-content {
    max-width: 1200px;
    margin: 0 auto;
    line-height: 1.8;
}

.documentation-content h1 {
    font-size: 26px;
    font-weight: 600;
    color: #2c3e50;
    margin: 40px 0 20px 0;
    padding-bottom: 10px;
    border-bottom: 3px solid #ecf0f1;
}

.documentation-content h2 {
    font-size: 22px;
    font-weight: 600;
    color: #34495e;
    margin: 35px 0 18px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #3498db;
}

.documentation-content h3 {
    font-size: 18px;
    font-weight: 600;
    color: #2c3e50;
    margin: 25px 0 12px 0;
}

.documentation-content h4 {
    font-size: 16px;
    font-weight: 600;
    color: #34495e;
    margin: 20px 0 10px 0;
}

.documentation-content p {
    margin: 10px 0;
    color: #333;
}

.documentation-content code {
    background-color: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 13px;
    color: #c7254e;
}

.documentation-content pre {
    background-color: #f8f8f8;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 15px;
    overflow-x: auto;
    margin: 15px 0;
}

.documentation-content pre code {
    background-color: transparent;
    padding: 0;
    font-size: 13px;
    color: #333;
}

.documentation-content table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    background-color: #fff;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.documentation-content table th {
    background-color: #2c3e50;
    color: #fff;
    padding: 12px;
    text-align: left;
    font-weight: 600;
    font-size: 14px;
}

.documentation-content table td {
    padding: 10px 12px;
    border-bottom: 1px solid #ecf0f1;
    font-size: 14px;
}

.documentation-content table tr:nth-child(even) {
    background-color: #f8f9fa;
}

.documentation-content table tr:hover {
    background-color: #e8f4fd;
}

.documentation-content ul, .documentation-content ol {
    margin: 15px 0;
    padding-left: 30px;
}

.documentation-content li {
    margin: 8px 0;
    line-height: 1.6;
}

.documentation-content blockquote {
    border-left: 4px solid #3498db;
    padding: 10px 20px;
    margin: 20px 0;
    background-color: #f8f9fa;
    color: #555;
}

.documentation-content hr {
    border: none;
    border-top: 2px solid #ecf0f1;
    margin: 30px 0;
}

.documentation-content strong {
    font-weight: 600;
    color: #2c3e50;
}

.documentation-content em {
    font-style: italic;
    color: #555;
}

/* LaTeX/Math equation styling */
.documentation-content .MathJax,
.documentation-content .katex {
    font-size: 1.1em;
}

.documentation-content p:has(.MathJax),
.documentation-content p:has(.katex) {
    margin: 15px 0;
}

.documentation-content p>has(.MathJax),
.documentation-content p>has(.katex) {
    display: inline-block;
}

/* Factor Information Section Styles */
.factor-info-section {
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    padding: 25px;
    margin-bottom: 30px;
}

.factor-info-section h2 {
    font-size: 20px;
    font-weight: 600;
    color: #34495e;
    margin: 0 0 20px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #3498db;
}

.factor-info-content {
    line-height: 1.7;
}

.factor-name {
    font-size: 18px;
    font-weight: 600;
    color: #2c3e50;
    margin: 0 0 15px 0;
}

.factor-description {
    color: #555;
    margin: 0 0 25px 0;
    font-size: 15px;
}

.factor-formula-block {
    background-color: #f8f9fa;
    border-radius: 6px;
    padding: 20px;
    margin-bottom: 20px;
    border-left: 4px solid #3498db;
}

.factor-formula-block h4 {
    font-size: 16px;
    font-weight: 600;
    color: #2c3e50;
    margin: 0 0 15px 0;
}

.factor-formula-block h5 {
    font-size: 14px;
    font-weight: 600;
    color: #34495e;
    margin: 20px 0 10px 0;
}

.formula-latex {
    background-color: #fff;
    padding: 15px 20px;
    border-radius: 4px;
    margin: 10px 0 20px 0;
    overflow-x: auto;
    border: 1px solid #e0e0e0;
}

.formula-latex math {
    font-size: 1.1em;
}

.latex-fallback {
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 13px;
    background-color: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    color: #333;
}

.formula-description {
    margin: 15px 0;
}

.calculation-steps {
    margin: 10px 0;
    padding-left: 25px;
}

.calculation-steps li {
    margin: 8px 0;
    color: #555;
    line-height: 1.6;
}

.formula-variables {
    margin-top: 15px;
}

.formula-variables .overview-table {
    margin: 10px 0;
}

.formula-variables .overview-table code {
    background-color: #e8f4fd;
    padding: 2px 8px;
    border-radius: 3px;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 13px;
    color: #2980b9;
}
"""
