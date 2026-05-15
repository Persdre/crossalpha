"""
backtest_doc.py - Documentation compilation module for backtesting.

This module provides documentation compilers for both cross-sectional (CS) and
time-series (TS) backtesting. Classes read markdown documentation files and
compile them into HTML format for embedding in backtest reports.
"""

from pathlib import Path
from typing import Optional
import markdown


class _BacktestDocBase:
    """
    Base class for documentation compilers with shared markdown processing methods.

    Attributes:
        docs_dir (Path): Directory containing markdown documentation files
    """

    docs_dir: Path  # To be set by subclasses

    def _read_markdown_file(self, filename: str) -> str:
        """
        Read a markdown file from the docs directory.

        Args:
            filename: Name of the markdown file (e.g., "1_overview.md")

        Returns:
            str: Contents of the markdown file

        Raises:
            FileNotFoundError: If the specified file doesn't exist
        """
        filepath = self.docs_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Documentation file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def _markdown_to_html(self, markdown_text: str) -> str:
        """
        Convert markdown text to HTML.

        LaTeX is left as-is so the report's MathJax renderer can typeset it.
        This keeps documentation generation robust even when optional LaTeX
        conversion dependencies are unavailable.

        Args:
            markdown_text: Markdown formatted text with LaTeX math

        Returns:
            str: HTML formatted text
        """
        # Process markdown normally
        md = markdown.Markdown(extensions=[
            'fenced_code',
            'tables',
            'nl2br',
            'sane_lists',
        ])
        
        return md.convert(markdown_text)

    def compile_documentation(self, backtest_mode: str = None) -> str:
        """
        Compile complete documentation.

        Reads all markdown files, converts to HTML, and returns compiled documentation.

        Args:
            backtest_mode: Optional, kept for backward compatibility (no longer used)

        Returns:
            str: Complete HTML documentation

        Raises:
            FileNotFoundError: If any required documentation file is missing
        """
        doc_files = [
            "1_overview.md",
            "2_factor_to_position.md",
            "3_metrics.md",
            "4_plots.md"
        ]

        doc_sections = [self._read_markdown_file(f) for f in doc_files]
        combined_markdown = "\n\n---\n\n".join(doc_sections)
        html_content = self._markdown_to_html(combined_markdown)

        return f"""
<div class="documentation-content">
{html_content}
</div>
"""

    def get_available_files(self) -> list[str]:
        """
        Get list of available documentation files.

        Returns:
            list: Sorted list of markdown filenames in the docs directory
        """
        if not self.docs_dir.exists():
            return []

        md_files = sorted([f.name for f in self.docs_dir.glob("*.md")])
        return md_files


class CS_Backtest_Doc(_BacktestDocBase):
    """
    Documentation compiler for cross-sectional backtesting.

    Reads markdown files from the resources/docs/cs/ directory and compiles them
    into HTML format for embedding in backtest reports.
    """

    def __init__(self, docs_dir: Optional[Path] = None):
        """
        Initialize the CS documentation compiler.

        Args:
            docs_dir: Path to documentation directory. If None, uses 'resources/docs/cs'
                      subdirectory relative to this module's location.
        """
        if docs_dir is None:
            self.docs_dir = Path(__file__).parent / "resources" / "docs" / "cs"
        else:
            self.docs_dir = Path(docs_dir)

        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Documentation directory not found: {self.docs_dir}")


class TS_Backtest_Doc(_BacktestDocBase):
    """
    Documentation compiler for time-series backtesting.

    Reads markdown files from the resources/docs/ts/ directory and compiles them
    into HTML format for embedding in backtest reports.
    """

    def __init__(self, docs_dir: Optional[Path] = None):
        """
        Initialize the TS documentation compiler.

        Args:
            docs_dir: Path to documentation directory. If None, uses 'resources/docs/ts'
                      subdirectory relative to this module's location.
        """
        if docs_dir is None:
            self.docs_dir = Path(__file__).parent / "resources" / "docs" / "ts"
        else:
            self.docs_dir = Path(docs_dir)

        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Documentation directory not found: {self.docs_dir}")


# Example usage
if __name__ == "__main__":
    # Test CS documentation
    print("=== Cross-Sectional (CS) Documentation ===")
    cs_compiler = CS_Backtest_Doc()
    print("Available files:")
    for filename in cs_compiler.get_available_files():
        print(f"  - {filename}")
    try:
        html_doc = cs_compiler.compile_documentation()
        print(f"Compiled: {len(html_doc)} characters of HTML\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # Test TS documentation
    print("=== Time-Series (TS) Documentation ===")
    ts_compiler = TS_Backtest_Doc()
    print("Available files:")
    for filename in ts_compiler.get_available_files():
        print(f"  - {filename}")
    try:
        html_doc = ts_compiler.compile_documentation()
        print(f"Compiled: {len(html_doc)} characters of HTML")
    except Exception as e:
        print(f"Error: {e}")
