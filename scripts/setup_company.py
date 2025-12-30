#!/usr/bin/env python3
"""
setup_company.py - Interactive setup for a new company analysis.

This script automates the setup of a new company for stock pitch analysis:
1. Creates .env file with SEC_USER_AGENT
2. Creates/updates COMPANY.md with company context
3. Creates folder structure for all document types
4. Runs `uv run pitch fetch` to download SEC filings
5. Optionally prints competitor research instructions

Usage:
    python scripts/setup_company.py              # Interactive mode
    python scripts/setup_company.py AAPL         # Pre-specify ticker
    python scripts/setup_company.py --research   # Include research instructions
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
from pathlib import Path
from typing import NoReturn


# ANSI color codes for terminal output
class Colors:
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    RESET = "\033[0m"


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== {text} ==={Colors.RESET}\n")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}[OK]{Colors.RESET} {text}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}[WARN]{Colors.RESET} {text}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}[ERROR]{Colors.RESET} {text}")


def print_info(text: str) -> None:
    """Print an info message."""
    print(f"{Colors.CYAN}[INFO]{Colors.RESET} {text}")


def handle_interrupt(signum: int, frame) -> NoReturn:
    """Handle Ctrl+C gracefully."""
    print(f"\n\n{Colors.YELLOW}Setup cancelled by user.{Colors.RESET}")
    sys.exit(1)


def prompt(message: str, default: str = "", required: bool = False) -> str:
    """
    Prompt for user input with optional default value.

    Args:
        message: The prompt message
        default: Default value shown in brackets
        required: If True, keep prompting until value provided

    Returns:
        User input or default value
    """
    if default:
        prompt_text = f"{message} [{default}]: "
    else:
        prompt_text = f"{message}: "

    while True:
        try:
            value = input(prompt_text).strip()
            if not value:
                value = default

            if required and not value:
                print_warning("This field is required. Please enter a value.")
                continue

            return value
        except EOFError:
            # Handle piped input ending
            return default


def prompt_choice(message: str, choices: list[str], default: str = "") -> str:
    """
    Prompt for a choice from a list of options.

    Args:
        message: The prompt message
        choices: List of valid choices
        default: Default choice

    Returns:
        Selected choice
    """
    choices_str = "/".join(choices)
    if default:
        prompt_text = f"{message} ({choices_str}) [{default}]: "
    else:
        prompt_text = f"{message} ({choices_str}): "

    while True:
        value = input(prompt_text).strip().lower()
        if not value:
            value = default.lower()

        # Match partial input
        matches = [c for c in choices if c.lower().startswith(value)]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            print_warning(f"Ambiguous input. Please specify: {', '.join(matches)}")
        else:
            print_warning(f"Invalid choice. Please enter one of: {', '.join(choices)}")


def confirm(message: str, default: bool = True) -> bool:
    """
    Prompt for yes/no confirmation.

    Args:
        message: The prompt message
        default: Default value (True = yes, False = no)

    Returns:
        True if confirmed, False otherwise
    """
    default_str = "Y/n" if default else "y/N"
    prompt_text = f"{message} [{default_str}]: "

    value = input(prompt_text).strip().lower()
    if not value:
        return default

    return value in ("y", "yes", "true", "1")


def create_env_file(project_root: Path, sec_email: str) -> bool:
    """
    Create or update .env file with SEC_USER_AGENT.

    Args:
        project_root: Root directory of the project
        sec_email: Email for SEC user agent

    Returns:
        True if file was created/updated, False if skipped
    """
    env_path = project_root / ".env"
    template_path = project_root / ".env.template"

    sec_user_agent = f"StockPitchAnalysis {sec_email}"

    if env_path.exists():
        # Check if SEC_USER_AGENT already set
        content = env_path.read_text()
        if "SEC_USER_AGENT" in content and "your.email@example.com" not in content:
            print_info(".env already has SEC_USER_AGENT configured")
            if not confirm("Overwrite SEC_USER_AGENT?", default=False):
                return False

        # Update existing file
        lines = content.splitlines()
        updated_lines = []
        found = False
        for line in lines:
            if line.startswith("SEC_USER_AGENT"):
                updated_lines.append(f"SEC_USER_AGENT={sec_user_agent}")
                found = True
            else:
                updated_lines.append(line)

        if not found:
            updated_lines.append(f"SEC_USER_AGENT={sec_user_agent}")

        env_path.write_text("\n".join(updated_lines) + "\n")
        print_success("Updated .env with SEC_USER_AGENT")
        return True

    elif template_path.exists():
        # Create from template
        content = template_path.read_text()
        content = content.replace("SEC_USER_AGENT=YourName your.email@example.com", f"SEC_USER_AGENT={sec_user_agent}")
        env_path.write_text(content)
        print_success("Created .env from template")
        return True

    else:
        # Create minimal .env
        env_content = f"""# Anthropic API
ANTHROPIC_API_KEY=your_api_key_here

# SEC EDGAR (for user agent identification)
SEC_USER_AGENT={sec_user_agent}

# Embedding model
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DEVICE=cuda
"""
        env_path.write_text(env_content)
        print_success("Created .env file")
        print_warning("Remember to add your ANTHROPIC_API_KEY to .env")
        return True


def generate_company_md(
    ticker: str,
    company_name: str,
    position: str,
    fiscal_year_end: str,
    thesis_points: list[str],
) -> str:
    """
    Generate COMPANY.md content.

    Args:
        ticker: Stock ticker symbol
        company_name: Full company name
        position: Long or Short
        fiscal_year_end: Fiscal year end description
        thesis_points: List of thesis points

    Returns:
        Markdown content for COMPANY.md
    """
    # Format thesis points for the narrative section
    thesis_narrative = ""
    if thesis_points:
        for i, point in enumerate(thesis_points, 1):
            thesis_narrative += f"""
### {i}. {point}
- Details here
"""
    else:
        thesis_narrative = """
### 1. [Add first thesis point]
- Details here

### 2. [Add second thesis point]
- Details here
"""

    # Determine position-specific language
    position_upper = position.upper()
    thesis_type = "Short Thesis" if position_upper == "SHORT" else "Long Thesis"

    # Format thesis summary
    if thesis_points:
        thesis_summary = " / ".join(thesis_points[:3])  # First 3 points
        if len(thesis_points) > 3:
            thesis_summary += " / ..."
    else:
        thesis_summary = "Add thesis summary here"

    # Format fiscal year info
    if fiscal_year_end:
        fiscal_info = f"""- Fiscal year ends **{fiscal_year_end}**
- [Add quarter mappings if non-calendar fiscal year]"""
    else:
        fiscal_info = """- Fiscal year ends **[specify fiscal year end]**
- [Add quarter mappings if non-calendar fiscal year]"""

    content = f"""# Company Context: {company_name} ({ticker})

## Ticker

{ticker} ({company_name})

## Position

**{position_upper}** - {thesis_summary}

## Fiscal Calendar

{fiscal_info}

## Key Metrics

<!-- Customize for industry - examples below -->
- Revenue and YoY growth
- Gross margin and operating margin trends
- Free cash flow
- [Add industry-specific metrics]

## Investment Narrative ({thesis_type})
{thesis_narrative}
## Available Documents (Indexed)

<!-- Update after running `uv run pitch inventory` -->

- SEC Filings: 0 docs
- Transcripts: 0 earnings calls
- Conferences: 0 transcripts
- Presentations: 0
- Analyst Reports: 0

**Total Index:** 0 chunks across 0 documents

---

## Key Competitors to Research

<!-- Add competitors relevant to this company -->

### Primary Competitors
- [Add competitor 1]
- [Add competitor 2]
- [Add competitor 3]

### Adjacent Competitors
- [Add adjacent competitors]

### Emerging Threats
- [Add emerging technology or business model threats]
"""
    return content


def create_company_md(
    project_root: Path,
    ticker: str,
    company_name: str,
    position: str,
    fiscal_year_end: str,
    thesis_points: list[str],
) -> bool:
    """
    Create or update COMPANY.md file.

    Returns:
        True if file was created/updated, False if skipped
    """
    company_md_path = project_root / "COMPANY.md"

    if company_md_path.exists():
        print_warning("COMPANY.md already exists")
        if not confirm("Overwrite COMPANY.md?", default=False):
            return False

    content = generate_company_md(
        ticker=ticker,
        company_name=company_name,
        position=position,
        fiscal_year_end=fiscal_year_end,
        thesis_points=thesis_points,
    )

    company_md_path.write_text(content)
    print_success("Created COMPANY.md")
    return True


def create_directory_structure(project_root: Path, ticker: str) -> None:
    """
    Create the folder structure for company analysis.

    Args:
        project_root: Root directory of the project
        ticker: Stock ticker symbol (uppercase)
    """
    ticker_upper = ticker.upper()

    # Directories to create (with ticker subdirectory)
    ticker_dirs = [
        "data",
        "transcripts",
        "analyst",
        "presentations",
        "conferences",
        "misc",
        "model",
        "index",
        "processed",
    ]

    # Directories without ticker subdirectory
    plain_dirs = [
        "output",
        "notes",
        "imports",
        "research",
        "research/competitors",
        f"questions/{ticker_upper}",
    ]

    # Create ticker-specific directories
    for dir_name in ticker_dirs:
        dir_path = project_root / dir_name / ticker_upper
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            # Add .gitkeep
            gitkeep = dir_path / ".gitkeep"
            gitkeep.touch()
            print_success(f"Created {dir_name}/{ticker_upper}/")
        else:
            print_info(f"{dir_name}/{ticker_upper}/ already exists")

    # Create plain directories
    for dir_name in plain_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            # Add .gitkeep to leaf directories
            if not any(dir_path.iterdir()):
                gitkeep = dir_path / ".gitkeep"
                gitkeep.touch()
            print_success(f"Created {dir_name}/")
        else:
            print_info(f"{dir_name}/ already exists")


def run_fetch_command(project_root: Path, ticker: str, years: int) -> bool:
    """
    Run the pitch fetch command to download SEC filings.

    Args:
        project_root: Root directory of the project
        ticker: Stock ticker symbol
        years: Number of years of filings to fetch

    Returns:
        True if successful, False otherwise
    """
    ticker_upper = ticker.upper()

    # Build the command
    cmd = [
        "uv",
        "run",
        "pitch",
        "fetch",
        ticker_upper,
        "-t",
        "10-K,10-Q,8-K,DEF 14A",
        "-y",
        str(years),
    ]

    print_info(f"Running: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            check=False,
        )

        if result.returncode == 0:
            print_success("SEC filings fetched successfully")
            return True
        else:
            print_error(f"Fetch command failed with exit code {result.returncode}")
            return False

    except FileNotFoundError:
        print_error("uv not found. Make sure uv is installed and in PATH.")
        print_info("You can run the fetch command manually:")
        print(f"  uv run pitch fetch {ticker_upper} -t '10-K,10-Q,8-K,DEF 14A' -y {years}")
        return False

    except Exception as e:
        print_error(f"Error running fetch: {e}")
        return False


def print_research_instructions(ticker: str) -> None:
    """Print instructions for competitor research."""
    ticker_upper = ticker.upper()

    print_header("Competitor Research Instructions")

    print(f"""To research competitors for {ticker_upper}, follow these steps:

{Colors.BOLD}1. Identify Key Competitors{Colors.RESET}
   - Check the company's 10-K Item 1 (Business) for named competitors
   - Look at industry analyst reports
   - Search for market share data

{Colors.BOLD}2. Create Competitor Profiles{Colors.RESET}
   Create files in research/competitors/ for each major competitor:
   - research/competitors/competitor1.md
   - research/competitors/competitor2.md

{Colors.BOLD}3. Research Areas for Each Competitor{Colors.RESET}
   - Business model and revenue streams
   - Product/service comparison
   - Market share and positioning
   - Recent strategic moves
   - Financial metrics (revenue, margins, growth)
   - Strengths and weaknesses vs {ticker_upper}

{Colors.BOLD}4. Update COMPANY.md{Colors.RESET}
   Add identified competitors to the "Key Competitors" section

{Colors.BOLD}5. Use the Research Agent{Colors.RESET}
   After indexing documents, use the /research command to query:
   - uv run pitch search {ticker_upper} "competitor"
   - uv run pitch ask {ticker_upper} "What are the main competitors?"
""")


def print_next_steps(ticker: str, fetched: bool) -> None:
    """Print next steps after setup."""
    ticker_upper = ticker.upper()

    print_header("Next Steps")

    print(f"""{Colors.BOLD}1. Review Generated Files{Colors.RESET}
   - Edit COMPANY.md with company-specific details
   - Update .env with API keys if needed

{Colors.BOLD}2. Add Source Materials{Colors.RESET}
   Upload documents to the appropriate folders:
   - transcripts/{ticker_upper}/ - Earnings call transcripts (PDF)
   - conferences/{ticker_upper}/ - Conference presentations (PDF)
   - analyst/{ticker_upper}/ - Bank analyst reports (PDF)
   - presentations/{ticker_upper}/ - Investor day materials (PDF)
   - model/{ticker_upper}/ - Financial models (Excel)

   For bulk imports: Drop ZIP files in imports/ for Claude Code to sort
""")

    if not fetched:
        print(f"""{Colors.BOLD}3. Fetch SEC Filings{Colors.RESET}
   uv run pitch fetch {ticker_upper} -t "10-K,10-Q,8-K,DEF 14A" -y 6
""")

    print(f"""{Colors.BOLD}{"3" if fetched else "4"}. Build Index and Summarize{Colors.RESET}
   uv run pitch index {ticker_upper} --source all
   uv run pitch summarize {ticker_upper} --latest

{Colors.BOLD}{"4" if fetched else "5"}. Run Inventory{Colors.RESET}
   uv run pitch inventory
   # Then update COMPANY.md with document counts

{Colors.BOLD}{"5" if fetched else "6"}. Start Research{Colors.RESET}
   uv run pitch search {ticker_upper} "revenue growth"
   uv run pitch ask {ticker_upper} "What is the company's competitive advantage?"
""")


def main() -> int:
    """Main entry point."""
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, handle_interrupt)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Set up a new company for stock pitch analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/setup_company.py              # Interactive mode
  python scripts/setup_company.py AAPL         # Pre-specify ticker
  python scripts/setup_company.py --research   # Include research instructions
        """,
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        default="",
        help="Stock ticker symbol (optional, will prompt if not provided)",
    )
    parser.add_argument(
        "--research",
        action="store_true",
        help="Print competitor research instructions at the end",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip fetching SEC filings",
    )

    args = parser.parse_args()

    # Determine project root (script is in scripts/ directory)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    print_header("Stock Pitch Company Setup")
    print(f"Project root: {project_root}")

    # Check for uv
    import shutil

    if not shutil.which("uv"):
        print_warning("uv not found in PATH. Install it from: https://docs.astral.sh/uv/")

    # Collect required inputs
    print_header("Company Information")

    # Ticker
    ticker = args.ticker.upper() if args.ticker else ""
    if not ticker:
        ticker = prompt("Ticker symbol", required=True).upper()
    else:
        print(f"Ticker: {ticker}")

    # Company Name
    company_name = prompt("Company name", required=True)

    # Position
    position = prompt_choice("Position", ["Long", "Short"], default="Long")

    # SEC User Agent Email
    print()
    print_info("SEC requires identification for API access.")
    sec_email = prompt("Your email for SEC access", required=True)

    # Optional: Fiscal Year End
    print()
    print_info("Fiscal year end helps with quarter mapping (e.g., 'First Sunday of February', 'December 31').")
    fiscal_year_end = prompt("Fiscal year end (or press Enter for calendar year)", default="December 31")

    # Optional: Filing Years
    print()
    years_str = prompt("Years of SEC filings to fetch", default="6")
    try:
        years = int(years_str)
    except ValueError:
        years = 6

    # Optional: Thesis Points
    print()
    print_info("Enter key thesis points separated by commas (optional).")
    print_info("Example: Valuation concerns, Competitive pressure, Management quality")
    thesis_str = prompt("Key thesis points", default="")
    thesis_points = [p.strip() for p in thesis_str.split(",") if p.strip()]

    # Confirm before proceeding
    print_header("Configuration Summary")
    print(f"  Ticker:          {ticker}")
    print(f"  Company:         {company_name}")
    print(f"  Position:        {position}")
    print(f"  SEC Email:       {sec_email}")
    print(f"  Fiscal Year End: {fiscal_year_end}")
    print(f"  Filing Years:    {years}")
    print(f"  Thesis Points:   {', '.join(thesis_points) if thesis_points else '(none)'}")
    print()

    if not confirm("Proceed with setup?", default=True):
        print("Setup cancelled.")
        return 1

    # Execute setup steps
    print_header("Step 1: Creating .env File")
    create_env_file(project_root, sec_email)

    print_header("Step 2: Creating COMPANY.md")
    create_company_md(
        project_root=project_root,
        ticker=ticker,
        company_name=company_name,
        position=position,
        fiscal_year_end=fiscal_year_end,
        thesis_points=thesis_points,
    )

    print_header("Step 3: Creating Directory Structure")
    create_directory_structure(project_root, ticker)

    fetched = False
    if not args.skip_fetch:
        print_header("Step 4: Fetching SEC Filings")
        if confirm(f"Fetch {years} years of SEC filings for {ticker}?", default=True):
            fetched = run_fetch_command(project_root, ticker, years)
        else:
            print_info("Skipping SEC filing fetch")
    else:
        print_info("Skipping SEC filing fetch (--skip-fetch)")

    # Print research instructions if requested
    if args.research:
        print_research_instructions(ticker)

    # Print next steps
    print_next_steps(ticker, fetched)

    print_header("Setup Complete!")
    print(f"Company analysis for {Colors.BOLD}{ticker}{Colors.RESET} is ready.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
