"""SEC filing section patterns, regex matchers, and chunk size configuration.

This module centralizes all filing-type-specific patterns used for parsing
10-K, 10-Q, 8-K, and DEF 14A (proxy) filings.
"""

import re

# ============================================================================
# SECTION PATTERNS - Filing-type-specific section definitions
# ============================================================================

# Standard 10-K section patterns
SECTION_PATTERNS_10K = {
    # Part I
    "Item 1": "Business",
    "Item 1A": "Risk Factors",
    "Item 1B": "Unresolved Staff Comments",
    "Item 1C": "Cybersecurity",
    "Item 2": "Properties",
    "Item 3": "Legal Proceedings",
    "Item 4": "Mine Safety Disclosures",
    # Part II
    "Item 5": "Market for Registrant's Common Equity",
    "Item 6": "Reserved",
    "Item 7": "Management's Discussion and Analysis",
    "Item 7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "Item 8": "Financial Statements and Supplementary Data",
    "Item 9": "Changes in and Disagreements with Accountants",
    "Item 9A": "Controls and Procedures",
    "Item 9B": "Other Information",
    "Item 9C": "Disclosure Regarding Foreign Jurisdictions",
    # Part III
    "Item 10": "Directors, Executive Officers and Corporate Governance",
    "Item 11": "Executive Compensation",
    "Item 12": "Security Ownership",
    "Item 13": "Certain Relationships and Related Transactions",
    "Item 14": "Principal Accountant Fees and Services",
    # Part IV
    "Item 15": "Exhibits and Financial Statement Schedules",
    "Item 16": "Form 10-K Summary",
}

# Standard 10-Q section patterns (use Part prefix to avoid key conflicts)
SECTION_PATTERNS_10Q = {
    # Part I - Financial Information
    "Part I, Item 1": "Financial Statements",
    "Part I, Item 2": "Management's Discussion and Analysis",
    "Part I, Item 3": "Quantitative and Qualitative Disclosures About Market Risk",
    "Part I, Item 4": "Controls and Procedures",
    # Part II - Other Information
    "Part II, Item 1": "Legal Proceedings",
    "Part II, Item 1A": "Risk Factors",
    "Part II, Item 2": "Unregistered Sales of Equity Securities",
    "Part II, Item 3": "Defaults Upon Senior Securities",
    "Part II, Item 4": "Mine Safety Disclosures",
    "Part II, Item 5": "Other Information",
    "Part II, Item 6": "Exhibits",
}

# Standard 8-K item patterns (Item X.XX format)
SECTION_PATTERNS_8K = {
    # Section 1 - Registrant's Business and Operations
    "Item 1.01": "Entry into a Material Definitive Agreement",
    "Item 1.02": "Termination of a Material Definitive Agreement",
    "Item 1.03": "Bankruptcy or Receivership",
    "Item 1.04": "Mine Safety - Reporting of Shutdowns and Patterns of Violations",
    # Section 2 - Financial Information
    "Item 2.01": "Completion of Acquisition or Disposition of Assets",
    "Item 2.02": "Results of Operations and Financial Condition",
    "Item 2.03": "Creation of a Direct Financial Obligation",
    "Item 2.04": "Triggering Events That Accelerate or Increase a Direct Financial Obligation",
    "Item 2.05": "Costs Associated with Exit or Disposal Activities",
    "Item 2.06": "Material Impairments",
    # Section 3 - Securities and Trading Markets
    "Item 3.01": "Notice of Delisting or Failure to Satisfy a Continued Listing Rule",
    "Item 3.02": "Unregistered Sales of Equity Securities",
    "Item 3.03": "Material Modification to Rights of Security Holders",
    # Section 4 - Matters Related to Accountants and Financial Statements
    "Item 4.01": "Changes in Registrant's Certifying Accountant",
    "Item 4.02": "Non-Reliance on Previously Issued Financial Statements",
    # Section 5 - Corporate Governance and Management
    "Item 5.01": "Changes in Control of Registrant",
    "Item 5.02": "Departure of Directors or Certain Officers; Election of Directors",
    "Item 5.03": "Amendments to Articles of Incorporation or Bylaws",
    "Item 5.04": "Temporary Suspension of Trading Under Employee Benefit Plans",
    "Item 5.05": "Amendment to the Registrant's Code of Ethics",
    "Item 5.06": "Change in Shell Company Status",
    "Item 5.07": "Submission of Matters to a Vote of Security Holders",
    "Item 5.08": "Shareholder Nominations Pursuant to Exchange Act Rule 14a-11",
    # Section 6 - Asset-Backed Securities
    "Item 6.01": "ABS Informational and Computational Material",
    "Item 6.02": "Change of Servicer or Trustee",
    "Item 6.03": "Change in Credit Enhancement or Other External Support",
    "Item 6.04": "Failure to Make a Required Distribution",
    "Item 6.05": "Securities Act Updating Disclosure",
    # Section 7 - Regulation FD
    "Item 7.01": "Regulation FD Disclosure",
    # Section 8 - Other Events
    "Item 8.01": "Other Events",
    # Section 9 - Financial Statements and Exhibits
    "Item 9.01": "Financial Statements and Exhibits",
}

# DEF 14A (Proxy Statement) section patterns
SECTION_PATTERNS_DEF14A = {
    "Executive Compensation": "Executive Compensation",
    "Summary Compensation Table": "Summary Compensation Table",
    "Compensation Discussion": "Compensation Discussion and Analysis",
    "Related Party Transactions": "Certain Relationships and Related Party Transactions",
    "Corporate Governance": "Corporate Governance",
    "Board of Directors": "Board of Directors",
    "Director Compensation": "Director Compensation",
    "Stock Ownership": "Stock Ownership",
    "Proposal": "Proposal",
}

# Backward compatibility alias
SECTION_PATTERNS = SECTION_PATTERNS_10K

# ============================================================================
# CHUNK SIZE CONFIGURATION - Section-specific optimal chunk sizes
# ============================================================================

# Content-type-specific chunk sizes (tokens) - Quick Win #1
# Different content types benefit from different chunk sizes
CHUNK_SIZE_BY_SECTION = {
    # Executive summary / overview sections - preserve narrative flow
    "Item 1": 1200,  # Business description
    "Item 5": 1200,  # Market info
    # Risk factors - one risk per chunk works better
    "Item 1A": 1500,  # Risk Factors - risks can be lengthy
    "Part II, Item 1A": 1500,  # 10-Q Risk Factors
    # MD&A - preserve analytical context
    "Item 7": 1500,  # MD&A needs more context
    "Part I, Item 2": 1500,  # 10-Q MD&A
    # Financial statements - smaller, table-focused
    "Item 8": 800,  # Financial Statements
    "Part I, Item 1": 800,  # 10-Q Financial Statements
    # Compensation / governance - moderate
    "Executive Compensation": 1200,
    "Compensation Discussion": 1200,
    "Board of Directors": 1000,
    # Default for unlisted sections
    "default": 1000,
}

# ============================================================================
# REGEX PATTERNS - Section header detection
# ============================================================================

# Regex to match section headers (10-K/10-Q style: Item 1, Item 1A, etc.)
SECTION_HEADER_REGEX = re.compile(
    r"Item\s+(\d+[A-C]?)\.?\s*[-–—]?\s*(.{0,100})",
    re.IGNORECASE,
)

# Regex to match 8-K section headers (Item X.XX format)
SECTION_HEADER_REGEX_8K = re.compile(
    r"Item\s+(\d+\.\d{2})\.?\s*[-–—]?\s*(.{0,100})",
    re.IGNORECASE,
)

# Regex to match DEF 14A section headers (text-based)
SECTION_HEADER_REGEX_DEF14A = re.compile(
    r"^(Executive\s+Compensation|Compensation\s+Discussion|"
    r"Related\s+Party|Corporate\s+Governance|Board\s+of\s+Directors|"
    r"Director\s+Compensation|Stock\s+Ownership|Proposal\s+\d+)",
    re.IGNORECASE | re.MULTILINE,
)

# ============================================================================
# HELPER FUNCTIONS - Pattern lookup by filing type
# ============================================================================


def get_section_patterns(filing_type: str) -> dict[str, str]:
    """Get section patterns for a specific filing type.

    Args:
        filing_type: Filing type string (e.g., "10-K", "10-Q", "8-K", "DEF 14A")

    Returns:
        Dictionary mapping item keys to section titles
    """
    filing_type_upper = filing_type.upper().replace("-", "")
    if filing_type_upper == "10K":
        return SECTION_PATTERNS_10K
    elif filing_type_upper == "10Q":
        return SECTION_PATTERNS_10Q
    elif filing_type_upper == "8K":
        return SECTION_PATTERNS_8K
    elif filing_type_upper in ("DEF14A", "DEF 14A", "PROXY"):
        return SECTION_PATTERNS_DEF14A
    else:
        # Default to 10-K patterns
        return SECTION_PATTERNS_10K


def get_section_regex(filing_type: str) -> re.Pattern:
    """Get section header regex for a specific filing type.

    Args:
        filing_type: Filing type string (e.g., "10-K", "10-Q", "8-K", "DEF 14A")

    Returns:
        Compiled regex pattern for matching section headers
    """
    filing_type_upper = filing_type.upper().replace("-", "")
    if filing_type_upper == "8K":
        return SECTION_HEADER_REGEX_8K
    elif filing_type_upper in ("DEF14A", "DEF 14A", "PROXY"):
        return SECTION_HEADER_REGEX_DEF14A
    else:
        return SECTION_HEADER_REGEX
