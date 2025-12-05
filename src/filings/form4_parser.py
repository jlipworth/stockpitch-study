"""Form 4 XML parser for insider trading data."""

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from src.rag.base_parser import BaseParser

logger = logging.getLogger(__name__)


@dataclass
class InsiderInfo:
    """Information about the reporting insider."""

    name: str
    cik: str = ""
    is_director: bool = False
    is_officer: bool = False
    is_ten_percent_owner: bool = False
    is_other: bool = False
    officer_title: str = ""

    @property
    def role(self) -> str:
        """Get a human-readable role description."""
        roles = []
        if self.officer_title:
            roles.append(self.officer_title)
        elif self.is_officer:
            roles.append("Officer")
        if self.is_director:
            roles.append("Director")
        if self.is_ten_percent_owner:
            roles.append("10% Owner")
        if self.is_other:
            roles.append("Other")
        return ", ".join(roles) if roles else "Unknown"


@dataclass
class _TransactionBase:
    """Common fields extracted from transaction elements (internal use)."""

    security_title: str
    transaction_date: date
    transaction_code: str
    shares: Decimal
    price_per_share: Decimal | None
    acquired_disposed: str


@dataclass
class Transaction:
    """A single transaction from Form 4."""

    security_title: str
    transaction_date: date
    transaction_code: str  # P=Purchase, S=Sale, A=Grant, M=Exercise, etc.
    shares: Decimal
    price_per_share: Decimal | None
    acquired_disposed: str  # A=Acquired, D=Disposed
    shares_owned_after: Decimal | None = None
    direct_indirect: str = "D"  # D=Direct, I=Indirect
    footnotes: list[str] = field(default_factory=list)

    @property
    def transaction_type(self) -> str:
        """Human-readable transaction type."""
        codes = {
            "P": "Purchase",
            "S": "Sale",
            "A": "Grant/Award",
            "M": "Exercise/Conversion",
            "C": "Conversion",
            "G": "Gift",
            "F": "Tax Withholding",
            "J": "Other",
            "K": "Equity Swap",
            "U": "Disposition to Issuer",
            "X": "Exercise of Out-of-Money",
            "W": "Exercise of In-Money",
        }
        return codes.get(self.transaction_code, f"Other ({self.transaction_code})")

    @property
    def total_value(self) -> Decimal | None:
        """Total transaction value if price is known."""
        if self.price_per_share is not None:
            return self.shares * self.price_per_share
        return None

    @property
    def is_buy(self) -> bool:
        """Is this a purchase/acquisition?"""
        return self.acquired_disposed == "A" and self.transaction_code in (
            "P",
            "A",
            "M",
            "C",
            "W",
            "X",
        )

    @property
    def is_sell(self) -> bool:
        """Is this a sale/disposition?"""
        return self.acquired_disposed == "D" or self.transaction_code in ("S", "F", "G", "U")


@dataclass
class Holding:
    """A non-derivative or derivative holding (not from a transaction)."""

    security_title: str
    shares: Decimal
    direct_indirect: str = "D"
    nature_of_ownership: str = ""


@dataclass
class ParsedForm4:
    """Complete parsed Form 4 filing."""

    issuer_name: str
    issuer_ticker: str
    issuer_cik: str
    insider: InsiderInfo
    filing_date: date
    period_of_report: date | None
    transactions: list[Transaction]
    holdings: list[Holding]
    source_path: str

    @property
    def net_shares(self) -> Decimal:
        """Net shares acquired (positive) or disposed (negative)."""
        total = Decimal(0)
        for t in self.transactions:
            if t.acquired_disposed == "A":
                total += t.shares
            else:
                total -= t.shares
        return total

    @property
    def total_value_transacted(self) -> Decimal:
        """Total dollar value of all transactions."""
        total = Decimal(0)
        for t in self.transactions:
            if t.total_value:
                total += t.total_value
        return total

    def filter_transactions(
        self,
        transaction_type: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[Transaction]:
        """Filter transactions by type and/or date range."""
        result = self.transactions

        if transaction_type:
            type_lower = transaction_type.lower()
            if type_lower in ("buy", "purchase"):
                result = [t for t in result if t.is_buy]
            elif type_lower in ("sell", "sale"):
                result = [t for t in result if t.is_sell]
            else:
                result = [t for t in result if t.transaction_code == transaction_type]

        if start_date:
            result = [t for t in result if t.transaction_date >= start_date]

        if end_date:
            result = [t for t in result if t.transaction_date <= end_date]

        return result


class Form4Parser(BaseParser["ParsedForm4"]):
    """Parser for SEC Form 4 XML filings."""

    # XML namespaces used in Form 4
    NAMESPACES = {
        "": "http://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=",
    }

    def parse_file(self, file_path: Path) -> ParsedForm4:
        """
        Parse a Form 4 XML file.

        Args:
            file_path: Path to Form 4 XML file

        Returns:
            ParsedForm4 with extracted data
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Form 4 file not found: {file_path}")

        content = file_path.read_text(encoding="utf-8", errors="replace")
        return self.parse_xml(content, str(file_path))

    def parse_xml(self, xml_content: str, source_path: str = "") -> ParsedForm4:
        """
        Parse Form 4 XML content.

        Args:
            xml_content: Raw XML string
            source_path: Source file path for reference

        Returns:
            ParsedForm4 with extracted data
        """
        # Clean up XML (some Form 4s have issues)
        xml_content = self._clean_xml(xml_content)

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            raise ValueError(f"Invalid Form 4 XML: {e}")

        # Extract issuer info
        issuer_name = self._get_text(root, ".//issuerName", "")
        issuer_ticker = self._get_text(root, ".//issuerTradingSymbol", "")
        issuer_cik = self._get_text(root, ".//issuerCik", "")

        # Extract insider info
        insider = self._parse_insider(root)

        # Extract dates
        filing_date = self._parse_date(self._get_text(root, ".//periodOfReport", ""))
        period_of_report = filing_date  # Usually the same

        # Extract transactions (non-derivative)
        transactions = []
        for txn_elem in root.findall(".//nonDerivativeTransaction"):
            txn = self._parse_transaction(txn_elem)
            if txn:
                transactions.append(txn)

        # Extract derivative transactions
        for txn_elem in root.findall(".//derivativeTransaction"):
            txn = self._parse_derivative_transaction(txn_elem)
            if txn:
                transactions.append(txn)

        # Extract holdings (non-transaction)
        holdings = []
        for hold_elem in root.findall(".//nonDerivativeHolding"):
            hold = self._parse_holding(hold_elem)
            if hold:
                holdings.append(hold)

        return ParsedForm4(
            issuer_name=issuer_name,
            issuer_ticker=issuer_ticker,
            issuer_cik=issuer_cik,
            insider=insider,
            filing_date=filing_date or date.today(),
            period_of_report=period_of_report,
            transactions=transactions,
            holdings=holdings,
            source_path=source_path,
        )

    def _clean_xml(self, xml_content: str) -> str:
        """Clean up common XML issues in Form 4 filings."""
        # Remove XML declaration if malformed
        xml_content = re.sub(r"<\?xml[^>]*\?>", "", xml_content)
        # Remove DOCTYPE if present
        xml_content = re.sub(r"<!DOCTYPE[^>]*>", "", xml_content)
        # Ensure we start with the root element
        match = re.search(r"<ownershipDocument", xml_content)
        if match:
            xml_content = xml_content[match.start() :]
        return xml_content.strip()

    def _get_text(self, elem: ET.Element, path: str, default: str = "") -> str:
        """Get text content from an element path."""
        found = elem.find(path)
        if found is not None and found.text:
            return found.text.strip()
        return default

    def _parse_date(self, date_str: str) -> date | None:
        """Parse date string in various formats."""
        if not date_str:
            return None

        formats = ["%Y-%m-%d", "%m/%d/%Y", "%Y%m%d"]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue

        return None

    def _parse_decimal(self, value_str: str) -> Decimal | None:
        """Parse a decimal value, handling various formats."""
        if not value_str:
            return None

        try:
            # Remove commas and whitespace
            cleaned = value_str.replace(",", "").replace(" ", "").strip()
            if not cleaned:
                return None
            return Decimal(cleaned)
        except (InvalidOperation, ValueError) as e:
            logger.debug(f"Could not parse decimal '{value_str}': {e}")
            return None

    def _parse_insider(self, root: ET.Element) -> InsiderInfo:
        """Parse insider/reporting owner information."""
        # Try to find owner info
        owner_elem = root.find(".//reportingOwner")
        if owner_elem is None:
            owner_elem = root

        # Get name
        name = self._get_text(owner_elem, ".//rptOwnerName", "")
        if not name:
            # Try alternative paths
            name = self._get_text(root, ".//reportingOwnerName", "")

        cik = self._get_text(owner_elem, ".//rptOwnerCik", "")

        # Get relationship flags
        relationship = owner_elem.find(".//reportingOwnerRelationship")
        is_director = False
        is_officer = False
        is_ten_percent = False
        is_other = False
        officer_title = ""

        if relationship is not None:
            is_director = self._get_text(relationship, "isDirector", "0") == "1"
            is_officer = self._get_text(relationship, "isOfficer", "0") == "1"
            is_ten_percent = self._get_text(relationship, "isTenPercentOwner", "0") == "1"
            is_other = self._get_text(relationship, "isOther", "0") == "1"
            officer_title = self._get_text(relationship, "officerTitle", "")

        return InsiderInfo(
            name=name,
            cik=cik,
            is_director=is_director,
            is_officer=is_officer,
            is_ten_percent_owner=is_ten_percent,
            is_other=is_other,
            officer_title=officer_title,
        )

    def _parse_transaction_base(self, txn_elem: ET.Element) -> _TransactionBase | None:
        """Extract common transaction fields (shared by derivative and non-derivative).

        Returns None if transaction date cannot be parsed.
        """
        security_title = self._get_text(txn_elem, ".//securityTitle/value", "")

        # Transaction date
        txn_date_str = self._get_text(txn_elem, ".//transactionDate/value", "")
        txn_date = self._parse_date(txn_date_str)
        if not txn_date:
            return None

        # Transaction coding
        coding = txn_elem.find(".//transactionCoding")
        txn_code = ""
        if coding is not None:
            txn_code = self._get_text(coding, "transactionCode", "")

        # Transaction amounts
        amounts = txn_elem.find(".//transactionAmounts")
        shares = Decimal(0)
        price = None
        acquired_disposed = "A"

        if amounts is not None:
            shares = self._parse_decimal(self._get_text(amounts, ".//transactionShares/value", "0")) or Decimal(0)
            price = self._parse_decimal(self._get_text(amounts, ".//transactionPricePerShare/value", ""))
            acquired_disposed = self._get_text(amounts, ".//transactionAcquiredDisposedCode/value", "A")

        return _TransactionBase(
            security_title=security_title,
            transaction_date=txn_date,
            transaction_code=txn_code,
            shares=shares,
            price_per_share=price,
            acquired_disposed=acquired_disposed,
        )

    def _parse_transaction(self, txn_elem: ET.Element) -> Transaction | None:
        """Parse a non-derivative transaction element."""
        base = self._parse_transaction_base(txn_elem)
        if base is None:
            return None

        # Post-transaction holdings (non-derivative specific)
        post_amounts = txn_elem.find(".//postTransactionAmounts")
        shares_after = None
        if post_amounts is not None:
            shares_after = self._parse_decimal(
                self._get_text(post_amounts, ".//sharesOwnedFollowingTransaction/value", "")
            )

        # Ownership nature (non-derivative specific)
        ownership = txn_elem.find(".//ownershipNature")
        direct_indirect = "D"
        if ownership is not None:
            direct_indirect = self._get_text(ownership, "directOrIndirectOwnership/value", "D")

        return Transaction(
            security_title=base.security_title,
            transaction_date=base.transaction_date,
            transaction_code=base.transaction_code,
            shares=base.shares,
            price_per_share=base.price_per_share,
            acquired_disposed=base.acquired_disposed,
            shares_owned_after=shares_after,
            direct_indirect=direct_indirect,
        )

    def _parse_derivative_transaction(self, txn_elem: ET.Element) -> Transaction | None:
        """Parse a derivative transaction element."""
        base = self._parse_transaction_base(txn_elem)
        if base is None:
            return None

        # For derivatives, also check exercise/conversion price if not in amounts
        price = base.price_per_share
        if price is None:
            price = self._parse_decimal(self._get_text(txn_elem, ".//conversionOrExercisePrice/value", ""))

        return Transaction(
            security_title=base.security_title,
            transaction_date=base.transaction_date,
            transaction_code=base.transaction_code,
            shares=base.shares,
            price_per_share=price,
            acquired_disposed=base.acquired_disposed,
            direct_indirect="D",
        )

    def _parse_holding(self, hold_elem: ET.Element) -> Holding | None:
        """Parse a non-derivative holding element."""
        security_title = self._get_text(hold_elem, ".//securityTitle/value", "")

        post_amounts = hold_elem.find(".//postTransactionAmounts")
        shares = Decimal(0)
        if post_amounts is not None:
            shares = self._parse_decimal(
                self._get_text(post_amounts, ".//sharesOwnedFollowingTransaction/value", "0")
            ) or Decimal(0)

        ownership = hold_elem.find(".//ownershipNature")
        direct_indirect = "D"
        nature = ""
        if ownership is not None:
            direct_indirect = self._get_text(ownership, "directOrIndirectOwnership/value", "D")
            nature = self._get_text(ownership, "natureOfOwnership/value", "")

        return Holding(
            security_title=security_title,
            shares=shares,
            direct_indirect=direct_indirect,
            nature_of_ownership=nature,
        )


def parse_form4(file_path: Path) -> ParsedForm4:
    """
    Convenience function to parse a Form 4 file.

    Args:
        file_path: Path to Form 4 XML file

    Returns:
        ParsedForm4 with extracted data
    """
    parser = Form4Parser()
    return parser.parse_file(file_path)
