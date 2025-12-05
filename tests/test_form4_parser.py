"""Tests for Form 4 XML parser."""

from datetime import date
from decimal import Decimal
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from src.filings.form4_parser import (
    Form4Parser,
    ParsedForm4,
    parse_form4,
)

# Sample Form 4 XML - CEO stock sale
SAMPLE_FORM4_SALE = """<?xml version="1.0"?>
<ownershipDocument>
    <schemaVersion>X0306</schemaVersion>
    <documentType>4</documentType>
    <periodOfReport>2024-01-15</periodOfReport>
    <issuer>
        <issuerCik>0000320193</issuerCik>
        <issuerName>Apple Inc.</issuerName>
        <issuerTradingSymbol>AAPL</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001214156</rptOwnerCik>
            <rptOwnerName>COOK TIMOTHY D</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>1</isDirector>
            <isOfficer>1</isOfficer>
            <isTenPercentOwner>0</isTenPercentOwner>
            <isOther>0</isOther>
            <officerTitle>Chief Executive Officer</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle>
                <value>Common Stock</value>
            </securityTitle>
            <transactionDate>
                <value>2024-01-15</value>
            </transactionDate>
            <transactionCoding>
                <transactionFormType>4</transactionFormType>
                <transactionCode>S</transactionCode>
                <equitySwapInvolved>0</equitySwapInvolved>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares>
                    <value>50000</value>
                </transactionShares>
                <transactionPricePerShare>
                    <value>185.50</value>
                </transactionPricePerShare>
                <transactionAcquiredDisposedCode>
                    <value>D</value>
                </transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction>
                    <value>3280557</value>
                </sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
            <ownershipNature>
                <directOrIndirectOwnership>
                    <value>D</value>
                </directOrIndirectOwnership>
            </ownershipNature>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>
"""

# Sample Form 4 XML - CFO stock purchase
SAMPLE_FORM4_PURCHASE = """<?xml version="1.0"?>
<ownershipDocument>
    <periodOfReport>2024-02-20</periodOfReport>
    <issuer>
        <issuerCik>0000320193</issuerCik>
        <issuerName>Apple Inc.</issuerName>
        <issuerTradingSymbol>AAPL</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001234567</rptOwnerCik>
            <rptOwnerName>MAESTRI LUCA</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>0</isDirector>
            <isOfficer>1</isOfficer>
            <isTenPercentOwner>0</isTenPercentOwner>
            <isOther>0</isOther>
            <officerTitle>CFO</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle>
                <value>Common Stock</value>
            </securityTitle>
            <transactionDate>
                <value>2024-02-20</value>
            </transactionDate>
            <transactionCoding>
                <transactionCode>P</transactionCode>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares>
                    <value>10000</value>
                </transactionShares>
                <transactionPricePerShare>
                    <value>175.25</value>
                </transactionPricePerShare>
                <transactionAcquiredDisposedCode>
                    <value>A</value>
                </transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction>
                    <value>150000</value>
                </sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>
"""

# Sample Form 4 with stock option exercise
SAMPLE_FORM4_OPTION_EXERCISE = """<?xml version="1.0"?>
<ownershipDocument>
    <periodOfReport>2024-03-10</periodOfReport>
    <issuer>
        <issuerCik>0000320193</issuerCik>
        <issuerName>Apple Inc.</issuerName>
        <issuerTradingSymbol>AAPL</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerName>DOE JANE</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>1</isDirector>
            <isOfficer>0</isOfficer>
            <isTenPercentOwner>0</isTenPercentOwner>
        </reportingOwnerRelationship>
    </reportingOwner>
    <derivativeTable>
        <derivativeTransaction>
            <securityTitle>
                <value>Stock Option (Right to Buy)</value>
            </securityTitle>
            <conversionOrExercisePrice>
                <value>120.00</value>
            </conversionOrExercisePrice>
            <transactionDate>
                <value>2024-03-10</value>
            </transactionDate>
            <transactionCoding>
                <transactionCode>M</transactionCode>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares>
                    <value>25000</value>
                </transactionShares>
                <transactionAcquiredDisposedCode>
                    <value>D</value>
                </transactionAcquiredDisposedCode>
            </transactionAmounts>
        </derivativeTransaction>
    </derivativeTable>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle>
                <value>Common Stock</value>
            </securityTitle>
            <transactionDate>
                <value>2024-03-10</value>
            </transactionDate>
            <transactionCoding>
                <transactionCode>M</transactionCode>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares>
                    <value>25000</value>
                </transactionShares>
                <transactionPricePerShare>
                    <value>120.00</value>
                </transactionPricePerShare>
                <transactionAcquiredDisposedCode>
                    <value>A</value>
                </transactionAcquiredDisposedCode>
            </transactionAmounts>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>
"""

# Sample Form 4 with multiple transactions
SAMPLE_FORM4_MULTIPLE = """<?xml version="1.0"?>
<ownershipDocument>
    <periodOfReport>2024-04-01</periodOfReport>
    <issuer>
        <issuerCik>0000789123</issuerCik>
        <issuerName>Test Corp</issuerName>
        <issuerTradingSymbol>TEST</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerName>SMITH JOHN</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>0</isDirector>
            <isOfficer>1</isOfficer>
            <officerTitle>SVP Sales</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2024-04-01</value></transactionDate>
            <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>5000</value></transactionShares>
                <transactionPricePerShare><value>50.00</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2024-04-02</value></transactionDate>
            <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>3000</value></transactionShares>
                <transactionPricePerShare><value>52.00</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2024-04-03</value></transactionDate>
            <transactionCoding><transactionCode>A</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>10000</value></transactionShares>
                <transactionPricePerShare><value>0</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>
"""

# Sample with 10% owner
SAMPLE_FORM4_TEN_PERCENT_OWNER = """<?xml version="1.0"?>
<ownershipDocument>
    <periodOfReport>2024-05-15</periodOfReport>
    <issuer>
        <issuerName>Small Corp</issuerName>
        <issuerTradingSymbol>SMLL</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerName>BIG FUND LP</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>0</isDirector>
            <isOfficer>0</isOfficer>
            <isTenPercentOwner>1</isTenPercentOwner>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2024-05-15</value></transactionDate>
            <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>1000000</value></transactionShares>
                <transactionPricePerShare><value>12.50</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>
"""


class TestForm4ParserBasics:
    """Test basic Form 4 parsing functionality."""

    def test_parse_sale_transaction(self):
        """Test parsing a stock sale."""
        parser = Form4Parser()
        result = parser.parse_xml(SAMPLE_FORM4_SALE)

        assert isinstance(result, ParsedForm4)
        assert result.issuer_name == "Apple Inc."
        assert result.issuer_ticker == "AAPL"
        assert result.insider.name == "COOK TIMOTHY D"
        assert len(result.transactions) == 1

    def test_parse_purchase_transaction(self):
        """Test parsing a stock purchase."""
        parser = Form4Parser()
        result = parser.parse_xml(SAMPLE_FORM4_PURCHASE)

        assert result.insider.name == "MAESTRI LUCA"
        assert result.insider.officer_title == "CFO"
        assert len(result.transactions) == 1
        assert result.transactions[0].transaction_code == "P"
        assert result.transactions[0].is_buy

    def test_parse_option_exercise(self):
        """Test parsing option exercise with derivative transaction."""
        parser = Form4Parser()
        result = parser.parse_xml(SAMPLE_FORM4_OPTION_EXERCISE)

        assert result.insider.name == "DOE JANE"
        assert result.insider.is_director
        assert not result.insider.is_officer
        # Should have both derivative and non-derivative transactions
        assert len(result.transactions) == 2


class TestInsiderInfo:
    """Test insider information extraction."""

    def test_ceo_info(self):
        """Test CEO information extraction."""
        parser = Form4Parser()
        result = parser.parse_xml(SAMPLE_FORM4_SALE)

        insider = result.insider
        assert insider.name == "COOK TIMOTHY D"
        assert insider.is_director
        assert insider.is_officer
        assert not insider.is_ten_percent_owner
        assert insider.officer_title == "Chief Executive Officer"
        assert "Chief Executive Officer" in insider.role

    def test_cfo_info(self):
        """Test CFO information extraction."""
        parser = Form4Parser()
        result = parser.parse_xml(SAMPLE_FORM4_PURCHASE)

        insider = result.insider
        assert insider.is_officer
        assert not insider.is_director
        assert insider.officer_title == "CFO"

    def test_director_only(self):
        """Test director (non-officer) information."""
        parser = Form4Parser()
        result = parser.parse_xml(SAMPLE_FORM4_OPTION_EXERCISE)

        insider = result.insider
        assert insider.is_director
        assert not insider.is_officer
        assert "Director" in insider.role

    def test_ten_percent_owner(self):
        """Test 10% owner information."""
        parser = Form4Parser()
        result = parser.parse_xml(SAMPLE_FORM4_TEN_PERCENT_OWNER)

        insider = result.insider
        assert insider.is_ten_percent_owner
        assert not insider.is_officer
        assert not insider.is_director
        assert "10% Owner" in insider.role


class TestTransactionDetails:
    """Test transaction detail extraction."""

    def test_sale_details(self):
        """Test sale transaction details."""
        parser = Form4Parser()
        result = parser.parse_xml(SAMPLE_FORM4_SALE)

        txn = result.transactions[0]
        assert txn.security_title == "Common Stock"
        assert txn.transaction_date == date(2024, 1, 15)
        assert txn.transaction_code == "S"
        assert txn.shares == Decimal("50000")
        assert txn.price_per_share == Decimal("185.50")
        assert txn.acquired_disposed == "D"
        assert txn.shares_owned_after == Decimal("3280557")
        assert txn.is_sell
        assert not txn.is_buy

    def test_purchase_details(self):
        """Test purchase transaction details."""
        parser = Form4Parser()
        result = parser.parse_xml(SAMPLE_FORM4_PURCHASE)

        txn = result.transactions[0]
        assert txn.transaction_code == "P"
        assert txn.shares == Decimal("10000")
        assert txn.price_per_share == Decimal("175.25")
        assert txn.acquired_disposed == "A"
        assert txn.is_buy
        assert not txn.is_sell

    def test_transaction_value(self):
        """Test total transaction value calculation."""
        parser = Form4Parser()
        result = parser.parse_xml(SAMPLE_FORM4_SALE)

        txn = result.transactions[0]
        expected_value = Decimal("50000") * Decimal("185.50")
        assert txn.total_value == expected_value

    def test_transaction_type_names(self):
        """Test human-readable transaction type names."""
        parser = Form4Parser()

        sale_result = parser.parse_xml(SAMPLE_FORM4_SALE)
        assert sale_result.transactions[0].transaction_type == "Sale"

        purchase_result = parser.parse_xml(SAMPLE_FORM4_PURCHASE)
        assert purchase_result.transactions[0].transaction_type == "Purchase"


class TestMultipleTransactions:
    """Test handling of multiple transactions."""

    def test_multiple_transactions_count(self):
        """Test that all transactions are parsed."""
        parser = Form4Parser()
        result = parser.parse_xml(SAMPLE_FORM4_MULTIPLE)

        assert len(result.transactions) == 3

    def test_net_shares_calculation(self):
        """Test net shares calculation across transactions."""
        parser = Form4Parser()
        result = parser.parse_xml(SAMPLE_FORM4_MULTIPLE)

        # -5000 (sale) -3000 (sale) +10000 (grant) = +2000 net
        assert result.net_shares == Decimal("2000")

    def test_total_value_transacted(self):
        """Test total value across all transactions."""
        parser = Form4Parser()
        result = parser.parse_xml(SAMPLE_FORM4_MULTIPLE)

        # 5000*50 + 3000*52 + 10000*0 = 250000 + 156000 + 0 = 406000
        assert result.total_value_transacted == Decimal("406000")

    def test_filter_by_type_sell(self):
        """Test filtering transactions by sell type."""
        parser = Form4Parser()
        result = parser.parse_xml(SAMPLE_FORM4_MULTIPLE)

        sells = result.filter_transactions(transaction_type="sell")
        assert len(sells) == 2
        assert all(t.is_sell for t in sells)

    def test_filter_by_type_buy(self):
        """Test filtering transactions by buy type."""
        parser = Form4Parser()
        result = parser.parse_xml(SAMPLE_FORM4_MULTIPLE)

        buys = result.filter_transactions(transaction_type="buy")
        # The grant (A) is an acquisition
        assert len(buys) == 1

    def test_filter_by_date_range(self):
        """Test filtering transactions by date range."""
        parser = Form4Parser()
        result = parser.parse_xml(SAMPLE_FORM4_MULTIPLE)

        filtered = result.filter_transactions(
            start_date=date(2024, 4, 2),
            end_date=date(2024, 4, 2),
        )
        assert len(filtered) == 1
        assert filtered[0].transaction_date == date(2024, 4, 2)


class TestFileOperations:
    """Test file-based operations."""

    def test_parse_file(self):
        """Test parsing from a file."""
        with NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(SAMPLE_FORM4_SALE)
            f.flush()

            parser = Form4Parser()
            result = parser.parse_file(Path(f.name))

            assert result.issuer_ticker == "AAPL"
            assert result.source_path == f.name

    def test_convenience_function(self):
        """Test parse_form4 convenience function."""
        with NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(SAMPLE_FORM4_PURCHASE)
            f.flush()

            result = parse_form4(Path(f.name))
            assert result.insider.officer_title == "CFO"

    def test_file_not_found(self):
        """Test error handling for missing file."""
        parser = Form4Parser()
        with pytest.raises(FileNotFoundError):
            parser.parse_file(Path("/nonexistent/form4.xml"))


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_malformed_xml_cleanup(self):
        """Test that malformed XML is cleaned up."""
        # XML with some issues that should be handled
        messy_xml = """
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE something>
        Some garbage here
        <ownershipDocument>
            <periodOfReport>2024-01-01</periodOfReport>
            <issuer>
                <issuerName>Test</issuerName>
                <issuerTradingSymbol>TST</issuerTradingSymbol>
            </issuer>
            <reportingOwner>
                <reportingOwnerId>
                    <rptOwnerName>TEST PERSON</rptOwnerName>
                </reportingOwnerId>
            </reportingOwner>
        </ownershipDocument>
        """
        parser = Form4Parser()
        result = parser.parse_xml(messy_xml)
        assert result.issuer_ticker == "TST"

    def test_missing_optional_fields(self):
        """Test handling of missing optional fields."""
        minimal_xml = """
        <ownershipDocument>
            <periodOfReport>2024-01-01</periodOfReport>
            <issuer>
                <issuerName>Test</issuerName>
            </issuer>
            <reportingOwner>
                <reportingOwnerId>
                    <rptOwnerName>PERSON</rptOwnerName>
                </reportingOwnerId>
            </reportingOwner>
        </ownershipDocument>
        """
        parser = Form4Parser()
        result = parser.parse_xml(minimal_xml)

        assert result.issuer_ticker == ""
        assert result.insider.name == "PERSON"
        assert len(result.transactions) == 0

    def test_zero_price_transaction(self):
        """Test handling of transactions with zero price (grants)."""
        parser = Form4Parser()
        result = parser.parse_xml(SAMPLE_FORM4_MULTIPLE)

        grant = [t for t in result.transactions if t.transaction_code == "A"][0]
        assert grant.price_per_share == Decimal("0")
        assert grant.total_value == Decimal("0")


class TestDateParsing:
    """Test date parsing in various formats."""

    def test_standard_date_format(self):
        """Test YYYY-MM-DD format."""
        parser = Form4Parser()
        result = parser.parse_xml(SAMPLE_FORM4_SALE)
        assert result.filing_date == date(2024, 1, 15)

    def test_transaction_dates(self):
        """Test transaction date parsing."""
        parser = Form4Parser()
        result = parser.parse_xml(SAMPLE_FORM4_MULTIPLE)

        dates = [t.transaction_date for t in result.transactions]
        assert date(2024, 4, 1) in dates
        assert date(2024, 4, 2) in dates
        assert date(2024, 4, 3) in dates


class TestDecimalParsing:
    """Test decimal value parsing."""

    def test_integer_shares(self):
        """Test parsing integer share counts."""
        parser = Form4Parser()
        result = parser.parse_xml(SAMPLE_FORM4_SALE)
        assert result.transactions[0].shares == Decimal("50000")

    def test_decimal_price(self):
        """Test parsing decimal prices."""
        parser = Form4Parser()
        result = parser.parse_xml(SAMPLE_FORM4_SALE)
        assert result.transactions[0].price_per_share == Decimal("185.50")

    def test_large_share_count(self):
        """Test parsing large share counts."""
        parser = Form4Parser()
        result = parser.parse_xml(SAMPLE_FORM4_TEN_PERCENT_OWNER)
        assert result.transactions[0].shares == Decimal("1000000")
