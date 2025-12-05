# Excel Model Workflow

This document describes how to use Claude Code to build and modify Excel financial models for stock pitch case studies.

## Overview

The workflow integrates:

1. **Tegus Model** - Third-party 3-statement model with detailed projections
1. **Custom Worksheets** - DCF, APV, Tear Sheet, Comps Table, Drivers
1. **CapIQ Formulas** - Real-time market data integration

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DRIVERS SHEET                           │
│  Scenario Selector (Bull/Base/Bear) → CHOOSE/MATCH formulas    │
│  Key Assumptions: Tax Rate, Risk-Free, ERP, Terminal Growth    │
│  Annual Projections: Revenue Growth, Margins, CapEx %, NWC %   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                         MODEL SHEET                             │
│  Tegus 3-Statement Model (1,400+ rows)                         │
│  - Income Statement (rows 1-500)                               │
│  - Balance Sheet (rows 1100-1220)                              │
│  - Cash Flow Statement (rows 1050-1100)                        │
│  Column Structure: Historical | Quarterly | Annual FY          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        TEAR SHEET                               │
│  One-Page Financial Summary                                     │
│  - Historical: CIQ formulas (FY24, FY25)                       │
│  - Projected: Model references (FY26-FY30)                     │
│  - KPIs: ARR, NRR, Customers (CIQ for consensus)               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│      DCF        │     │   COMPS TABLE   │
│  Two-Stage      │     │  Trading Comps  │
│  - Stage 1:     │     │  - CIQ formulas │
│    FY27-FY31    │     │  - Peer group   │
│  - Stage 2:     │     │  - Mean/Median  │
│    Fade Period  │     │                 │
└─────────────────┘     └─────────────────┘
```

## Working with Excel via Python

### Library: openpyxl

```python
import openpyxl
from openpyxl.utils import get_column_letter, column_index_from_string
from openpyxl.styles import Font, Alignment, Border

# Load workbook (preserve formulas)
wb = openpyxl.load_workbook("Model.xlsx", data_only=False)
ws = wb["Sheet Name"]

# Access cells
cell = ws["A1"]
cell = ws.cell(row=1, column=1)

# Read/write values and formulas
value = cell.value  # Returns formula string if formula, else value
cell.value = "=A1+B1"  # Set formula
cell.value = 100  # Set value

# Save
wb.save("Model.xlsx")
wb.close()
```

### Key Patterns

**1. Find Row by Label**

```python
def find_row_by_label(ws, search_term, column=1):
    """Search column A for a label containing search_term"""
    for row in range(1, ws.max_row + 1):
        label = ws.cell(row=row, column=column).value
        if label and search_term.lower() in str(label).lower():
            return row
    return None
```

**2. Column Letter Conversion**

```python
# Letter to index
col_idx = column_index_from_string('BP')  # Returns 68

# Index to letter
col_letter = get_column_letter(68)  # Returns 'BP'
```

**3. Bulk Formula Updates**

```python
# Replace CIQ formulas with Model references
for col in ['D', 'E', 'F', 'G', 'H']:
    model_col = col_mapping[col]
    for ts_row, model_row in row_mapping.items():
        ws[f"{col}{ts_row}"] = f"=Model!{model_col}{model_row}"
```

## Workflow Steps

### Phase 1: DCF/APV Fixes

Common issues and fixes:

| Issue                    | Fix                                           |
| ------------------------ | --------------------------------------------- |
| Wrong cell references    | Update to correct Drivers row/column          |
| Hardcoded values         | Replace with `=Drivers!$C$6` style references |
| Missing WACC build-up    | Add Rf + Beta × ERP formula                   |
| NWC as absolute vs delta | Change to `=(Current NWC) - (Prior NWC)`      |

**WACC Build-up Pattern:**

```
Cost of Equity = Risk-Free + Beta × ERP
WACC = (E/V) × Ke + (D/V) × Kd × (1-t)
```

**Two-Stage DCF with Fade:**

- Stage 1 (FY27-FY31): Explicit Model projections
- Stage 2 (FY32-FY36): Growth fades linearly to terminal rate
- Terminal Value: Exit Multiple × Terminal EBITDA

### Phase 2: Connect Tear Sheet to Model

**Column Mapping (Tegus Model):**

| Tear Sheet | Fiscal Year | Model Column |
| ---------- | ----------- | ------------ |
| D          | FY26        | BP           |
| E          | FY27        | BU           |
| F          | FY28        | BZ           |
| G          | FY29        | CE           |
| H          | FY30        | CJ           |
| I          | FY31        | BY           |

**Note:** Model uses 5-column quarters (e.g., BL-BP = FY26 Q1-Q4 + Annual). FY31 annual is BY, not CO.

**Row Mapping:**

| Tear Sheet Row | Metric           | Model Row |
| -------------- | ---------------- | --------- |
| 7              | Revenue          | 136       |
| 10             | Gross Profit     | 163       |
| 14             | S&M Expense      | 191       |
| 16             | R&D Expense      | 178       |
| 18             | G&A Expense      | 204       |
| 22             | Operating Income | 219       |
| 25             | D&A              | 1059+1060 |
| 29             | Interest         | 552       |
| 31             | Taxes            | 559       |
| 36             | SBC              | 264       |
| 40             | Cash             | 1161      |
| 41             | Total Debt       | 1198      |
| 43             | Deferred Revenue | 1187      |
| 44             | Total Assets     | 1179      |
| 45             | Total Equity     | 1207      |
| 48             | Cash from Ops    | 1075      |
| 49             | CapEx            | 933       |
| 62             | Diluted EPS      | 580       |
| 63             | Diluted Shares   | 586       |

**Calculated rows (keep formulas):**

- Row 9 (COGS): `=Revenue - Gross Profit`
- Row 20 (Total OpEx): `=S&M + R&D + G&A`
- Row 26 (EBITDA): `=Operating Income + D&A`
- Row 33 (Net Income): `=Pre-Tax - Taxes`
- Row 50 (FCF): `=CFO - ABS(CapEx)`

### Phase 3: Comps Table Fixes

Common issues:

1. **Blank rows** - Delete with `ws.delete_rows(row_num)`
1. **Wrong CIQ references** - Fix `$A6` to `$A{current_row}`
1. **Font inconsistency** - Standardize to Book Antiqua
1. **Units** - Market Cap/EV in billions (`/1000`)
1. **Mean/Median ranges** - Update after row deletions

**CIQ Formula Pattern:**

```
=ciq($A5,"IQ_MARKETCAP")                    # Basic (current)
=CIQ($A5,"IQ_TOTAL_REV",IQ_LTM)             # With period constant
=CIQ($A5,"IQ_TOTAL_REV","FY2024")           # Historical (4-digit year)
=CIQ($A5,"IQ_TOTAL_REV_EST","FY2026")       # Estimate (_EST suffix)
=IFERROR(CIQ(...),"-")                      # With error handling
```

**CIQ Period Format Rules:**

- Always use 4-digit years in quotes: `"FY2024"`, `"FY2025"`, `"FY2026"`
- Historical data: Use item code as-is (e.g., `IQ_TOTAL_REV`)
- Forward estimates: Add `_EST` suffix (e.g., `IQ_TOTAL_REV_EST`, `IQ_EBITDA_EST`)
- Period constants: `IQ_LTM`, `IQ_NTM`, `IQ_FY0`, `IQ_FY1` (no quotes)

### Phase 4: Scenario-Driven Model

**Drivers Sheet Structure:**

```
Row 3: Scenario Selector ("Bull", "Base", "Bear")
Row 6-10: Static Assumptions (Tax, Rf, ERP, Terminal Growth, Exit Multiple)
Row 15-22: Active Drivers (pull from scenario tables via CHOOSE/MATCH)
Row 28-35: Bull Case Table
Row 38-45: Base Case Table
Row 48-55: Bear Case Table
```

**CHOOSE/MATCH Pattern:**

```
=CHOOSE(MATCH($C$3,{"Bull","Base","Bear"},0), BullValue, BaseValue, BearValue)
```

## Formatting Standards

### Font

- **Primary**: Book Antiqua
- **Headers**: Bold
- **Inputs**: Blue font (hardcoded assumptions)

### Number Formats

- Currency: `#,##0.0` or `$#,##0.0`
- Percentages: `0.0%`
- Multiples: `0.0x`
- Large numbers: Billions for Market Cap/EV

### Structure

- Freeze panes on headers
- Print area set
- Double underline on totals

## Backup Strategy

Create versioned backups before major changes:

```bash
cp Model.xlsx Model_v{N}_pre_{change}.xlsx
```

Example sequence:

- `v5_pre_dcf_fix.xlsx`
- `v6_pre_wacc_fix.xlsx`
- `v7_pre_fade.xlsx`
- `v8_pre_nwc_fix.xlsx`
- `v9_pre_sheet_rename.xlsx`
- `v10_pre_comps_fix.xlsx`
- `v11_pre_tearsheet_model.xlsx`
- `v14_pre_ciq_period_fix.xlsx`
- `v15_pre_FY31.xlsx`

## Validation Checklist

**Philosophy:** Fail-fast on structural checks. Company-specific ranges validated manually.

### Critical Structural Checks (Automated)

- [ ] Balance sheet balances: Total Assets = Total Liabilities + Equity
- [ ] Cash flow reconciliation: Beginning Cash + Net Change = Ending Cash
- [ ] FCF = CFO - CapEx
- [ ] No formula errors (#REF!, #VALUE!, #DIV/0!)

### Manual Sanity Checks (Company-Specific)

- [ ] Share price in reasonable range for thesis
- [ ] WACC appropriate for company risk
- [ ] Growth rates align with investment thesis
- [ ] Margins consistent with company/industry history

### Comps Table Validation

- [ ] All peers pull correct ticker
- [ ] Mean/Median exclude subject company
- [ ] Units consistent (billions)

## Common Tegus Model Row Locations

These are typical for Tegus SaaS models (verify for each company):

| Section       | Row Range | Key Rows                                               |
| ------------- | --------- | ------------------------------------------------------ |
| Revenue       | 20-140    | Total Revenue: ~136                                    |
| Gross Profit  | 95-175    | GAAP GP: ~163                                          |
| OpEx          | 175-235   | S&M: ~191, R&D: ~178, G&A: ~204, Op Inc: ~219          |
| SBC           | 260-280   | Total SBC: ~264                                        |
| EBITDA        | 540-560   | Non-GAAP EBITDA: ~549                                  |
| EPS           | 575-590   | Adj EPS: ~580, Shares: ~586                            |
| Cash Flow     | 1055-1090 | D&A: 1059+1060, CFO: ~1075, CapEx: ~933                |
| Balance Sheet | 1160-1210 | Cash: ~1161, Debt: ~1198, Assets: ~1179, Equity: ~1207 |

## Tips for Claude Code Usage

1. **Always create backups** before modifying Excel files
1. **Read before writing** - Understand structure first
1. **Use `data_only=False`** to preserve formulas
1. **Verify after changes** - Read back and check
1. **Document row mappings** in separate MD file for reference
1. **Dispatch agents** for complex analysis tasks
1. **Incremental saves** - Save after each phase
