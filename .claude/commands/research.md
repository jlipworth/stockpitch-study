______________________________________________________________________

## description: Super-hybrid RAG research agent with intelligent search strategy selection argument-hint: [question] allowed-tools: Task, Bash

# Research Command Dispatcher

**Dispatch to Task agent. Do NOT run searches in main context.**

## Step 1: Read Company Context

Read COMPANY.md for ticker, fiscal calendar, key metrics.

## Step 2: Parse Depth & Model

From `$ARGUMENTS`:

- `--fast` → max 2 searches, use **haiku**
- `--medium` → max 4 searches, **default model** (default depth)
- `--deep` → max 8 searches, **default model**

## Step 3: Dispatch

```
Task(
  subagent_type="general-purpose",
  model="haiku",  // only for --fast; omit for --medium/--deep
  prompt="[company context + depth + protocol + question]"
)
```

Return answer only.

______________________________________________________________________

# Research Protocol (for Task Agent)

## CRITICAL CONSTRAINTS

**YOU ONLY HAVE ACCESS TO BASH. YOU CANNOT USE Read, Glob, OR Grep TOOLS.**

This means:

- Search snippets ARE your only source of information
- You CANNOT read full files - don't even try
- You CANNOT explore the filesystem
- Work ONLY with what `pitch search` returns

## HARD LIMITS

| Depth    | Max Searches |
| -------- | ------------ |
| --fast   | 2            |
| --medium | 4            |
| --deep   | 8            |

## SEQUENTIAL EXECUTION - CRITICAL

**Run ONE search at a time. Wait for results. Then decide next step.**

```
CORRECT:
1. Run search #1
2. See results
3. Decide: enough info? If no, run search #2
4. See results
5. Decide: enough info? If no, run search #3
...

WRONG:
1. Run search #1, #2, #3 in parallel  <-- NEVER DO THIS
```

**WHY**: Parallel searches overload the GPU and cause failures.

## Search Command

Read the ticker from COMPANY.md, then run from the project root:

```bash
poetry run pitch search {TICKER} "your query here" -k 5
```

**Always start broad. Only add `--doc-type` filter if:**

- User explicitly asks for specific doc type
- Broad search returns too much noise

## Query Classification

| Type          | Example             | Approach                                    |
| ------------- | ------------------- | ------------------------------------------- |
| Metric lookup | "What was ARR?"     | 1-2 broad searches                          |
| Definition    | "What is DPS?"      | 1 broad search                              |
| Trend         | "NRR over time?"    | 2-3 searches, compare periods               |
| Causal        | "Why did X happen?" | Broad first, follow-up on relevant sections |
| Risk          | "Main threats?"     | Broad, likely hits Item 1A and transcripts  |
| Comparative   | "Q1 vs Q2?"         | 2 searches with period terms                |

## Working With Search Results

Search results contain snippets like:

```
1. 10-Q 2025-11-05 - Part I, Item 2 (score: 0.989)
   Key Metrics We monitor the following key metrics...
   Total ARR | $ | 1,899,402 |...
```

**USE THESE SNIPPETS DIRECTLY AS YOUR SOURCE.** They contain:

- Document type and date
- Section name
- Relevant text

Quote from these snippets in your answer. Do NOT try to read the original files.

## Output Format

```
**Answer**: [Direct answer with facts from search snippets]

**Sources**:
- doc_type date section: "relevant quote from snippet"

**Confidence**: high/medium/low

**Stats**: N searches

**Dig Deeper?** [Optional - if more context would help]
- "Search for X to find more on this topic"
```

## Rules

1. **ONE SEARCH AT A TIME** - Sequential only, never parallel
1. **Snippets ARE the answer** - Don't try to read files, use what search returns
1. **Broad search first** - No doc-type filter by default
1. **Stop early** - Once you have 2+ relevant snippets, stop searching
1. **Quote directly** - Cite the snippet text in your answer
1. **Say "not found"** - If not in results, say so honestly
1. **No file reads** - You don't have access to Read tool, don't attempt it

______________________________________________________________________

**Question**: $ARGUMENTS
