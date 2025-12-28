# Roadmap

Future improvements and planned features for the Stock Pitch Case Template.

## RAG Pipeline Enhancements

### Semantic Chunking (Medium Priority)

Create chunks at semantic boundaries rather than arbitrary token counts.

**Goal**: More coherent chunks with better embeddings.

**Approach**:

- Identify logical units (complete thoughts, arguments, tables)
- Keep related content together
- Split only at major semantic boundaries
- Variable-size chunks based on content

### Multi-Hop Retrieval (Low Priority)

Decompose complex questions and perform iterative retrieval.

**Use cases**:

- "How has gross margin changed over the last 3 years and what were the drivers?"
- "Compare capital allocation priorities in 2023 vs 2024"

**Approach**:

- Step 1: Decompose question using LLM
- Step 2: Answer each sub-question with RAG
- Step 3: Synthesize final answer from hop results

### Enhanced Chunk Metadata with NER (Low Priority)

Enrich chunks with named entities, metrics, and semantic metadata.

**Benefits**:

- Filter searches: "Find chunks mentioning EBITDA"
- Boost results: Prefer chunks with numbers for metric queries
- Better prompts: Tell Claude what type of content it's seeing

### Reranker Fine-Tuning (Low Priority)

Fine-tune the cross-encoder reranker on domain-specific data.

**Approach**:

- Collect positive pairs from high-scoring search results
- Generate hard negatives (high similarity but wrong answer)
- Fine-tune BGE Reranker on financial document pairs

## Research Agent

- [ ] Continue testing `/research` agent, update prompts based on feedback patterns from `research_feedback.jsonl`
- [ ] Evolve `/research` into MCP server once patterns are understood

## Hybrid Search Tuning

- [ ] Analyze feedback log to determine optimal default strategies per query type
- [ ] Consider making summary-first the default for trend/overview questions

## Performance Optimization

- Query result caching for frequently repeated queries
- Embedding caching to avoid re-computation
- Index partitioning for large indexes by filing type or date

## CI/CD Improvements

### Re-enable Embedding Tests in CI (Medium Priority)

Currently embedding/reranker tests are marked `@pytest.mark.integration` and skipped in CI because:

- BGE-M3 model is ~2GB download
- Reranker model is ~1GB download
- Full PyTorch with CUDA is ~10GB (sentence-transformers pulls it in)

**Goal**: Run embedding tests in CI with CPU-only PyTorch.

**Approach**:

- Configure Poetry to use PyTorch CPU-only index for CI
- Or pre-download models into CI image (adds ~3GB but enables tests)
- Remove `@pytest.mark.integration` from embedding tests once working
