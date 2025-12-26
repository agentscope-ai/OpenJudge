# Text Graders

Algorithm-based graders for evaluating text similarity, string matching, and numerical accuracy. These graders don't require LLMs—they rely purely on algorithms and rules, offering fast execution, zero cost, and deterministic results.


## Overview

| Grader | Purpose | Type | Score Range | Key Use Case |
|--------|---------|------|-------------|--------------|
| `SimilarityGrader` | Compute text similarity (15+ algorithms) | Code-Based | [0, 1] | Translation, summarization, answer matching |
| `StringMatchGrader` | String pattern matching (9+ modes) | Code-Based | {0, 1} | Format validation, keyword detection |
| `NumberAccuracyGrader` | Numerical accuracy checks | Code-Based | [0, 1] | Math calculations, data reports |

## SimilarityGrader

Unified text similarity grader supporting multiple mainstream similarity algorithms. Choose the most suitable algorithm based on your scenario.

**When to use:**
- Translation quality assessment (BLEU)
- Text summarization evaluation (ROUGE)
- Answer matching evaluation (F1 Score)
- Semantic similarity computation (Cosine)
- Fuzzy text matching (Fuzzy Match)

**Supported Algorithms:**

| Category | Algorithm | Description | Typical Use Case |
|----------|-----------|-------------|------------------|
| **N-gram Matching** | `bleu` | Standard BLEU, sacrebleu implementation | Machine translation |
| | `sentence_bleu` | Sentence-level BLEU, NLTK implementation | Single sentence translation |
| | `gleu` | Google BLEU, more lenient | Grammar correction |
| | `chrf` | Character-level F-score | Morphologically rich languages |
| **Recall-Oriented** | `rouge1` | Unigram recall | Content coverage |
| | `rouge2` | Bigram recall | Semantic coherence |
| | `rougeL` | Longest common subsequence | Summary quality |
| | `rouge3/4/5` | Higher-order N-grams | Long text matching |
| **Balanced Metrics** | `f1_score` | Token-based F1 | Q&A systems |
| | `meteor` | Considers synonyms and word order | Comprehensive translation quality |
| **Semantic Similarity** | `cosine` | TF-IDF + cosine similarity | Document similarity |
| | `jaccard` | Set-based similarity | Keyword overlap |
| **Fuzzy Matching** | `fuzzy_match` | Levenshtein distance | Spelling tolerance |
| | `edit_distance` | Normalized edit distance | Text difference

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `reference_response` | str | Yes | Reference text |
| `response` | str | Yes | Text to evaluate |
| `algorithm` | str | Yes | Algorithm name (see table above) |
| `normalize` | bool | No | Whether to normalize text (default True) |
| `case_sensitive` | bool | No | Whether case-sensitive (default False) |
| `**kwargs` | Any | No | Algorithm-specific parameters |

**Scoring:**
- Score range: `0.0` - `1.0`
- Specific meaning depends on chosen algorithm
- Generally: `1.0` = perfect match, `0.0` = no match

**Examples:**

### BLEU Algorithm - Machine Translation Evaluation

```python
import asyncio
from openjudge.graders.text.similarity import SimilarityGrader

async def main():
    grader = SimilarityGrader(algorithm="bleu")

    result = await grader.aevaluate(
        reference_response="The cat is on the mat.",
        response="The cat sits on the mat.",
    )

    print(f"Score: {result.score}")  # 0.75-0.85 (good partial match)
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

### ROUGE-L Algorithm - Summarization Quality

```python
import asyncio
from openjudge.graders.text.similarity import SimilarityGrader

async def main():
    grader = SimilarityGrader(algorithm="rougeL")

    # Evaluate summarization quality
    result = await grader.aevaluate(
        reference_response="Artificial intelligence is transforming the technology industry.",
        response="AI is changing tech.",
    )

    print(f"Score: {result.score}")  # Based on longest common subsequence
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

### F1 Score Algorithm - Q&A System Evaluation

```python
import asyncio
from openjudge.graders.text.similarity import SimilarityGrader

async def main():
    grader = SimilarityGrader(algorithm="f1_score", normalize=True)

    result = await grader.aevaluate(
        reference_response="Paris is the capital of France",
        response="The capital of France is Paris",
    )

    print(f"Score: {result.score}")  # ~1.0 (same tokens)
    print(f"Precision: {result.metadata['precision']}")
    print(f"Recall: {result.metadata['recall']}")

asyncio.run(main())
```

### Cosine Similarity Algorithm - Semantic Similarity

```python
import asyncio
from openjudge.graders.text.similarity import SimilarityGrader

async def main():
    grader = SimilarityGrader(algorithm="cosine")

    result = await grader.aevaluate(
        reference_response="machine learning and artificial intelligence",
        response="AI and ML technologies",
        use_tfidf=True,
    )

    print(f"Score: {result.score}")
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

### Fuzzy Match Algorithm - Fuzzy Matching

```python
import asyncio
from openjudge.graders.text.similarity import SimilarityGrader

async def main():
    grader = SimilarityGrader(algorithm="fuzzy_match")

    # Fuzzy matching with spelling tolerance
    result = await grader.aevaluate(
        reference_response="Hello World",
        response="Helo Wrld",
        method="ratio",  # 'ratio', 'partial_ratio', 'token_sort_ratio'
        threshold=0.8,
    )

    print(f"Score: {result.score}")
    print(f"Matched: {result.metadata['matched']}")

asyncio.run(main())
```

**Algorithm-Specific Parameters:**

**BLEU Series:**
- `max_ngram_order` (int): Maximum N-gram order (default 4)
- `smooth_method` (str): Smoothing method (`exp`, `floor`, `add-k`)

**ROUGE Series:**
- `use_stemmer` (bool): Whether to use stemming (default True)
- `score_key` (str): Score type (`fmeasure`, `precision`, `recall`)

**METEOR:**
- `alpha` (float): Precision weight (default 0.9)
- `beta` (float): Recall weight (default 3.0)
- `gamma` (float): Chunking penalty (default 0.5)

**Cosine:**
- `use_tfidf` (bool): Whether to use TF-IDF (default True)
- `ngram_range` (tuple): N-gram range (default (1, 2))
- `max_features` (int): Maximum features (default None)


## StringMatchGrader

Unified string matching grader supporting multiple matching patterns. Use for format validation, keyword detection, and pattern matching.

**When to use:**
- Format validation (email, phone numbers)
- Keyword detection
- Prefix/suffix checking
- Exact answer verification
- Regular expression matching

**Supported Algorithms:**

| Algorithm | Description | Return Value | Typical Use Case |
|-----------|-------------|--------------|------------------|
| `exact_match` | Exact string match | 1.0/0.0 | Answer verification |
| `prefix_match` | Check if response starts with text | 1.0/0.0 | Completion check |
| `suffix_match` | Check if response ends with text | 1.0/0.0 | Extension validation |
| `regex_match` | Regular expression matching | 1.0/0.0 | Format validation |
| `substring_match` | Substring containment | 1.0/0.0 | Keyword detection |
| `contains_all` | Contains all substrings | 0.0-1.0 | Multiple keywords |
| `contains_any` | Contains any substring | 1.0/0.0 | Alternative keywords |
| `word_overlap` | Word overlap ratio | 0.0-1.0 | Content coverage |
| `char_overlap` | Character overlap ratio | 0.0-1.0 | Character coverage |

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `reference_response` | str | Yes* | Reference text or pattern |
| `response` | str | Yes | Text to evaluate |
| `algorithm` | str | Yes | Algorithm name (see table above) |
| `case_sensitive` | bool | No | Whether case-sensitive (default False) |
| `ignore_whitespace` | bool | No | Whether to ignore whitespace (default False) |
| `**kwargs` | Any | No | Algorithm-specific parameters |

!!! note
    For `contains_all` and `contains_any` algorithms, `reference_response` can be empty and pass substrings via the `substrings` parameter.

**Scoring:**
- Boolean algorithms: `1.0` (match) or `0.0` (no match)
- Overlap algorithms: `0.0` - `1.0` (overlap ratio)

**Examples:**

### Exact Match - Answer Verification

```python
import asyncio
from openjudge.graders.text.string_match import StringMatchGrader

async def main():
    grader = StringMatchGrader(
        algorithm="exact_match",
        case_sensitive=False,
        ignore_whitespace=True
    )

    result = await grader.aevaluate(
        reference_response="Paris",
        response="paris",
    )

    print(f"Score: {result.score}")  # 1.0
    print(f"Matched: {result.metadata['matched']}")  # True

asyncio.run(main())
```

### Regular Expression - Format Validation

```python
import asyncio
from openjudge.graders.text.string_match import StringMatchGrader

async def main():
    grader = StringMatchGrader(algorithm="regex_match")

    # Validate email format
    result = await grader.aevaluate(
        reference_response=r"[\w.-]+@[\w.-]+\.\w+",
        response="user@example.com",
    )

    print(f"Score: {result.score}")  # 1.0
    print(f"Reason: {result.reason}")

    # Validate phone number format
    result = await grader.aevaluate(
        reference_response=r"\d{3}-\d{4}",
        response="My phone is 123-4567",
    )

    print(f"Score: {result.score}")  # 1.0

asyncio.run(main())
```

### Keyword Detection - Contains All

```python
import asyncio
from openjudge.graders.text.string_match import StringMatchGrader

async def main():
    grader = StringMatchGrader(algorithm="contains_all", case_sensitive=False)

    # Check if response contains all required keywords
    result = await grader.aevaluate(
        reference_response="",  # reference_response not used
        response="The quick brown fox jumps over the lazy dog",
        substrings=["fox", "dog", "jumps"],
    )

    print(f"Score: {result.score}")  # 1.0 - all keywords found
    print(f"Matched: {result.metadata['matched']}")  # True

    # Partial match
    result = await grader.aevaluate(
        reference_response="",
        response="The quick brown fox jumps over the lazy dog",
        substrings=["fox", "cat", "dog"],
    )

    print(f"Score: {result.score}")  # 0.67 - 2/3 keywords found
    print(f"Missing: {result.metadata['missing_substrings']}")  # ['cat']

asyncio.run(main())
```

### Keyword Detection - Contains Any

```python
import asyncio
from openjudge.graders.text.string_match import StringMatchGrader

async def main():
    grader = StringMatchGrader(algorithm="contains_any")

    # Check if response contains any of the keywords
    result = await grader.aevaluate(
        reference_response="",
        response="The weather is sunny today",
        substrings=["sunny", "cloudy", "rainy"],
    )

    print(f"Score: {result.score}")  # 1.0
    print(f"Matched: {result.metadata['matched_substrings']}")  # ['sunny']

asyncio.run(main())
```

### Prefix/Suffix Matching

```python
import asyncio
from openjudge.graders.text.string_match import StringMatchGrader

async def main():
    # Prefix matching
    prefix_grader = StringMatchGrader(algorithm="prefix_match")
    result = await prefix_grader.aevaluate(
        reference_response="Hello",
        response="Hello World",
    )
    print(f"Prefix Score: {result.score}")  # 1.0

    # Suffix matching
    suffix_grader = StringMatchGrader(algorithm="suffix_match")
    result = await suffix_grader.aevaluate(
        reference_response="World",
        response="Hello World",
    )
    print(f"Suffix Score: {result.score}")  # 1.0

asyncio.run(main())
```

### Word Overlap - Content Coverage

```python
import asyncio
from openjudge.graders.text.string_match import StringMatchGrader

async def main():
    grader = StringMatchGrader(algorithm="word_overlap", case_sensitive=False)

    result = await grader.aevaluate(
        reference_response="the cat sat on the mat",
        response="the dog sat on the rug",
    )

    # Overlapping words: "the", "sat", "on" (3)
    # Unique words in reference_response: "the", "cat", "sat", "on", "mat" (5)
    print(f"Score: {result.score}")  # 0.6 (3/5)
    print(f"Overlap Ratio: {result.metadata['overlap_ratio']}")

asyncio.run(main())
```

**Algorithm-Specific Parameters:**

**exact_match:**
- `case_sensitive` (bool): Case-sensitive matching (default True)
- `ignore_whitespace` (bool): Ignore whitespace (default False)

**regex_match:**
- `pattern` (str): Regular expression pattern (can replace ground_truth)
- `case_sensitive` (bool): Case-sensitive matching (default True)

**substring_match:**
- `bidirectional` (bool): Bidirectional matching (default False)

**contains_all/contains_any:**
- `substrings` (List[str]): List of substrings to detect


## NumberAccuracyGrader

Check numerical calculation accuracy by comparing numbers extracted from text.

**When to use:**
- Math calculation verification
- Data report accuracy
- Quantitative metric checking
- Numerical Q&A evaluation

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `response` | str | Yes | Text to evaluate |
| `reference_response` | str | Yes | Reference answer text |
| `tolerance` | float | No | Numerical tolerance (default 1e-6) |

**Scoring:**
- Score range: `0.0` - `1.0`
- Calculation: correct numbers / total reference numbers
- `1.0`: All numbers correct
- `0.5`: Half numbers correct
- `0.0`: No correct numbers or no numbers to compare

**Examples:**

### Basic Numerical Verification

```python
import asyncio
from openjudge.graders.text.number_accuracy import NumberAccuracyGrader

async def main():
    grader = NumberAccuracyGrader(tolerance=1e-6)

    # Perfect match
    result = await grader.aevaluate(
        response="The result is 3.14159",
        reference_response="The result is 3.14159",
    )

    print(f"Score: {result.score}")  # 1.0
    print(f"Reason: {result.reason}")  # "Number accuracy: 1/1 numbers correct"

asyncio.run(main())
```

### Multiple Number Verification

```python
import asyncio
from openjudge.graders.text.number_accuracy import NumberAccuracyGrader

async def main():
    grader = NumberAccuracyGrader(tolerance=0.01)

    result = await grader.aevaluate(
        response="Temperature readings: 25.5°C, 30.2°C, 28.7°C",
        reference_response="Expected values: 25.5°C, 30.0°C, 28.7°C",
    )

    # Number matching: 25.5 ✓, 30.2 ✗ (vs 30.0), 28.7 ✓
    print(f"Score: {result.score}")  # 0.67 (2/3)
    print(f"Correct: {result.metadata['correct_numbers']}")  # 2
    print(f"Total: {result.metadata['total_reference_response_numbers']}")  # 3

asyncio.run(main())
```

### Math Calculation Verification

```python
import asyncio
from openjudge.graders.text.number_accuracy import NumberAccuracyGrader

async def main():
    grader = NumberAccuracyGrader(tolerance=1e-6)

    # Verify calculation results
    result = await grader.aevaluate(
        response="Area = 78.54 square units, Perimeter = 31.42 units",
        reference_response="Area = 78.54, Perimeter = 31.42",
    )

    print(f"Score: {result.score}")  # 1.0
    print(f"Response Numbers: {result.metadata['response_numbers']}")
    print(f"Reference Numbers: {result.metadata['reference_response_numbers']}")

asyncio.run(main())
```

### Custom Tolerance

```python
import asyncio
from openjudge.graders.text.number_accuracy import NumberAccuracyGrader

async def main():
    # Loose tolerance - for approximate calculations
    loose_grader = NumberAccuracyGrader(tolerance=0.1)

    result = await loose_grader.aevaluate(
        response="The value is approximately 3.14",
        reference_response="The exact value is 3.14159",
    )

    print(f"Score (loose): {result.score}")  # 1.0 (3.14 vs 3.14159 within tolerance)

    # Strict tolerance - for high precision
    strict_grader = NumberAccuracyGrader(tolerance=1e-9)

    result = await strict_grader.aevaluate(
        response="The value is approximately 3.14",
        reference_response="The exact value is 3.14159",
    )

    print(f"Score (strict): {result.score}")  # 0.0 (exceeds strict tolerance)

asyncio.run(main())
```

**How It Works:**

1. Extract all numbers (integers and floats) from both texts
2. Compare numbers in order of appearance
3. Use specified tolerance to determine matches
4. Return match ratio as score

**Important Notes:**

- Numbers are matched in order of appearance, position-independent
- Supports negative numbers and floats
- Non-numeric text content is ignored
- Returns 0.0 if reference text has no numbers


## Best Practices

### 1. Choose Appropriate Normalization

```python
# Case-insensitive scenario
grader = SimilarityGrader(
    algorithm="f1_score",
    normalize=True,  # converts to lowercase
    case_sensitive=False
)

# Strict format scenario
grader = StringMatchGrader(
    algorithm="exact_match",
    case_sensitive=True,
    ignore_whitespace=False
)
```

### 2. Combine Multiple Algorithms for Comprehensive Evaluation

```python
    # Evaluate both exactness and similarity
    exact_grader = StringMatchGrader(algorithm="exact_match")
    fuzzy_grader = SimilarityGrader(algorithm="fuzzy_match")

    exact_result = await exact_grader.aevaluate(reference_response="...", response="...")
    fuzzy_result = await fuzzy_grader.aevaluate(reference_response="...", response="...")

# Decision logic: prioritize exact match, fallback to fuzzy
if exact_result.score == 1.0:
    final_score = 1.0
elif fuzzy_result.score > 0.8:
    final_score = 0.8
else:
    final_score = fuzzy_result.score
```

### 3. Tune Parameters Based on Data Characteristics

```python
# Science calculations - strict tolerance
scientific_grader = NumberAccuracyGrader(tolerance=1e-9)

# Engineering calculations - loose tolerance
engineering_grader = NumberAccuracyGrader(tolerance=0.01)

# Short text - high-order N-grams
short_grader = SimilarityGrader(algorithm="bleu", max_ngram_order=2)

# Long text - standard N-grams
long_grader = SimilarityGrader(algorithm="bleu", max_ngram_order=4)
```

### 4. Deep Analysis Using Metadata

```python
result = await grader.aevaluate(reference_response="...", response="...")

# Check detailed metrics
print(f"Score: {result.score}")
print(f"Precision: {result.metadata.get('precision', 'N/A')}")
print(f"Recall: {result.metadata.get('recall', 'N/A')}")
print(f"Algorithm: {result.metadata['algorithm']}")

# Adjust strategy based on metadata
if result.metadata.get('recall', 0) < 0.5:
    print("Warning: Low recall - response may be incomplete")
```


## Performance Characteristics

| Grader | Avg Latency | Throughput | Memory | Thread-Safe |
|--------|-------------|------------|--------|-------------|
| `SimilarityGrader` (BLEU) | < 1ms | > 10K/s | Very Low | ✓ |
| `SimilarityGrader` (ROUGE) | < 5ms | > 5K/s | Low | ✓ |
| `SimilarityGrader` (Cosine) | < 10ms | > 2K/s | Moderate | ✓ |
| `StringMatchGrader` | < 0.5ms | > 20K/s | Very Low | ✓ |
| `NumberAccuracyGrader` | < 1ms | > 10K/s | Very Low | ✓ |

!!! note "Performance Note"
    Performance metrics are based on typical text length (100-500 tokens). Actual performance may vary based on text length and hardware configuration.


## Next Steps

- [Format Graders](format.md) — Validate structured outputs and formatting
- [Create Custom Graders](../building_graders/create_custom_graders.md) — Build specialized text graders
- [Build Reward for Training](../get_started/build_reward.md) — Combine graders for RLHF rewards

