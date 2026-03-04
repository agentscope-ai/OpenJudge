# Paper Review Skill — Reference

## PipelineConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | `"gpt-4o"` | LiteLLM model string |
| `api_key` | str | `""` | API key for the model provider |
| `base_url` | str \| None | `None` | Custom API endpoint (proxies, self-hosted) |
| `temperature` | float | `0.7` | Generation temperature |
| `timeout` | int | `7500` | Request timeout in seconds. Increase for very long papers. |
| `enable_safety_checks` | bool | `True` | Jailbreak detection + format check |
| `enable_correctness` | bool | `True` | Objective error detection |
| `enable_review` | bool | `True` | Overall quality/novelty review (score 1–6) |
| `enable_criticality` | bool | `True` | Severity check (only runs if correctness score > 1) |
| `enable_bib_verification` | bool | `True` | BibTeX reference cross-check |
| `crossref_mailto` | str \| None | `None` | Email for CrossRef API; improves rate limits |
| `discipline` | str \| DisciplineConfig \| None | `None` | Discipline ID or custom config |
| `venue` | str \| None | `None` | Target venue name, applied on top of discipline criteria |
| `instructions` | str \| None | `None` | Free-form reviewer guidance, e.g. "Focus on experimental design" |
| `language` | str \| None | `None` | Output language: `"en"` (default) or `"zh"` (Simplified Chinese) |
| `use_vision_for_pdf` | bool | `False` | Render PDF pages as images (needs `pypdfium2`) |
| `vision_max_pages` | int \| None | `30` | Max pages when using vision mode |
| `format_vision_max_pages` | int \| None | `10` | Max pages for Format grader in vision mode |

## Disciplines

| ID | Name | Key venues |
|----|------|-----------|
| `cs` | Computer Science & AI/ML | NeurIPS, ICML, ICLR, CVPR, ACL, AAAI |
| `medicine` | Medicine & Clinical Research | NEJM, The Lancet, JAMA, BMJ, Nature Medicine |
| `physics` | Physics | Physical Review Letters, Nature Physics, JHEP, PRD |
| `chemistry` | Chemistry | JACS, Angewandte Chemie, Nature Chemistry, JCTC |
| `biology` | Biology & Life Sciences | Cell, Nature, Science, eLife, PLOS Biology, Nature Genetics |
| `economics` | Economics | AER, QJE, JPE, Econometrica, REStud |
| `psychology` | Psychology | Psychological Review, JEP:General, Psychological Science |
| `environmental_science` | Environmental Science | Nature Climate Change, Environmental Science & Technology |
| `mathematics` | Mathematics | Annals of Mathematics, Inventiones Mathematicae, JAMS |
| `social_sciences` | Social Sciences | American Sociological Review, APSR, American Journal of Sociology |

## Model Strings (LiteLLM format)

| Provider | Example model string | API key env var |
|----------|---------------------|-----------------|
| OpenAI | `gpt-4o`, `gpt-4.1`, `o3`, `o4-mini` | `OPENAI_API_KEY` |
| Anthropic | `claude-opus-4-5`, `claude-sonnet-4-5`, `claude-haiku-3-5` | `ANTHROPIC_API_KEY` |
| DashScope / Qwen | `qwen-plus`, `qwen-max`, `qwen-turbo` | `DASHSCOPE_API_KEY` |
| Azure OpenAI | `azure/gpt-4o` | `AZURE_API_KEY` + `AZURE_API_BASE` |
| Local (Ollama) | `ollama/llama3.1` | — (use `--base-url http://localhost:11434`) |

## Script CLI Reference

### review_paper.py

```
positional:
  pdf                    Path to the PDF file

options:
  --bib BIB_FILE         .bib file for BibTeX verification
  --model MODEL          Model name (default: gpt-4o)
  --api-key KEY          API key
  --base-url URL         Custom API base URL
  --discipline DISC      One of: cs medicine physics chemistry biology
                         economics psychology environmental_science
                         mathematics social_sciences
  --venue VENUE          Target venue, e.g. "NeurIPS 2025"
  --paper-name NAME      Paper title for report header
  --output FILE          Output .md file (default: <name>_review.md)
  --email EMAIL          CrossRef email for BibTeX rate limits
  --no-safety            Skip safety/jailbreak checks
  --no-correctness       Skip correctness check
  --no-criticality       Skip criticality verification
  --no-bib               Skip BibTeX verification
  --language LANG        Output language: en (default) or zh
  --instructions TEXT    Free-form reviewer guidance
  --vision               Use vision mode (render pages as images)
  --vision-max-pages N   Max pages in vision mode (default: 30, 0 = all)
  --format-vision-max-pages N
                         Max pages for format check in vision mode (default: 10)
  --timeout SECONDS      API timeout (default: 7500)
```

### review_tex.py

```
positional:
  PACKAGE                Path to .tar.gz or .zip TeX source package

or:
  --bib-only BIB_FILE    Verify a standalone .bib file only (no review)

options:
  --model MODEL          Model name (default: gpt-4o)
  --api-key KEY          API key
  --base-url URL         Custom API base URL
  --discipline DISC      Academic discipline
  --venue VENUE          Target venue
  --paper-name NAME      Paper title for report header
  --output FILE          Output .md report file
  --email EMAIL          CrossRef email for BibTeX rate limits
  --no-safety            Skip safety/jailbreak checks
  --no-correctness       Skip correctness check
  --no-criticality       Skip criticality verification
  --no-bib               Skip BibTeX verification
  --language LANG        Output language: en (default) or zh
  --instructions TEXT    Free-form reviewer guidance
  --timeout SECONDS      API timeout (default: 7500)
```

## Output: PaperReviewResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `is_safe` | bool | False if jailbreaking detected |
| `safety_issues` | list[str] | Safety check failure reasons |
| `format_compliant` | bool | True if paper format is acceptable |
| `correctness` | CorrectnessResult \| None | Objective error check |
| `review` | ReviewResult \| None | Overall review with score 1–6 |
| `criticality` | CriticalityResult \| None | Error severity assessment |
| `bib_verification` | dict[str, BibVerificationSummary] \| None | BibTeX results per file |
| `tex_info` | TexPackageInfo \| None | TeX package metadata (TeX review only) |

### CorrectnessResult

| Field | Description |
|-------|-------------|
| `score` | 1 = no errors, 2 = minor, 3 = major |
| `reasoning` | Step-by-step explanation |
| `key_issues` | List of specific errors with locations |

### ReviewResult

| Field | Description |
|-------|-------------|
| `score` | 1–6 (1–2 reject, 3–4 borderline, 5–6 accept) |
| `review` | Full detailed review text |

### BibVerificationSummary

| Field | Description |
|-------|-------------|
| `total_references` | Total entries in .bib file |
| `verified` | Confirmed in CrossRef/arXiv/DBLP |
| `suspect` | Title/author mismatch or not found |
| `errors` | Parse or API errors |
| `verification_rate` | verified / total |
| `suspect_references` | List of suspect reference titles |

## Custom Discipline

For disciplines not in the registry, create a `DisciplineConfig` directly:

```python
from cookbooks.paper_review.disciplines.base import DisciplineConfig
from cookbooks.paper_review import PaperReviewPipeline, PipelineConfig

my_discipline = DisciplineConfig(
    id="my_field",
    name="My Research Field",
    venues=["Top Conference A", "Top Journal B"],
    reviewer_context="You specialize in ...",
    evaluation_dimensions=[
        "Dimension 1: ...",
        "Dimension 2: ...",
    ],
    correctness_categories=[
        "Error type 1 - description",
        "Error type 2 - description",
    ],
    correctness_context="Pay attention to ...",
    scoring_notes="For this field, ... lowers the score.",
)

config = PipelineConfig(
    model_name="gpt-4o",
    api_key="...",
    discipline=my_discipline,
)
pipeline = PaperReviewPipeline(config)
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'cookbooks'`**
Run scripts from the project root, or install with `pip install -e .`

**`ModuleNotFoundError: No module named 'litellm'`**
```bash
pip install litellm
```

**BibTeX verification returns all "suspect"**
Provide `--email your@email.com` to avoid CrossRef rate limiting.

**Timeout errors on long papers**
Increase `--timeout 15000` or enable vision mode with page limits:
```bash
python review_paper.py long_paper.pdf --vision --timeout 15000
```

**Vision mode: `ModuleNotFoundError: No module named 'pypdfium2'`**
```bash
pip install pypdfium2
```
