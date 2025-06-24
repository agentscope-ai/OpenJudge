# 1. Reward Model Benchmark Survey

## 1.1 [RewardBench](https://arxiv.org/abs/2403.13787)
### Overview
A comprehensive benchmark for reward model evaluation covering multiple domains:
- **Data Composition**:
  - Chat (358 samples): Basic chat response discrimination
  - Chat-Hard (456 samples): Subtle instructions & adversarial examples
  - Safety (740 samples): Sensitive content handling
  - Reasoning (1431 samples): Code/math reasoning
  - Prior Sets: Existing dataset subsets for compatibility

### Key Features
- **Evaluation Framework**:
  - Pairwise accuracy metric: Compares scores between chosen and rejected responses
  - Weighted scoring system:
    - Primary categories (Chat/Chat-Hard/Safety/Reasoning) weighted equally
    - Prior Sets weighted at 0.5x
  - Unified evaluation framework supporting both classifier-based and DPO-trained models

### Evaluation Metrics
- Accuracy = (Correctly predicted trios) / (Total trios)
- Domain-specific metrics:
  - Safety: Rejection accuracy for sensitive content
  - Reasoning: Code/math problem solving capability
  - Chat: Natural conversation flow assessment


## 1.2 [RM-Bench](https://arxiv.org/abs/2410.16184)
### Overview
Focuses on evaluating reward models' sensitivity to subtle differences and style bias robustness:
- **Key Components**:
  - GPT-4O generated (chosen,rejected) pairs with minimal differences
  - Style-controlled variants (length/format)
  - Domain coverage: Chat (129), Safety (441), Math (529), Code (228)

### Evaluation Framework
- **Style-Substance Matrix**: 3x3 matrix comparing different style combinations
- **Three Accuracy Metrics**:
  1. **Normal Accuracy**: Same-style comparison (content-only assessment)
  2. **Easy Accuracy**: Chosen has favorable style (style-assisted assessment)
  3. **Hard Accuracy**: Chosen has unfavorable style (style-robust assessment)

### Domain-Specific Analysis
- Code/Math: Unit test validation
- Chat/Safety: Human annotation verification
- Style dimensions: Length vs Markdown format

## 1.3 [RMB](https://arxiv.org/abs/2410.09893)
### Overview
Comprehensive benchmark with real-world user queries:
- **Data Composition**:
  - 25,845 samples across 49 scenarios
  - 37 Helpfulness scenarios + 12 Harmlessness scenarios
  - Real user queries from WildChat (3,197 prompts)
  - 14 different LLMs for response generation

### Evaluation Paradigms
- **Pairwise Accuracy**:
  - Discriminative RM: s(x_chosen) > s(x_rejected)
  - Generative RM: Direct selection accuracy
- **Best-of-N (BoN) Accuracy**:
  - Measures ability to identify optimal response in N candidates
  - s(x_winner) > s(x_loser) for all losers

### Correlation Analysis
- **Spearman Rank Correlation**:
  - RMB rankings vs downstream alignment performance
  - Measures benchmark validity
- **Task Correlation Analysis**:
  - Cross-scenario performance consistency
  - Multi-objective model analysis

## 1.4 Benchmark Comparison

| Benchmark    | Key Focus                     | Scenario Coverage       | Evaluation Methodology                |
|--------------|-------------------------------|-------------------------|----------------------------------------|
| RewardBench  | Broad capability assessment     | 4 primary domains       | Weighted accuracy scoring             |
| RM-Bench     | Subtle difference detection     | 4 focused domains       | Style-robustness matrix analysis      |
| RMB          | Real-world alignment evaluation | 49 detailed scenarios   | Pairwise/Best-of-N + correlation analysis |
# 2. Built-in Scenarios

## 2.1 Alignment
The Alignment module provides reward models for evaluating and optimizing model outputs against human values alignment, covering safety, helpfulness, and factual accuracy described above. Below is a comprehensive technical overview:

### Core Reward Models Overview
| Scenario | Description | Reward Model |
|------------|------------|--------------------|
| Helpfuness| The assistant aims to answer questions, avoiding harmful behaviors such as spreading misinformation, spreading harmful ideas, or engaging in other harmful activities. | base_helpfulness_pointwise/base_helpfulness_listwise |
| Harmlessness| The assistant aims to provide helpful and informative responses to users, responding to their queries with relevant and accurate information. | base_harmlessness_pointwise/base_harmlessness_listwise |
| Honesty| The assistant aims to truthfully answer the user's questions with no bias or prejudice. | base_honesty_pointwise/base_honesty_listwise|


### Hramlessness
|  Scenario | Source |Scenario | Reward Model |
|------------|------------|------------|--------------------|
| Safety| RewardBench2 | Safety: Comply with or refuse prompts related to harmful use cases as well as general compliance behaviors. | safety_pointwise_reward |

### Helpfulness

| Scenario | Source | Description| Reward Model |
|------------|------------|------------|--------------------|
| Brainstorming| RMBBench | Brainstorming: Generating text to come up with new ideas or solutions, with an emphasis on creativity and driving thinking. |brainstorming_listwise_reward |
| Chat| RMBBench | Chat: Simulates human conversation and communicates a variety of topics through text understanding and generation, emphasizing coherence and natural flow of interaction. | chat_listwise_reward |
| Classification | RMBBench | Classification: Entails assigning predefined categories or labels to text based on its content. | classification_listwise_reward |
| Closed QA | RMBBench | Closed QA: Search for direct answers to specific questions in given text sources (i.e. given context, given options). | closed_qa_listwise_reward |
| Code | RMBBench | Code: Involves generating, understanding, or modifying programming language code within text. | code_listwise_reward |
| Generation | RMBBench | Generation: Creating new textual content, from articles to stories, with an emphasis on originality and creativity. | generation_listwise_reward |
| Open QA | RMBBench | Open QA: Search for answers across a wide range of text sources. The challenge is to process large amounts of information and understand complex questions. |  open_qa_listwise_reward |
| Reasoning | RMBBench | Reasoning: Involves processing and analyzing text to draw inferences, make predictions, or solve problems, requiring an understanding of underlying concepts and relationships within the text. | reasoning_listwise_reward |
| Rewrite | RMBBench | Rewrite: the assitant aims to modifies existing text to alter its style while preserving the original information and intent. | rewrite_listwise_reward |
| Role Playing | RMBBench | Role Playing: Entails adopting specific characters or personas within text-based scenarios, engaging in dialogues or actions that reflect the assigned roles. | role_palying_listwise_reward |
| Summarization | RMBBench | Summarization: The text is compressed into a short form, retaining the main information, which is divided into extraction (directly selected from the original text) and production (rewriting the information). | summarization_listwise_reward |
| Translation | RMBBench | Translation: Converting text from one language to another. | translation_listwise_reward |
| Focus| RMBBench | Focus: Detectes high-quality, on-topic answers to general user queries | focus_pointwise_reward |
| Math | RewardBench2 | Math: Solves problems at math, on open-ended human prompts ranging from middle school physics and geometry to college-level chemistry, calculus, combinatorics, and more. | math_pointwise_reward |
| Precise IF| RewardBench2 | Precise Instruction Following : Follows precise instructions, such as 'Answer without the letter u'. | precise_if_pointwise_reward|

Click [here](autoprinciple.ipynb#6) to view relevant evaluation results

### Honesty


|  Scenario | Source | Description | Reward Model |
|------------|---------------|---------------|-----------------|
| Factuality| RewardBench2 | Factuality: Detectes hallucinations and other basic errors in completions. | factuality_pointwise_reward|


## 2.2 Rule based reward

The rule-based reward system provides various evaluation criteria for different aspects of generated content. Below are the available reward models organized by category:

### 2.2.1 Math Evaluation Rewards

| Scenario | Source | Description | Reward Model |
|------|----------|----------|------------|
| Math Verify | | Verifies mathematical expressions using the math_verify library, supporting both LaTeX and plain expressions | MathVerifyReward |

### 2.2.2 Code Quality Rewards

| Scenario | Source | Description | Reward Model |
|------|----------|----------|------------|
| Code Syntax |  | Check code syntax using Abstract Syntax Tree to validate Python code blocks |  SyntaxCheckReward |
| Code Style |  | Basic code style checking including indentation consistency and naming conventions | CodeStyleReward |
| Patch Similarity |  | Calculate similarity between generated patch and oracle patch using difflib.SequenceMatcher | PatchSimilarityReward |
| Code Execution |  | Executes code against test cases and evaluates correctness based on test case results | CodeExecutionReward |

### 2.2.3 General Evaluation Rewards

| Scenario | Source | Description | Reward Model |
|------|----------|----------|--------------|
| Accuracy |  | Calculate accuracy (exact match rate) between generated content and reference answer | AccuracyReward |
| F1 Score |  | Calculate F1 score between generated content and reference answer at word level with configurable tokenizer | F1ScoreReward |
| ROUGE |  | ROUGE-L similarity evaluation using longest common subsequence | RougeReward |
| Number Accuracy |  | Check numerical calculation accuracy by comparing numbers in generated vs reference content | NumberAccuracyReward |

### 2.2.4 Format and Style Rewards

| Scenario | Source | Description | Reward Model |
|------|----------|----------|--------------|
| Reasoning Format |  |  Check format reward for thinking format and answer format with proper tags | ReasoningFormatReward |
| Tool Call Format |  | Check tool call format including think, answer and tool_call tags with JSON validation | ReasoningToolCallFormatReward |
| Length Penalty |  | Text length based penalty for content that is too short or too long | LengthPenaltyReward |
| N-gram Repetition |  | Calculate N-gram repetition penalty supporting Chinese processing and multiple penalty strategies | NgramRepetitionPenaltyReward |
| Privacy Leakage |  |Privacy information leakage detection for emails, phone numbers, ID cards, credit cards, and IP addresses | PrivacyLeakageReward |
