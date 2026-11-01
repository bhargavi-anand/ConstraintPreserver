# ConstraintPreserver

### Investigating Information Loss in AI-Generated Summaries

**The Disappearing Detail Problem: What Important Information Disappears When AI Summarizes Text?**

ConstraintPreserver is an NLP research project investigating whether important information gets lost when large language models summarize long, information-dense documents.

The project focuses on information that is particularly important for accurate interpretation and decision-making: conditions, exceptions, deadlines, responsibilities, decisions, and numerical limitations.

Rather than evaluating summaries only on general semantic similarity, ConstraintPreserver examines whether specific constraints and obligations survive the summarization process.

---

## Research Motivation

AI-generated summaries can communicate the general meaning of a document while omitting details that significantly affect its interpretation.

For example:

**Original statement:**

> Students may submit applications after the deadline only if they have received prior written approval from the department.

**Potentially incomplete summary:**

> Students may submit applications after the deadline with approval.

The summary preserves the general idea but loses the requirement that approval must be both prior and written.

This is the information-loss problem that ConstraintPreserver aims to investigate.

## Research Question

**Which linguistic and structural properties make important information more likely to disappear from AI-generated summaries?**

The project also investigates whether constraint-aware prompting can improve the preservation of critical information.

## Research Objectives

* Identify critical constraints in institutional documents.
* Generate ordinary and constraint-aware summaries using an LLM.
* Measure how much critical information survives summarization.
* Compare preservation across different constraint categories.
* Investigate which linguistic and structural features are associated with information loss.
* Validate automated evaluations through human annotation.

---

## Types of Information Studied

ConstraintPreserver focuses on six categories:

| Category         | Description                                   | Examples                              |
| ---------------- | --------------------------------------------- | ------------------------------------- |
| Decisions        | Explicit decisions and mandatory requirements | Must submit, required to attend       |
| Conditions       | Prerequisites and conditional requirements    | Only if, provided that                |
| Exceptions       | Exclusions and special circumstances          | Unless, except when                   |
| Deadlines        | Time restrictions and deadlines               | Within 30 days, before submission     |
| Responsibilities | Assigned roles and obligations                | Department must approve               |
| Numerical Limits | Quantities, thresholds and percentages        | Maximum of 5 attempts, 75% attendance |

---

## Research Methodology

The research pipeline consists of five major stages.

### 1. Document Preprocessing

The dataset contains 136 institutional generative AI policy documents.

Documents undergo:

* Text extraction and cleaning.
* Document structuring.
* Paragraph-based chunking.
* Preparation for constraint extraction and summarization.

### 2. Constraint Extraction

An LLM identifies critical constraints from the original documents.

Each extracted constraint includes:

* Constraint ID
* Constraint category
* Original constraint text
* Supporting source evidence

### 3. Comparative Summarization

Each document is summarized using two different prompting strategies.

**A. Baseline Summarization**

Uses a conventional summarization prompt focused on capturing the main ideas, purpose, and important information.

**B. Constraint-Aware Summarization**

Uses a modified prompt explicitly instructing the model to preserve:

* Decisions
* Conditions
* Exceptions
* Deadlines
* Responsibilities
* Numerical limits

Both approaches use the same underlying language model to make the prompting strategies comparable.

### 4. Constraint Preservation Evaluation

Each extracted constraint is checked against both generated summaries.

Evaluation labels:

| Label     | Score | Meaning                                                         |
| --------- | ----: | --------------------------------------------------------------- |
| PRESERVED |   1.0 | The constraint is accurately retained                           |
| PARTIAL   |   0.5 | Some information is retained, but important details are missing |
| LOST      |   0.0 | The constraint is absent or inaccurately represented            |

The system records supporting evidence and reasoning for each evaluation.

### 5. Statistical Analysis and Human Validation

The evaluation results are analyzed to compare:

* Overall preservation rates.
* Document-level performance.
* Category-wise information loss.
* Differences between baseline and constraint-aware summaries.

A human annotation dataset is also generated to support independent validation of automated evaluation results.

---

## Preliminary Pilot Results

An initial pilot experiment was conducted on three documents, with 31 extracted constraints.

| Metric               | Baseline | Constraint-Aware |
| -------------------- | -------: | ---------------: |
| Overall preservation |   73.91% |           96.97% |

**Improvement: +23.06 percentage points**

### Category-wise Pilot Results

| Category       | Baseline | Constraint-Aware | Improvement |
| -------------- | -------: | ---------------: | ----------: |
| Decision       |   69.23% |           96.15% |   +26.92 pp |
| Condition      |   50.00% |           90.00% |   +40.00 pp |
| Exception      |  100.00% |          100.00% |        0 pp |
| Responsibility |   80.00% |          100.00% |   +20.00 pp |
| Deadline       |  100.00% |          100.00% |        0 pp |

*These are preliminary, automated pilot results from three documents, not final research findings. The full dataset evaluation and human validation are still required. Category-wise results are calculated from pooled constraint-level evaluations.*

---

## Technology Stack

| Component              | Technology                 |
| ---------------------- | -------------------------- |
| Programming Language   | Python                     |
| Language Model         | Google Gemini API          |
| LLM Integration        | Google Gen AI SDK          |
| Text Processing        | Python                     |
| Semantic Retrieval     | FAISS                      |
| Embeddings             | Sentence Transformers      |
| Data Storage           | JSON, CSV                  |
| Data Analysis          | Python                     |
| Environment Management | Python Virtual Environment |

---

## Project Structure

```text
ConstraintPreserver/
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── cleaned_documents.json
│   │   └── chunks.json
│   │
│   ├── constraints/
│   ├── index/
│   ├── summaries/
│   ├── evaluation/
│   ├── annotations/
│   └── analysis/
│
├── src/
│   ├── preprocessing/
│   │   ├── build_documents_json.py
│   │   ├── clean_documents.py
│   │   └── chunk_documents.py
│   │
│   ├── constraints/
│   │   └── extractor.py
│   │
│   ├── retrieval/
│   │   ├── build_index.py
│   │   └── search.py
│   │
│   ├── generation/
│   │   └── answer.py
│   │
│   ├── summarization/
│   │   └── generate_summaries.py
│   │
│   ├── evaluation/
│   │   ├── evaluate_preservation.py
│   │   └── create_annotation_dataset.py
│   │
│   └── analysis/
│       └── analyze_results.py
│
├── configs/
├── notebooks/
├── tests/
├── docs/
├── .env.example
├── .gitignore
└── README.md
```

*Some directories are created as the pipeline runs. The structure may evolve as the research progresses.*

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/bhargavi-anand/ConstraintPreserver.git
cd ConstraintPreserver
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install google-genai python-dotenv faiss-cpu sentence-transformers pandas
```

### 4. Configure the Gemini API

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=your_available_gemini_model
```

Never commit your `.env` file or expose your API key.

### 5. Run the Summarization Pipeline

Run a three-document pilot:

```bash
python -m src.summarization.generate_summaries --limit 3
```

Run the full dataset:

```bash
python -m src.summarization.generate_summaries --limit 0
```

The summarization pipeline supports checkpointing and automatic resumption. Completed documents are saved incrementally, allowing interrupted runs to continue without repeating completed work.

### 6. Evaluate Constraint Preservation

```bash
python -m src.evaluation.evaluate_preservation
```

### 7. Generate the Human Annotation Dataset

```bash
python -m src.evaluation.create_annotation_dataset
```

### 8. Analyze the Results

```bash
python -m src.analysis.analyze_results
```

---

## Current Research Status

| Component                           | Status      |
| ----------------------------------- | ----------- |
| Document preprocessing              | Completed   |
| Dataset preparation (136 documents) | Completed   |
| Text chunking                       | Completed   |
| Constraint extraction module        | Implemented |
| Baseline summarization              | Implemented |
| Constraint-aware summarization      | Implemented |
| Checkpoint and resume mechanism     | Implemented |
| Preservation evaluation             | Implemented |
| Pilot experiment (3 documents)      | Completed   |
| Preliminary statistical analysis    | Completed   |
| Human annotation dataset generation | Completed   |
| Full-dataset summarization          | In progress |
| Full-dataset evaluation             | Pending     |
| Human annotation and validation     | Pending     |
| Linguistic feature analysis         | Planned     |

---

## Future Work

1. **Full-Scale Evaluation:** Extend the experiment from the pilot sample to all 136 documents.
2. **Human Validation:** Compare human judgments with automated LLM-based evaluations.
3. **Inter-Annotator Agreement:** Measure annotation consistency where multiple annotators are available.
4. **Linguistic Feature Analysis:** Investigate the effects of modal verbs, conditional expressions, negation, numerical expressions, clause complexity, and constraint position.
5. **Statistical Significance:** Apply appropriate paired statistical tests and confidence intervals.
6. **Generalizability:** Investigate whether the findings extend to other document types and summarization models.

---

## Research Contribution

ConstraintPreserver approaches summarization evaluation from an information-preservation perspective.

Instead of asking only whether an AI-generated summary captures the general meaning of a document, it asks:

**Does the summary preserve the specific details that determine what someone is allowed, required, or prohibited from doing?**

The goal is to develop a systematic way of identifying, measuring, and understanding the disappearance of critical information during AI summarization.

---

**Project:** ConstraintPreserver
**Research Area:** Natural Language Processing | Large Language Models | Summarization Evaluation | Information Preservation
