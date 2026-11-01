import os
import json
import re
import time
import argparse

from pathlib import Path
from dotenv import load_dotenv
from google import genai


# =====================================================
# CONFIGURATION
# =====================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY missing from .env")

client = genai.Client(api_key=API_KEY)

ROOT = Path(__file__).resolve().parents[2]

DOCUMENTS_FILE = ROOT / "data/processed/cleaned_documents.json"

SUMMARIES_FILE = ROOT / "data/summaries/summaries_pilot.json"

OUTPUT_DIR = ROOT / "data/evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "preservation_results_pilot.json"

CATEGORIES = [
    "decision",
    "condition",
    "exception",
    "deadline",
    "responsibility",
    "numerical_limit"
]


# =====================================================
# API REQUEST WITH RETRIES
# =====================================================

def generate_response(prompt, max_attempts=5):

    for attempt in range(max_attempts):

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )

            if not response.text:
                raise ValueError("Empty Gemini response")

            return response.text.strip()

        except Exception as error:

            message = str(error)

            temporary = any(
                code in message
                for code in ["429", "500", "502", "503", "504"]
            )

            if not temporary or attempt == max_attempts - 1:
                raise

            wait = 2 ** (attempt + 1)

            print(f"Temporary API error. Retrying in {wait}s...")
            time.sleep(wait)


# =====================================================
# JSON RESPONSE PARSER
# =====================================================

def parse_json(response):

    cleaned = re.sub(
        r"```(?:json)?",
        "",
        response,
        flags=re.IGNORECASE
    ).strip()

    cleaned = cleaned.replace("```", "").strip()

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:

        start = cleaned.find("{")

        if start == -1:
            raise ValueError("No JSON object found in model response.")

        decoder = json.JSONDecoder()

        result, _ = decoder.raw_decode(cleaned[start:])

        return result


# =====================================================
# STEP 1: EXTRACT CONSTRAINTS FROM SOURCE DOCUMENT
# =====================================================

def extract_constraints(document):

    document_text = document["text"]

    prompt = f"""
You are performing research for a project called ConstraintPreserver.

Your task is to identify important constraints and requirements
in the original document.

Identify these six categories:

1. decision
2. condition
3. exception
4. deadline
5. responsibility
6. numerical_limit

Definitions:

DECISION:
A rule, decision, permission, prohibition or mandatory requirement.

CONDITION:
A prerequisite or circumstance under which something is allowed,
required or prohibited.

EXCEPTION:
An explicit exemption, exclusion or special case.

DEADLINE:
A date, time limit, duration or temporal restriction.

RESPONSIBILITY:
A person, role or group assigned an action or obligation.

NUMERICAL_LIMIT:
A number, threshold, quantity, percentage or numerical restriction
that affects an obligation or rule.

Instructions:

- Extract only information explicitly supported by the source.
- Preserve the original meaning.
- Keep conditions and exceptions specific.
- Do not invent constraints.
- Do not create duplicate constraints.
- Include the exact supporting quotation from the source.
- If a category is absent, do not invent an example.

Return ONLY valid JSON in this format:

{{
  "constraints": [
    {{
      "constraint_id": "C1",
      "category": "condition",
      "constraint": "Students must disclose their use of AI.",
      "source_evidence": "Exact quotation from source."
    }}
  ]
}}

SOURCE DOCUMENT:

{document_text}
"""

    response = generate_response(prompt)

    result = parse_json(response)

    constraints = result.get("constraints", [])

    for index, constraint in enumerate(constraints, start=1):

        constraint["constraint_id"] = f"C{index}"

        category = constraint.get("category", "").lower()

        if category not in CATEGORIES:
            raise ValueError(
                f"Unexpected constraint category: {category}"
            )

        if not constraint.get("constraint"):
            raise ValueError("Constraint text is missing.")

        if not constraint.get("source_evidence"):
            raise ValueError("Source evidence is missing.")

    return constraints


# =====================================================
# STEP 2: EVALUATE A SUMMARY
# =====================================================

def evaluate_summary(constraints, summary, summary_type):

    if not constraints:

        return []

    constraints_json = json.dumps(
        constraints,
        indent=2,
        ensure_ascii=False
    )

    prompt = f"""
You are an evaluator for the ConstraintPreserver research project.

Your job is to determine whether important constraints from an
original document survived in an AI-generated summary.

You will receive:

1. A list of constraints extracted from the original document.
2. An AI-generated summary.

Evaluate EVERY constraint independently.

Possible labels:

PRESERVED:
The summary accurately retains the constraint and its important details.

PARTIAL:
The summary mentions the general idea but loses an important detail,
such as who, when, under what condition, an exception, or a number.

LOST:
The constraint is absent or materially misrepresented.

Important rules:

- Do not assume that a constraint is preserved merely because
  the summary discusses the same general topic.
- Check exact conditions, exceptions, deadlines and responsibilities.
- Do not reward information that is contradicted by the summary.
- Do not use external knowledge.
- Do not invent evidence.
- Be strict and consistent.

Return ONLY valid JSON:

{{
  "evaluations": [
    {{
      "constraint_id": "C1",
      "status": "PRESERVED",
      "summary_evidence": "Exact supporting phrase from the summary.",
      "reason": "Why this constraint received this label."
    }}
  ]
}}

SUMMARY TYPE: {summary_type}

ORIGINAL CONSTRAINTS:

{constraints_json}

SUMMARY:

{summary}
"""

    response = generate_response(prompt)

    result = parse_json(response)

    evaluations = result.get("evaluations", [])

    expected_ids = {
        constraint["constraint_id"]
        for constraint in constraints
    }

    received_ids = [
        item.get("constraint_id")
        for item in evaluations
    ]

    if (
        len(received_ids) != len(expected_ids)
        or set(received_ids) != expected_ids
    ):
        raise ValueError(
            "Evaluator did not return exactly one result per constraint."
        )

    valid_statuses = {"PRESERVED", "PARTIAL", "LOST"}

    for item in evaluations:

        if item.get("status") not in valid_statuses:
            raise ValueError(
                f"Invalid evaluation status: {item.get('status')}"
            )

    return evaluations


# =====================================================
# STEP 3: CALCULATE PRESERVATION SCORES
# =====================================================

def calculate_scores(evaluations):

    if not evaluations:

        return {
            "total_constraints": 0,
            "preserved": 0,
            "partial": 0,
            "lost": 0,
            "preservation_score": None,
            "category_scores": {}
        }

    scores = {
        "PRESERVED": 1.0,
        "PARTIAL": 0.5,
        "LOST": 0.0
    }

    total = len(evaluations)

    preserved = sum(
        item["status"] == "PRESERVED"
        for item in evaluations
    )

    partial = sum(
        item["status"] == "PARTIAL"
        for item in evaluations
    )

    lost = sum(
        item["status"] == "LOST"
        for item in evaluations
    )

    overall_score = (
        sum(scores[item["status"]] for item in evaluations)
        / total
    ) * 100

    category_scores = {}

    for category in CATEGORIES:

        category_items = [
            item for item in evaluations
            if item["category"] == category
        ]

        if not category_items:
            continue

        category_score = (
            sum(scores[item["status"]] for item in category_items)
            / len(category_items)
        ) * 100

        category_scores[category] = {
            "count": len(category_items),
            "preservation_score": round(category_score, 2)
        }

    return {
        "total_constraints": total,
        "preserved": preserved,
        "partial": partial,
        "lost": lost,
        "preservation_score": round(overall_score, 2),
        "category_scores": category_scores
    }


# =====================================================
# STEP 4: RUN EXPERIMENT
# =====================================================

def main(limit):

    with DOCUMENTS_FILE.open(
        "r", encoding="utf-8"
    ) as file:

        documents = json.load(file)

    with SUMMARIES_FILE.open(
        "r", encoding="utf-8"
    ) as file:

        summaries = json.load(file)

    document_lookup = {
        document["document_id"]: document
        for document in documents
    }

    results = []

    print("=" * 65)
    print("CONSTRAINTPRESERVER - PRESERVATION EVALUATION")
    print("=" * 65)

    if limit > 0:
        summaries = summaries[:limit]

    for index, item in enumerate(summaries, start=1):

        document_id = item["document_id"]

        document = document_lookup.get(document_id)

        if document is None:
            raise ValueError(
                f"Original document not found: {document_id}"
            )

        print(
            f"\n[{index}/{len(summaries)}] "
            f"Processing {document_id}"
        )

        # Extract constraints once from the original source.
        print("   Extracting original constraints...")

        constraints = extract_constraints(document)

        print(f"   Found {len(constraints)} constraints.")

        # Evaluate ordinary summary.
        print("   Evaluating ordinary summary...")

        baseline_evaluations = evaluate_summary(
            constraints,
            item["baseline_summary"],
            "ordinary"
        )

        # Evaluate constraint-aware summary.
        print("   Evaluating constraint-aware summary...")

        aware_evaluations = evaluate_summary(
            constraints,
            item["constraint_aware_summary"],
            "constraint-aware"
        )

        # Attach category information.
        category_lookup = {
            c["constraint_id"]: c["category"]
            for c in constraints
        }

        for evaluation in (
            baseline_evaluations + aware_evaluations
        ):

            evaluation["category"] = category_lookup[
                evaluation["constraint_id"]
            ]

        baseline_scores = calculate_scores(
            baseline_evaluations
        )

        aware_scores = calculate_scores(
            aware_evaluations
        )

        result = {

            "document_id": document_id,

            "filename": item["filename"],

            "model": MODEL,

            "constraints": constraints,

            "baseline": {
                "evaluations": baseline_evaluations,
                "scores": baseline_scores
            },

            "constraint_aware": {
                "evaluations": aware_evaluations,
                "scores": aware_scores
            },

            "score_difference": (
                round(
                    aware_scores["preservation_score"]
                    - baseline_scores["preservation_score"],
                    2
                )
                if baseline_scores["preservation_score"] is not None
                and aware_scores["preservation_score"] is not None
                else None
            )
        }

        results.append(result)

        # Save after every document.
        with OUTPUT_FILE.open(
            "w", encoding="utf-8"
        ) as file:

            json.dump(
                results,
                file,
                indent=4,
                ensure_ascii=False
            )

        print(
            "   Ordinary preservation:",
            baseline_scores["preservation_score"]
        )

        print(
            "   Constraint-aware preservation:",
            aware_scores["preservation_score"]
        )

        print("   Document completed.")

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETED")
    print("Documents evaluated:", len(results))
    print("Results:", OUTPUT_FILE)
    print("=" * 65)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=3
    )

    args = parser.parse_args()

    main(args.limit)