import json
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EVALUATION_FILE = ROOT / "data/evaluation/preservation_results_pilot.json"
SUMMARY_FILE = ROOT / "data/summaries/summaries_pilot.json"

OUTPUT_DIR = ROOT / "data/annotations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "human_annotation_dataset.csv"


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_value(record, possible_keys, default=""):
    for key in possible_keys:
        value = record.get(key)
        if value is not None:
            return value
    return default


def get_summary_document(summaries, document_id):
    if isinstance(summaries, list):
        for item in summaries:
            if item.get("document_id") == document_id:
                return item

    elif isinstance(summaries, dict):
        if document_id in summaries:
            return summaries[document_id]

        for item in summaries.values():
            if isinstance(item, dict):
                if item.get("document_id") == document_id:
                    return item

    return {}


def main():
    evaluations = load_json(EVALUATION_FILE)
    summaries = load_json(SUMMARY_FILE)

    rows = []

    for document in evaluations:
        document_id = document["document_id"]
        filename = document.get("filename", "")

        baseline_evaluations = document["baseline"]["evaluations"]
        aware_evaluations = document["constraint_aware"]["evaluations"]

        # Map original constraints using constraint IDs
        constraints_by_id = {
            item["constraint_id"]: item
            for item in document.get("constraints", [])
        }

        # Retrieve summaries
        summary_document = get_summary_document(
            summaries, document_id
        )

        ordinary_summary = get_value(
            summary_document,
            ["baseline_summary", "ordinary_summary", "baseline"]
        )

        aware_summary = get_value(
            summary_document,
            ["constraint_aware_summary", "aware_summary"]
        )

        if isinstance(ordinary_summary, dict):
            ordinary_summary = ordinary_summary.get("summary", "")

        if isinstance(aware_summary, dict):
            aware_summary = aware_summary.get("summary", "")

        if len(baseline_evaluations) != len(aware_evaluations):
            raise ValueError(
                f"Evaluation count mismatch for {document_id}"
            )

        # Generate one annotation row per constraint
        for index, baseline in enumerate(baseline_evaluations):

            aware = aware_evaluations[index]

            constraint_id = baseline["constraint_id"]

            # Retrieve original constraint and source evidence
            original = constraints_by_id.get(constraint_id, {})

            constraint = original.get("constraint", "")
            evidence = original.get("source_evidence", "")

            category = original.get(
                "category",
                baseline.get("category", "")
            )

            if not constraint:
                raise ValueError(
                    f"Missing original constraint: "
                    f"{document_id} - {constraint_id}"
                )

            rows.append({
                "document_id": document_id,
                "filename": filename,
                "constraint_number": index + 1,
                "constraint_category": category,
                "original_constraint": constraint,
                "source_evidence": evidence,
                "ordinary_summary": ordinary_summary,
                "constraint_aware_summary": aware_summary,
                "gemini_ordinary_status": baseline.get("status", ""),
                "gemini_aware_status": aware.get("status", ""),
                "human_ordinary_status": "",
                "human_aware_status": "",
                "human_notes": ""
            })

    if not rows:
        raise ValueError("No annotation records were generated.")

    fields = list(rows[0].keys())

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 60)
    print("CONSTRAINTPRESERVER - HUMAN ANNOTATION DATASET")
    print("=" * 60)
    print("Documents:", len(evaluations))
    print("Constraints:", len(rows))
    print(
        "Constraints with source evidence:",
        sum(1 for row in rows if row["source_evidence"])
    )
    print(
        "Output:",
        OUTPUT_FILE
    )
    print("=" * 60)


if __name__ == "__main__":
    main()