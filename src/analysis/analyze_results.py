import json
import csv
from pathlib import Path
from collections import defaultdict
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data/evaluation/preservation_results_pilot.json"
)

OUTPUT_DIR = ROOT / "data/analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SCORES = {
    "PRESERVED": 1.0,
    "PARTIAL": 0.5,
    "LOST": 0.0
}


def load_results():

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def calculate_average(results, summary_type):

    scores = []

    for document in results:

        score = document[summary_type]["scores"][
            "preservation_score"
        ]

        if score is not None:
            scores.append(score)

    return round(mean(scores), 2) if scores else None


def analyze_categories(results):

    category_data = defaultdict(
        lambda: {
            "ordinary": [],
            "constraint_aware": []
        }
    )

    for document in results:

        for summary_type in [
            "baseline",
            "constraint_aware"
        ]:

            for evaluation in document[summary_type]["evaluations"]:

                category = evaluation["category"]

                score = SCORES[evaluation["status"]]

                label = (
                    "ordinary"
                    if summary_type == "baseline"
                    else "constraint_aware"
                )

                category_data[category][label].append(score)

    report = {}

    for category, values in category_data.items():

        ordinary = values["ordinary"]
        aware = values["constraint_aware"]

        ordinary_score = (
            round(mean(ordinary) * 100, 2)
            if ordinary else None
        )

        aware_score = (
            round(mean(aware) * 100, 2)
            if aware else None
        )

        report[category] = {

            "constraint_count": len(ordinary),

            "ordinary_preservation": ordinary_score,

            "constraint_aware_preservation": aware_score,

            "improvement": (
                round(aware_score - ordinary_score, 2)
                if ordinary_score is not None
                and aware_score is not None
                else None
            )
        }

    return report


def export_document_csv(results):

    output = OUTPUT_DIR / "document_comparison.csv"

    fields = [
        "document_id",
        "filename",
        "total_constraints",
        "ordinary_score",
        "constraint_aware_score",
        "improvement"
    ]

    with output.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        for document in results:

            writer.writerow({

                "document_id": document["document_id"],

                "filename": document["filename"],

                "total_constraints": document["baseline"][
                    "scores"
                ]["total_constraints"],

                "ordinary_score": document["baseline"][
                    "scores"
                ]["preservation_score"],

                "constraint_aware_score": document[
                    "constraint_aware"
                ]["scores"]["preservation_score"],

                "improvement": document["score_difference"]
            })

    print("Created:", output)


def export_category_csv(category_report):

    output = OUTPUT_DIR / "category_comparison.csv"

    fields = [
        "category",
        "constraint_count",
        "ordinary_preservation",
        "constraint_aware_preservation",
        "improvement"
    ]

    with output.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        for category, values in category_report.items():

            writer.writerow({
                "category": category,
                **values
            })

    print("Created:", output)


def main():

    results = load_results()

    if not results:
        print("No evaluation results found.")
        return

    ordinary_average = calculate_average(
        results,
        "baseline"
    )

    aware_average = calculate_average(
        results,
        "constraint_aware"
    )

    improvement = round(
        aware_average - ordinary_average,
        2
    )

    category_report = analyze_categories(results)

    report = {

        "experiment": "ConstraintPreserver Pilot",

        "documents_evaluated": len(results),

        "average_ordinary_preservation": ordinary_average,

        "average_constraint_aware_preservation": aware_average,

        "average_improvement_percentage_points": improvement,

        "category_analysis": category_report

    }

    output = OUTPUT_DIR / "research_report.json"

    with output.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    export_document_csv(results)

    export_category_csv(category_report)

    print("\n" + "=" * 60)
    print("CONSTRAINTPRESERVER - RESEARCH ANALYSIS")
    print("=" * 60)

    print("Documents evaluated:", len(results))

    print(
        "Ordinary preservation:",
        ordinary_average,
        "%"
    )

    print(
        "Constraint-aware preservation:",
        aware_average,
        "%"
    )

    print("Improvement:", improvement, "percentage points")

    print("\nCATEGORY-WISE RESULTS")

    for category, values in category_report.items():

        print("\n", category.upper())

        print(
            "Ordinary:",
            values["ordinary_preservation"]
        )

        print(
            "Constraint-aware:",
            values["constraint_aware_preservation"]
        )

        print(
            "Improvement:",
            values["improvement"]
        )

    print("\nResearch report:", output)
    print("=" * 60)


if __name__ == "__main__":
    main()