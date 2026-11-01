import os
import json
import time
import argparse

from pathlib import Path
from dotenv import load_dotenv
from google import genai


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "cleaned_documents.json"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "summaries"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PILOT_OUTPUT_FILE = OUTPUT_DIR / "summaries_pilot.json"
FULL_OUTPUT_FILE = OUTPUT_DIR / "summaries_full.json"

MAX_CHARS = 16000

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. "
        "Add it to your project's .env file."
    )

client = genai.Client(api_key=API_KEY)


# --------------------------------------------------
# PROMPTS
# --------------------------------------------------

BASELINE_PROMPT = """
You are an expert document summarization system.

Produce a concise, accurate summary of the supplied text.

Requirements:
- Capture the main purpose and central ideas.
- Include the most important information.
- Avoid unnecessary repetition.
- Do not invent information.
- Maintain factual accuracy.

Do not add information that is absent from the source.
"""


CONSTRAINT_AWARE_PROMPT = """
You are an expert document summarization system.

Produce a concise and accurate summary of the supplied text.

In addition to capturing its main ideas, carefully preserve
the following types of information whenever present:

1. Decisions and mandatory requirements.
2. Conditions and prerequisites.
3. Exceptions, exclusions and limitations.
4. Deadlines and time restrictions.
5. Responsibilities and assigned roles.
6. Numerical thresholds, quantities and percentages.

Do not replace specific conditions with vague statements.
Do not omit exceptions simply because they appear less frequently.
Preserve who is responsible for an action and when it must happen.

Do not invent information or infer unsupported requirements.
"""


# --------------------------------------------------
# CHECKPOINT MANAGEMENT
# --------------------------------------------------

def load_checkpoint(output_file):

    if not output_file.exists():
        print("No previous checkpoint found. Starting fresh.")
        return {}

    with output_file.open("r", encoding="utf-8") as file:
        saved_results = json.load(file)

    if not isinstance(saved_results, list):
        raise ValueError("Checkpoint file must contain a list.")

    results = {}

    for item in saved_results:
        document_id = item.get("document_id")

        if document_id:
            results[document_id] = item

    print(
        f"Checkpoint loaded: {len(results)} previously saved records."
    )

    return results


def save_checkpoint(results, documents, output_file):

    ordered_results = []

    for document in documents:
        document_id = document.get("document_id")

        if document_id in results:
            ordered_results.append(results[document_id])

    temporary_file = output_file.with_suffix(".tmp")

    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(
            ordered_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Replace the old checkpoint only after the new file is written.
    temporary_file.replace(output_file)


def is_complete(result):

    return (
        bool(result.get("baseline_summary", "").strip())
        and bool(result.get("constraint_aware_summary", "").strip())
    )


# --------------------------------------------------
# TEXT PROCESSING
# --------------------------------------------------

def split_text(text, max_chars=MAX_CHARS):

    paragraphs = [
        p.strip()
        for p in text.split("\n")
        if p.strip()
    ]

    segments = []
    current = ""

    for paragraph in paragraphs:

        if len(paragraph) > max_chars:

            if current:
                segments.append(current)
                current = ""

            for start in range(0, len(paragraph), max_chars):
                segments.append(
                    paragraph[start:start + max_chars]
                )

            continue

        if len(current) + len(paragraph) > max_chars:

            if current:
                segments.append(current)

            current = paragraph

        else:
            current += "\n\n" + paragraph if current else paragraph

    if current:
        segments.append(current)

    return segments


# --------------------------------------------------
# GEMINI GENERATION
# --------------------------------------------------

def generate_response(prompt, text):

    max_attempts = 5

    for attempt in range(max_attempts):

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=f"{prompt}\n\nSOURCE TEXT:\n{text}",
            )

            if not response.text:
                raise ValueError(
                    "The model returned an empty response."
                )

            return response.text.strip()

        except Exception as error:

            error_message = str(error)

            temporary_error = any(
                code in error_message
                for code in ["429", "500", "502", "503", "504"]
            )

            if not temporary_error:
                raise

            if attempt == max_attempts - 1:
                raise

            wait_time = 2 ** (attempt + 1)

            print(
                f"Temporary Gemini error. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)


def summarize_document(text, prompt):

    segments = split_text(text)
    partial_summaries = []

    for index, segment in enumerate(segments, start=1):

        print(
            f"      Summarizing segment "
            f"{index}/{len(segments)}"
        )

        summary = generate_response(prompt, segment)
        partial_summaries.append(summary)

        time.sleep(1)

    combined_text = "\n\n".join(partial_summaries)

    if len(segments) == 1:
        return combined_text

    final_prompt = prompt + """

You are given intermediate summaries of different parts
of the same document.

Combine them into one coherent summary of approximately
200-250 words.

Preserve important information and avoid repetition.
Do not introduce facts absent from the intermediate summaries.
"""

    return generate_response(final_prompt, combined_text)


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main(limit):

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    if not isinstance(documents, list):
        raise ValueError("Expected a list of documents.")

    # Separate pilot and full experiment outputs.
    output_file = (
        FULL_OUTPUT_FILE if limit == 0
        else PILOT_OUTPUT_FILE
    )

    if limit > 0:
        documents = documents[:limit]

    # Load previously completed work.
    results = load_checkpoint(output_file)

    print("=" * 60)
    print("CONSTRAINTPRESERVER - SUMMARIZATION")
    print("=" * 60)

    print("Model:", MODEL)
    print("Documents in this run:", len(documents))
    print("Output file:", output_file.name)

    completed_count = 0
    skipped_count = 0

    for index, document in enumerate(documents, start=1):

        document_id = document.get("document_id")
        text = document.get("text", "")

        if not document_id:
            print("Skipping document without an ID.")
            continue

        if not text.strip():
            print(f"Skipping empty document: {document_id}")
            continue

        # Skip documents that already have both summaries.
        if (
            document_id in results
            and is_complete(results[document_id])
        ):

            print(
                f"\n[{index}/{len(documents)}] "
                f"{document_id} - Already completed. Skipping."
            )

            skipped_count += 1
            continue

        print(
            f"\n[{index}/{len(documents)}] "
            f"{document_id}"
        )

        try:

            print("   Generating ordinary summary...")

            baseline = summarize_document(
                text,
                BASELINE_PROMPT
            )

            print("   Generating constraint-aware summary...")

            constraint_aware = summarize_document(
                text,
                CONSTRAINT_AWARE_PROMPT
            )

            results[document_id] = {

                "document_id": document_id,

                "filename": document.get("filename"),

                "source_word_count": len(text.split()),

                "model": MODEL,

                "baseline_summary": baseline,

                "constraint_aware_summary": constraint_aware,

            }

            # Save immediately after each completed document.
            save_checkpoint(
                results,
                documents,
                output_file
            )

            completed_count += 1

            print("   Completed and checkpoint saved.")

        except Exception as error:

            # Preserve all previously completed documents.
            save_checkpoint(
                results,
                documents,
                output_file
            )

            print("\nERROR processing:", document_id)
            print("Reason:", error)
            print("Checkpoint saved.")
            print("Run the same command again to resume.")

            raise

    print("\n" + "=" * 60)
    print("SUMMARIZATION RUN FINISHED")
    print("=" * 60)

    print("Newly processed:", completed_count)
    print("Already completed/skipped:", skipped_count)
    print(
        "Total completed:",
        sum(1 for item in results.values() if is_complete(item))
    )

    print("Output:", output_file)


# --------------------------------------------------
# COMMAND LINE
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of documents to process. Use 0 for all."
    )

    args = parser.parse_args()

    main(args.limit)