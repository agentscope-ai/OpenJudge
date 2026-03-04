# -*- coding: utf-8 -*-
"""Example: Standalone correctness check."""

import asyncio
import os

from cookbooks.paper_review.graders import CorrectnessGrader
from cookbooks.paper_review.utils import encode_pdf_base64, load_pdf_bytes
from openjudge.models.openai_chat_model import OpenAIChatModel


async def main():
    print("=" * 60)
    print("CORRECTNESS CHECK")
    print("=" * 60)

    # Initialize model
    model = OpenAIChatModel(
        model="gpt-4o",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    # Create grader
    grader = CorrectnessGrader(model)

    # Load and encode PDF
    pdf_bytes = load_pdf_bytes("paper.pdf")
    pdf_data = encode_pdf_base64(pdf_bytes)

    # Run evaluation
    print("\nAnalyzing paper for objective errors...")
    result = await grader.aevaluate(pdf_data=pdf_data)

    # Display results
    score_labels = {
        1: "No objective errors detected",
        2: "Minor errors present",
        3: "Major errors present",
    }

    print(f"\nScore: {result.score}/3 - {score_labels.get(result.score, 'Unknown')}")
    print(f"\nReasoning:\n{result.reason[:1000]}...")

    key_issues = result.metadata.get("key_issues", [])
    if key_issues:
        print(f"\nKey Issues ({len(key_issues)}):")
        for i, issue in enumerate(key_issues, 1):
            print(f"  {i}. {issue}")


if __name__ == "__main__":
    asyncio.run(main())
