# -*- coding: utf-8 -*-
"""Example: BibTeX reference verification."""

from cookbooks.paper_review.processors.bib_checker import BibChecker, VerificationStatus


def main():
    # Initialize checker (provide email for better rate limits)
    checker = BibChecker(mailto="your-email@example.com")

    # Example 1: Check a .bib file
    print("=" * 60)
    print("BIB FILE VERIFICATION")
    print("=" * 60)

    # Create a sample bib content for demonstration
    sample_bib = """
    @article{vaswani2017attention,
        title={Attention is All You Need},
        author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob},
        journal={Advances in Neural Information Processing Systems},
        year={2017}
    }

    @article{devlin2018bert,
        title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
        author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
        journal={arXiv preprint arXiv:1810.04805},
        year={2018}
    }

    @article{fake2024paper,
        title={This Paper Does Not Exist: A Fake Reference},
        author={Fake, Author and Nobody, Else},
        journal={Journal of Imaginary Research},
        year={2024}
    }
    """

    # Parse and verify
    references = checker.parse_bib_file(sample_bib)
    print(f"\nFound {len(references)} references\n")

    results = checker.verify_all(references, max_workers=3)

    for r in results:
        status_icon = "✓" if r.status == VerificationStatus.VERIFIED else "✗"
        print(f"{status_icon} [{r.status.value.upper()}] {r.reference.title}")
        print(f"  Confidence: {r.confidence:.0%}")
        print(f"  Message: {r.message}\n")

    # Summary
    verified = sum(1 for r in results if r.status == VerificationStatus.VERIFIED)
    suspect = sum(1 for r in results if r.status == VerificationStatus.SUSPECT)

    print("-" * 60)
    print(f"Summary: {verified} verified, {suspect} suspect")
    print(f"Verification rate: {verified / len(results):.0%}")


if __name__ == "__main__":
    main()
