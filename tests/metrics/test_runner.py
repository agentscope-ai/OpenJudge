"""
Test Runner for Grader Architecture

Quick test to verify core Grader functionality works before running full test suite.
"""

import pytest


def test_basic_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    # Test imports for Grader architecture - avoid circular import by importing from specific modules
    from rm_gallery.core.metrics.format_check.json_match import (  # noqa: F401
        JsonMatchGrader,
    )
    from rm_gallery.core.metrics.string_check.exact_match import (  # noqa: F401
        ExactMatchGrader,
    )
    from rm_gallery.core.metrics.text_similarity.fuzzy import (  # noqa: F401
        FuzzyMatchGrader,
    )

    print("✓ All imports successful")


@pytest.mark.asyncio
async def test_fuzzy_match():
    """Test fuzzy match grader"""
    print("\nTesting Fuzzy Match...")

    from rm_gallery.core.metrics.text_similarity.fuzzy import FuzzyMatchGrader

    grader = FuzzyMatchGrader()
    result = await grader.evaluate(reference="hello world", candidate="hello world")

    if result.score == 1.0:
        print(f"✓ Fuzzy match works: {result.score}")
        return True
    else:
        print(f"✗ Unexpected score: {result.score}")
        return False


@pytest.mark.asyncio
async def test_exact_match():
    """Test exact match grader"""
    print("\nTesting Exact Match...")

    from rm_gallery.core.metrics.string_check.exact_match import ExactMatchGrader

    grader = ExactMatchGrader()
    result = await grader.evaluate(reference="test", candidate="test")

    if result.score == 1.0:
        print(f"✓ Exact match works: {result.score}")
        return True
    else:
        print(f"✗ Unexpected score: {result.score}")
        return False


@pytest.mark.asyncio
async def test_json_match():
    """Test JSON match grader"""
    print("\nTesting JSON Match...")

    from rm_gallery.core.metrics.format_check.json_match import JsonMatchGrader

    grader = JsonMatchGrader()
    result = await grader.evaluate(
        reference='{"name": "Alice"}', candidate='{"name": "Alice"}'
    )

    if result.score == 1.0:
        print(f"✓ JSON match works: {result.score}")
        return True
    else:
        print(f"✗ Unexpected score: {result.score}")
        return False


@pytest.mark.asyncio
async def test_multiple_graders():
    """Test using multiple graders"""
    print("\nTesting Multiple Graders...")

    from rm_gallery.core.metrics.nlp_metrics.bleu import BLEUGrader
    from rm_gallery.core.metrics.string_check.exact_match import ExactMatchGrader
    from rm_gallery.core.metrics.text_similarity.fuzzy import FuzzyMatchGrader

    graders = [
        FuzzyMatchGrader(),
        ExactMatchGrader(),
        BLEUGrader(),
    ]

    reference = "the cat sat on the mat"
    candidate = "the cat sat on the mat"

    for grader in graders:
        result = await grader.evaluate(reference=reference, candidate=candidate)
        print(f"  - {grader.name}: {result.score:.4f}")

    print("✓ Multiple graders work")
    return True


def main():
    """Run all quick tests"""
    import asyncio

    print("=" * 60)
    print("Grader Architecture Test Runner")
    print("=" * 60)

    # Run async tests
    loop = asyncio.get_event_loop()

    tests = [
        test_basic_imports,
    ]

    async_tests = [
        test_fuzzy_match,
        test_exact_match,
        test_json_match,
        test_multiple_graders,
    ]

    passed = 0
    failed = 0

    # Run sync tests
    for test in tests:
        try:
            if test() is not False:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} raised exception: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    # Run async tests
    for test in async_tests:
        try:
            if loop.run_until_complete(test()) is not False:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} raised exception: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
