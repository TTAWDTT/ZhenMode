"""Migration reporting must refuse lost, colliding or silently altered tests."""
import pytest

from scripts.capture_test_collection import compare_collections


def test_mapping_preserves_full_parameter_identifiers_and_records_new_nodes():
    result = compare_collections(
        {"nodeids": ["tests/test_old.py::test_state[float32]", "tests/test_old.py::test_state[float64]"]},
        {"nodeids": ["tests/fd/test_old.py::test_state[float32]", "tests/fd/test_old.py::test_state[float64]", "tests/new.py::test_contract"]},
        {"tests/test_old.py": "tests/fd/test_old.py"},
    )
    assert result["preserved_nodes"] == 2
    assert result["new_nodes"] == ["tests/new.py::test_contract"]


@pytest.mark.parametrize("candidate,moves,error", [
    (["tests/fd/test_old.py::test_state[float64]"], {"tests/test_old.py": "tests/fd/test_old.py"}, "missing original"),
    (["tests/fd/test_old.py::test_state[float32]"] * 2, {"tests/test_old.py": "tests/fd/test_old.py"}, "duplicate"),
    (["tests/fd/test_old.py::test_state[float32]"], {}, "no declared move"),
])
def test_mapping_refuses_missing_changed_duplicate_or_unmapped_nodes(candidate, moves, error):
    with pytest.raises(ValueError, match=error):
        compare_collections({"nodeids": ["tests/test_old.py::test_state[float32]"]},
                            {"nodeids": candidate}, moves)
