"""Shared repository location; production imports use the installed package."""
import pytest

from tests.support.paths import REPOSITORY_ROOT


@pytest.fixture
def repository_root():
    return REPOSITORY_ROOT
