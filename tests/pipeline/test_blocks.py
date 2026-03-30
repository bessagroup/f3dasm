"""Tests for built-in pipeline blocks."""

import pytest

from f3dasm._src.pipeline.blocks import CollectArrayResults

pytestmark = pytest.mark.smoke


class TestCollectArrayResults:
    def test_default_cleanup(self):
        block = CollectArrayResults()
        assert block.cleanup is True

    def test_no_cleanup(self):
        block = CollectArrayResults(cleanup=False)
        assert block.cleanup is False
