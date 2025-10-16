"""
Basic tests for blackrl package.
"""

import blackrl


def test_version():
    """Test that version is defined."""
    assert hasattr(blackrl, "__version__")
    assert isinstance(blackrl.__version__, str)
    assert len(blackrl.__version__) > 0


def test_package_import():
    """Test that the package can be imported."""
    import blackrl

    assert blackrl is not None
