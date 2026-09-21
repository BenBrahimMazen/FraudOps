"""Phase 0 smoke test: the package imports and exposes its version."""


def test_package_imports_with_version() -> None:
    import fraudops

    assert isinstance(fraudops.__version__, str)
    assert fraudops.__version__ == "0.1.0"
