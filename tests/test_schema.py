"""Feature schema integrity: disjoint groups, no duplicates."""

from fraudops.features import schema


def test_numeric_and_categorical_groups_are_disjoint() -> None:
    numeric = schema.NUMERIC_TX + schema.NUMERIC_ID
    categorical = schema.CATEGORICAL_TX + schema.CATEGORICAL_ID
    overlap = set(numeric) & set(categorical)
    assert not overlap, f"columns in both numeric and categorical groups: {sorted(overlap)}"


def test_no_column_listed_twice() -> None:
    all_lists = {
        "CATEGORICAL_TX": schema.CATEGORICAL_TX,
        "NUMERIC_TX": schema.NUMERIC_TX,
        "CATEGORICAL_ID": schema.CATEGORICAL_ID,
        "NUMERIC_ID": schema.NUMERIC_ID,
        "FLAG_COLUMNS": schema.FLAG_COLUMNS,
    }
    for name, cols in all_lists.items():
        assert len(cols) == len(set(cols)), f"duplicates in {name}"


def test_schema_matches_loader_dtype_classification() -> None:
    """Storage dtype and model role are related but distinct contracts.

    The loader classifies *storage* (string columns must be read as category,
    numeric columns as float32). The schema classifies *model role*. Every
    string column must take the categorical path in the model — a string
    cannot be a numeric feature. The converse is NOT required: numeric-valued
    columns (card2/card5, addr1/addr2) may be categorical features.
    """
    from fraudops.data import loader

    loader_categorical = set(loader._TX_CATEGORY_COLUMNS + loader._ID_CATEGORY_COLUMNS)
    schema_categorical = set(schema.CATEGORICAL_TX + schema.CATEGORICAL_ID)
    assert loader_categorical <= schema_categorical, (
        "string columns that the model would treat as numeric: "
        f"{sorted(loader_categorical - schema_categorical)}"
    )
