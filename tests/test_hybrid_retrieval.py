from __future__ import annotations

from collections.abc import Sequence

from semantic_text2sql.context import build_context_plan, model_context_payload
from semantic_text2sql.glossary import BusinessGlossary, GlossaryTerm
from semantic_text2sql.hybrid_retrieval import HybridSchemaRetriever, metadata_requests
from semantic_text2sql.models import (
    ColumnInfo,
    ColumnProfile,
    ContextRequest,
    DatabaseProfile,
    ForeignKeyInfo,
    PlannerMetadataRequirement,
    SchemaInfo,
    SemanticContract,
    TableInfo,
    TableProfile,
    ValueFrequency,
)


class KeywordEncoder:
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vocabulary = ("customer", "currency", "order", "date", "amount")
        return [[float(text.casefold().count(token)) for token in vocabulary] for text in texts]


class FixedReranker:
    def score(self, features: list[list[float]]) -> list[float]:
        return [float(index) for index, _ in enumerate(features)]


class BrokenReranker:
    def score(self, features: list[list[float]]) -> list[float]:
        raise RuntimeError("artifact failure")


def _column_profile(
    table: str,
    column: str,
    data_type: str,
    semantic_type: str,
    *,
    values: list[str] | None = None,
    observed_format: str | None = None,
    minimum: str | None = None,
    maximum: str | None = None,
    null_count: int = 0,
) -> ColumnProfile:
    return ColumnProfile(
        table=table,
        column=column,
        database_type=data_type,
        semantic_type=semantic_type,
        row_count=10,
        null_count=null_count,
        null_ratio=null_count / 10,
        distinct_count=2,
        top_values=[ValueFrequency(value=value, count=5) for value in values or []],
        examples=values or [],
        observed_format=observed_format,
        minimum=minimum,
        maximum=maximum,
    )


def _schema() -> SchemaInfo:
    return SchemaInfo(
        db_id="shop",
        tables=[
            TableInfo(
                name="customers",
                create_sql="",
                columns=[
                    ColumnInfo(name="customer_id", data_type="INTEGER", primary_key=True),
                    ColumnInfo(name="currency", data_type="TEXT", aliases=["payment currency"]),
                ],
            ),
            TableInfo(
                name="orders",
                create_sql="",
                columns=[
                    ColumnInfo(name="order_id", data_type="INTEGER", primary_key=True),
                    ColumnInfo(name="customer_id", data_type="INTEGER"),
                    ColumnInfo(name="order_date", data_type="TEXT", semantic_type="date"),
                    ColumnInfo(name="amount", data_type="REAL"),
                ],
            ),
        ],
        relationships=[
            ForeignKeyInfo(
                from_table="orders",
                from_column="customer_id",
                to_table="customers",
                to_column="customer_id",
            )
        ],
    )


def _profile() -> DatabaseProfile:
    return DatabaseProfile(
        db_id="shop",
        dialect="sqlite",
        profiled_at="2026-08-21",
        tables=[
            TableProfile(table="customers", summary="customer accounts", grain="customer_id"),
            TableProfile(table="orders", summary="purchases", grain="order_id"),
        ],
        columns=[
            _column_profile("customers", "customer_id", "INTEGER", "identifier"),
            _column_profile("customers", "currency", "TEXT", "categorical", values=["EUR", "CZK"]),
            _column_profile("orders", "order_id", "INTEGER", "identifier"),
            _column_profile("orders", "customer_id", "INTEGER", "identifier"),
            _column_profile(
                "orders",
                "order_date",
                "TEXT",
                "date",
                values=["2026-01-04", "2026-03-21"],
                observed_format="YYYY-MM-DD",
                minimum="2026-01-04",
                maximum="2026-03-21",
            ),
            _column_profile("orders", "amount", "REAL", "numeric", minimum="1.0", maximum="500.0"),
        ],
    )


def test_hybrid_retrieval_fuses_rankings_and_preserves_keys() -> None:
    retriever = HybridSchemaRetriever(KeywordEncoder(), max_tables=2, max_columns_per_table=2)
    _, selection, trace = retriever.retrieve(
        "EUR customer order amount", None, _schema(), _profile(), None
    )

    assert selection.tables == ["customers", "orders"]
    assert "customer_id" in selection.columns["customers"]
    assert "customer_id" in selection.columns["orders"]
    assert trace.bm25_ranks
    assert trace.embedding_ranks
    assert trace.value_match_ranks["column:customers.currency"] == 1
    assert trace.rrf_scores


def test_optional_ml_reranker_scores_rrf_pool_before_key_restoration() -> None:
    retriever = HybridSchemaRetriever(
        KeywordEncoder(), reranker=FixedReranker(), max_tables=2, max_columns_per_table=2
    )
    _, selection, trace = retriever.retrieve(
        "EUR customer order amount", None, _schema(), _profile(), None
    )

    assert trace.ml_reranker_applied is True
    assert trace.ml_reranker_scores
    assert "customer_id" in selection.columns["customers"]
    assert "customer_id" in selection.columns["orders"]


def test_ml_reranker_failure_falls_back_to_rrf() -> None:
    retriever = HybridSchemaRetriever(KeywordEncoder(), reranker=BrokenReranker())
    _, _, trace = retriever.retrieve("customer currency", None, _schema(), _profile(), None)

    assert trace.ml_reranker_applied is False
    assert trace.ml_reranker_scores == {}


def test_glossary_dependencies_and_bridge_tables_are_restored() -> None:
    schema = _schema()
    bridge = TableInfo(
        name="customer_regions",
        create_sql="",
        columns=[
            ColumnInfo(name="customer_id", data_type="INTEGER"),
            ColumnInfo(name="region_id", data_type="INTEGER"),
        ],
    )
    regions = TableInfo(
        name="regions",
        create_sql="",
        columns=[ColumnInfo(name="region_id", data_type="INTEGER", primary_key=True)],
    )
    schema = schema.model_copy(
        update={
            "tables": [*schema.tables, bridge, regions],
            "relationships": [
                *schema.relationships,
                ForeignKeyInfo(
                    from_table="customer_regions",
                    from_column="customer_id",
                    to_table="customers",
                    to_column="customer_id",
                ),
                ForeignKeyInfo(
                    from_table="customer_regions",
                    from_column="region_id",
                    to_table="regions",
                    to_column="region_id",
                ),
            ],
        }
    )
    glossary = BusinessGlossary(
        db_id="shop",
        version="1",
        source="test",
        precedence=[],
        terms=[
            GlossaryTerm(
                term="regional customer",
                definition="customer assigned to a region",
                columns=["regions.region_id", "customers.customer_id"],
            )
        ],
    )
    retriever = HybridSchemaRetriever(KeywordEncoder(), max_tables=1, max_columns_per_table=1)
    _, selection, trace = retriever.retrieve(
        "regional customer", None, schema, _profile(), glossary
    )

    assert {"customers", "customer_regions", "regions"} <= set(selection.tables)
    assert trace.bridge_tables_added == ["customer_regions"]
    assert "region_id" in selection.columns["regions"]


def test_grounding_propagates_grain_null_text_and_date_metadata() -> None:
    schema = _schema()
    profile = _profile()
    request = ContextRequest(
        tables=["customers", "orders"],
        columns={
            "customers": ["customer_id", "currency"],
            "orders": ["customer_id", "order_date", "amount"],
        },
        metadata_requirements=[
            PlannerMetadataRequirement(kind="DATE_FORMAT", targets=["orders.order_date"]),
            PlannerMetadataRequirement(kind="CATEGORY_VALUES", targets=["customers.currency"]),
            PlannerMetadataRequirement(kind="NUMERIC_PROFILE", targets=["orders.amount"]),
        ],
    )
    plan = build_context_plan(
        schema, profile, SemanticContract(), "EUR orders above 100 in 2026", None, request
    )
    payload = model_context_payload(
        plan, profile, SemanticContract(), None, "EUR orders above 100 in 2026", "sqlite"
    )

    assert payload["tables"]["orders"]["grain"] == "order_id"
    currency = payload["tables"]["customers"]["columns"]["currency"]
    assert currency["observed_nulls"] is False
    assert currency["top_values"] == ["EUR", "CZK"]
    date = payload["tables"]["orders"]["columns"]["order_date"]
    assert date["observed_format"] == "YYYY-MM-DD"
    assert "examples" not in date
    assert "minimum" not in date
    assert "maximum" not in date
    assert "safe_date_operations" not in date
    assert payload["tables"]["orders"]["columns"]["amount"]["maximum"] == "500.0"


def test_metadata_selection_is_conditional_after_retrieval() -> None:
    requests = metadata_requests("orders over 100 in 2026", _schema(), _profile())

    assert ("DATE_FORMAT", "orders.order_date") in requests
    assert ("CATEGORY_VALUES", "customers.currency") in requests
    assert ("NUMERIC_PROFILE", "orders.amount") in requests
