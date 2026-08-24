from __future__ import annotations

import json

from semantic_text2sql.historical import HistoricalQueryStore
from semantic_text2sql.models import ColumnInfo, SchemaInfo, TableInfo


def test_only_one_strong_same_database_history_match_is_returned(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps(
            [
                {
                    "db_id": "shop",
                    "question": "What is the ratio of EUR customers to CZK customers?",
                    "sql": (
                        "SELECT 1.0 * SUM(currency = 'EUR') / "
                        "NULLIF(SUM(currency = 'CZK'), 0) FROM customers"
                    ),
                    "success": True,
                },
                {
                    "db_id": "other",
                    "question": "What is the ratio of EUR customers to CZK customers?",
                    "sql": "SELECT 1",
                    "success": True,
                },
            ]
        ),
        encoding="utf-8",
    )

    examples = HistoricalQueryStore(path).search(
        "What is the ratio of EUR customers to CZK customers?",
        "shop",
        top_k=1,
        min_score=0.85,
        candidate_tables={"customers"},
    )

    assert len(examples) == 1
    assert examples[0].db_id == "shop"


def test_weak_history_match_is_not_returned(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps(
            [
                {
                    "db_id": "shop",
                    "question": "List pending orders by date",
                    "sql": "SELECT * FROM orders WHERE status = 'pending' ORDER BY order_date",
                    "success": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    examples = HistoricalQueryStore(path).search(
        "What is the ratio of EUR customers to CZK customers?",
        "shop",
        top_k=1,
        min_score=0.85,
        candidate_tables={"customers"},
    )

    assert examples == []


def test_historical_sql_produces_candidate_features_and_supports_leave_one_out() -> None:
    records: list[dict[str, object]] = [
        {
            "dataset_index": 7,
            "db_id": "shop",
            "question": "What is the ratio of EUR customers to CZK customers?",
            "sql": "SELECT SUM(currency = 'EUR') FROM customers",
            "success": True,
        }
    ]
    schema = SchemaInfo(
        db_id="shop",
        tables=[
            TableInfo(
                name="customers",
                create_sql="",
                columns=[ColumnInfo(name="currency", data_type="TEXT")],
            )
        ],
    )
    store = HistoricalQueryStore.from_records(records)

    evidence = store.schema_evidence(
        "What is the ratio of EUR customers to CZK customers?", "shop", schema
    )
    excluded = store.schema_evidence(
        "What is the ratio of EUR customers to CZK customers?",
        "shop",
        schema,
        exclude_record_id="7",
    )

    assert evidence["table:customers"]["support_count"] == 1
    assert evidence["column:customers.currency"]["max_similarity"] >= 0.65
    assert excluded == {}
