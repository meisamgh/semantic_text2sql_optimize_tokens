"""Deterministic schema, glossary, and dependency verification."""

from __future__ import annotations

from typing import Any

from semantic_text2sql.models import (
    AggregationStage,
    ContextRequest,
    DatabaseProfile,
    MetricSelector,
    PlannerMetadataRequirement,
    SchemaInfo,
    SemanticContract,
)


def verify_context_request(
    requested: ContextRequest,
    schema: SchemaInfo,
    deterministic: SemanticContract,
    profile: DatabaseProfile | None = None,
    approved_concepts: set[str] | None = None,
) -> ContextRequest:
    """Drop hallucinated identifiers and force-add correctness-critical dependencies."""
    table_map = {table.name: table for table in schema.tables}
    live_columns = {
        f"{table.name}.{column.name}" for table in schema.tables for column in table.columns
    }
    formula_ids = {item.id for item in deterministic.structural_formulas}
    approved = formula_ids | (approved_concepts or set())
    valid_business_concepts = [
        normalized
        for item in requested.business_concepts
        if (normalized := item.casefold().replace(" ", "_")) in approved
    ]
    derived_names = {item.output for item in requested.aggregations if item.output is not None}
    named_metrics = approved | derived_names
    selected_tables = [name for name in requested.tables if name in table_map]
    columns: dict[str, list[str]] = {}
    for table, names in requested.columns.items():
        if table not in table_map:
            continue
        live = {item.name for item in table_map[table].columns}
        columns[table] = list(dict.fromkeys(name for name in names if name in live))

    def add_column(qualified: str) -> None:
        if qualified not in live_columns:
            return
        table, column = qualified.split(".", 1)
        if table not in selected_tables:
            selected_tables.append(table)
        columns.setdefault(table, [])
        if column not in columns[table]:
            columns[table].append(column)

    # Key metadata is a deterministic dependency, not an optional relevance decision.
    for table_name in list(selected_tables):
        for column in table_map[table_name].columns:
            if column.primary_key:
                add_column(f"{table_name}.{column.name}")

    valid_outputs = [
        item for item in requested.outputs if item in live_columns or item in named_metrics
    ]
    for item in [*deterministic.outputs, *valid_outputs, *deterministic.required_columns]:
        add_column(item)
    planner_filters = [
        item
        for item in requested.filters
        if item.operand in live_columns or item.operand in named_metrics
    ]
    deterministic_operands = {item.operand for item in deterministic.filters}
    valid_filters = [
        *deterministic.filters,
        *(item for item in planner_filters if item.operand not in deterministic_operands),
    ]
    for semantic_filter in valid_filters:
        add_column(semantic_filter.operand)
    valid_measures = [
        item for item in requested.measures if item in live_columns or item in named_metrics
    ]
    for item in [*deterministic.measures, *valid_measures, *requested.group_by]:
        add_column(item)
    for aggregation in requested.aggregations:
        add_column(aggregation.input)
        for item in aggregation.group_by:
            add_column(item)
    for ranking in requested.ranking:
        add_column(ranking.metric)
    for formula in deterministic.structural_formulas:
        if formula.id in valid_business_concepts or formula.id in {
            item.operand for item in valid_filters
        }:
            for argument in formula.arguments:
                add_column(argument)

    selected = set(selected_tables)
    for live_relationship in schema.relationships:
        if live_relationship.from_table in selected and live_relationship.to_table in selected:
            add_column(f"{live_relationship.from_table}.{live_relationship.from_column}")
            add_column(f"{live_relationship.to_table}.{live_relationship.to_column}")
    if profile is not None:
        for profiled_relationship in profile.relationships:
            if (
                profiled_relationship.parent_table in selected
                and profiled_relationship.child_table in selected
            ):
                add_column(
                    f"{profiled_relationship.parent_table}.{profiled_relationship.parent_column}"
                )
                add_column(
                    f"{profiled_relationship.child_table}.{profiled_relationship.child_column}"
                )

    metadata = _verified_metadata(
        requested.metadata_requirements, selected_tables, live_columns, approved
    )
    if profile is not None:
        profile_columns = {f"{item.table}.{item.column}": item for item in profile.columns}
        exact_filters = {
            item.operand: item.value for item in valid_filters if item.operator in {"=", "IN"}
        }
        metadata = [
            item
            for item in metadata
            if not (
                item.kind == "CATEGORY_VALUES"
                and all(
                    _literal_is_grounded(exact_filters.get(target), profile_columns.get(target))
                    for target in item.targets
                )
            )
        ]
    if len(selected_tables) > 1:
        metadata.extend(
            [
                PlannerMetadataRequirement(kind="RELATIONSHIP", targets=selected_tables),
                PlannerMetadataRequirement(kind="JOIN_CARDINALITY", targets=selected_tables),
                PlannerMetadataRequirement(kind="TABLE_GRAIN", targets=selected_tables),
            ]
        )
    for formula in deterministic.structural_formulas:
        if formula.id in valid_business_concepts or formula.id in {
            item.operand for item in valid_filters
        }:
            metadata.append(
                PlannerMetadataRequirement(kind="FORMULA_DEFINITION", targets=[formula.id])
            )
    unique_metadata = {(item.kind, tuple(item.targets)): item for item in metadata}
    return requested.model_copy(
        update={
            "outputs": list(dict.fromkeys([*deterministic.outputs, *valid_outputs])),
            "tables": selected_tables,
            "columns": columns,
            "filters": _unique_models(valid_filters),
            "measures": list(dict.fromkeys([*deterministic.measures, *valid_measures])),
            "business_concepts": valid_business_concepts,
            "metadata_requirements": list(unique_metadata.values()),
        }
    )


def reconcile_context_contract(
    deterministic: SemanticContract, requested: ContextRequest
) -> SemanticContract:
    """Merge verified WHAT-level hints while deterministic and glossary facts win conflicts."""
    planner_aggregations = (
        []
        if deterministic.outputs and deterministic.filters and deterministic.grain
        else requested.aggregations
    )
    primary_aggregation = planner_aggregations[0].function if planner_aggregations else None
    ranking_limit = next((item.limit for item in requested.ranking if item.limit is not None), None)
    required_columns = [
        f"{table}.{column}" for table, names in requested.columns.items() for column in names
    ]
    aggregation_stages = [
        AggregationStage(
            name=item.output or f"{item.function}_{item.input}",
            function=item.function,
            input=item.input,
            group_by=item.group_by or requested.group_by,
            output_grain=item.group_by or requested.group_by,
        )
        for item in planner_aggregations
    ]
    selectors = [
        MetricSelector(
            name=f"select_{item.metric}",
            function="argmax" if item.direction == "DESC" else "argmin",
            metric=item.metric,
            scope="across_groups",
            tie_policy="deterministic_one" if item.limit == 1 else "return_all",
        )
        for item in requested.ranking
    ]
    formula_inputs = {item.input for item in planner_aggregations}
    formulas = [
        item.model_copy(update={"usage": "OUTPUT" if item.id in formula_inputs else item.usage})
        for item in deterministic.structural_formulas
    ]
    return deterministic.model_copy(
        update={
            "aggregation": deterministic.aggregation or primary_aggregation,
            "requires_grouping": deterministic.requires_grouping or bool(requested.group_by),
            "requires_ordering": deterministic.requires_ordering or bool(requested.ranking),
            "limit": deterministic.limit or ranking_limit,
            "outputs": deterministic.outputs or requested.outputs,
            "proposed_tables": requested.tables,
            "required_columns": list(
                dict.fromkeys([*deterministic.required_columns, *required_columns])
            ),
            "filters": _unique_models([*deterministic.filters, *requested.filters]),
            "measures": list(dict.fromkeys([*deterministic.measures, *requested.measures])),
            "grain": deterministic.grain or requested.group_by,
            "aggregation_stages": deterministic.aggregation_stages or aggregation_stages,
            "selectors": deterministic.selectors or selectors,
            "structural_formulas": formulas,
            "named_outputs": list(
                dict.fromkeys(
                    [
                        *deterministic.named_outputs,
                        *(item.output for item in planner_aggregations if item.output),
                    ]
                )
            ),
        }
    )


def fallback_context_request(contract: SemanticContract) -> ContextRequest:
    """Represent the deterministic context retained for optimization turns."""
    columns: dict[str, list[str]] = {}
    for qualified in contract.required_columns:
        if "." not in qualified:
            continue
        table, column = qualified.split(".", 1)
        columns.setdefault(table, []).append(column)
    return ContextRequest(
        outputs=contract.outputs,
        tables=contract.proposed_tables or list(columns),
        columns=columns,
        filters=contract.filters,
        measures=contract.measures,
        business_concepts=[item.id for item in contract.structural_formulas],
    )


def _verified_metadata(
    requirements: list[PlannerMetadataRequirement],
    tables: list[str],
    live_columns: set[str],
    approved_concepts: set[str],
) -> list[PlannerMetadataRequirement]:
    verified: list[PlannerMetadataRequirement] = []
    live_tables = set(tables)
    for item in requirements:
        if item.kind in {"RELATIONSHIP", "TABLE_GRAIN"}:
            targets = [target for target in item.targets if target in live_tables]
        elif item.kind == "JOIN_CARDINALITY":
            targets = [
                target for target in item.targets if target in live_columns or target in live_tables
            ]
        elif item.kind == "FORMULA_DEFINITION":
            targets = [target for target in item.targets if target in approved_concepts]
        else:
            targets = [target for target in item.targets if target in live_columns]
        if targets:
            verified.append(item.model_copy(update={"targets": targets}))
    return verified


def _unique_models(values: list[Any]) -> list[Any]:
    return list({item.model_dump_json(): item for item in values}.values())


def _literal_is_grounded(value: Any, profile: Any) -> bool:
    if value is None or profile is None:
        return False
    requested = value if isinstance(value, list) else [value]
    known = {str(item.value).casefold() for item in profile.top_values} | {
        str(item).casefold() for item in profile.allowed_values
    }
    return bool(known) and all(str(item).casefold() in known for item in requested)
