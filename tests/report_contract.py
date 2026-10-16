"""Compare program contracts while keeping analysis work and origin separate.

Unknown fields are retained. This deliberately has no count/statistics suffix
rule: native accesses, dependence records, domains and assigned IDs describe
the program even when their values happen to be integers.
"""

_WORK_FIELDS = frozenset(
    {
        "completion_statistics",
        "scope_completion_facts",
        "scope_anchored_facts",
        "physical_facts",
        "potential_work",
        "activation_checks",
        "pending_frontier_steps",
        "analysis_work",
        "verification_mode",
        "completion_math",
        "debug_verification",
        "proof_path_queries",
        "completion_batch_queries",
        "completion_transfer_passes",
        "completion_fact_transfers",
        "coverage_queries",
        "scalar_capture_queries",
        "binary_lifetime_queries",
        "sequence_boundary_coverage_queries",
        "sequence_child_projection_statistics",
        "sequence_completion_transfers",
        "sequence_completion_facts",
        "sequence_completion_edges",
        "sequence_completion_sites",
        "invocation_proof_queries",
        "invocation_fact_transfers",
        "invocation_completion_edges",
        "invocation_completion_facts",
        "invocation_completion_transfers",
        "invocation_completion_sites",
        "invocation_child_projection_statistics",
        "parent_access_visits",
        "parent_binary_lifetime_queries",
        "child_parent_lifetime_queries",
        "self_queries",
        "self_batches",
        "sharing_queries",
        "sharing_batches",
        "merge_candidates",
        "merge_checks",
        "merges",
        "new_completion_checks",
        "query_event_reuse_checks_reused",
        "child_lifetime_queries_reused",
        "pool_statistics_reused",
        "counter_backlog_reused",
        "new_invocation_resource_queries",
        "new_invocation_resource_solver_calls",
        "memory_checks_reused",
        "new_memory_checks",
        "established_in",
        "verification_source",
        "verification_origin",
        "fact_nodes",
        "condition_facts",
        "peak_condition_facts",
        "fact_attempts",
        "fact_comparisons",
        "fact_index_steps",
        "fact_interval_joins",
        "fact_updates",
        "fact_transfers",
        "pending_index_steps",
        "pending_interval_joins",
        "coverage_batches",
        "coverage_requests",
        "coverage_fact_scans",
        "coverage_result_items",
    }
)


def semantic_report(report, *, _section=None):
    """Retain semantic results; callers compare work/proof origins separately."""
    if isinstance(report, dict):
        return {
            key: semantic_report(value, _section=key)
            for key, value in report.items()
            if key not in _WORK_FIELDS
            and not (_section == "invocation_resources" and key == "source")
        }
    if isinstance(report, (list, tuple)):
        return [semantic_report(value, _section=_section) for value in report]
    return report
