# Brain Observatory provenance graph

The Observatory center pane calls
`/api/observatory/provenance/{event_id}` to obtain a bounded neighborhood of
the selected canonical BrainEvent. The endpoint reads the same retained event
history used by live and replay sources; it does not scrape the timeline or a
panel-specific payload.

Supported explanation modes are `why_believed`, `why_selected`,
`why_rejected`, `dependents`, and `full`. Traversal follows typed parent,
support, contradiction, and reverse-dependent relationships. A visited set
terminates cycles. `max_nodes` defaults to 80 and is capped at 250; omitted
nodes and truncation are explicit.

Every event node carries its recorded event class, semantic origin, producer,
occurred time, confidence, uncertainty, freshness, trust, and disposition.
Origin classification distinguishes direct evidence, inference, recall,
learned/model output, human instruction, LLM hypotheses, configuration, gaps,
and otherwise-recorded events. References absent from retained history become
visible `missing_reference` nodes. Transport gaps crossing the displayed
interval become `capture_gap` nodes.

Model/configuration artifacts and raw payload references are opt-in expansions.
Calibration epochs are retained alongside their event. The UI displays asset
metadata only; it never downloads a raw image, depth cloud, audio sample, or
vector until a future explicit asset action requests one. Switching selection
aborts the previous graph request so delayed retrieval cannot overwrite the
new inspection context.
