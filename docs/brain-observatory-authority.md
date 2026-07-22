# Brain Observatory authority and safety flow

The authority view is read-only. It derives a bounded, time-scoped flow from
canonical BrainEvents and combines that with the server's existing live session
and hardware-control status. It neither creates a control route nor grants an
observability component a lease.

`/api/observatory/authority?at_ms=...&window_ms=...` preserves every proposal
in the selected window, including losing goals. Events are classified into the
recorded pipeline stages from goal/drive through evaluation, arbitration,
behavior, skill, proposal, operator context, autonomic and motor gates,
possession lease, brainstem command, acknowledgement, and outcome. Event IDs,
command IDs, goal IDs, parent IDs, scores, TTLs, reasons, trust, and disposition
are carried through unchanged.

The UI distinguishes human direct control, Reign assist/suggest, LLM advisory,
learned shadow output, deterministic fallback, brainstem-local reflexes,
motherbrain recovery, autonomic safety, and ordinary runtime output. Advisory,
shadow, and suggest cards use a dashed outline and are explicitly described as
non-authoritative.

Lease ID/TTL, heartbeat age, STOP state, E-stop, safety latch, and brainstem
authority appear only when recorded payloads report them. Missing values read
`not observed`; they are never treated as safe defaults. Bump, cliff, and wheel
drop indicators come from the selected live body's existing sensor state.
Preemption, veto, expiry, rejection, supersession, failure, and completion stay
distinct, including failed STOP acknowledgements and reflex preemption.
