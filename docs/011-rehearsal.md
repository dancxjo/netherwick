# End-to-end rehearsal

The local development entrypoint is `just run`, which forwards arguments to the `netherwick-tools` binary.

Run the full model rehearsal:

```sh
just run sim --steps 200 --ledger data/ledger/sim1
just run train behavior danger --ledger data/ledger/sim1
just run train behavior charge --ledger data/ledger/sim1
just run train behavior future --ledger data/ledger/sim1
just run evaluate behavior danger --ledger data/ledger/sim1
just run model-status
just run sim --steps 200 --danger-checkpoint data/models/danger_v0 --danger-mode shadow-infer
```

Run the read-only Cockpit capture pulse:

```sh
just run robot --mode read-only --cockpit sim --steps 20 --capture data/captures/mock-readonly
just run replay-capture --capture data/captures/mock-readonly
```

Or run both sequences as one local rehearsal:

```sh
just rehearse-models
```
