# Demo data status

| Generation | Official demo status | Local status |
|---|---|---|
| MIMIC-I | No separate demo; open waveform database is 23.5 GB | Partial, paused and resumable |
| MIMIC-II | Deprecated 4,000-patient flat-file demo | Download in progress |
| MIMIC-III | Official 100-patient clinical demo v1.4 | Download in progress |
| MIMIC-IV | Official 100-patient clinical demo v2.2 | Complete; all checksums pass |

The first implementation targets MIMIC-IV because its demo has the same schema and
structure as the corresponding full release. Cross-version adapters follow after the
MIMIC-IV architecture smoke test.
