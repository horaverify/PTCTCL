# verifyta.csv

The UPPAAL baseline as ONE file: `results_old/verifyta_only.csv` (600 s budget) with every row
that the 2 h re-run (`results_old/verifyta_2h.csv`) covered replaced by its 2 h result. Both
sources are kept in `results_old/`; nothing was discarded.

96 rows, 92 OK / 4 TIMEOUT. Merge identity is (family, model, tool, threads, rep); the 2 h row
always wins.

Replaced (8 rows), and what the longer budget bought:

| model | 600 s | 2 h |
|---|---|---|
| fischer14 | TIMEOUT | OK, 1289 s |
| phil15 | TIMEOUT | OK, 3707-3743 s |
| twolock14 | TIMEOUT | OK, 1310 s |
| twolock15 | TIMEOUT | OK, 4926-4940 s |
| viking20 | TIMEOUT | **OK, 1254-1268 s** |
| phil16 | TIMEOUT | TIMEOUT at 7200 s |
| single8 | TIMEOUT | TIMEOUT at 7200 s |
| twolock16 | TIMEOUT | TIMEOUT at 7200 s |

Plus 5 rep-2 rows that exist only in the 2 h run.

viking20 is the row the 13.8x headline rests on: verifyta 1253.98 s against PTCTLMC's ~91 s,
both exploring 143,654,910 states. It is a TIMEOUT in the 600 s file, so that file alone cannot
support the claim.

## One row to read carefully

`phil17` is a **600 s** TIMEOUT, not a 2 h one -- the 2 h re-run used `--strike-limit 1`, so once
phil16 blew the budget phil17 was never attempted. Its `seconds` column reads 600.042, which is
when it was abandoned, NOT a measured cost. phil16 does time out at the full 2 h, so phil17
would too; the row is a weaker statement than the others, not a wrong one.
