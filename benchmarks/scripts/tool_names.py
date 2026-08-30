"""Canonical tool labels for the recorded benchmark CSVs.

The checker was called `ptctlmc` until it was renamed `hora`, so result files recorded before
the rename carry the old label in their `tool` column. Those files are measurement OUTPUT -- what
a binary actually printed -- so they are not rewritten. Readers normalise instead.

Use `canonical(row["tool"])` wherever a `tool` field is compared against a name.
"""

ALIASES = {
    "ptctlmc": "hora",
    "ptctlmc-omp": "hora-omp",
    "ptctlmc-seq": "hora-seq",
    "ptctlmc_gsim": "hora_gsim",
}


def canonical(tool):
    """Return the current label for a possibly pre-rename tool name."""
    return ALIASES.get(tool, tool)


def normalise_rows(rows):
    """Rewrite the `tool` field of every row in place, and return them."""
    for row in rows:
        if "tool" in row:
            row["tool"] = canonical(row["tool"])
    return rows
