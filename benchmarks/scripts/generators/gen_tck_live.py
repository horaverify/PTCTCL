#!/usr/bin/env python3
"""Generate TChecker models for local UPPAAL leads-to properties.

For ``Process.p --> Process.q``, each location of Process is duplicated into
``clear`` and ``pending`` variants. At a p-state, the observer may start
tracking that occurrence; the pending branch has no transition to q. Thus an
accepting pending cycle is precisely a counterexample to the leads-to property.
"""
import argparse
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
BENCH = HERE.parent.parent
SOURCE_DIR = BENCH / "models" / "tck"
QUERY_DIR = BENCH / "queries" / "live"
DEFAULT_OUTPUT = BENCH / "models" / "tck_live"
PENDING_LABEL = "TC_LIVE_PENDING"
QUERY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*-->\s*"
                      r"([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)$")
LOCATION_RE = re.compile(r"^location:([^:]+):([^\{]+)(\{.*\})$")
EDGE_RE = re.compile(r"^edge:([^:]+):([^:]+):([^:]+):([^\{]+)(\{.*\})$")
ATTRIBUTE_RE = re.compile(r"(?:^|:)([A-Za-z_][A-Za-z0-9_]*):")


def family_for(model_name):
    match = re.match(r"([A-Za-z_]+)\d+$", model_name)
    return match.group(1) if match else None


def parse_query(path):
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{path}: expected exactly one non-empty query line")
    match = QUERY_RE.fullmatch(lines[0])
    if not match:
        raise ValueError(f"{path}: only Process.location --> Process.location is supported")
    trigger_process, trigger_location, response_process, response_location = match.groups()
    if trigger_process != response_process:
        raise ValueError(f"{path}: cross-process leads-to properties require a product observer")
    return trigger_process, trigger_location, response_location, lines[0]


def parse_attributes(text):
    inner = text[1:-1]
    matches = list(ATTRIBUTE_RE.finditer(inner))
    attributes = []
    for index, match in enumerate(matches):
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(inner)
        attributes.append((match.group(1), inner[match.end():value_end]))
    return attributes


def format_attributes(attributes):
    return "{" + ":".join(f"{key}:{value}" for key, value in attributes) + "}"


def location_attributes(attributes, pending, initial):
    result = [(key, value) for key, value in attributes if key != "initial"]
    if initial:
        result.insert(0, ("initial", ""))
    if pending:
        for index, (key, value) in enumerate(result):
            if key == "labels":
                result[index] = (key, f"{value},{PENDING_LABEL}" if value else PENDING_LABEL)
                break
        else:
            result.append(("labels", PENDING_LABEL))
    return format_attributes(result)


def instrument(source, process, trigger, response):
    locations = {}
    for line in source.splitlines():
        match = LOCATION_RE.fullmatch(line)
        if match and match.group(1) == process:
            locations[match.group(2)] = parse_attributes(match.group(3))
    if trigger not in locations or response not in locations:
        raise ValueError(f"{process}.{trigger} --> {process}.{response}: location missing from model")

    output = []
    for line in source.splitlines():
        location = LOCATION_RE.fullmatch(line)
        if location and location.group(1) == process:
            name, attributes = location.group(2), parse_attributes(location.group(3))
            initial = any(key == "initial" for key, _ in attributes)
            output.append(f"location:{process}:{name}__clear" +
                          location_attributes(attributes, False, initial))
            output.append(f"location:{process}:{name}__pending" +
                          location_attributes(attributes, True, initial and name == trigger))
            continue

        edge = EDGE_RE.fullmatch(line)
        if edge and edge.group(1) == process:
            _, source_location, target_location, event, attributes = edge.groups()
            targets = [("clear", "clear")]
            if target_location == trigger:
                targets.append(("clear", "pending"))
            if target_location != response:
                targets.append(("pending", "pending"))
            for source_suffix, target_suffix in targets:
                output.append(f"edge:{process}:{source_location}__{source_suffix}:"
                              f"{target_location}__{target_suffix}:{event}{attributes}")
            continue

        output.append(line)
    return "\n".join(output) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    parser.add_argument("--queries", type=Path, default=QUERY_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    arguments.outdir.mkdir(parents=True, exist_ok=True)
    manifest = {"pending_label": PENDING_LABEL, "models": {}}
    for model in sorted(arguments.source.glob("*.tck")):
        family = family_for(model.stem)
        query_file = arguments.queries / f"{family}.q" if family else None
        if query_file is None or not query_file.exists():
            continue
        process, trigger, response, query = parse_query(query_file)
        output_path = arguments.outdir / model.name
        output_path.write_text(instrument(model.read_text(), process, trigger, response))
        manifest["models"][model.name] = {
            "query": query,
            "accepting_label": PENDING_LABEL,
            "command": f"tck-liveness -a couvscc -l {PENDING_LABEL} {output_path}",
        }
    (arguments.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {len(manifest['models'])} liveness observer models in {arguments.outdir}")


if __name__ == "__main__":
    main()