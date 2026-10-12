"""Traceable Ensembl BioMart human-mouse orthology utilities."""

from __future__ import annotations

import hashlib
import io
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd


MART_NAME = "ENSEMBL_MART_ENSEMBL"
HUMAN_DATASET = "hsapiens_gene_ensembl"
MOUSE_DATASET = "mmusculus_gene_ensembl"
QUERY_ATTRIBUTES = (
    "ensembl_gene_id",
    "mmusculus_homolog_ensembl_gene",
    "mmusculus_homolog_orthology_type",
    "mmusculus_homolog_orthology_confidence",
    "mmusculus_homolog_subtype",
    "mmusculus_homolog_perc_id",
    "mmusculus_homolog_perc_id_r1",
    "mmusculus_homolog_goc_score",
    "mmusculus_homolog_wga_coverage",
)
EXPORT_COLUMNS = (
    "human_ensembl_id",
    "mouse_ensembl_id",
    "orthology_type",
    "orthology_confidence",
    "last_common_ancestor",
    "target_percent_identity",
    "query_percent_identity",
    "gene_order_conservation",
    "whole_genome_alignment_coverage",
)
_HUMAN_ID = re.compile(r"^ENSG[0-9]+$")
_MOUSE_ID = re.compile(r"^ENSMUSG[0-9]+$")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse_registry(registry_xml: bytes | str, mart_name: str = MART_NAME) -> dict[str, object]:
    """Resolve one Ensembl Genes mart and its release from registry XML."""
    root = ET.fromstring(registry_xml)
    matches = [node for node in root if node.attrib.get("name") == mart_name]
    if len(matches) != 1:
        raise ValueError(f"registry contains {len(matches)} entries for {mart_name}")
    attributes = dict(matches[0].attrib)
    database = attributes.get("database", "")
    match = re.fullmatch(r"ensembl_mart_([0-9]+)", database)
    if match is None:
        raise ValueError(f"unexpected Ensembl mart database: {database!r}")
    release = int(match.group(1))
    display_name = attributes.get("displayName", "")
    if display_name != f"Ensembl Genes {release}":
        raise ValueError(
            f"registry display/release mismatch: {display_name!r} vs {release}"
        )
    return {"release": release, **attributes}


def parse_dataset_catalog(dataset_tsv: bytes | str) -> dict[str, dict[str, str]]:
    """Parse the BioMart dataset service response used for assembly provenance."""
    text = dataset_tsv.decode("utf-8") if isinstance(dataset_tsv, bytes) else dataset_tsv
    result: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) < 9:
            continue
        dataset = fields[1]
        result[dataset] = {
            "type": fields[0],
            "dataset": dataset,
            "display_name": fields[2],
            "visible": fields[3],
            "assembly": fields[4],
            "interface": fields[7],
            "modified": fields[8],
            "raw_line": line,
        }
    return result


def parse_dataset_configuration(configuration_xml: bytes | str) -> dict[str, str]:
    root = ET.fromstring(configuration_xml)
    if root.tag != "DatasetConfig":
        raise ValueError(f"unexpected configuration root: {root.tag}")
    return {str(key): str(value) for key, value in root.attrib.items()}


def parse_attribute_catalog(attribute_tsv: bytes | str) -> pd.DataFrame:
    """Parse BioMart attributes without assuming unverified field names."""
    text = attribute_tsv.decode("utf-8") if isinstance(attribute_tsv, bytes) else attribute_tsv
    rows = []
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) < 7:
            continue
        rows.append({
            "name": fields[0],
            "display_name": fields[1],
            "description": fields[2],
            "page": fields[3],
            "formats": fields[4],
            "table": fields[5],
            "field": fields[6],
        })
    return pd.DataFrame(rows)


def resolve_query_attributes(attribute_catalog: pd.DataFrame) -> pd.DataFrame:
    """Return the exact homolog-page definitions for the locked query fields."""
    selected = attribute_catalog[
        attribute_catalog["name"].isin(QUERY_ATTRIBUTES)
        & attribute_catalog["page"].eq("homologs")
    ].copy()
    counts = selected["name"].value_counts()
    missing = [name for name in QUERY_ATTRIBUTES if int(counts.get(name, 0)) != 1]
    if missing:
        raise ValueError(
            "BioMart homolog attributes are absent or ambiguous: " + ", ".join(missing)
        )
    selected["_order"] = selected["name"].map({
        name: index for index, name in enumerate(QUERY_ATTRIBUTES)
    })
    return selected.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def build_biomart_query(software_version: str) -> str:
    """Build the deterministic query after fields/config have been validated."""
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", software_version):
        raise ValueError(f"invalid BioMart software version: {software_version!r}")
    attributes = "".join(
        f'    <Attribute name="{name}"/>\n' for name in QUERY_ATTRIBUTES
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE Query>\n'
        '<Query virtualSchemaName="default" formatter="TSV" header="1" '
        f'uniqueRows="1" count="" datasetConfigVersion="{software_version}">\n'
        f'  <Dataset name="{HUMAN_DATASET}" interface="default">\n'
        f"{attributes}"
        "  </Dataset>\n"
        "</Query>\n"
    )


def parse_biomart_export(raw_tsv: bytes, resolved_attributes: pd.DataFrame) -> pd.DataFrame:
    """Parse a raw export and verify that BioMart returned the requested columns."""
    if raw_tsv.lstrip().startswith(b"Query ERROR"):
        raise ValueError(raw_tsv.decode("utf-8", errors="replace")[:1000])
    frame = pd.read_csv(
        io.BytesIO(raw_tsv), sep="\t", dtype="string", keep_default_na=False
    )
    expected_headers = resolved_attributes["display_name"].astype(str).tolist()
    if frame.columns.astype(str).tolist() != expected_headers:
        raise ValueError(
            f"BioMart export headers changed: {frame.columns.tolist()} != {expected_headers}"
        )
    if frame.shape[1] != len(EXPORT_COLUMNS):
        raise ValueError("BioMart export column count does not match the locked query")
    frame.columns = list(EXPORT_COLUMNS)
    return frame


def high_confidence_edges(raw_export: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Keep all confidence=1 Ensembl-ID edges, including one-to-many edges."""
    required = set(EXPORT_COLUMNS)
    missing = required - set(raw_export.columns)
    if missing:
        raise ValueError(f"raw orthology export is missing columns: {sorted(missing)}")
    frame = raw_export[list(EXPORT_COLUMNS)].copy()
    human = frame["human_ensembl_id"].astype(str)
    invalid_human = ~human.map(lambda value: bool(_HUMAN_ID.fullmatch(value)))
    if invalid_human.any():
        raise ValueError(f"invalid human Ensembl ID: {human[invalid_human].iloc[0]!r}")

    mapped = frame[frame["mouse_ensembl_id"].astype(str).ne("")].copy()
    mouse = mapped["mouse_ensembl_id"].astype(str)
    invalid_mouse = ~mouse.map(lambda value: bool(_MOUSE_ID.fullmatch(value)))
    if invalid_mouse.any():
        raise ValueError(f"invalid mouse Ensembl ID: {mouse[invalid_mouse].iloc[0]!r}")
    confidence_values = set(mapped["orthology_confidence"].astype(str).unique())
    if not confidence_values.issubset({"0", "1"}):
        raise ValueError(f"unexpected orthology confidence values: {sorted(confidence_values)}")
    if mapped["orthology_type"].astype(str).eq("").any():
        raise ValueError("mapped BioMart row lacks orthology_type")
    if not mapped["orthology_type"].astype(str).str.startswith("ortholog_").all():
        raise ValueError("BioMart returned a non-ortholog homology type")

    high = mapped[mapped["orthology_confidence"].astype(str).eq("1")].copy()
    pair_columns = ["human_ensembl_id", "mouse_ensembl_id"]
    for _, group in high.groupby(pair_columns, sort=False):
        if len(group.drop_duplicates()) != 1:
            pair = tuple(group.iloc[0][pair_columns].astype(str))
            raise ValueError(f"conflicting BioMart metadata for edge {pair}")
    high = high.drop_duplicates().sort_values(
        pair_columns + ["orthology_type"], kind="stable"
    ).reset_index(drop=True)
    stats = {
        "raw_rows": int(len(frame)),
        "raw_human_genes": int(frame["human_ensembl_id"].nunique()),
        "mapped_rows": int(len(mapped)),
        "low_confidence_rows_excluded": int(
            mapped["orthology_confidence"].astype(str).eq("0").sum()
        ),
        "high_confidence_edges": int(len(high)),
        "high_confidence_human_genes": int(high["human_ensembl_id"].nunique()),
        "high_confidence_mouse_genes": int(high["mouse_ensembl_id"].nunique()),
        "orthology_type_counts": {
            str(key): int(value)
            for key, value in high["orthology_type"].value_counts().sort_index().items()
        },
    }
    return high, stats


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        # Lexical root choice keeps the structure deterministic as well as the output.
        lower, upper = sorted((left_root, right_root))
        self.parent[upper] = lower


def build_homology_groups(
    edges: pd.DataFrame, release: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build deterministic connected components over a species-tagged bipartite graph."""
    if release < 1:
        raise ValueError("release must be positive")
    required = {"human_ensembl_id", "mouse_ensembl_id"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"orthology edges are missing columns: {sorted(missing)}")
    pairs = edges.sort_values(
        ["human_ensembl_id", "mouse_ensembl_id"], kind="stable"
    ).reset_index(drop=True)
    if pairs.empty:
        raise ValueError("orthology edge table is empty")
    if pairs.duplicated(["human_ensembl_id", "mouse_ensembl_id"]).any():
        raise ValueError("orthology edge table contains duplicate ID pairs")
    disjoint = _DisjointSet()
    for row in pairs.itertuples(index=False):
        disjoint.union(
            f"human:{row.human_ensembl_id}",
            f"mouse:{row.mouse_ensembl_id}",
        )
    components: dict[str, list[str]] = defaultdict(list)
    for node in sorted(disjoint.parent):
        components[disjoint.find(node)].append(node)
    edge_count_by_root: dict[str, int] = defaultdict(int)
    for row in pairs.itertuples(index=False):
        edge_count_by_root[disjoint.find(f"human:{row.human_ensembl_id}")] += 1

    group_by_node: dict[str, str] = {}
    group_rows = []
    group_stats = []
    seen_group_ids: set[str] = set()
    ordered_components = sorted(
        components.items(), key=lambda item: tuple(item[1])
    )
    for root, members in ordered_components:
        humans = sorted(node.split(":", 1)[1] for node in members if node.startswith("human:"))
        mice = sorted(node.split(":", 1)[1] for node in members if node.startswith("mouse:"))
        if not humans or not mice:
            raise ValueError("a bipartite component lacks one species")
        payload = "\n".join([*(f"human:{item}" for item in humans), *(f"mouse:{item}" for item in mice)])
        group_id = f"ensembl_r{release}_hm_cc_{hashlib.sha256(payload.encode()).hexdigest()[:20]}"
        if group_id in seen_group_ids:
            raise ValueError("homology group hash collision")
        seen_group_ids.add(group_id)
        if len(humans) == 1 and len(mice) == 1:
            relationship = "one_to_one"
        elif len(humans) == 1:
            relationship = "human_to_many_mouse"
        elif len(mice) == 1:
            relationship = "many_human_to_mouse"
        else:
            relationship = "many_to_many_component"
        edge_count = int(edge_count_by_root[root])
        group_stats.append({
            "homology_group": group_id,
            "human_genes": len(humans),
            "mouse_genes": len(mice),
            "edges": edge_count,
            "relationship_class": relationship,
        })
        for species, identifiers in (("human", humans), ("mouse", mice)):
            for identifier in identifiers:
                group_by_node[f"{species}:{identifier}"] = group_id
                group_rows.append({
                    "ensembl_id": identifier,
                    "species": species,
                    "homology_group": group_id,
                    "component_human_genes": len(humans),
                    "component_mouse_genes": len(mice),
                    "component_edges": edge_count,
                    "relationship_class": relationship,
                    "ensembl_release": release,
                })
    groups = pd.DataFrame(group_rows).sort_values(
        ["species", "ensembl_id"], kind="stable"
    ).reset_index(drop=True)
    if groups["ensembl_id"].duplicated().any():
        raise ValueError("an Ensembl ID belongs to more than one homology group")
    edge_groups = pairs.copy()
    edge_groups["homology_group"] = [
        group_by_node[f"human:{identifier}"]
        for identifier in edge_groups["human_ensembl_id"].astype(str)
    ]
    components_frame = pd.DataFrame(group_stats).sort_values(
        "homology_group", kind="stable"
    ).reset_index(drop=True)
    stats = {
        "components": int(len(components_frame)),
        "group_rows": int(len(groups)),
        "relationship_class_counts": {
            str(key): int(value)
            for key, value in components_frame["relationship_class"].value_counts().sort_index().items()
        },
        "largest_component_human_genes": int(components_frame["human_genes"].max()),
        "largest_component_mouse_genes": int(components_frame["mouse_genes"].max()),
        "largest_component_edges": int(components_frame["edges"].max()),
    }
    return groups, edge_groups, stats


__all__ = [
    "EXPORT_COLUMNS",
    "HUMAN_DATASET",
    "MART_NAME",
    "MOUSE_DATASET",
    "QUERY_ATTRIBUTES",
    "build_biomart_query",
    "build_homology_groups",
    "high_confidence_edges",
    "parse_attribute_catalog",
    "parse_biomart_export",
    "parse_dataset_catalog",
    "parse_dataset_configuration",
    "parse_registry",
    "resolve_query_attributes",
    "sha256_bytes",
]
