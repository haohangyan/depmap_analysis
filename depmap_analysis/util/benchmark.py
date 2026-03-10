"""DepMap scratch workflow split into runnable commands.

Typical flow:
  1) Build base norand inet from statements (gene-filtered):
     python scratch.py build-norand --input-stmts /path/to/stmts.pkl --source-type all

  1b) Build base norand inet by streaming unique statements TSV:
     python scratch.py build-norand --input-unique-stmts /path/to/unique_statements.tsv.gz \
         --source-counts-pkl /path/to/source_counts.pkl --source-type all

  2) Build shuffled inet files (requires xswap, e.g., Python 3.9):
     python scratch.py build-inets --source-type all

  3) Run DepMap explanation benchmark (can run in Python 3.12):
     python scratch.py run-depmap --source-types all --rand-types norand,randxswap,randlabels
"""

import argparse
import csv
import gzip
import logging
import os
import pickle
import random
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from indra_db.readonly_dumping.util import clean_json_loads

logger = logging.getLogger("depmap_scratch")

DEFAULT_OUTPUT_DIR = os.path.expanduser("~/.data/dp2026")
DEFAULT_EXPL_FUNCS = (
    "expl_ab",
    "expl_ba",
    "find_cp",
    "apriori_explained",
    "parent_connections",
    "common_reactome_paths",
)

SIF_COL_NAMES = [
    "agA_name",
    "agB_name",
    "agA_ns",
    "agA_id",
    "agB_ns",
    "agB_id",
    "residue",
    "position",
    "stmt_type",
    "evidence_count",
    "stmt_hash",
    "belief",
    "source_counts",
    "initial_sign",
]


def inet_filename(source_type: str, rand_type: str) -> str:
    return f"bioexp_depmap_{source_type}_stmts_{rand_type}_inet.pkl"


def parse_csv_arg(arg: str) -> List[str]:
    return [item.strip() for item in arg.split(",") if item.strip()]


def _extract_stmts(loaded_obj):
    """Handle a few common statement-pickle shapes."""
    if isinstance(loaded_obj, list):
        return loaded_obj
    if isinstance(loaded_obj, tuple) and loaded_obj and isinstance(loaded_obj[0], list):
        return loaded_obj[0]
    if isinstance(loaded_obj, dict) and isinstance(loaded_obj.get("statements"), list):
        return loaded_obj["statements"]
    raise ValueError(
        "Could not parse statement input. Expected list[Statement], "
        "(list[Statement], ...), or {'statements': [...]}"
    )


def filter_stmts(
    stmts,
    genes_only: bool = True,
    human_only: bool = True,
    require_named_agents: bool = True,
    require_two_agents: bool = False,
):
    from indra.tools import assemble_corpus as ac

    if genes_only:
        stmts = ac.filter_genes_only(stmts)
    if human_only:
        stmts = ac.filter_human_only(stmts)
    if require_named_agents:
        stmts = [s for s in stmts if None not in s.agent_list()]
        stmts = [
            s for s in stmts
            if all(getattr(ag, "name", None) for ag in s.agent_list())
        ]
    if require_two_agents:
        stmts = [s for s in stmts if len(s.agent_list()) == 2]
    return stmts


def _load_mitogenes() -> List[str]:
    from depmap_analysis.scripts.depmap_script2 import mito_file

    mitocarta = pd.read_excel(mito_file, sheet_name=1)
    return list(mitocarta.Symbol.values)


def apply_mitocarta_exclusion(stmts, mitogenes: Sequence[str]):
    from indra.tools import assemble_corpus as ac

    return ac.filter_gene_list(stmts, mitogenes, policy="all", invert=True)


def _filter_and_rows_from_batch(
    batch,
    args: argparse.Namespace,
    source_counts,
    mitogenes: Optional[Sequence[str]],
):
    from indra.assemblers.indranet import statement_to_rows

    filtered = filter_stmts(
        batch,
        genes_only=args.genes_only,
        human_only=args.human_only,
        require_named_agents=args.require_named_agents,
        require_two_agents=args.require_two_agents,
    )
    if mitogenes is not None:
        filtered = apply_mitocarta_exclusion(filtered, mitogenes)

    rows = []
    for stmt in filtered:
        rows.extend(
            statement_to_rows(
                stmt,
                complex_members=args.complex_members,
                source_counts=source_counts,
            )
        )
    return filtered, rows


def _build_norand_from_stmts(args: argparse.Namespace, out_file: str) -> None:
    from indra.tools import assemble_corpus as ac
    from indra.assemblers.indranet import IndraNetAssembler
    from depmap_analysis.network_functions.net_functions import sif_dump_df_to_digraph

    logger.info("Loading statements from %s", args.input_stmts)
    loaded = ac.load_statements(args.input_stmts)
    stmts = _extract_stmts(loaded)
    logger.info("Loaded %d statements", len(stmts))

    before = len(stmts)
    stmts = filter_stmts(
        stmts,
        genes_only=args.genes_only,
        human_only=args.human_only,
        require_named_agents=args.require_named_agents,
        require_two_agents=args.require_two_agents,
    )
    logger.info("After filters: %d -> %d", before, len(stmts))

    if args.exclude_mitocarta:
        mitogenes = _load_mitogenes()
        pre_mito = len(stmts)
        stmts = apply_mitocarta_exclusion(stmts, mitogenes)
        logger.info("After MitoCarta exclusion: %d -> %d", pre_mito, len(stmts))

    logger.info("Assembling norand inet")
    ina = IndraNetAssembler(statements=stmts)
    sif_df = ina.make_df(complex_members=args.complex_members)
    inet = sif_dump_df_to_digraph(df=sif_df, date=args.indra_date)

    with open(out_file, "wb") as fh:
        pickle.dump(inet, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved norand inet -> %s", out_file)


def _build_norand_from_unique_tsv(args: argparse.Namespace, out_file: str) -> None:
    from indra.statements import stmt_from_json
    from depmap_analysis.network_functions.net_functions import sif_dump_df_to_digraph

    source_counts = None
    if args.source_counts_pkl:
        logger.info("Loading source counts from %s", args.source_counts_pkl)
        with open(args.source_counts_pkl, "rb") as fh:
            source_counts = pickle.load(fh)

    mitogenes = _load_mitogenes() if args.exclude_mitocarta else None

    batch = []
    rows = []
    total = 0
    parsed = 0
    skipped = 0
    kept = 0

    logger.info("Streaming unique statements from %s", args.input_unique_stmts)
    with gzip.open(args.input_unique_stmts, "rt") as fi:
        reader = csv.reader(fi, delimiter="\t")
        for row in reader:
            if args.max_unique_rows and total >= args.max_unique_rows:
                break
            total += 1

            if not row:
                skipped += 1
                continue

            stmt_json_str = row[-1]
            try:
                stmt_json = clean_json_loads(stmt_json_str)
                stmt = stmt_from_json(stmt_json)
                if source_counts is not None:
                    # Keep memory lower for this path; evidence_count comes from source_counts.
                    stmt.evidence = []
                batch.append(stmt)
                parsed += 1
            except Exception:
                skipped += 1
                continue

            if len(batch) >= args.batch_size:
                filtered, batch_rows = _filter_and_rows_from_batch(
                    batch, args=args, source_counts=source_counts, mitogenes=mitogenes
                )
                kept += len(filtered)
                rows.extend(batch_rows)
                batch = []

    if batch:
        filtered, batch_rows = _filter_and_rows_from_batch(
            batch, args=args, source_counts=source_counts, mitogenes=mitogenes
        )
        kept += len(filtered)
        rows.extend(batch_rows)

    logger.info(
        "Unique TSV processed: total=%d parsed=%d skipped=%d kept_after_filters=%d rows=%d",
        total,
        parsed,
        skipped,
        kept,
        len(rows),
    )

    logger.info("Converting rows to DataFrame and building norand inet")
    sif_df = pd.DataFrame(rows, columns=SIF_COL_NAMES, dtype=object)
    inet = sif_dump_df_to_digraph(df=sif_df, date=args.indra_date)

    with open(out_file, "wb") as fh:
        pickle.dump(inet, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved norand inet -> %s", out_file)


def build_norand(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = args.output_file or os.path.join(
        args.output_dir, inet_filename(args.source_type, "norand")
    )

    if args.input_unique_stmts:
        _build_norand_from_unique_tsv(args, out_file)
    else:
        _build_norand_from_stmts(args, out_file)


def shuffle_xswap(net, seed: int = 1):
    """Degree-preserving edge randomization with xswap."""
    try:
        import xswap
        import networkx as nx
    except ImportError as err:
        raise RuntimeError(
            "xswap/networkx is required for build-inets. "
            "Run this subcommand in your Python 3.9 env."
        ) from err

    node_to_int = {node: ix for ix, node in enumerate(net.nodes)}
    int_to_node = {ix: node for node, ix in node_to_int.items()}
    relabeled_edges = [(node_to_int[u], node_to_int[v]) for u, v in net.edges]

    permuted_edges_int, _ = xswap.permute_edge_list(
        relabeled_edges,
        allow_self_loops=True,
        allow_antiparallel=True,
        multiplier=10,
        seed=seed,
    )
    permuted_edges_node = [(int_to_node[u], int_to_node[v])
                           for u, v in permuted_edges_int]

    shuffled = nx.DiGraph()
    shuffled.add_nodes_from(net.nodes(data=True))
    shuffled.add_edges_from(permuted_edges_node)
    if "node_by_ns_id" in net.graph:
        shuffled.graph["node_by_ns_id"] = net.graph["node_by_ns_id"]
    return shuffled


def shuffle_labels(net, seed: int = 1):
    """Node-label permutation randomization."""
    import networkx as nx

    rng = random.Random(seed)
    old_nodes = list(net.nodes())
    shuffled_nodes = old_nodes[:]
    rng.shuffle(shuffled_nodes)
    old_to_new = dict(zip(old_nodes, shuffled_nodes))

    shuffled = nx.relabel.relabel_nodes(net, old_to_new, copy=True)
    if "node_by_ns_id" in net.graph:
        shuffled.graph["node_by_ns_id"] = net.graph["node_by_ns_id"]
    return shuffled


def build_inets(args: argparse.Namespace) -> None:
    source_type = args.source_type
    norand_file = args.norand_file or os.path.join(
        args.output_dir, inet_filename(source_type, "norand")
    )
    randxswap_file = os.path.join(
        args.output_dir, inet_filename(source_type, "randxswap")
    )
    randlabels_file = os.path.join(
        args.output_dir, inet_filename(source_type, "randlabels")
    )

    logger.info("Loading base inet: %s", norand_file)
    with open(norand_file, "rb") as fh:
        inet = pickle.load(fh)

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Building randlabels inet")
    randlabels_inet = shuffle_labels(inet, seed=args.seed)
    with open(randlabels_file, "wb") as fh:
        pickle.dump(randlabels_inet, fh, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Building randxswap inet")
    randxswap_inet = shuffle_xswap(inet, seed=args.seed)
    with open(randxswap_file, "wb") as fh:
        pickle.dump(randxswap_inet, fh, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Saved randlabels -> %s", randlabels_file)
    logger.info("Saved randxswap -> %s", randxswap_file)


def get_corr_bins(lower: float, upper: float, n_points: int) -> List[Tuple[float, Optional[float]]]:
    corr_range = np.linspace(lower, upper, n_points)
    corr_bins = []
    for ix in range(len(corr_range)):
        corr_lb = float(corr_range[ix])
        corr_ub = None if (ix + 1) >= len(corr_range) else float(corr_range[ix + 1])
        corr_bins.append((corr_lb, corr_ub))
    return corr_bins


def load_or_calc_corr_bin_counts(
    depmap_corr_file: str,
    corr_bin_ct_file: str,
    corr_bins: Sequence[Tuple[float, Optional[float]]],
    recalculate: bool,
) -> List[Tuple[Tuple[float, Optional[float]], int]]:
    from depmap_analysis.network_functions.depmap_network_functions import get_pairs

    if not recalculate and os.path.exists(corr_bin_ct_file):
        logger.info("Loading correlation counts from %s", corr_bin_ct_file)
        with open(corr_bin_ct_file, "rb") as fh:
            return pickle.load(fh)

    logger.info("Recalculating correlation bin counts from %s", depmap_corr_file)
    dep_z = pd.read_hdf(depmap_corr_file)
    corr_bin_counts = []
    for corr_lb, corr_ub in corr_bins:
        logger.info("Filtering correlation matrix to range (%s, %s)", corr_lb, corr_ub)
        if corr_ub is None:
            dep_subset = dep_z[dep_z.abs() >= corr_lb]
        else:
            dep_subset = dep_z[(dep_z.abs() >= corr_lb) & (dep_z.abs() < corr_ub)]
        bin_count = get_pairs(dep_subset)
        corr_bin_counts.append(((corr_lb, corr_ub), bin_count))

    with open(corr_bin_ct_file, "wb") as fh:
        pickle.dump(corr_bin_counts, fh)
    logger.info("Saved correlation counts to %s", corr_bin_ct_file)
    return corr_bin_counts


def run_depmap_wrapper(
    inet_file: str,
    output_file: str,
    sd_range: Tuple[float, Optional[float]],
    count: int,
    depmap_corr_file: str,
    reactome_file: str,
    depmap_date: str,
    expl_funcs: Sequence[str],
    max_pairs: int,
) -> None:
    from depmap_analysis.scripts.depmap_script2 import main as run_depmap
    from depmap_analysis.scripts.depmap_script2 import mito_file

    sample_size = max_pairs if count > max_pairs else None
    run_depmap(
        inet_file,
        depmap_corr_file,
        output_file,
        "unsigned",
        sd_range,
        sample_size=sample_size,
        apriori_explained=mito_file,
        reactome_path=reactome_file,
        overwrite=True,
        depmap_date=depmap_date,
        expl_funcs=tuple(expl_funcs),
        n_chunks=1,
    )


def iter_expl_jobs(
    source_types: Iterable[str],
    rand_types: Iterable[str],
    corr_bin_counts: Sequence[Tuple[Tuple[float, Optional[float]], int]],
    output_dir: str,
):
    for source_type in source_types:
        for rand_type in rand_types:
            inet_file = os.path.join(output_dir, inet_filename(source_type, rand_type))
            for (corr_lb, corr_ub), count in reversed(corr_bin_counts):
                output_stem = f"bioexp_depmap_{source_type}_{rand_type}_{corr_lb}_{corr_ub}"
                output_file = os.path.join(output_dir, "expl", output_stem)
                yield (source_type, rand_type, corr_lb, corr_ub, count, inet_file, output_file)


def run_depmap_phase(args: argparse.Namespace) -> None:
    source_types = parse_csv_arg(args.source_types)
    rand_types = parse_csv_arg(args.rand_types)

    depmap_corr_file = args.depmap_corr_file or os.path.join(args.output_dir, "dep_z.h5")
    corr_bin_ct_file = args.corr_bin_count_file or os.path.join(args.output_dir, "dep_corr_bin_counts.pkl")
    reactome_file = args.reactome_file or os.path.join(args.output_dir, "reactome_pathways.pkl")
    os.makedirs(os.path.join(args.output_dir, "expl"), exist_ok=True)

    corr_bins = get_corr_bins(args.corr_lower, args.corr_upper, args.corr_points)
    corr_bin_counts = load_or_calc_corr_bin_counts(
        depmap_corr_file=depmap_corr_file,
        corr_bin_ct_file=corr_bin_ct_file,
        corr_bins=corr_bins,
        recalculate=args.recalculate_corr_bins,
    )

    for source_type, rand_type, corr_lb, corr_ub, count, inet_file, output_file in iter_expl_jobs(
        source_types=source_types,
        rand_types=rand_types,
        corr_bin_counts=corr_bin_counts,
        output_dir=args.output_dir,
    ):
        if not os.path.exists(inet_file):
            raise FileNotFoundError(
                f"Missing inet file for {source_type}/{rand_type}: {inet_file}"
            )
        logger.info(
            "Running depmap: source=%s rand=%s z_range=(%s, %s) pairs=%s",
            source_type,
            rand_type,
            corr_lb,
            corr_ub,
            count,
        )
        run_depmap_wrapper(
            inet_file=inet_file,
            output_file=output_file,
            sd_range=(corr_lb, corr_ub),
            count=count,
            depmap_corr_file=depmap_corr_file,
            reactome_file=reactome_file,
            depmap_date=args.depmap_date,
            expl_funcs=parse_csv_arg(args.expl_funcs),
            max_pairs=args.max_pairs,
        )


def run_single(args: argparse.Namespace) -> None:
    """Run a single depmap job, matching the notebook-style call."""
    from depmap_analysis.scripts.depmap_script2 import main as run_depmap
    from depmap_analysis.scripts.depmap_script2 import mito_file

    sd_range = (args.sd_lower, args.sd_upper)
    expl_funcs = parse_csv_arg(args.expl_funcs)

    run_depmap(
        args.inet_file,
        args.depmap_corr_file,
        args.output_file,
        "unsigned",
        sd_range,
        sample_size=args.sample_size,
        apriori_explained=mito_file,
        reactome_path=args.reactome_file,
        overwrite=True,
        depmap_date=args.depmap_date,
        expl_funcs=expl_funcs,
        n_chunks=args.n_chunks,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split depmap scratch into env-specific commands."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_norand = subparsers.add_parser(
        "build-norand",
        help="Build base norand inet from statement input or streamed unique statements.",
    )
    source_group = p_norand.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--input-stmts",
        help="Input statement pickle/json path accepted by indra.tools.assemble_corpus.load_statements",
    )
    source_group.add_argument(
        "--input-unique-stmts",
        help="Path to unique_statements.tsv.gz to stream rows from.",
    )
    p_norand.add_argument("--source-counts-pkl", default=None,
                          help="Optional source_counts.pkl for unique-statements mode.")
    p_norand.add_argument("--max-unique-rows", type=int, default=None,
                          help="Optional cap on number of rows read from unique statements.")
    p_norand.add_argument("--batch-size", type=int, default=100000)
    p_norand.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p_norand.add_argument("--source-type", default="all")
    p_norand.add_argument("--output-file", default=None)
    p_norand.add_argument("--genes-only", action=argparse.BooleanOptionalAction, default=False) #
    p_norand.add_argument("--human-only", action=argparse.BooleanOptionalAction, default=True)
    p_norand.add_argument("--require-named-agents", action=argparse.BooleanOptionalAction, default=True)
    p_norand.add_argument("--require-two-agents", action=argparse.BooleanOptionalAction, default=False)
    p_norand.add_argument("--exclude-mitocarta", action=argparse.BooleanOptionalAction, default=True)
    p_norand.add_argument("--complex-members", type=int, default=3)
    p_norand.add_argument("--indra-date", default="20220802")
    p_norand.set_defaults(func=build_norand)

    p_build = subparsers.add_parser(
        "build-inets",
        help="Build randxswap/randlabels inet files from a base norand inet (py3.9 + xswap).",
    )
    p_build.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p_build.add_argument("--source-type", default="all")
    p_build.add_argument("--norand-file", default=None)
    p_build.add_argument("--seed", type=int, default=1)
    p_build.set_defaults(func=build_inets)

    p_run = subparsers.add_parser(
        "run-depmap",
        help="Run depmap explanation benchmark from existing inet files (py3.12).",
    )
    p_run.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p_run.add_argument("--source-types", default="all")
    p_run.add_argument("--rand-types", default="norand,randxswap,randlabels")
    p_run.add_argument("--depmap-corr-file", default=None)
    p_run.add_argument("--corr-bin-count-file", default=None)
    p_run.add_argument("--reactome-file", default=None)
    p_run.add_argument("--recalculate-corr-bins", action="store_true")
    p_run.add_argument("--corr-lower", type=float, default=0.0)
    p_run.add_argument("--corr-upper", type=float, default=16.0)
    p_run.add_argument("--corr-points", type=int, default=33)
    p_run.add_argument("--depmap-date", default="21q2")
    p_run.add_argument("--max-pairs", type=int, default=1_000_000)
    p_run.add_argument("--expl-funcs", default=",".join(DEFAULT_EXPL_FUNCS))
    p_run.set_defaults(func=run_depmap_phase)

    p_single = subparsers.add_parser(
        "run-single",
        help="Run one depmap job directly (notebook-style run_depmap call).",
    )
    p_single.add_argument("--inet-file", required=True)
    p_single.add_argument("--depmap-corr-file", required=True)
    p_single.add_argument("--output-file", required=True)
    p_single.add_argument("--sd-lower", type=float, required=True)
    p_single.add_argument("--sd-upper", type=float, default=None)
    p_single.add_argument("--sample-size", type=int, default=None)
    p_single.add_argument("--reactome-file", required=True)
    p_single.add_argument("--depmap-date", default="21q2")
    p_single.add_argument("--expl-funcs", default=",".join(DEFAULT_EXPL_FUNCS))
    p_single.add_argument("--n-chunks", type=int, default=1)
    p_single.set_defaults(func=run_single)

    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
