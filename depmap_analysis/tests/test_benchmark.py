import argparse
import sys
import types

import pandas as pd
from indra.statements import Agent, Evidence, Activation, Inhibition
from depmap_analysis.util import benchmark as bm
from depmap_analysis.network_functions.net_functions import sif_dump_df_to_digraph


def test_inet_generation_from_synthetic_statements():
    a = Agent("TP53", db_refs={"HGNC": "11998"})
    b = Agent("CDKN1A", db_refs={"HGNC": "1784"})

    st1 = Activation(a, b, evidence=[Evidence(source_api="reach")])
    st2 = Inhibition(b, a, evidence=[Evidence(source_api="sparser")])

    _filtered, rows = bm._filter_and_rows_from_batch(
        [st1, st2], source_counts=None, mitogenes=None
    )

    sif_df = pd.DataFrame(rows, columns=bm.SIF_COL_NAMES, dtype=object)
    inet = sif_dump_df_to_digraph(
        df=sif_df,
        date="test",
        graph_type="digraph",
        include_entity_hierarchies=False,
    )

    assert inet.has_edge("TP53", "CDKN1A")
    assert inet.has_edge("CDKN1A", "TP53")


def test_run_single_invokes_run_depmap(monkeypatch):
    captured = {}

    fake_script = types.ModuleType("depmap_analysis.scripts.depmap_script2")

    def _fake_main(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    fake_script.main = _fake_main
    fake_script.mito_file = "fake_mito.tsv"
    monkeypatch.setitem(sys.modules, "depmap_analysis.scripts.depmap_script2", fake_script)

    args = argparse.Namespace(
        inet_file="inet.pkl",
        depmap_corr_file="dep_z.h5",
        output_file="/tmp/out_expl",
        sd_lower=3.0,
        sd_upper=3.5,
        reactome_file="reactome.pkl",
    )

    bm.run_single(args)

    assert captured["args"][:5] == (
        "inet.pkl",
        "dep_z.h5",
        "/tmp/out_expl",
        "unsigned",
        (3.0, 3.5),
    )
    assert captured["kwargs"]["apriori_explained"] == "fake_mito.tsv"
    assert captured["kwargs"]["reactome_path"] == "reactome.pkl"
    assert captured["kwargs"]["expl_funcs"] == bm.DEFAULT_EXPL_FUNCS
