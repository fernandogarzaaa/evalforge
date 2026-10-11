"""Streamlit curation dashboard: dataset explorer, review queue, growth.

    pip install "evalforge[dashboard]"
    EVALFORGE_DB=evalforge.db streamlit run evalforge/dashboard.py

A thin view over ``evalforge.curation`` and ``evalforge.store``; all decisions are
recorded through the same functions the CLI uses.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any

import pandas as pd
import streamlit as st

from evalforge.curation import review, review_queue, reviewer_stats, similar_cases
from evalforge.dedup import coverage
from evalforge.store import Store

st.set_page_config(page_title="evalforge", layout="wide")


@st.cache_resource
def get_store(path: str) -> Store:
    return Store(path)


def case_row(c: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": c["id"],
        "status": c["status"],
        "category": c["category"],
        "difficulty": c["difficulty"],
        "polarity": c["polarity"],
        "quality": c["observed"]["quality"],
        "behavior": c["expected"]["behavior"],
        "confidence": c["confidence"],
        "source": c["label_source"],
        "prompt": c["input"]["prompt"][:120],
    }


def show_case(store: Store, c: dict[str, Any], key: str) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("**Prompt**")
        st.code(c["input"]["prompt"], language=None)
        st.markdown("**Observed response**")
        st.code(c["observed"]["response"] or "(empty / filtered)", language=None)
        st.caption(f"flags: {', '.join(c.get('flags', [])) or 'none'} | judge runs: {c.get('judge_run_qualities')}")
    with right:
        st.markdown("**Labels**")
        st.json(
            {
                "expected": c["expected"],
                "assertions": c["assertions"],
                "quality": c["observed"]["quality"],
                "difficulty": c["difficulty"],
                "confidence": c["confidence"],
                "rationale": c.get("rationale"),
            }
        )
        sims = similar_cases(store, c["id"], k=3)
        if sims:
            st.markdown("**Similar existing cases**")
            for s in sims:
                st.caption(f"{s['similarity']:.2f} · {s['input']['prompt'][:100]}")
    reviewer = st.session_state.get("reviewer", "reviewer")
    a, b, d = st.columns(3)
    if a.button("Approve", key=f"ap-{key}"):
        review(store, c["id"], reviewer, "approve")
        st.rerun()
    if d.button("Reject", key=f"rj-{key}"):
        review(store, c["id"], reviewer, "reject")
        st.rerun()
    with b.popover("Edit labels"):
        text = st.text_area(
            "Edits (JSON)",
            value=json.dumps({"behavior": c["expected"]["behavior"], "must_contain": c["assertions"]["must_contain"]}),
            key=f"ed-{key}",
        )
        if st.button("Save and approve", key=f"sv-{key}"):
            review(store, c["id"], reviewer, "edit", json.loads(text))
            st.rerun()


def main() -> None:
    store = get_store(os.environ.get("EVALFORGE_DB", "evalforge.db"))
    st.sidebar.title("evalforge")
    st.session_state["reviewer"] = st.sidebar.text_input("Reviewer", value=os.environ.get("USER", "reviewer"))
    page = st.sidebar.radio("View", ("Review queue", "Explorer", "Growth & coverage"))

    if page == "Review queue":
        queue = review_queue(store)
        st.header(f"Review queue ({len(queue)})")
        stats = reviewer_stats(store.reviews())
        st.caption(f"reviews: {stats['reviews']} · per reviewer: {stats['per_reviewer']} · kappa: {stats['kappa']}")
        for i, c in enumerate(queue[:25]):
            with st.expander(f"{c['confidence']:.2f} · {c['category']} · {c['input']['prompt'][:90]}"):
                show_case(store, c, f"q{i}")

    elif page == "Explorer":
        cases = store.cases()
        df = pd.DataFrame([case_row(c) for c in cases])
        if df.empty:
            st.info("No cases yet. Run `evalforge run --inbox <dir>` first.")
            return
        cols = st.columns(4)
        filters = {
            f: cols[i].multiselect(f, sorted(df[f].dropna().unique()))
            for i, f in enumerate(("status", "category", "difficulty", "polarity"))
        }
        for f, vals in filters.items():
            if vals:
                df = df[df[f].isin(vals)]
        st.dataframe(df, use_container_width=True, hide_index=True)
        pick = st.selectbox("Open case", [""] + list(df["id"]))
        if pick:
            case = store.get_case(pick)
            if case:
                show_case(store, case, "x")

    else:
        runs = store.pipeline_runs()
        st.header("Dataset growth")
        if runs:
            growth = pd.DataFrame(
                [
                    {"day": r["day"], "active": r.get("total_active", 0), "pending": r.get("total_pending", 0)}
                    for r in runs
                ]
            ).set_index("day")
            st.line_chart(growth)
        active = store.cases("active")
        cov = coverage(active, [c["name"] for c in store.categories()], 8)
        st.header("Coverage by category")
        st.bar_chart(pd.Series(cov["per_category"]).sort_values(ascending=False).head(40))
        st.caption(f"{len(cov['sparse'])} categories below {cov['target']} cases")
        st.header("Distribution")
        c1, c2, c3 = st.columns(3)
        c1.bar_chart(pd.Series(Counter(c["difficulty"] for c in active)))
        c2.bar_chart(pd.Series(Counter(c["polarity"] for c in active)))
        c3.bar_chart(pd.Series(Counter(c["label_source"] for c in active)))


main()
