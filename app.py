# app.py — Streamlit pilot (Py3.13-compatible)
import io
import json
import pandas as pd
import streamlit as st
from solver_core import (
    solve_from_dataframes,
    REQUIRED_HEADERS,
    ORTOOLS_AVAILABLE,
    ortools_version_hint,
)

st.set_page_config(page_title="Class Builder 2026 (Pilot)", layout="wide")
st.title("Class Builder 2026 — Pilot")
st.caption("Uploads stay in-memory for this browser session only. No database.")

with st.sidebar:
    st.header("Run settings")
    time_limit = st.slider("Time limit (seconds)", 5, 120, 20, step=5)
    st.divider()
    load_sample = st.button("Load sample data")

if not ORTOOLS_AVAILABLE:
    st.info(
        "OR-Tools isn't imported until you click Run. "
        "If it's missing, install the version listed in requirements.txt. "
        f"Expected version: {ortools_version_hint}"
    )

# Uploaders
with st.expander("Upload CSVs", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        f_students = st.file_uploader("students.csv", type=["csv"])
        f_classes = st.file_uploader("classes.csv", type=["csv"])
    with c2:
        f_friend = st.file_uploader("friendships.csv (optional)", type=["csv"])
        f_teacher = st.file_uploader("teacher_prefs.csv (optional)", type=["csv"])
    with c3:
        f_together = st.file_uploader("must_together.csv (optional)", type=["csv"])
        f_apart = st.file_uploader("must_apart.csv (optional)", type=["csv"])

    st.markdown("**Required headers**")
    st.code(
        "\n".join([
            f"students.csv -> {REQUIRED_HEADERS['students']}",
            f"classes.csv  -> {REQUIRED_HEADERS['classes']}",
        ]),
        language="text"
    )

# Sample data (no triple quotes)
S_STUDENTS = (
    "student_id,name,gender,year,achievement_band,support_level,behaviour_level,locked_class\n"
    "s01,Ava,F,6,3,0,0,\n"
    "s02,Noah,M,6,2,1,0,\n"
    "s03,Olivia,F,6,4,0,1,\n"
    "s04,Leo,M,6,3,2,1,\n"
    "s05,Isla,F,7,5,1,0,\n"
    "s06,Jack,M,7,2,0,0,\n"
    "s07,Emily,F,7,3,1,1,\n"
    "s08,Lucas,M,7,4,0,0,\n"
    "s09,Ruby,F,6,2,0,0,\n"
    "s10,Oliver,M,6,5,2,0,\n"
    "s11,Grace,F,7,3,1,0,\n"
    "s12,Charlie,M,7,1,0,2,\n"
)
S_CLASSES = (
    "class_id,name,capacity_min,capacity_max,target_gender_M,target_year6\n"
    "R1,Room 1,4,4,2,2\n"
    "R2,Room 2,4,4,2,2\n"
    "R3,Room 3,4,4,2,2\n"
)
S_FRIENDS = (
    "student_id,friend_id,weight\n"
    "s01,s02,2\n"
    "s03,s04,3\n"
    "s05,s06,2\n"
    "s07,s08,1\n"
    "s09,s10,2\n"
    "s11,s12,1\n"
)
S_TOGETHER = "a_id,b_id\ns01,s03\n"
S_APART    = "a_id,b_id\ns10,s12\n"

if load_sample:
    st.session_state["_use_sample"] = True
if "_use_sample" not in st.session_state:
    st.session_state["_use_sample"] = False

run = st.button("Run optimizer", type="primary", use_container_width=True)

if run:
    try:
        if st.session_state["_use_sample"]:
            students_df = pd.read_csv(io.StringIO(S_STUDENTS))
            classes_df  = pd.read_csv(io.StringIO(S_CLASSES))
            friends_df  = pd.read_csv(io.StringIO(S_FRIENDS))
            together_df = pd.read_csv(io.StringIO(S_TOGETHER))
            apart_df    = pd.read_csv(io.StringIO(S_APART))
            teacher_df  = None
        else:
            if not (f_students and f_classes):
                st.error("students.csv and classes.csv are required.")
                st.stop()
            students_df = pd.read_csv(f_students)
            classes_df  = pd.read_csv(f_classes)
            friends_df  = pd.read_csv(f_friend) if f_friend else None
            together_df = pd.read_csv(f_together) if f_together else None
            apart_df    = pd.read_csv(f_apart) if f_apart else None
            teacher_df  = pd.read_csv(f_teacher) if f_teacher else None

        with st.status("Solving...", expanded=False):
            assignments_df, metrics_dict, diagnostics = solve_from_dataframes(
                students_df, classes_df,
                friendships_df=friends_df,
                must_together_df=together_df,
                must_apart_df=apart_df,
                teacher_prefs_df=teacher_df,
                time_limit_s=time_limit,
            )
    except Exception as e:
        st.exception(e)
        st.stop()

    colL, colR = st.columns([2, 1], gap="large")
    with colL:
        st.subheader("Assignments")
        st.dataframe(assignments_df, use_container_width=True, height=420)
        st.download_button(
            "Download assignments.csv",
            data=assignments_df.to_csv(index=False).encode("utf-8"),
            file_name="assignments.csv",
            mime="text/csv"
        )

    with colR:
        st.subheader("Metrics")
        st.json(metrics_dict)
        st.download_button(
            "Download metrics.json",
            data=json.dumps(metrics_dict, indent=2).encode("utf-8"),
            file_name="metrics.json"
        )

    if diagnostics:
        st.warning("Diagnostics")
        for line in diagnostics:
            st.write("- " + line)
