# solver_core.py — lazy OR-Tools import; Py3.13 compatible
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pandas as pd

ORTOOLS_AVAILABLE = False
ortools_version_hint = "ortools==9.13.4784"  # Matches requirements.txt

try:
    import ortools  # probe only
    ORTOOLS_AVAILABLE = True
except Exception:
    ORTOOLS_AVAILABLE = False

REQUIRED_HEADERS = {
    "students": ["student_id","name","gender","year","achievement_band","support_level","behaviour_level","locked_class"],
    "classes":  ["class_id","name","capacity_min","capacity_max","target_gender_M","target_year6"],
}

@dataclass
class Student:
    id: str
    gender_M: int
    year6: int
    ach_band: int
    support: int
    behaviour: int
    locked_class: Optional[str]

@dataclass
class Klass:
    id: str
    cap_min: int
    cap_max: int
    target_gender_M: Optional[int]
    target_year6: Optional[int]

@dataclass
class ModelWeights:
    w_gender: int = 10
    w_yearmix: int = 10
    w_achiev: int = 5
    w_support: int = 8
    w_behaviour: int = 8
    w_friends: int = 6
    w_teacher: int = 4

def _to_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def _norm_students(df: pd.DataFrame) -> List[Student]:
    out: List[Student] = []
    for _, r in df.fillna("").iterrows():
        sid = str(r["student_id"]).strip()
        gM = 1 if str(r["gender"]).strip().upper() == "M" else 0
        year6 = 1 if _to_int(r.get("year", ""), 0) == 6 else 0
        ach = _to_int(r.get("achievement_band", ""), 3) or 3
        sup = _to_int(r.get("support_level", ""), 0) or 0
        beh = _to_int(r.get("behaviour_level", ""), 0) or 0
        locked = str(r.get("locked_class", "")).strip() or None
        if sid:
            out.append(Student(sid, gM, year6, ach, sup, beh, locked))
    return out

def _norm_classes(df: pd.DataFrame) -> List[Klass]:
    out: List[Klass] = []
    for _, r in df.fillna("").iterrows():
        kid = str(r["class_id"]).strip()
        cap_min = _to_int(r.get("capacity_min", ""), 0) or 0
        cap_max = _to_int(r.get("capacity_max", ""), 10) or 10
        tg = _to_int(r.get("target_gender_M", ""), None)
        ty = _to_int(r.get("target_year6", ""), None)
        if kid:
            out.append(Klass(kid, cap_min, cap_max, tg, ty))
    return out

def solve_from_dataframes(students_df: pd.DataFrame,
                          classes_df: pd.DataFrame,
                          friendships_df: Optional[pd.DataFrame] = None,
                          must_together_df: Optional[pd.DataFrame] = None,
                          must_apart_df: Optional[pd.DataFrame] = None,
                          teacher_prefs_df: Optional[pd.DataFrame] = None,
                          time_limit_s: int = 30):
    try:
        from ortools.sat.python import cp_model
    except Exception as e:
        raise RuntimeError(
            "OR-Tools is required to run the optimizer but is not available. "
            f"Install {ortools_version_hint}. Original error: {e}"
        )

    # Header checks
    def _check(df, required):
        if df is None:
            return [f"Missing required file with headers: {required}"]
        missing = [h for h in required if h not in df.columns]
        return [f"Missing headers in required CSV: {missing}"] if missing else []

    errs = []
    errs += _check(students_df, REQUIRED_HEADERS["students"])
    errs += _check(classes_df, REQUIRED_HEADERS["classes"])
    if errs:
        raise ValueError("; ".join(errs))

    students = _norm_students(students_df)
    klasses = _norm_classes(classes_df)

    # Edge building
    def _edges(df: Optional[pd.DataFrame], cols: List[str], weight_col: Optional[str] = None):
        if df is None: return {}
        df = df.fillna("")
        mx = {}
        for _, r in df.iterrows():
            a = str(r[cols[0]]).strip()
            b = str(r[cols[1]]).strip()
            if not a or not b or a == b: 
                continue
            w = _to_int(r.get(weight_col, ""), 1) or 1 if weight_col else 1
            key = tuple(sorted((a,b)))
            mx[key] = max(mx.get(key, 0), w)
        return mx

    friend_edges = _edges(friendships_df, ["student_id","friend_id"], "weight")
    must_together_edges = _edges(must_together_df, ["a_id","b_id"])
    must_apart_edges    = _edges(must_apart_df,    ["a_id","b_id"])

    # Model
    model = cp_model.CpModel()
    I = list(range(len(students)))
    K = list(range(len(klasses)))
    sid_to_i = {s.id: i for i, s in enumerate(students)}
    x = {(i,k): model.NewBoolVar(f"x_{i}_{k}") for i in I for k in K}

    # Assignment + locks
    for i in I:
        s = students[i]
        if s.locked_class is None:
            model.Add(sum(x[(i,k)] for k in K) == 1)
        else:
            k_lock = next((k for k, kk in enumerate(klasses) if kk.id == s.locked_class), None)
            if k_lock is None:
                model.Add(sum(x[(i,k)] for k in K) == 1)
            else:
                model.Add(x[(i,k_lock)] == 1)
                for k in K:
                    if k != k_lock: model.Add(x[(i,k)] == 0)

    # Caps
    for k in K:
        model.Add(sum(x[(i,k)] for i in I) >= klasses[k].cap_min)
        model.Add(sum(x[(i,k)] for i in I) <= klasses[k].cap_max)

    # Must-apart
    for (a_id,b_id) in must_apart_edges.keys():
        if a_id in sid_to_i and b_id in sid_to_i:
            a, b = sid_to_i[a_id], sid_to_i[b_id]
            for k in K: model.Add(x[(a,k)] + x[(b,k)] <= 1)

    # Must-together
    for (a_id,b_id) in must_together_edges.keys():
        if a_id in sid_to_i and b_id in sid_to_i:
            a, b = sid_to_i[a_id], sid_to_i[b_id]
            for k in K: model.Add(x[(a,k)] == x[(b,k)])

    # Aggregates
    g_sum  = {k: model.NewIntVar(0, len(I),      f"g_sum_{k}")  for k in K}
    y6_sum = {k: model.NewIntVar(0, len(I),      f"y6_sum_{k}") for k in K}
    ach_sum= {k: model.NewIntVar(0, 5*len(I),    f"ach_sum_{k}")for k in K}

    sup_sum= {k: model.NewIntVar(0, 100000,      f"sup_sum_{k}")for k in K}
    beh_sum= {k: model.NewIntVar(0, 100000,      f"beh_sum_{k}")for k in K}

    for k in K:
        model.Add(g_sum[k]  == sum(students[i].gender_M * x[(i,k)] for i in I))
        model.Add(y6_sum[k] == sum(students[i].year6    * x[(i,k)] for i in I))
        model.Add(ach_sum[k]== sum(students[i].ach_band * x[(i,k)] for i in I))
        model.Add(sup_sum[k]== sum(students[i].support  * x[(i,k)] for i in I))
        model.Add(beh_sum[k]== sum(students[i].behaviour* x[(i,k)] for i in I))

    def abs_dev(var, target, name):
        d = model.NewIntVar(0, 1000000, name)
        model.Add(var - target <= d)
        model.Add(target - var <= d)
        return d

    penalties = []
    for k in K:
        tg = klasses[k].target_gender_M
        if tg is not None: penalties.append(abs_dev(g_sum[k], tg, f"dev_gender_{k}"))
        ty = klasses[k].target_year6
        if ty is not None: penalties.append(abs_dev(y6_sum[k], ty, f"dev_year6_{k}"))

    for k1 in K:
        for k2 in K:
            if k2 <= k1: continue
            dA = model.NewIntVar(0, 1000000, f"dev_ach_{k1}_{k2}")
            model.Add(ach_sum[k1] - ach_sum[k2] <= dA)
            model.Add(ach_sum[k2] - ach_sum[k1] <= dA)
            penalties.append(dA)
            dS = model.NewIntVar(0, 1000000, f"dev_sup_{k1}_{k2}")
            dB = model.NewIntVar(0, 1000000, f"dev_beh_{k1}_{k2}")
            model.Add(sup_sum[k1] - sup_sum[k2] <= dS)
            model.Add(sup_sum[k2] - sup_sum[k1] <= dS)
            model.Add(beh_sum[k1] - beh_sum[k2] <= dB)
            model.Add(beh_sum[k2] - beh_sum[k1] <= dB)
            penalties.extend([dS,dB])

    def pair_terms(edge_map, prefix):
        terms = []
        for (a_id,b_id), w in edge_map.items():
            if a_id in sid_to_i and b_id in sid_to_i:
                a, b = sid_to_i[a_id], sid_to_i[b_id]
                for k in K:
                    y = model.NewBoolVar(f"{prefix}_{a}_{b}_{k}")
                    model.Add(y <= x[(a,k)])
                    model.Add(y <= x[(b,k)])
                    model.Add(y >= x[(a,k)] + x[(b,k)] - 1)
                    terms.append((y,w))
        return terms

    friend_terms = pair_terms(friend_edges, "y_friend")

    w = ModelWeights()
    obj = []
    for p in penalties:
        name = p.Name()
        if name.startswith("dev_gender"): obj.append(w.w_gender * p)
        elif name.startswith("dev_year6"): obj.append(w.w_yearmix * p)
        elif name.startswith("dev_ach"):   obj.append(w.w_achiev * p)
        elif name.startswith("dev_sup"):   obj.append(w.w_support * p)
        elif name.startswith("dev_beh"):   obj.append(w.w_behaviour * p)
        else: obj.append(p)
    for y, wt in friend_terms: obj.append(- w.w_friends * wt * y)
    model.Minimize(sum(obj))

    from ortools.sat.python import cp_model
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 8
    solver.parameters.log_search_progress = False
    status = solver.Solve(model)

    # Extract solution
    assign = {}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in I:
            for k in K:
                if solver.Value(x[(i,k)]) == 1:
                    assign[students[i].id] = klasses[k].id
                    break

    assignments_df = pd.DataFrame(
        [(sid, cid) for sid, cid in assign.items()],
        columns=["student_id", "class_id"]
    ).sort_values("student_id")

    # Basic metrics
    per_class = {k.id: {"size":0, "gender_M":0, "year6":0, "ach_sum":0, "support_sum":0, "behaviour_sum":0} for k in klasses}
    s_by_id = {s.id: s for s in students}
    for sid, cid in assign.items():
        s = s_by_id[sid]
        m = per_class[cid]
        m["size"] += 1
        m["gender_M"] += s.gender_M
        m["year6"] += s.year6
        m["ach_sum"] += s.ach_band
        m["support_sum"] += s.support
        m["behaviour_sum"] += s.behaviour

    def spread(key):
        vals = [m[key] for m in per_class.values()]
        return max(vals) - min(vals) if vals else None

    if friend_edges:
        sat = sum(1 for (a,b) in friend_edges.keys() if assign.get(a) and assign.get(a) == assign.get(b))
        rate = sat / len(friend_edges)
    else:
        rate = None

    metrics = {
        "friend_satisfaction_rate": rate,
        "size_spread": spread("size"),
        "genderM_spread": spread("gender_M"),
        "year6_spread": spread("year6"),
        "support_spread": spread("support_sum"),
        "behaviour_spread": spread("behaviour_sum"),
        "achievement_spread": spread("ach_sum"),
    }

    diagnostics = []
    cap_min_total = sum(k.cap_min for k in klasses)
    cap_max_total = sum(k.cap_max for k in klasses)
    n = len(students)
    if cap_min_total > n:
        diagnostics.append(f"Infeasible: sum(cap_min)={cap_min_total} > students={n}")
    if cap_max_total < n:
        diagnostics.append(f"Infeasible: sum(cap_max)={cap_max_total} < students={n}")

    return assignments_df, metrics, diagnostics
