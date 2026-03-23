# from modules.utils.load_config import load_settings
# from modules.utils.redis import RedisCache

import json
import time
import yaml
import requests

from typing import TypedDict, Any, Dict, List, Optional, Sequence, Set, Tuple, Protocol, Literal, Iterable
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from dataclasses import dataclass
import pandas as pd

# ----------------------------------------------------------------------
# schema
# ----------------------------------------------------------------------
@dataclass
class OnlineRetrievalResult:
    student_id : str
    candidates : List[Dict[str,Any]]
    meta       : Dict[str,Any]

@dataclass(frozen=True)
class RetrievalDebug:
    """Explainability information for a single retrieval candidate"""
    best_qi : int
    best_raw_score : float
    best_weight : float
    aggregated_score : float

@dataclass(frozen=True)
class ScoreAggregationConfig:
    enabled: bool = False
    mode : str = "linear"
    weights : Dict[str,float] = None
    clamp_inputs: bool = True
    renormalize:bool = True
    missing_subscore_value:float = 0.0
    tie_breakers: Tuple[str,...] = ("vector_score","recency")

class RerankItem(TypedDict):
    feed_id: str
    final_score: float

    
# ----------------------------------------------------------------------
# Helper function
# ----------------------------------------------------------------------
def prettyjson(txt:str) -> str:
    return str(json.dumps(txt,indent=4, ensure_ascii=False))

########################################################################

########################################################################
# online_score_aggregation.py
########################################################################
### def main <- | functions.core.online_score_aggregation.py (line:82) *customs
def extract_query_weights_and_labels(hq:List[Dict]):
    # - Modify datalake to keep this weights from HyDE
    # hq = [
    #     {'query_id': 'Q1',
    #     'query_text': 'แนวทางทำพอร์ต Data Analyst โปรเจกต์ Python SQL',
    #     'weight': 1.0,
    #     'intent_label': 'history_aligned'},
    #     {'query_id': 'Q2',
    #     'query_text': 'เทคนิคเตรียมสัมภาษณ์ฝึกงาน Data Analyst โจทย์ SQL Python',
    #     'weight': 1.0,
    #     'intent_label': 'history_aligned'},
    #     {'query_id': 'Q3',
    #     'query_text': 'ตัวอย่างโปรเจกต์ Data Analyst สำหรับฝึกงานพร้อมโค้ด',
    #     'weight': 1.0,
    #     'intent_label': 'practical'},
    #     {'query_id': 'Q4',
    #     'query_text': 'สร้าง Dashboard ด้วย Power BI หรือ Tableau สำหรับ Data Analyst',
    #     'weight': 0.6,
    #     'intent_label': 'exploratory'},
    #     {'query_id': 'Q5',
    #     'query_text': 'เส้นทางอาชีพ Data Analyst ทักษะที่จำเป็นในอนาคต',
    #     'weight': 0.6,
    #     'intent_label': 'exploratory'}
    # ]
    weights : List[float] = []
    qids    : List[str]   = []
    intents : List[str]   = []
    for i,q in enumerate(hq):
        if not isinstance(q,dict):
            qids.append(f"Q{i+q}")
            intents.append("unknows")
            weights.append("1.0")
            continue

        qid = q.get("query_id")
        if not isinstance(qid,str) or not qid.strip():
            qid = f"Q{i+1}"
        qids.append(qid)

        intent = q.get("intent_label")
        if not isinstance(intent,str) or not intent.strip():
            intent = "unknow"
        intents.append(intent)

        try:
            wf = float(q.get("weight",1.0))
        except Exception:
            wf = 1.0
        weights.append(wf)
    return np.asanyarray(weights,dtype=np.float32), qids, intents

### def load_score_aggregation_config -> | functions.core.online_score_aggregation.py (line:186)
def _to_float(x:Any, default:float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)
    
### def aggregate_candidates -> | functions.core.online_score_aggregation.py (line:193)
def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x

### def load_score_aggregation_config -> | functions.core.online_score_aggregation.py (line:201)
def _safe_yaml_load(path:str) -> Dict[str,Any]:
    """Best-effort YAML loader"""
    try:
        p = Path(path)
        if not p.exists():
            return {}
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    
### def load_score_aggregation_config -> | functions.core.online_score_aggregation.py (line:218)
def _parse_score_aggregation(d:Dict[str,Any]) -> ScoreAggregationConfig:
    sa = d.get("score_aggregation") if isinstance(d,dict) else None
    if not isinstance(sa,dict):
        return ScoreAggregationConfig(enabled=False,weights={})
    enabled = bool(sa.get("enabled",False))
    mode = str(sa.get("mode","linear") or "linear").lower().strip()
    w_raw = sa.get("weights",{}) or {}     # load weight {'vector_score': 0.6, 'language_match': 0.3, 'recency': 0.05, 'popularity': 0.05}
    weights:Dict[str,float] = {}
    if isinstance(w_raw,dict):   # turn weight to dict
        for k,v in w_raw.items():
            if isinstance(k,str) and k.strip():
                weights[k.strip()] = _to_float(v,0.0)
    clamp_inputs = bool(sa.get("clamp_inputs",True))
    renormalize  = bool(sa.get("renormalize",True))
    missing      = _to_float(sa.get("missing_subscore_value",0.0),0.0)
    # print(f"clamp_inputs : {clamp_inputs}")
    # print(f"renormalize  : {renormalize}")
    # print(f"missing      : {missing}")
    tb = sa.get("tie_breakers",["vector_score","recency"])
    if isinstance(tb,(list,tuple)):
        tie_breakers = tuple(str(x) for x in tb if str(x).strip())
    else:
        tie_breakers = ("vector_score","recency")
    
    return ScoreAggregationConfig(
        enabled = enabled,
        mode    = mode,
        weights = weights,
        clamp_inputs = clamp_inputs,
        renormalize= renormalize,
        missing_subscore_value= missing,
        tie_breakers=tie_breakers or ("vector_score","recency")
    )

### def main -> | functions.core.online_score_aggregation.py (line:257)
def load_score_aggregation_config(path:str) -> Dict[str,Any]:
    raw = _safe_yaml_load(path)
    # print(f"raw -> {prettyjson(raw)}")
    cfg = _parse_score_aggregation(raw)
    return {
        "score_aggregation": {
            "enabled": bool(cfg.enabled),
            "mode": str(cfg.mode),
            "weights": dict(cfg.weights or {}),
            "clamp_inputs": bool(cfg.clamp_inputs),
            "renormalize": bool(cfg.renormalize),
            "missing_subscore_value": float(cfg.missing_subscore_value),
            "tie_breakers": list(cfg.tie_breakers),
        }
    }

### def aggregate_candidates -> | functions.core.online_score_aggregation.py (line:300)
def _get_feature_value(candidate: Dict[str, Any], key: str, missing: float) -> float:
    if not isinstance(candidate, dict):
        return float(missing)
    if key in candidate:
        return _to_float(candidate.get(key), missing)
    subs = candidate.get("subscores")
    if isinstance(subs, dict) and key in subs:
        return _to_float(subs.get(key), missing)
    return float(missing)

### def main -> | functions.core.online_score_aggregation.py (line:322)
def aggregate_candidates(candidates:List[Dict[str,Any]], cfg:Dict[str,Any]) -> List[Dict[str,Any]]:
    sa = cfg.get("score_aggregation") if isinstance(cfg, dict) else None
    if not isinstance(sa,dict) or not bool(sa.get("enabled",False)):
        return candidates
    mode = str(sa.get("mode","linear") or "linear").lower().strip()
    if mode != "linear":
        return candidates
    weights_raw = sa.get("weights",{}) or {} # {'vector_score': 0.6, 'language_match': 0.3, 'recency': 0.05, 'popularity': 0.05}
    weights: Dict[str,float] = {}
    if isinstance(weights_raw,dict):
        for k,v in weights_raw.items():
            if isinstance(k,str) and k.strip():
                weights[k.strip()] = _to_float(v,0.0)

    clamp_inputs = bool(sa.get("clamp_inputs",True))
    renormalize  = bool(sa.get("renormalize",True))
    missing      = _to_float(sa.get("missing_subscore_value",0.0),0.0)

    tie_breakers = sa.get("tie_breakers",["vector_score","recency"])
    if not isinstance(tie_breakers,(list,tuple)):
        tie_breakers = ["vector_score","recency"]
    tie_breakers = [str(x) for x in tie_breakers if str(x).strip()]
    # print(tie_breakers)

    # Renormalize weights if requested (normalize with summation of every weight)
    if renormalize:
        s = float(sum(max(0.0, float(v)) for v in weights.values()))
        if s > 0.0:
            weights = {k: float(v)/s for k,v in weights.items()}
    out: List[Dict[str,Any]] = []
    for c in candidates:
        # print(f"candidate -> {c}")          # {'feed_id': 'TH_F023', 'vector_score': 0.9138078093528748, 'subscores': {'language_match': 1.0, 'recency': 0.12537116975009344, 'popularity': 0.7422222182490312}}
        row = dict(c) if isinstance(c,dict) else {"feed_id":None}
        final_score = 0.0
        for feat, w in weights.items():
            # print(f"feat -> {feat:15s} | w -> {w}")
            if float(w) == 0.0:
                continue
            v = _get_feature_value(c,feat,missing)
            # print(f"v -> {v}")
            # ----------
            # candidate -> {'feed_id': 'TH_BIO_059', 'vector_score': 0.4929769831091946, 'subscores': {'language_match': 1.0, 'recency': 0.22812574777636782, 'popularity': 0.694063716589213}}
            # feat -> vector_score    | w -> 0.6
            # v -> 0.4929769831091946
            # feat -> language_match  | w -> 0.3
            # v -> 1.0
            # feat -> recency         | w -> 0.05
            # v -> 0.22812574777636782
            # feat -> popularity      | w -> 0.05
            # v -> 0.694063716589213
            # ----------
            if clamp_inputs:
                v = _clamp01(float(v))
            final_score += float(w) * float(v)        # Main of everytning
            # print(f"final_score : {final_score}")
        row["final_score"] = float(final_score)
        out.append(row)
        # print("-"*10)
    # Deterministic sorting
    def _sort_key(c:Dict[str,Any]):
        keys: List[float] = [float(c.get("final_score",0.0))]
        for tb in tie_breakers:
            v = _get_feature_value(c,tb,missing)
            if clamp_inputs:
                v = _clamp01(float(v))
            keys.append(float(v))
        return tuple(keys)
    out.sort(key = _sort_key, reverse=True)
    return out
########################################################################


########################################################################
# index_store.py
########################################################################
### def _ensure_popularity -> | functions.core.index_store.py (line:74)
def _coerce_int(x:Any, default:int = 0) -> int:
    """
    Best-effort case to int.
    """
    try:
        if x is None:
            return int(default)
        return int(float(x))
    except Exception:
        return int(default)
    
### def FeedIndexStore_customs -> | functions.core.index_store.py (line:93)
def _ensure_popularity(row:Dict[str,Any]) -> Dict[str,Any]:
    """
    Ensure output meta has a canonical integer `popularity` field.
    """
    out = dict(row)
    if "popularity" in out:
        out["popularity"] = _coerce_int(out.get("popularity"), default=0)
        return out
    if "views" in out:
        out["popularity"] = _coerce_int(out.get("views"), default=0)
        return out
    out["popularity"] = 0
    return out

### def main -> | functions.core.index_store.py (line:121)
def _validate_query_weights(nq:int, query_weights: Optional[np.ndarray]) -> np.ndarray:
    """Validate and sanitize query weights"""
    if query_weights is None:
        return np.ones((nq,), dtype=np.float32)       # array([1., 1., 1., 1., 1.], dtype=float32)
    w = np.asarray(query_weights, dtype=np.float32).reshape(-1)
    if w.shape[0] != nq:
        raise ValueError(f"query_weights must have length {nq}, got {w.shape[0]}")
    # Deterministic sanitization
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.maximum(w, 0.0)
    return w.astype(np.float32)

### def main -> | functions.core.index_store.py / class FeedIndexStore: / def load (line:159)
def FeedIndexStore_customs():
    # TODO(Tun,260219):connect with the feed database to get metadata that use in process
    meta: List[Dict[str,Any]] = []
    meta_path = r"data/feeds_meta.jsonl"
    with open(meta_path,"r",encoding="utf-8") as f:
        for line_no,line in enumerate(f,start=1):
            line = line.strip()
            if not line:
                continue
            try:
                meta.append(_ensure_popularity(json.loads(line)))
            except Exception as e:
                raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e
    return meta

### def main -> | functions.core.index_store.py (line:265)
def get_feed_id(_meta, internal_idx:int) -> Optional[str]:
    """Convert a vector index internal id into a feed_id using aligned metadata"""
    i = int(internal_idx)
    if i < 0 or i >= len(_meta):
        return None
    row = _meta[i]          # find position of feeds
    fid = row.get("feed_id")    # Get feed_id
    return str(fid) if fid is not None else None
########################################################################


########################################################################
# functions.core.retrieval.py
########################################################################
# ---------------------------
# Structural validation
# ---------------------------
def _validate_structure(feed, scores, hyde_query):
    if len(feed) != len(scores):
        raise ValueError(
            f"Query dimension mismatch: len(feed)={len(feed)} vs len(scores)={len(scores)}"
        )

    if len(feed) != len(hyde_query):
        raise ValueError(
            f"Query dimension mismatch: len(feed)={len(feed)} vs len(hyde_query)={len(hyde_query)}"
        )

    for qi in range(len(feed)):
        if len(feed[qi]) != len(scores[qi]):
            raise ValueError(
                f"Candidate mismatch at qi={qi}: len(feed[qi])={len(feed[qi])} vs len(scores[qi])={len(scores[qi])}"
            )
        
AggMode = Literal["WEIGHTED_MAX","WEIGHTED_MEAN"]           
### def main <- | functions.core.retrieval.py (line:186) *customs
def retrieve_by_hyde_queries_weighted(
        *,
        # index_store: FeedIndexStoreLike,
        # hyde_query_embeddings:np.ndarray,
        query_weights:Optional[np.ndarray],
        top_k: int = 30,
        max_candidate: int = 100,
        agg_mode: AggMode = "WEIGHTED_MAX",
        return_debug: bool = False,
        scores,
        feed,
        hyde_query
) -> Tuple[List[Tuple[str, float]], Optional[Dict[str, RetrievalDebug]]]:
    # print(f"query_weights -> {query_weights}")
    # print(f"top_k         -> {top_k}")
    # print(f"agg_mode      -> {agg_mode}")
    # print(f"return_debug  -> {return_debug}")
    # print(f"score         -> {score}")
    # print(f"feed          -> {feed}")
    # print(f"hyde_query    -> {len(hyde_query)}")

    nq = len(hyde_query)
    ### --------- Validate / sanitize weights --------- ###
    w = _validate_query_weights(nq, query_weights)

    ### --------- Vector search (backent-agnostic) (customs) --------- ###
    # with open(r"data/shin_retrieval_output.json", "r") as f:
    #     data = json.load(f)
    # scores  = np.array(data["scores"] , dtype=np.float32)
    _validate_structure(feed, scores, hyde_query)
    # indices = np.array(data["indices"], dtype=np.int64)
    # print(f"scores  -> {scores}")
    # print(f"indices -> {indices}")
    # >>> indices
    #     array([[30, 32, 41, 38, 34, 52, 46, 50, 35, 39, 57, 47, 37, 31, 55, 58,
    #             44, 54, 36, 43, 33, 40, 45, 56, 17, 42, 51, 53, 15, 49],
    #         [32, 37, 30, 52, 34, 58, 43, 35, 56, 39, 47, 38, 54, 41, 33, 36,
    #             45, 50, 46, 51, 40, 53, 26, 44, 48, 31, 15, 17, 57, 19],
    #         [30, 52, 35, 38, 41, 32, 34, 46, 50, 37, 39, 45, 31, 40, 33, 43,
    #             47, 36, 58, 53, 44, 51, 57, 54, 48, 56, 55, 15, 13, 17],
    #         [34, 46, 30, 35, 32, 31, 41, 52, 38, 39, 54, 58, 57, 43, 45, 36,
    #             47, 50, 33, 37, 15, 56, 53, 44, 40, 42, 51, 19, 14, 18],
    #         [34, 41, 32, 35, 39, 30, 52, 43, 15, 10, 14, 33, 53, 38, 36, 57,
    #             31, 45, 18,  6, 54, 11, 58, 37, 56, 47, 46, 19, 44,  1]])
    # >>> scores
    #     array([[0.99, 0.91, 0.88, 0.88, 0.88, 0.87, 0.86, 0.85, 0.85, 0.84, 0.84,
    #             0.84, 0.84, 0.83, 0.83, 0.83, 0.83, 0.83, 0.82, 0.81, 0.81, 0.81,
    #             0.81, 0.81, 0.8 , 0.8 , 0.79, 0.79, 0.79, 0.78],
    #         [0.97, 0.91, 0.91, 0.9 , 0.87, 0.86, 0.86, 0.85, 0.85, 0.85, 0.85,
    #             0.85, 0.85, 0.84, 0.84, 0.83, 0.82, 0.8 , 0.79, 0.79, 0.79, 0.78,
    #             0.78, 0.78, 0.78, 0.78, 0.77, 0.76, 0.76, 0.76],
    #         [0.91, 0.91, 0.91, 0.9 , 0.9 , 0.9 , 0.89, 0.87, 0.86, 0.86, 0.85,
    #             0.85, 0.85, 0.85, 0.85, 0.84, 0.84, 0.83, 0.83, 0.83, 0.82, 0.82,
    #             0.81, 0.81, 0.81, 0.8 , 0.8 , 0.8 , 0.79, 0.79],
    #         [0.88, 0.88, 0.86, 0.86, 0.86, 0.86, 0.86, 0.85, 0.83, 0.83, 0.82,
    #             0.82, 0.82, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.8 , 0.8 , 0.79,
    #             0.79, 0.79, 0.79, 0.78, 0.77, 0.77, 0.77, 0.77],
    #         [0.94, 0.89, 0.88, 0.87, 0.87, 0.87, 0.86, 0.86, 0.85, 0.85, 0.84,
    #             0.84, 0.83, 0.83, 0.83, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82,
    #             0.81, 0.81, 0.81, 0.81, 0.8 , 0.8 , 0.8 , 0.8 ]], dtype=float32)
    # _meta
    # [
    #     {
    #         "feed_id": "TH_UNI_041",
    #         "title": "ปฏิทินการรับเข้าศึกษาใหม่ของมหาวิทยาลัยไทย (อัปเดตล่าสุด)",
    #         "feed_text": "สรุปช่วงเวลารับสมัครนักศึกษาใหม่ของมหาวิทยาลัยในไทย แยกรอบรับสมัครและรายการเอกสารที่ควรเตรียมให้ครบ",
    #         "tags": [
    #             "university",
    #             "admission",
    #             "education"
    #         ],
    #         "language": "th",
    #         "created_at": "2025-01-06T07:30:00Z",
    #         "source": "University News",
    #         "url": "https://example.com/th/university-admission-schedule",
    #         "views": 68400,
    #         "embedding_input": "title: ปฏิทินการรับเข้าศึกษาใหม่ของมหาวิทยาลัยไทย (อัปเดตล่าสุด)\nfeed_text: สรุปช่วงเวลารับสมัครนักศึกษาใหม                ม่ของมหาวิทยาลัยในไทย แยกรอบรับสมัครและรายการเอกสารที่ควรเตรียมให้ครบ\ntags: university, admission, education",
    #         "popularity": 68400
    #     },...
    # ]
    ### --------- Aggregation state --------- ###
    agg_scores   : Dict[str, float] = {}
    best_qi_map  : Dict[str, int]   = {}
    best_raw_map : Dict[str, float] = {}
    best_w_map   : Dict[str, float] = {}

    # For WEIGHTED_MEAN
    sum_w_map    : Dict[str, float] = {}
    sum_ws_map   : Dict[str, float] = {}

    ### def main -> | functions.core.index_store.py / class FeedIndexStore: / def load (line:159)
    # _meta = FeedIndexStore_customs()   # just load all of feed index
    # print(f"_meta -> {prettyjson(_meta)}")
    # assume len(feed) == len(scores) == len(hyde_query)

    assert len(w) == len(feed)
    for qi in range(len(feed)):
        wi = float(w[qi])
        # print(f"qi:{qi} -> wi:{wi}")
        limit = min(top_k, len(feed[qi]))
        for ki in range(limit):
            # print(f"- ki : {ki}")
            # idx = int(indices[qi,ki])     # indices[0,1] -> np.int64(32), indices[0,2] -> np.int64(41), ..., indices[0,29] -> np.int64(49)
            raw     = float(scores[qi][ki])   # float(scores[0,1]) -> 0.90618, float(scores[0,2]) -> 0.88088
            feed_id = feed[qi][ki]           # feed[0][0] -> TH_F001, feed[0][1] -> EN_F028
            # print(f" - feed_id : {feed_id} -> score : {raw}")
            if feed_id is None:
                continue

            if agg_mode == "WEIGHTED_MAX":
                # print("Weight max")
                cand = wi * raw   # weight x raw_score
                prev = agg_scores.get(feed_id)
                # print(f"  - cand -> {cand}")
                # print(f"  - prev -> {prev}")
                if prev is None or cand > prev:     # KEEP THE MAX of cand that we can calculate
                    agg_scores[feed_id]   = cand    # cand = weight x raw_score {'TH_F001': 0.99, 'TH_F003': 0.96,...,'TH_UNI_042': 0.47}
                    best_qi_map[feed_id]  = qi      # Order of Q {'TH_F001': 0, 'TH_F003': 1, 'TH_F012': 2,
                    best_raw_map[feed_id] = raw     # Raw score
                    best_w_map[feed_id]   = wi      # Weight of HyDE

            elif agg_mode == "WEIGHTED_MEAN":
                # print("Weight mean")
                if wi <= 0.0:
                    continue
                sum_ws_map[feed_id] = sum_ws_map.get(feed_id,0.0) + wi * raw
                sum_w_map[feed_id]  = sum_w_map.get(feed_id,0.0)  + wi
                # Track best raw score for explainability
                prev_best = best_raw_map.get(feed_id)
                if prev_best is None or raw > prev_best:
                    best_qi_map[feed_id]  = qi
                    best_raw_map[feed_id] = raw
                    best_w_map[feed_id]   = wi
            else:
                raise ValueError(f"Unknow agg_mode: {agg_mode}")
        # print("-"*50)
    # Finalize WEIGHTED_MEAN scores
    if agg_mode == "WEIGHTED_MEAN":
        for fid, ws in sum_ws_map.items():
            sw = sum_w_map.get(fid, 0.0)
            if sw > 0.0:
                agg_scores[fid] = float(ws/sw)
    # print(f"agg_scores  -> \n{agg_scores}")
    # print(f"best_qi_map -> \n{best_qi_map}")
    # print(f"best_raw_map-> \n{best_raw_map}")
    # print(f"best_w_map  -> \n{best_w_map}")
    # print(f"sum_w_map   -> \n{sum_w_map}")
    # print(f"sum_ws_map  -> \n{sum_ws_map}")

    # ----------------------------------------------------------------------
    # Deterministic sorting ***
    # ----------------------------------------------------------------------
    # Tie-breakers:
    # 1) aggregated score (desc)
    # 2) best query index (asc)
    # 3) feed_id (lex asc)
    def _sort_key(item: Tuple[str,float]) -> Tuple[float, int, str]:
        # print(f"items -> {item}") # ('TH_F001', 0.9919580221176147)
        fid, sc = item
        qi = best_qi_map.get(fid,10**9)
        return (-float(sc), int(qi), str(fid))
    candidates = sorted(agg_scores.items(), key=_sort_key)
    # Final candidate cap
    if max_candidate is not None and max_candidate > 0:
        candidates = candidates[:max_candidate]
    # print(f"candidates -> \n{candidates}")
    # ----------------------------------------------------------------------
    # Optional debug output
    # ----------------------------------------------------------------------     
    debug: Optional[Dict[str, RetrievalDebug]] = None
    if return_debug:
        debug = {
            fid: RetrievalDebug(
                best_qi=int(best_qi_map.get(fid, -1)),
                best_raw_score=float(best_raw_map.get(fid, 0.0)),
                best_weight=float(best_w_map.get(fid, 0.0)),
                aggregated_score=float(sc),
            )
            for fid, sc in candidates
        }
    # print(f"candidate -> \n{len(candidates)}")
    return candidates, debug
########################################################################


########################################################################
# interactions_store.py
########################################################################
### def _read_interactions_table -> | functions.core.interactions_store.py (line:26)
def _strip_cols(d:pd.DataFrame) -> pd.DataFrame:
    d.columns = [str(c).strip() for c in d.columns]
    return d

### def load_user_interactions -> | functions.core.interactions_store.py (line:10)
def _read_interactions_table(p:Path) -> pd.DataFrame:
    df = pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig")
    df = _strip_cols(df)
    if len(df.columns) == 1:
        # common case: TSV but sniff failed
        df2 = pd.read_csv(p, sep="\t", engine="python", encoding="utf-8-sig")
        df2 = _strip_cols(df2)
        if len(df2.columns) > 1:
            return df2
        # fallback: comma
        df3 = pd.read_csv(p, sep=",", engine="python", encoding="utf-8-sig")
        df3 = _strip_cols(df3)
        return df3
    return df

### def _get_seen_feed_ids_from_params -> | functions.core.interactions_store.py (line:47)
def load_user_interactions(interactions_path:str,*,student_id:str) -> pd.DataFrame:
    p = Path(interactions_path)
    if not p.exists():
        raise FileNotFoundError(f"interactions_path not found: {p}")
    df = _read_interactions_table(p)
    # normalize column names (strip already handled now lower for safety)
    df.columns = [c.lower().strip() for c in df.columns]
    id_col: Optional[str] = None
    if "student_id" in df.columns:
        id_col = "student_id"
    elif "user_id" in df.columns:
        id_col = "user_id"

    if id_col is None:
        raise ValueError("interactions.csv must contain 'student_id' or 'user_id' column")

    sid = str(student_id).strip()
    out = df[df[id_col].astype(str).str.strip() == sid].copy()
    return out
########################################################################





########################################################################
# functions.core.feeds_meta_store.py 
########################################################################
### def main -> | functions.core.feeds_meta_store.py (line:264)
# _FEED_META_CACHE : Dict[str,Dict[str,Dict[str,Any]]] = {}
# def load_feeds_meta_map(feed_index_dir:str) -> Dict[str,Dict[str,Any]]:
#     # cache_key = str(Path(feed_index_dir).resolve())
#     # cached    = _FEED_META_CACHE.get(cache_key)
#     # if cached is not None:
#     #     return cached
#     meta_path = Path("data/feeds_meta.jsonl")
    # print(f"meta_path -> {meta_path}")
#     out : Dict[str,Dict[str,Any]] = {}
#     # TODO
#     if meta_path.exists():
#         with meta_path.open("r",encoding="utf-8") as f:
#             for line_no, line in enumerate(f,start=1):
                # print(f"line_no -> {line_no}, line -> {line}")
#                 line = line.strip()
#                 if not line:
#                     continue
#                 # Best-effort parse JSON; skip invalid lines rather than failing serving
#                 try:
#                     obj = json.loads(line)
#                 except Exception:
#                     continue
#                 # Ensure the parsed JSON is a dict-like object
#                 if not isinstance(obj,dict):
#                     continue
#                 fid = obj.get("feed_id") or obj.get("id")
#                 if isinstance(fid,str) and fid.strip():
#                     out[fid.strip()] = obj
#     # _FEED_META_CACHE[cache_key] = out
#     return out



def load_feeds_meta_map(feed_ids: List[str],url_path:str) -> Dict[str, Dict[str, Any]]:
    url = url_path

    feed_ids = [f"feeds:{feed_id}" for feed_id in feed_ids]

    payload = {"ids": feed_ids}

    # print(payload)

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", {})

        # convert keys back: "feeds:EN_F028" -> "EN_F028"
        normalized_items = {}
        for k, v in items.items():
            raw_id = k.replace("feeds:", "", 1)
            normalized_items[raw_id] = v

        return normalized_items

    except Exception as e:
        # print(f"[load_feeds_meta_map] API error: {e}")
        return {}

########################################################################


# ########################################################################
# functions.core.history.py 
########################################################################
### def main -> | functions.core.history.py (line:371)
### --------- Public API : seen-feed extraction for exclude-seen policy --------- ###
def extract_seen_feed_ids(
    user_events: pd.DataFrame,
    event_types: Optional[Iterable[str]] = None,
    now_utc: Optional[datetime] = None,
    window_days: Optional[int] = None,
    max_unique: Optional[int] = None,
) -> Set[str]:
    if user_events is None or len(user_events) == 0:
        return set()
    if "feed_id" not in user_events.columns:
        return set()
    
    df = user_events.copy()
    df["feed_id"] = df["feed_id"].astype(str).str.strip()
    df = df[df["feed_id"].astype(bool)] # turn feed_id to bool (True)
    
    # Filter event types if possible.
    if event_types is not None and "event_type" in df.columns:
        # print('if1')
        allow = {str(x).lower().strip() for x in event_types if str(x).strip()}   # turn interaction from parameter to set
        df["event_type"] = df["event_type"].astype(str).str.lower().str.strip()
        df = df[df["event_type"].isin(allow)]   # crop only event in allow (event in the list that we want)
    ts_col = "ts" if "ts" in df.columns else None
    # Optional windowing by time (only if ts exists)
    if window_days is not None and int(window_days) > 0 and ts_col is not None:
        # print('if2')
        # TODO(Tun) : check time zone
        now_utc = now_utc or datetime.now(timezone.utc)
        ts = pd.to_datetime(df[ts_col],errors="coerce", utc=True)
        df = df.assign(_ts=ts).dropna(subset=["_ts"]) # select only time 16  stu_p003  TH_F006  2026-01-06T07:20:15Z       view     22000 2026-01-06 07:20:15+00:00
        # print(f"df -> \n{df}")
        # TODO(Tun) : do something with windown day
        if not df.empty:
            # print('if2.1')
            cutoff = now_utc - pd.Timedelta(days=int(window_days))
            df = df[df["_ts"] >= cutoff]
        # print(f"df -> \n{df}")
    # Optional deterministic cap by most-recent unique feeds.
    if max_unique is not None and int(max_unique) > 0 and ts_col is not None:
        # print('if3')
        ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)  # crop only ts_col then turn to datetime style
        df = df.assign(_ts=ts).dropna(subset=["_ts"]).sort_values("_ts", ascending=False)  # sort event by time
        # print(f"df -> \n{df}")
        seen: Set[str] = set()
        for fid in df["feed_id"].tolist():
            if fid in seen:
                continue
            seen.add(fid)
            if len(seen) >= int(max_unique):
                break
        return seen

    return set(df["feed_id"].tolist())
########################################################################

########################################################################
# functions.online.subscore.py
########################################################################
def score_language_match(
        feed_meta:Dict[str,Any],
        *,
        user_lang:Optional[str]
)->float:
    if not user_lang:
        return 0.0
    feed_lang = feed_meta.get("lang") or feed_meta.get("language")
    if isinstance(feed_lang,str) and feed_lang.strip().lower() == user_lang.strip().lower():
        return 1.0
    return 0.0

def _parse_ts_any(v:Any) -> Optional[datetime]:
    if isinstance(v,str):
        s = v.strip()
        if not s:
            return None
        try:
            # Normalize trailing Z
            if s.endswith("Z"):
                s = s[:-1]+"+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except:
            return None
    return None

def score_recency(
        feed_meta: Dict[str,Any],
        *,
        now_utc: datetime,
        half_life_days: float = 30.0
) -> float:
    for key in ("published_at","created_at","timestamp","date","ts"):
        dt = _parse_ts_any(feed_meta.get(key))
        if dt is None:
            continue
        age_days = (now_utc-dt).total_seconds()/86400.0
        if age_days < 0:
            age_days = 0.0
        if half_life_days <= 0:
            return 0.0
        return float(2.0 ** (-(age_days/half_life_days)))
    return 0.0

def score_popularity(feed_meta:Dict[str,Any]) -> float:
    for key in ("popularity", "views", "likes", "clicks", "impressions"):
        v = feed_meta.get(key)
        if isinstance(v,(int,float)) and v > 0:
            return float(
                min(
                    1.0,
                    np.log1p(float(v)) / np.log1p(1_000_000.),
                )
            )
    return 0.0
########################################################################



########################################################################
# functions.online.pipeline_3_online_retrieval.py (Helper function)
########################################################################
### def main() -> |pipeline_3_online_retrieval.py (line:102)
def _load_params_yaml(path: Path) -> Dict[str, Any]:
    """Load parameter from yaml file."""
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    return obj if isinstance(obj, dict) else {}

### def main() <- | pipeline_3_online_retrieval.py (line:125)
def _get_nested(d: Dict[str,Any], keys: Sequence[str], default:Any)->Any:
    '''Safe nested dict getter'''
    cur: Any = d
    for k in keys:
        if not isinstance(cur,dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

### def main() <- | pipeline_3_online_retrieval.py (line:140)
def _coalesce(*vals:Any) -> Any:
    for v in vals:
        if v is not None:
            return v
    return None

### def main () -> | pipeline_3_online_retrieval.py (line:151)
def _extract_feed_header(fmeta: Dict[str, Any], header_keys: List[str]) -> Optional[str]:
    if not isinstance(fmeta, dict):
        return None
    for k in header_keys:
        v = fmeta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

### def main <- | pipeline_3_online_retrieval.py (line:375)
def _ms(dt: float) -> float:
    """Convert perf_counter delta seconds -> milliseconds."""
    return float(dt*1000.0)

### def main -> | functions.core.pipeline_3_online_retrieval.py (line:264)
def _get_seen_feed_ids_from_params(
        *,
        params: Dict[str,Any],
        student_id : str,
        now_utc : datetime,
        metadata
) -> Tuple[bool, Set[str], Dict[str,Any]]:    # line : 264
    cfg     = _get_nested(params,["retrieval","exclude_seen"],{}) or {}
    enabled = bool(cfg.get("enabled",False))
    meta = {
        "exclude_seen_enabled": enabled,
        "exclude_seen_count":0
    }
    if not enabled:
        return False, set(), meta
    event_types  = cfg.get("event_types",["view","click","like","share","comment"])
    window_days  = cfg.get("window_days",30)
    max_unique   = cfg.get("max_unique",5000)

    # interactions_path = str(cfg.get("interactions_path","data/interactions.csv"))

    # # Load interactions filtered to this students.
    # df = load_user_interactions(
    #     interactions_path,
    #     student_id = student_id
    # )
    interactions = metadata.get("interaction",[])
    if not interactions:
        return True, set(), meta
    
    # turn metadata["interactive"] json -> pandas dataframe with work format
    df = pd.DataFrame(interactions)
    df.columns = [c.lower().strip() for c in df.columns]
    id_col = "student_id" if "student_id" in df.columns else "user_id"
    df = df[df[id_col].astype(str).str.strip() == str(student_id)]
    df = df.drop_duplicates()
    if "dwell_ms" in df.columns:
        df["dwell_ms"] = df["dwell_ms"].astype(int)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # print(f"interactions_path -> {interactions_path}")
    # print(f"df                -> \n{df}")
    # print(f"event_types       -> {event_types}")
    # print(f"window_days       -> {window_days}")
    # print(f"max_unique        -> {max_unique}")
    seen = extract_seen_feed_ids(
        df,
        event_types = event_types,
        now_utc     = now_utc,
        window_days = int(window_days) if window_days is not None else None,
        max_unique  = int(max_unique) if max_unique is not None else None,
    )
    # print(f"seen -> \n{seen}")
    meta.update(
        {
            "exclude_seen_enabled":True,
            "exclude_seen_count":int(len(seen)),
            # "exclude_seen_interactions_path": interactions_path,
            "exclude_seen_event_types":[str(x) for x in (event_types or [])],
            "exclude_seen_window_days": int(window_days) if window_days is not None else None,
            "exclude_seen_max_unique":int(max_unique) if max_unique is not None else None,
        }
    )
    # print(f"meta -> \n{prettyjson(meta)}")
    return True, seen, meta

def to_rerank_items(candidates: List[Dict[str, Any]]) -> List[RerankItem]:
    reranked: List[RerankItem] = []

    for c in candidates:
        if not isinstance(c, dict):
            continue

        # ---- feed_id ----
        fid = c.get("feed_id")
        if not isinstance(fid, str):
            continue

        # ---- final_score (robust key lookup) ----
        final_score = None
        for k, v in c.items():
            if "final_score" in k.replace(" ", "").lower():
                try:
                    final_score = float(v)
                    break
                except Exception:
                    pass

        if final_score is None:
            continue

        reranked.append(
            RerankItem(
                feed_id=fid,
                final_score=final_score
            )
        )

    return reranked

# ----------------------------------------------------------------------
# Initial value
# ----------------------------------------------------------------------
# Get directory of this file
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PARAMS_PATH = BASE_DIR / "parameters" / "parameters.yaml"
DEFAULT_SCORE_WEIGHTS_PATH = BASE_DIR / "parameters" / "retrieval_score_weights.yaml"

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
# hyde_query, metadata = Shin to pull from hyde_embedding.py
def calc_subscore(
    *,
    student_id : str,
    score : list[list[float]]              = None,
    feed  : list[list[str]],
    hyde_query : dict                      = None,
    metadata : dict,
    now_utc:Optional[datetime]             = None,
    include_feed_header:Optional[bool]     = None,
    max_candidates:Optional[int]           = None,
    agg_mode:Optional[str]                 = None,
    recency_half_life_days:Optional[float] = None,
    score_weights_path:Optional[str]       = None,
    ### add
    user_lang                              = 'th',
    return_debug                           = True 
):
    # ----------------------------------------------------------------------
    # Main
    # pipeline_3_online_retrieval.py | def run_online_retrieval()
    # ----------------------------------------------------------------------
    #########################################################
    ### ------------- Part 1 : Prepare data ------------- ###
    #########################################################
    t0_total = time.perf_counter()
    timing_ms : Dict[str,float] = {}
    ### --------- Load params (best-efford) --------- ###
    t0 = time.perf_counter()
    params = _load_params_yaml(DEFAULT_PARAMS_PATH)
    timing_ms["load_params_ms"] = _ms(time.perf_counter() - t0)
    # print(f"params -> {prettyjson(params)}")

    ### --------- Error Protection --------- ###
    num_queries       = _get_nested(params, ["eprotection", "num_queries"],5)
    # ---- duplicate feed if needed ----
    if len(feed) != num_queries:
        feed = [feed[0].copy() for _ in range(num_queries)]
    if score is None:
        candidate_queries = _get_nested(params, ["eprotection", "candidate_queries"],50)
        # print(f"if score is None | num_queries -> {num_queries} | candidate_queries -> {candidate_queries}")
        score = np.zeros((num_queries, candidate_queries), dtype=np.float32)
    if hyde_query is None:
        weights = _get_nested(params, ["eprotection", "fallback_hq_weights"], None)
        intent  = _get_nested(params, ["eprotection", "fallback_intent_label"], "fallback")
        qtext   = _get_nested(params, ["eprotection", "fallback_query_text"], "")
        if not isinstance(weights, list) or len(weights) == 0:
            weights = [1.0] * num_queries
        if len(weights) < num_queries:
            weights = weights + [weights[-1]] * (num_queries - len(weights))
        hyde_query = [
            {
                "query_id": f"Q{i+1}",
                "query_text": qtext,
                "weight": float(weights[i]),
                "intent_label": str(intent),
            }
            for i in range(num_queries)
        ]
    # print(prettyjson(hyde_query))

    ### --------- YAML toggles (CLI can overide) --------- ###
    yaml_include_header  = bool(_get_nested(params, ["retrieval", "include_feed_header"], False))    # line:695
    # # yaml_print_header    = bool(_get_nested(params, ["retrieval", "print_feed_header"], False))
    include_header_final = True    # line:699
    # print_header_final   = True
    ### --------- Retrieval defaults --------- ###
    now_utc         = now_utc or datetime.now(timezone.utc)    # line:156cls
    # Feed header config
    include_feed_header = bool(
        _coalesce(include_feed_header, _get_nested(params, ["retrieval", "include_feed_header"], False))
    )
    header_keys = _get_nested(params, ["retrieval", "feed_header_keys"], ["title", "header", "name", "feed_title"])

    top_k_per_query = int(_coalesce(_get_nested(d  = params, keys = ["retrieval","top_k_per_query"], default = 50)))    # line:398
    max_candidates  = int(_coalesce(max_candidates, _get_nested(params, ["retrieval", "max_candidates"], 200)))
    agg_mode        = str(_coalesce(agg_mode, _get_nested(params, ["retrieval", "agg_mode"], "WEIGHTED_MAX")))

    # print(now_utc)
    # print(include_feed_header)
    # print(header_keys)
    # print(top_k_per_query)
    # print(max_candidates)
    # print(agg_mode)

    ### --------- Query weights/labels aligned with embedding rows (customs) --------- ###
    t0 = time.perf_counter()    # line:451
    weights, qids, intents = extract_query_weights_and_labels(hyde_query)
    timing_ms["extract_query_weights_ms"] = _ms(time.perf_counter() - t0)
    # print(weights,qids,intents)

    ### --------- Vector retrieval (multi-query aggregation) --------- ###
    t0 = time.perf_counter()    #line:478
    mode = "WEIGHTED_MEAN" if str(agg_mode).upper() == "WEIGHTED_MEAN" else "WEIGHTED_MAX" 
    scored, debug_map = retrieve_by_hyde_queries_weighted(
        query_weights = weights,
        top_k         = int(top_k_per_query),
        agg_mode      = mode,
        return_debug  = True,
        # add 
        scores        = score,
        feed          = feed,
        hyde_query    = hyde_query,
        max_candidate = max_candidates
    )
    n = 0
    for i, j in scored:
        # print(f"{n:2d} Feed_index : {i:10s} sorted_score : {j}")
        n = n+1
    timing_ms["vector_retrieval_ms"] = _ms(time.perf_counter() - t0)
    ### --------- Exclude seen feeds BEFORE max_candidates cap --------- ###
    t0 = time.perf_counter()
    exclude_enabled, seen_set, seen_meta = _get_seen_feed_ids_from_params(
        params = params,
        student_id = student_id,
        now_utc = now_utc,
        metadata = metadata
    )


    pre_filter_len = len(scored) # -> 40
    if exclude_enabled and seen_set:
            scored = [(fid, sc) for (fid, sc) in scored if fid not in seen_set]
    post_filter_len = len(scored) # -. 35
    # print(f"pre_filter_len -> {pre_filter_len} | post_filter_len -> {post_filter_len}")
    # print(f"scored -> {scored}")
    never_seen = [i for i,_ in scored]
    # print(f"exclude_enabled ->\n{exclude_enabled}")
    # print(f"seen_meta       ->\n{seen_meta}")
    # print(f"seen_set ({len(seen_set)})    -> {seen_set}")
    # print(f"never seen ({len(never_seen)})  -> {never_seen}")

    # Cap after filtering to preserve return up to N unseen candidates
    scored = scored[: int(max(0,max_candidates))]    # select only maximum candidate feed that possible

    n = 0
    for i, j in scored:
        # print(f"{n:2d} Feed_index : {i:10s} sorted_score : {j}")
        n = n+1

    timing_ms["exclude_seen_ms"] = _ms(time.perf_counter()-t0)

    ### --------- Deterministic subscore --------- ###
    t0 = time.perf_counter()
    load_feeds_meta_map_path = str(_coalesce(_get_nested(d  = params, keys = ["retrieval","load_feeds_meta_map_path"], default = "https://hyde-cache-pipeline-api-810737581373.asia-southeast1.run.app/cache/get-many")))    # line:398
    feeds_meta_map = load_feeds_meta_map(
        feed_ids = never_seen,
        url_path = load_feeds_meta_map_path
        )
    # print(f"feeds_meta_map : \n{prettyjson(feeds_meta_map)}")


    #########################################################
    ### --------- Part 2 : Subscore calculation --------- ###
    ### if want to modify some subscore go to line:734    ###
    #########################################################
    # print(f"user_lang -> {user_lang}")

    recency_half_life_days  = int(_coalesce(recency_half_life_days, _get_nested(params, ["retrieval", "recency_half_life_days"], 200)))
    t0 = time.perf_counter()
    candidates: List[Dict[str,Any]] = []
    for fid,score in scored:
        # print(f"fid -> {fid:10s} | score -> {score}")
        # Feed metadata lookup by feed_id (independent of vector backend):
        fmeta = feeds_meta_map.get(fid,{}) if isinstance(feeds_meta_map,dict) else {}
        # print(f"fmeta -> {prettyjson(fmeta)}")

        subscores = {
            # 1
            "language_match": score_language_match(
                fmeta,
                user_lang = user_lang
            ),
            # 2
            "recency" : score_recency(
                fmeta,
                now_utc = now_utc,
                half_life_days = float(recency_half_life_days)
            ),
            # 3
            "popularity": score_popularity(fmeta)
        }
        # print(f"subscore -> \n{subscores}")
        # Debug : indicate which HyDE query contributed most (if available).
        dbg_obj = None
        if debug_map is not None:
            # print("debug_map mode")
            dbg = debug_map.get(fid)
            # print(f"dbg -> {dbg}")
            if dbg is not None:
                qi = int(dbg.best_qi)
                obg_obj = {
                    "best_query_id": qids[qi] if 0 <= qi <= len(qids) else None,
                    "best_query_intent": intents[qi] if 0 <= qi < len(intents) else None,
                    "best_raw_score": float(dbg.best_raw_score),
                    "best_weight": float(dbg.best_weight)
                }
                # print(f"obg_obj -> {prettyjson(obg_obj)}")
        row: Dict[str,Any] = {
            "feed_id":fid,
            "vector_score":float(score),
            "subscores":subscores
        }
        # print(f"row -> \n{row}")
        if include_feed_header:    
            # print(f"header_keys -> {header_keys}")  # ['title', 'header', 'name', 'feed_title']
            header = _extract_feed_header(fmeta if isinstance(fmeta, dict) else {}, header_keys)
            # print(header)
            row["feed_header"] = header
        if dbg_obj is not None:
            row["debug"] = dbg_obj
        candidates.append(row)
    timing_ms["build_candidates_ms"] = _ms(time.perf_counter() - t0)
    # print(f"candidates -> \n {prettyjson(candidates)}")

    # #########################################################
    # ### --------- Part 3 : Subscore aggregator --------- ###
    # #########################################################
    t0 = time.perf_counter()
    score_cfg:Dict[str,Any] = {}
    score_enabled = False
    score_weights_path = str(_coalesce(score_weights_path,_get_nested(params, ["retrieval", "score_weights_path"], DEFAULT_SCORE_WEIGHTS_PATH),))
    # print(Path(score_weights_path).exists())
    try:
        if score_weights_path and Path(score_weights_path).exists():
            score_cfg = load_score_aggregation_config(score_weights_path)
            # print(f"score_cfg -> \n{prettyjson(score_cfg)}")
            sa = score_cfg.get("score_aggregation") if isinstance(score_cfg,dict) else {}
            sa = sa if isinstance(sa,dict) else {}
            score_enabled = bool(sa.get("enabled",False))
    except Exception:
        # print("It's fail ")
        score_cfg = {}
        score_enabled = False

    # Downstream consumers should rely on this key when displaying or evaluating ordering:
    ranking_score_key = "vector_score"
    if score_enabled:
        # aggregate_candidates:
        # - computes final_score per candidate
        # - sorts candidates deterministically using tie breaker
        candidates = aggregate_candidates(candidates,score_cfg)
        # print(f"candidates -> \n{prettyjson(candidates)}")
        ranking_score_key = "final_score"
        # Backward-compat cleanup if any older naming leaked into results.
        for c in candidates:
            if "final_agg_score" in c:
                c.pop("final_agg_score",None)
        
    timing_ms["score_aggregation_ms"] = _ms(time.perf_counter() - t0)
    # Total wall time
    timing_ms["total_ms"] = _ms(time.perf_counter() - t0_total)

    # #########################################################
    # ### --------- Part 4 : Meta payload (for debugging + gold validation) --------- ###
    # #########################################################
    sa_out: Dict[str,Any] = {}
    if isinstance(score_cfg,dict):
        sa_raw = score_cfg.get("score_aggregation")
        if isinstance(sa_raw,dict):
            sa_out = sa_raw

    meta = {
        # retrieval config
        "top_k_per_query": int(top_k_per_query),
        "max_candidates": int(max_candidates),
        # "num_queries": int(hyde_q_emb.shape[0]),
        # "dim": int(hyde_q_emb.shape[1]) if hyde_q_emb.ndim == 2 else 0,
        "bundle_path": f"{student_id}.json",
        "agg_mode": mode,
        # policy / safety
        "missing_npy_policy": "HARD_FAIL",
        "index_cached": True,
        # scoring context
        "user_lang": user_lang,
        "recency_half_life_days": float(recency_half_life_days),
        # debug-friendly header config
        "include_feed_header": bool(include_feed_header),
        "feed_header_keys": header_keys,
        # provenance
        # "params_path": params_path,
        # "feed_index_dir": feed_index_dir,
        # "user_bundle_dir": user_bundle_dir,
        # score aggregation meta (safe subset)
        "score_weights_path": score_weights_path,
        "score_aggregation": {
            "enabled": bool(sa_out.get("enabled", False)) if sa_out else False,
            "mode": str(sa_out.get("mode", "linear")) if sa_out else "linear",
            "weights": dict(sa_out.get("weights", {}) or {})
            if (sa_out and isinstance(sa_out.get("weights"), dict))
            else {},
            "clamp_inputs": bool(sa_out.get("clamp_inputs", True)) if sa_out else True,
            "renormalize": bool(sa_out.get("renormalize", True)) if sa_out else True,
            "missing_subscore_value": float(sa_out.get("missing_subscore_value", 0.0)) if sa_out else 0.0,
            "tie_breakers": list(sa_out.get("tie_breakers", []) or []) if sa_out else [],
            "applied": bool(score_enabled),
        },
        # seen-feed meta from exclusion step
        **seen_meta,
        # timing
        "timing_ms": timing_ms,
        "timing_meta": {
            "retrieved_pairs_pre_filter": int(pre_filter_len),
            "retrieved_pairs_post_filter": int(post_filter_len),
            "returned_candidates": int(len(candidates)),
            "exclude_seen_enabled": bool(exclude_enabled),
            "exclude_seen_removed": int(max(0, pre_filter_len - post_filter_len)),
        },
        # tell consumers what score key represents ordering
        "ranking_score_key": ranking_score_key,
    }

    # #########################################################
    # ### --------- Part 5 : Output --------- ###
    # #########################################################
    # print(f"student_id -> {student_id}")
    # print(f"candidates -> {prettyjson(candidates)}")
    # print(f"meta -> {meta}")

    # res = OnlineRetrievalResult(student_id=student_id, candidates=candidates, meta=meta)

    ranked = to_rerank_items(candidates)
    return ranked






# # OPEN dump informat (stu_p000) 
# txt_mockup_path = "modules/functions/eg_input_for_subscore2.txt"
# with open(txt_mockup_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# student_id = data["student_id"]
# # score      = np.array(data["score_matrix"],dtype=np.float32)
# feed       = data["feed_matrix"]
# # hyde_query = [
# #     {"query_id":q["query_id"],"query": q["query_text"], "weight": q["weight"],"intent_label":q["intent_label"]}
# #     for q in data["hyde_query"]["hq"]
# # ]
# metadata   = data["metadata"]

# # print(score)


# # RUN main funnction
# ranked = calc_subscore(
#     student_id = student_id,
#     feed       = feed,
#     metadata   = metadata
# )

# # OUTPUT 
# print(ranked)
# # >>> [
# #     {'feed_id': 'EN_F028', 'final_score': 0.6700167136077243}, 
# #     {'feed_id': 'TH_F005', 'final_score': 0.6636993678001213}, 
# #     {'feed_id': 'EN_F027', 'final_score': 0.6451926602288541}, 
# #     {'feed_id': 'EN_F029', 'final_score': 0.6412319515316993}, 
# #     {'feed_id': 'TH_F013', 'final_score': 0.6390323392274545}, 
# #     {'feed_id': 'TH_F015', 'final_score': 0.6327290219303203}, 
# #     {'feed_id': 'TH_F026', 'final_score': 0.6224891612760681}, 
# #     {'feed_id': 'TH_F021', 'final_score': 0.6188191920670725}
# #    ]



