from typing import List
from config import RRF_K, FINAL_TOP_K
from utils import get_logger

logger = get_logger(__name__)


def reciprocal_rank_fusion(result_lists: List[list]) -> list:
    """
    Merge multiple ranked node lists using Reciprocal Rank Fusion (RRF).

    RRF score for a node = Σ  1 / (k + rank_i)
    where rank_i is the 1-based position in list i and k = RRF_K constant.

    Returns the top FINAL_TOP_K nodes, sorted descending by fused score.
    """
    scores: dict[str, float] = {}
    nodes_by_id: dict[str, object] = {}

    for result_list in result_lists:
        for rank, node_with_score in enumerate(result_list, start=1):
            node_id = node_with_score.node.node_id
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (RRF_K + rank)
            nodes_by_id[node_id] = node_with_score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top = [nodes_by_id[nid] for nid, _ in ranked[:FINAL_TOP_K]]

    logger.debug(f"RRF fusion: {sum(len(r) for r in result_lists)} → {len(top)} docs")
    return top
