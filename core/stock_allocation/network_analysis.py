"""US→JP 株式ネットワーク分析

イベントスタディの有意ペアからネットワークグラフを構築し、
ハブ銘柄・クラスター・影響伝播経路を可視化する。
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .event_study import EventStudyResult, PairResult

logger = logging.getLogger(__name__)


@dataclass
class NetworkNode:
    """ネットワーク上の銘柄ノード"""
    id: str              # ticker or code
    name: str            # 表示名
    side: str            # "US" or "JP"
    sector: str = ""
    degree: int = 0      # 接続数
    weighted_degree: float = 0.0  # CAR加重の接続強度
    avg_car: float = 0.0
    avg_hit_rate: float = 0.0


@dataclass
class NetworkEdge:
    """ネットワーク上のペアエッジ"""
    source: str    # US ticker
    target: str    # JP code
    weight: float  # |peak_car|
    car: float
    hit_rate: float
    peak_lag: int
    sector: str = ""


@dataclass
class StockNetwork:
    """構築済みネットワーク"""
    nodes: list[NetworkNode] = field(default_factory=list)
    edges: list[NetworkEdge] = field(default_factory=list)
    # ランキング
    us_hubs: list[NetworkNode] = field(default_factory=list)     # 影響力の高いUS株
    jp_receivers: list[NetworkNode] = field(default_factory=list) # 感応的なJP株
    clusters: list[dict] = field(default_factory=list)            # 銘柄クラスター


def build_network(
    result: EventStudyResult,
    min_hit_rate: float = 0.0,
) -> StockNetwork:
    """イベントスタディ結果からネットワークを構築する。

    Args:
        result: EventStudyResult (significant_pairsを使う)
        min_hit_rate: 方向一致率の最低閾値

    Returns:
        StockNetwork
    """
    pairs = result.significant_pairs
    if min_hit_rate > 0:
        pairs = [p for p in pairs if p.direction_hit_rate >= min_hit_rate]

    if not pairs:
        return StockNetwork()

    # ノードとエッジの構築
    us_nodes: dict[str, NetworkNode] = {}
    jp_nodes: dict[str, NetworkNode] = {}
    edges: list[NetworkEdge] = []

    for p in pairs:
        # USノード
        if p.us_ticker not in us_nodes:
            us_nodes[p.us_ticker] = NetworkNode(
                id=p.us_ticker, name=p.us_ticker,
                side="US", sector=p.sector_label,
            )

        # JPノード
        if p.jp_code not in jp_nodes:
            jp_nodes[p.jp_code] = NetworkNode(
                id=p.jp_code, name=p.jp_name or p.jp_code,
                side="JP", sector=p.sector_label,
            )

        # エッジ
        edges.append(NetworkEdge(
            source=p.us_ticker,
            target=p.jp_code,
            weight=abs(p.peak_car),
            car=p.peak_car,
            hit_rate=p.direction_hit_rate,
            peak_lag=p.peak_lag,
            sector=p.sector_label,
        ))

        # 次数を更新
        us_nodes[p.us_ticker].degree += 1
        us_nodes[p.us_ticker].weighted_degree += abs(p.peak_car)
        jp_nodes[p.jp_code].degree += 1
        jp_nodes[p.jp_code].weighted_degree += abs(p.peak_car)

    # 平均CARとヒット率を計算
    for p in pairs:
        us_n = us_nodes[p.us_ticker]
        us_n.avg_car = us_n.weighted_degree / max(us_n.degree, 1)
        us_n.avg_hit_rate += p.direction_hit_rate / max(us_n.degree, 1)

        jp_n = jp_nodes[p.jp_code]
        jp_n.avg_car = jp_n.weighted_degree / max(jp_n.degree, 1)
        jp_n.avg_hit_rate += p.direction_hit_rate / max(jp_n.degree, 1)

    all_nodes = list(us_nodes.values()) + list(jp_nodes.values())

    # ハブランキング
    us_hubs = sorted(us_nodes.values(), key=lambda n: n.weighted_degree, reverse=True)
    jp_receivers = sorted(jp_nodes.values(), key=lambda n: n.weighted_degree, reverse=True)

    # クラスター検出 (連結成分ベース)
    clusters = _detect_clusters(edges, us_nodes, jp_nodes)

    return StockNetwork(
        nodes=all_nodes,
        edges=edges,
        us_hubs=us_hubs,
        jp_receivers=jp_receivers,
        clusters=clusters,
    )


def _detect_clusters(
    edges: list[NetworkEdge],
    us_nodes: dict,
    jp_nodes: dict,
) -> list[dict]:
    """連結成分でクラスターを検出する。"""
    # Union-Find
    parent: dict[str, str] = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for e in edges:
        union(e.source, e.target)

    # クラスターに分類
    groups: dict[str, list[str]] = {}
    all_ids = list(us_nodes.keys()) + list(jp_nodes.keys())
    for nid in all_ids:
        root = find(nid)
        groups.setdefault(root, []).append(nid)

    clusters = []
    for root, members in groups.items():
        us_members = [m for m in members if m in us_nodes]
        jp_members = [m for m in members if m in jp_nodes]
        if us_members and jp_members:
            # 代表セクター
            sectors = set()
            for m in us_members:
                if us_nodes[m].sector:
                    sectors.add(us_nodes[m].sector)
            clusters.append({
                "us": [us_nodes[m].name for m in us_members],
                "jp": [jp_nodes[m].name for m in jp_members],
                "sectors": list(sectors),
                "size": len(members),
                "n_edges": sum(1 for e in edges if e.source in us_members),
            })

    clusters.sort(key=lambda c: c["n_edges"], reverse=True)
    return clusters


def create_network_figure(
    network: StockNetwork,
    height: int = 700,
) -> go.Figure:
    """Plotlyでインタラクティブなネットワークグラフを作成する。"""
    if not network.nodes or not network.edges:
        fig = go.Figure()
        fig.add_annotation(text="有意なペアがありません", showarrow=False)
        return fig

    # レイアウト: US左側、JP右側に配置
    us_nodes = [n for n in network.nodes if n.side == "US"]
    jp_nodes = [n for n in network.nodes if n.side == "JP"]

    # 次数でソート（ハブを中央に）
    us_nodes.sort(key=lambda n: n.degree, reverse=True)
    jp_nodes.sort(key=lambda n: n.degree, reverse=True)

    pos: dict[str, tuple[float, float]] = {}
    # US: x=0, yを均等配置
    for i, n in enumerate(us_nodes):
        y = 1.0 - (i / max(len(us_nodes) - 1, 1))
        pos[n.id] = (0.0, y)

    # JP: x=1, yを均等配置
    for i, n in enumerate(jp_nodes):
        y = 1.0 - (i / max(len(jp_nodes) - 1, 1))
        pos[n.id] = (1.0, y)

    fig = go.Figure()

    # エッジ
    for e in network.edges:
        if e.source in pos and e.target in pos:
            x0, y0 = pos[e.source]
            x1, y1 = pos[e.target]
            opacity = min(0.8, 0.2 + e.weight * 10)
            width = max(0.5, e.weight * 80)

            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line={"width": width, "color": f"rgba(255,128,0,{opacity})"},
                hoverinfo="text",
                text=f"{e.source}→{e.target}<br>CAR: {e.car:+.1%}<br>一致率: {e.hit_rate:.0%}<br>ラグ: t+{e.peak_lag}",
                showlegend=False,
            ))

    # USノード
    us_x = [pos[n.id][0] for n in us_nodes]
    us_y = [pos[n.id][1] for n in us_nodes]
    us_sizes = [max(15, n.degree * 12) for n in us_nodes]
    us_text = [
        f"{n.name}<br>接続数: {n.degree}<br>平均CAR: {n.avg_car:.1%}"
        for n in us_nodes
    ]
    us_labels = [n.name for n in us_nodes]

    fig.add_trace(go.Scatter(
        x=us_x, y=us_y,
        mode="markers+text",
        marker={"size": us_sizes, "color": "#2196F3", "line": {"width": 1, "color": "white"}},
        text=us_labels,
        textposition="middle left",
        textfont={"size": 10},
        hoverinfo="text",
        hovertext=us_text,
        name="US",
    ))

    # JPノード
    jp_x = [pos[n.id][0] for n in jp_nodes]
    jp_y = [pos[n.id][1] for n in jp_nodes]
    jp_sizes = [max(15, n.degree * 12) for n in jp_nodes]
    jp_text = [
        f"{n.name}<br>接続数: {n.degree}<br>平均CAR: {n.avg_car:.1%}"
        for n in jp_nodes
    ]
    jp_labels = [n.name for n in jp_nodes]

    fig.add_trace(go.Scatter(
        x=jp_x, y=jp_y,
        mode="markers+text",
        marker={"size": jp_sizes, "color": "#FF5722", "line": {"width": 1, "color": "white"}},
        text=jp_labels,
        textposition="middle right",
        textfont={"size": 10},
        hoverinfo="text",
        hovertext=jp_text,
        name="JP",
    ))

    fig.update_layout(
        title="US → JP 銘柄ネットワーク",
        showlegend=True,
        height=height,
        xaxis={"visible": False, "range": [-0.3, 1.3]},
        yaxis={"visible": False, "range": [-0.05, 1.05]},
        plot_bgcolor="white",
        annotations=[
            {"x": 0, "y": 1.08, "text": "US", "showarrow": False, "font": {"size": 16, "color": "#2196F3"}},
            {"x": 1, "y": 1.08, "text": "JP", "showarrow": False, "font": {"size": 16, "color": "#FF5722"}},
        ],
    )

    return fig


def build_hub_table(network: StockNetwork) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ハブ銘柄のランキングテーブルを作成する。"""
    us_rows = []
    for n in network.us_hubs:
        # 接続先のJP銘柄を取得
        targets = [e.target for e in network.edges if e.source == n.id]
        target_names = []
        for t in targets:
            jp_n = next((nn for nn in network.nodes if nn.id == t), None)
            if jp_n:
                target_names.append(jp_n.name)
        us_rows.append({
            "US銘柄": n.name,
            "接続数": n.degree,
            "加重強度": n.weighted_degree,
            "接続先JP": ", ".join(target_names[:5]) + ("..." if len(target_names) > 5 else ""),
        })

    jp_rows = []
    for n in network.jp_receivers:
        sources = [e.source for e in network.edges if e.target == n.id]
        jp_rows.append({
            "JP銘柄": n.name,
            "接続数": n.degree,
            "加重強度": n.weighted_degree,
            "接続元US": ", ".join(sources[:5]) + ("..." if len(sources) > 5 else ""),
        })

    return pd.DataFrame(us_rows), pd.DataFrame(jp_rows)
