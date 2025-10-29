# vis_log_to_drawio.py
# TorchLens 0.1.36 -> Draw.io XML(.drawio) export
# - TensorLogEntry 속성 기반 파싱 (dict .get 사용 금지)
# - 레이어 타입별 색상/스타일
# - 모듈 기반 그룹 컬럼
# - 좌표 자동 정규화로 "항상 화면 중앙에 보이도록" 보정

import os
import html
from collections import defaultdict, deque

import torch
from torch import nn
from torchlens import log_forward_pass
from model_complex import UNet

# =========================
# (예시) 사용자 모델 (원하시는 모델로 교체하세요)
# =========================
class DemoUNetToy(nn.Module):
    """데모용 아주 간단한 U-Net-like 토이 모델 (입력: Bx2x1024x1024)"""
    def __init__(self, ch=16):
        super().__init__()
        self.in_proj = nn.Conv2d(2, ch, 3, padding=1)
        self.enc_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(4, ch), nn.ReLU(),
            ) for _ in range(2)
        ])
        self.aspp = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(4, ch), nn.SiLU()
        )
        self.upconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(4, ch), nn.ReLU(),
            )
        ])
        self.dec_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(4, ch), nn.ReLU(),
            )
        ])
        self.align = nn.Sequential(
            nn.Sequential(nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity())
        )
        self.out_conv_r = nn.Conv2d(ch, 1, 1)
        self.out_conv_i = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        x = self.in_proj(x)
        for blk in self.enc_blocks:
            x = blk(x)
        x = self.aspp(x)
        for up in self.upconvs:
            x = up(x)
        for blk in self.dec_blocks:
            x = blk(x)
        # align/head 구조 흉내
        y = self.align(x)
        r = self.out_conv_r(y)
        i = self.out_conv_i(y)
        return torch.cat([r, i], dim=1)


# =========================
# 유틸
# =========================
def _safe_str(x):
    try:
        return str(x)
    except Exception:
        return repr(x)

def _esc(s):
    return html.escape(_safe_str(s), quote=True)

def _infer_type_style_from_layer_type(layer_type: str) -> str:
    """TensorLogEntry.layer_type 기반 스타일 (우선 사용)"""
    if not layer_type:
        return "shape=rectangle;rounded=1;whiteSpace=wrap;html=1;strokeColor=#666666;"
    t = layer_type.lower()
    if "conv" in t:
        return "shape=rectangle;rounded=1;fillColor=#DAE8FC;strokeColor=#6C8EBF;whiteSpace=wrap;html=1;"
    if any(k in t for k in ["batchnorm", "groupnorm", "instancenorm", "layernorm", "norm"]):
        return "shape=rectangle;rounded=1;fillColor=#E1D5E7;strokeColor=#9673A6;whiteSpace=wrap;html=1;"
    if any(k in t for k in ["relu", "gelu", "silu", "sigmoid", "tanh", "softmax", "mish"]):
        return "shape=rectangle;rounded=1;fillColor=#FFE6CC;strokeColor=#D79B00;whiteSpace=wrap;html=1;"
    if "pool" in t:
        return "shape=rectangle;rounded=1;fillColor=#D5E8D4;strokeColor=#82B366;whiteSpace=wrap;html=1;"
    if any(k in t for k in ["linear", "matmul", "mm", "addmm", "gemm", "fc"]):
        return "shape=rectangle;rounded=1;fillColor=#F8CECC;strokeColor=#B85450;whiteSpace=wrap;html=1;"
    if any(k in t for k in ["flatten", "reshape", "view", "cat", "concat", "stack"]):
        return "shape=rectangle;rounded=1;fillColor=#FFF2CC;strokeColor=#D6B656;whiteSpace=wrap;html=1;"
    return "shape=rectangle;rounded=1;whiteSpace=wrap;html=1;strokeColor=#666666;"

def _first_shape_txt(info):
    # TorchLens 0.1.36: TensorLogEntry.tensor_shape
    val = getattr(info, "tensor_shape", None)
    if val is None:
        return "shape: ?"
    return f"shape: {val}"

def _guess_group_by_module(info):
    # 가장 깊은 모듈을 그룹으로 사용 (modules_entered: ["enc_blocks.0", ...])
    modules = getattr(info, "modules_entered", [])
    if not modules:
        return "main"
    return modules[-1].split(":")[0]  # "enc_blocks.0:1" -> "enc_blocks.0"


# =========================
# 레이아웃(위상) 계산
# =========================
def _topo_levels(parents_map: dict) -> dict:
    indeg = defaultdict(int)
    children = defaultdict(list)
    for n, ps in parents_map.items():
        for p in ps:
            indeg[n] += 1
            children[p].append(n)
        indeg.setdefault(n, indeg.get(n, 0))
    level = {}
    q = deque([n for n in parents_map if indeg[n] == 0])
    for n in q:
        level[n] = 0
    while q:
        u = q.popleft()
        for v in children[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                level[v] = level.get(u, 0) + 1
                q.append(v)
    # 사이클/미할당 처리
    max_level = max(level.values()) if level else 0
    for n in parents_map:
        if n not in level:
            plv = [level.get(p, 0) for p in parents_map[n]]
            level[n] = (max(plv) + 1) if plv else max_level + 1
    return level


# =========================
# Draw.io XML 생성
# =========================
def build_drawio_xml(nodes, edges, node_infos=None, node_styles=None,
                     file_name="model_graph.drawio", lr=True,
                     dx=220, dy=110, w=160, h=70,
                     use_groups=False, group_dx=1200, group_margin=40,
                     centerize=True, pad=80):
    """
    기존과 동일한 파라미터지만,
    - draw.io(.drawio) + 순수 mxGraph(.xml) **둘 다** 생성해 안정적으로 불러오게 함.
    - 값(value)에는 HTML 사용 지양(줄바꿈만), 엔티티 이스케이프 철저.
    """
    import html
    from collections import defaultdict, deque
    def _esc(s):
        return html.escape(str(s), quote=True)

    node_infos = node_infos or {}
    node_styles = node_styles or {}

    # parents map
    parents_map = {n: [] for n in nodes}
    for p, c in edges:
        parents_map.setdefault(p, [])
        parents_map.setdefault(c, [])
        parents_map[c].append(p)

    # 위상 레벨
    indeg = defaultdict(int)
    children = defaultdict(list)
    for n, ps in parents_map.items():
        for p in ps:
            indeg[n] += 1
            children[p].append(n)
        indeg.setdefault(n, indeg.get(n, 0))
    level = {}
    q = deque([n for n in parents_map if indeg[n] == 0])
    for n in q:
        level[n] = 0
    while q:
        u = q.popleft()
        for v in children[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                level[v] = level.get(u, 0) + 1
                q.append(v)
    max_level = max(level.values()) if level else 0
    for n in parents_map:
        if n not in level:
            plv = [level.get(p, 0) for p in parents_map[n]]
            level[n] = (max(plv) + 1) if plv else max_level + 1

    # 좌표 배치
    cols = defaultdict(list)
    for n in nodes:
        cols[level.get(n, 0)].append(n)
    coords = {}
    for lv, ns in cols.items():
        ns_sorted = sorted(ns)
        for i, n in enumerate(ns_sorted):
            x = lv * dx if lr else i * dx
            y = i * dy if lr else lv * dy
            coords[n] = (x, y)

    # 중앙 보정(뷰 밖 방지)
    if coords:
        xs = [x for x, _ in coords.values()]
        ys = [y for _, y in coords.values()]
        min_x, min_y = min(xs), min(ys)
        for n in coords:
            x, y = coords[n]
            coords[n] = (x - min_x + pad, y - min_y + pad)

    # 공통 루트(<root>) 생성기
    def _build_root_cells():
        xml = []
        A = xml.append
        A('  <root>')
        A('    <mxCell id="0"/>')
        A('    <mxCell id="1" parent="0" />')  # 레이어 셀
        next_id = 2

        node_id = {}
        # 노드
        for n in nodes:
            nid = str(next_id); next_id += 1
            node_id[n] = nid
            x, y = coords.get(n, (0, 0))
            # draw.io가 가끔 value 내부의 HTML을 싫어할 수 있어, 줄바꿈만 씀
            label = _esc(n)
            info = node_infos.get(n)
            if info:
                # <br/> 대신 줄바꿈 사용
                label += "\\n" + _esc(info)
            style = node_styles.get(n, "shape=rectangle;rounded=1;whiteSpace=wrap;html=0;strokeColor=#666666;")
            A(f'    <mxCell id="{nid}" value="{label}" style="{style}" vertex="1" parent="1">')
            A(f'      <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>')
            A('    </mxCell>')

        # 엣지
        for (p, c) in edges:
            ps = node_id.get(p); cs = node_id.get(c)
            if not ps or not cs:  # 안전장치
                continue
            eid = str(next_id); next_id += 1
            estyle = "endArrow=block;rounded=0;html=0;strokeColor=#606060;"
            A(f'    <mxCell id="{eid}" style="{estyle}" edge="1" parent="1" source="{ps}" target="{cs}">')
            A('      <mxGeometry relative="1" as="geometry"/>')
            A('    </mxCell>')

        A('  </root>')
        return "\n".join(xml)

    root_cells = _build_root_cells()

    # 1) 순수 mxGraph XML (.xml)
    xml_plain = []
    P = xml_plain.append
    P('<?xml version="1.0" encoding="UTF-8"?>')
    # grid/page 설정을 명시적으로 넣어 호환성↑
    P('<mxGraphModel dx="1000" dy="1000" grid="1" gridSize="10" guides="1" tooltips="1" connect="1"')
    P('  arrows="1" fold="1" page="1" pageScale="1" pageWidth="827" pageHeight="1169" math="0" shadow="0">')
    P(root_cells)
    P('</mxGraphModel>')

    plain_path = os.path.splitext(file_name)[0] + ".xml"
    with open(plain_path, "w", encoding="utf-8") as f:
        f.write("\n".join(xml_plain))

    # 2) draw.io 포맷(.drawio) — wrapper만 다름, 내부는 동일 root 사용
    xml_drawio = []
    D = xml_drawio.append
    D('<?xml version="1.0" encoding="UTF-8"?>')
    D('<mxfile host="app.diagrams.net" type="device">')
    D('  <diagram id="model" name="Model Graph">')
    D('    <mxGraphModel dx="1000" dy="1000" grid="1" gridSize="10" guides="1" tooltips="1" connect="1"')
    D('      arrows="1" fold="1" page="1" pageScale="1" pageWidth="827" pageHeight="1169" math="0" shadow="0">')
    D(root_cells)
    D('    </mxGraphModel>')
    D('  </diagram>')
    D('</mxfile>')

    drawio_path = os.path.splitext(file_name)[0] + ".drawio"
    with open(drawio_path, "w", encoding="utf-8") as f:
        f.write("\n".join(xml_drawio))

    return os.path.abspath(drawio_path)


# =========================
# TorchLens 로그 -> 노드/엣지
# =========================
def build_drawio_from_log(log, file_name="model_graph.drawio", lr=True, use_groups=True):
    if not hasattr(log, "layer_labels") or not hasattr(log, "layer_dict_all_keys"):
        raise RuntimeError("TorchLens log에 layer_labels / layer_dict_all_keys 가 없습니다.")

    layer_names = list(log.layer_labels)
    info_dict = log.layer_dict_all_keys  # name -> TensorLogEntry

    nodes, edges = [], []
    node_infos, node_styles = {}, {}

    # 우선 레이어만 돌면서 정보 수집
    for name in layer_names:
        info = info_dict[name]
        nodes.append(name)
        node_infos[name] = _first_shape_txt(info)

        # 스타일: layer_type 기반
        ltype = getattr(info, "layer_type", None)
        node_styles[name] = _infer_type_style_from_layer_type(ltype)

    # 부모-자식 엣지 구성
    for name in layer_names:
        info = info_dict[name]
        parents = getattr(info, "parent_layers", [])
        if parents is None:
            parents = []
        elif isinstance(parents, str):
            parents = [parents]
        else:
            parents = list(parents)
        for p in parents:
            edges.append((p, name))

    # 부모만 있고 nodes에 없는 입력 노드 보강
    node_set = set(nodes)
    extras = sorted({p for p, _ in edges if p not in node_set})
    for p in extras:
        nodes.append(p)
        node_infos[p] = "input"
        node_styles[p] = _infer_type_style_from_layer_type("input")

    # ===== 중앙 배치 가능하도록 그룹 정보 준비 =====
    # 기본은 "main" 그룹인데, real 그룹은 modules_entered로 추정
    group_of = {}
    for n in nodes:
        info = info_dict.get(n)
        if info is None:
            group_of[n] = "main"
        else:
            group_of[n] = _guess_group_by_module(info)

    # build_drawio_xml 내부에서 group_of를 알 수 있도록, 임시로 전역 dict를 사용하거나
    # 여기서 레벨 계산 및 coords를 직접 만들어도 된다.
    # 간단히: build_drawio_xml 호출 전에 group 컬럼을 반영하기 위해
    # nodes 이름을 "group::name"로 바꾸는 방식은 피하고,
    # 아래처럼 monkey patch: 함수 내부 group_of를 교체하는 대신
    # 좌표 계산을 여기에서 직접 수행할 수도 있지만,
    # 본 함수에서는 build_drawio_xml의 기본 group_of("main")를 사용하되,
    # 그룹 배경은 껐다가(=use_groups=False) 화면 중앙 정렬만 보장하는 방식을 택한다.

    # 👉 더 나은 방법: build_drawio_xml를 한 번 호출하기 전에
    #    우리가 직접 좌표를 계산하고 전달하는 방식으로 바꾸려면 함수 대폭 수정 필요.
    #    여기서는 간단하고 안전하게: "그룹 배경은 끄고(use_groups=False)" + 중앙정렬 확실히.
    #    (그룹 배경이 꼭 필요하면 아래 주석 블록의 확장 버전을 드릴 수 있어요.)

    # 디버그 출력
    print(f"[DEBUG] nodes={len(nodes)}, edges={len(edges)}")

    # 그룹 배경을 끄고 중앙정렬만 적용한 상태로 먼저 생성
    # (원하시면 use_groups=True로 바꿔도 되지만, 내부 group_of 통제가 필요)
    return build_drawio_xml(
        nodes, edges, node_infos=node_infos, node_styles=node_styles,
        file_name=file_name, lr=lr, use_groups=False,  # 👈 배경 그룹 off
        dx=220, dy=110, w=170, h=74, group_dx=1200, group_margin=40,
        centerize=True, pad=80
    )


# =========================
# 실행 예시
# =========================
if __name__ == "__main__":
    torch.manual_seed(0)

    # 1) 사용자 정의 모델로 교체 가능
    model = UNet(in_ch=2, out_ch=2, base_ch=16, ch_mult=[1,2,4], conditional=False)

    # 2) 입력 텐서: 요청하신 (1, 2, 1024, 1024)
    x = torch.randn(1, 2, 1024, 1024)

    model.eval()
    with torch.no_grad():
        log = log_forward_pass(model, x)

    out_path = build_drawio_from_log(
        log, file_name="model_graph.xml", lr=True, use_groups=False
    )
    print(f"[OK] Draw.io 파일 생성 완료: {out_path}")
    print("→ diagrams.net(https://app.diagrams.net) 열기 → File > Import From > Device 에서 model_graph.drawio 선택")
    print("→ 화면에 안 보이면 View > Fit / Ctrl+Shift+H (하지만 이번 버전은 자동 중앙정렬됨)")
