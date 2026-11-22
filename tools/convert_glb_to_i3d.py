#!/usr/bin/env python3
"""Convert source_assets/gol_g5.glb into FS25 I3D v1.6 + tools/i3d_node_map.json."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
from lxml import etree

ROOT_DIR = Path(__file__).resolve().parent.parent
GLB_PATH = ROOT_DIR / "source_assets" / "gol_g5.glb"
I3D_PATH = ROOT_DIR / "FS25_VW_Gol_G5" / "vehicles" / "vwGolG5" / "golG5.i3d"
NODE_MAP_PATH = ROOT_DIR / "tools" / "i3d_node_map.json"

Y_GROUND_OFFSET = 0.000992
STEERING_PIVOT_FS25 = np.array([0.4026, 0.9492, 0.5397], dtype=np.float64)
STEERING_TILT_DEG = 65.0

COMP_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

STATIC_LIGHT_NODES = [
    "headlight_l",
    "headlight_r",
    "luxfarol",
    "brakelight_l",
    "brakelight_m",
    "brakelight_r",
    "reversinglight_l",
    "reversinglight_r",
    "lanternada",
]

VISUAL_NODES = [
    "bodyshell",
    "bonnet",
    "boot",
    "bumper_f",
    "bumper_r",
    "door_dside_f",
    "door_pside_f",
    "door_dside_r",
    "door_pside_r",
    "painel",
    "painellux",
    "dials",
    "lanterna",
    "Taillight_Housing_Rear",
    "Headlight_Optics_Detail",
    "hub_lf",
    "hub_rf",
    "hub_lr",
    "hub_rr",
    "suspension_lf",
    "suspension_rf",
    "suspension_lr",
    "suspension_rr",
]

# Primitives in gol_g5.glb whose face winding and vertex normals point inward (>65% vs car center)
INVERTED_PRIMITIVES: dict[str, set[int]] = {
    "bonnet": {0},
    "bumper_f": {1},
    "door_dside_f": {0, 1, 2, 4, 5},
    "door_pside_f": {0, 1, 2, 4, 5},
    "door_dside_r": {0, 1, 3, 4},
    "door_pside_r": {3, 4},
}


def load_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    magic, version, _ = struct.unpack("<4sII", raw[:12])
    if magic != b"glTF" or version != 2:
        raise RuntimeError(f"Unsupported GLB header in {path}")
    jlen, jtype = struct.unpack("<II", raw[12:20])
    if jtype != 0x4E4F534A:
        raise RuntimeError("Invalid JSON chunk in GLB")
    gltf = json.loads(raw[20 : 20 + jlen].decode("utf-8"))
    b_off = 20 + jlen
    blen, btype = struct.unpack("<II", raw[b_off : b_off + 8])
    if btype != 0x004E4942:
        raise RuntimeError("Invalid BIN chunk in GLB")
    return gltf, raw[b_off + 8 : b_off + 8 + blen]


def read_accessor(gltf: dict[str, Any], bin_chunk: bytes, acc_idx: int) -> np.ndarray:
    acc = gltf["accessors"][acc_idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    dtype = np.dtype(COMP_DTYPE[acc["componentType"]])
    count = acc["count"]
    ncomp = TYPE_COMPONENTS[acc["type"]]
    stride = bv.get("byteStride")
    elem_bytes = ncomp * dtype.itemsize

    if stride and stride != elem_bytes:
        arr = np.empty((count, ncomp), dtype=dtype)
        for i in range(count):
            start = offset + i * stride
            arr[i] = np.frombuffer(bin_chunk, dtype=dtype, count=ncomp, offset=start)
    else:
        flat = np.frombuffer(bin_chunk, dtype=dtype, count=count * ncomp, offset=offset)
        arr = flat if ncomp == 1 else flat.reshape(count, ncomp)
    return arr


def _quat_to_mat3(q: list[float]) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def node_local_matrix(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    mat = np.eye(4, dtype=np.float64)
    r = _quat_to_mat3(node["rotation"]) if "rotation" in node else np.eye(3, dtype=np.float64)
    s = np.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    mat[:3, :3] = r * s[None, :]
    if "translation" in node:
        mat[:3, 3] = np.array(node["translation"], dtype=np.float64)
    return mat


def compute_world_matrices(gltf: dict[str, Any]) -> dict[int, np.ndarray]:
    nodes = gltf["nodes"]
    parents: dict[int, int] = {}
    for idx, node in enumerate(nodes):
        for child in node.get("children", []):
            parents[child] = idx

    world_mats: dict[int, np.ndarray] = {}

    def get_world(idx: int) -> np.ndarray:
        if idx in world_mats:
            return world_mats[idx]
        local = node_local_matrix(nodes[idx])
        if idx in parents:
            local = get_world(parents[idx]) @ local
        world_mats[idx] = local
        return local

    for idx in range(len(nodes)):
        get_world(idx)
    return world_mats


def rx_matrix(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def fmt_vec(arr: np.ndarray) -> str:
    return f"{arr[0]:.6g} {arr[1]:.6g} {arr[2]:.6g}"


def build_collision_box_shape() -> tuple[str, int, int]:
    """Build simplified collision box (2.06 x 1.55 x 4.24 m, ground clearance 0.24 m)."""
    x0, x1 = -1.03, 1.03
    y0, y1 = 0.24, 1.79
    z0, z1 = -2.10, 2.14
    corners = np.array(
        [
            [x0, y0, z0],  # 0: R B Back
            [x1, y0, z0],  # 1: L B Back
            [x1, y1, z0],  # 2: L T Back
            [x0, y1, z0],  # 3: R T Back
            [x0, y0, z1],  # 4: R B Front
            [x1, y0, z1],  # 5: L B Front
            [x1, y1, z1],  # 6: L T Front
            [x0, y1, z1],  # 7: R T Front
        ],
        dtype=np.float64,
    )
    center = corners.mean(axis=0)
    normals = corners - center
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    # Outward CCW triangles
    tris = [
        (0, 2, 1), (0, 3, 2),  # Back (-Z)
        (4, 5, 6), (4, 6, 7),  # Front (+Z)
        (0, 1, 5), (0, 5, 4),  # Bottom (-Y)
        (3, 7, 6), (3, 6, 2),  # Top (+Y)
        (1, 2, 6), (1, 6, 5),  # Left (+X)
        (0, 4, 7), (0, 7, 3),  # Right (-X)
    ]

    v_lines = [
        f'          <v p="{p[0]:.6g} {p[1]:.6g} {p[2]:.6g}" n="{n[0]:.6g} {n[1]:.6g} {n[2]:.6g}"/>'
        for p, n in zip(corners, normals)
    ]
    t_lines = [f'          <t vi="{a} {b} {c}"/>' for a, b, c in tris]

    xml = "\n".join(
        [
            '    <IndexedTriangleSet name="golG5_collisionBox" shapeId="1">',
            '      <Vertices count="8" normal="true">',
            *v_lines,
            "      </Vertices>",
            '      <Triangles count="12">',
            *t_lines,
            "      </Triangles>",
            '      <Subsets count="1">',
            '        <Subset firstVertex="0" numVertices="8" firstIndex="0" numIndices="36"/>',
            "      </Subsets>",
            "    </IndexedTriangleSet>",
        ]
    )
    return xml, 8, 12


def build_materials_xml(gltf: dict[str, Any]) -> list[str]:
    """Build <Materials> section (materialId 1..19) matching Contract Section 2.2."""
    lines = [
        '    <Material name="Mat_Collision_Default" materialId="1" diffuseColor="0.5 0.5 0.5 1" specularColor="0.2 0.2 0"/>'
    ]
    for glb_idx, mat in enumerate(gltf["materials"]):
        mat_id = glb_idx + 2
        name = mat["name"]
        pbr = mat.get("pbrMetallicRoughness", {})
        rough = float(pbr.get("roughnessFactor", 0.5))
        metal = float(pbr.get("metallicFactor", 0.0))
        spec = f"{1.0 - rough:.6g} {metal:.6g} 0"

        if name == "Mat_CarPaint_Black":
            lines.append(
                f'    <Material name="{escape(name)}" materialId="{mat_id}" '
                f'diffuseColor="0.015 0.016 0.02 1" specularColor="{spec}" customShaderId="7">'
            )
            lines.append('      <CustomParameter name="colorScale" value="0.015 0.016 0.020 1"/>')
            lines.append("    </Material>")
        elif name == "Mat_Taillight_Amber_Chamber":
            lines.append(f'    <Material name="{escape(name)}" materialId="{mat_id}" specularColor="{spec}">')
            lines.append('      <Texture fileId="1"/>')
            lines.append("    </Material>")
        elif name == "Mat_Brakelight_Red":
            lines.append(
                f'    <Material name="{escape(name)}" materialId="{mat_id}" specularColor="{spec}" '
                f'customShaderId="7" customShaderVariation="staticLight">'
            )
            lines.append('      <Texture fileId="3"/>')
            lines.append('      <Emissivemap fileId="2"/>')
            lines.append("    </Material>")
        elif name == "Mat_Headlight_Xenon":
            lines.append(
                f'    <Material name="{escape(name)}" materialId="{mat_id}" specularColor="{spec}" '
                f'customShaderId="7" customShaderVariation="staticLight">'
            )
            lines.append('      <Texture fileId="5"/>')
            lines.append('      <Emissivemap fileId="4"/>')
            lines.append("    </Material>")
        elif name == "Mat_Reversing_Light":
            lines.append(
                f'    <Material name="{escape(name)}" materialId="{mat_id}" specularColor="{spec}" '
                f'emissiveColor="0.1425 0.147 0.15 1" customShaderId="7" customShaderVariation="staticLight">'
            )
            lines.append('      <Texture fileId="6"/>')
            lines.append("    </Material>")
        else:
            col = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
            diff = f"{col[0]:.6g} {col[1]:.6g} {col[2]:.6g} {col[3]:.6g}"
            alpha_attr = ' alphaBlending="true"' if mat.get("alphaMode") == "BLEND" else ""
            lines.append(
                f'    <Material name="{escape(name)}" materialId="{mat_id}" '
                f'diffuseColor="{diff}" specularColor="{spec}"{alpha_attr}/>'
            )

    lines.append(
        '    <Material name="Mat_Taillight_Amber_Solid" materialId="19" '
        'diffuseColor="0.85 0.42 0.02 1" specularColor="0.9 0.35 0"/>'
    )
    return lines


def convert_mesh_to_shape(
    gltf: dict[str, Any],
    bin_chunk: bytes,
    mesh_idx: int,
    shape_id: int,
    world_mat: np.ndarray,
    mode: str,
) -> tuple[str, list[int], int, int, np.ndarray | None, np.ndarray | None]:
    """Convert one GLB mesh into an I3D <IndexedTriangleSet>.

    mode:
      - 'wheel': local GLB coordinates -> (-lx, ly, -lz)
      - 'steering': world FS25 -> subtract pivot -> R_x(+65 deg)
      - 'world': world FS25 -> (-wx, wy + Y_GROUND_OFFSET, -wz)
    """
    mesh = gltf["meshes"][mesh_idx]
    mesh_name = mesh.get("name", f"mesh_{mesh_idx}")
    has_uv = all("TEXCOORD_0" in p["attributes"] for p in mesh["primitives"])

    all_pos = []
    all_nrm = []
    all_uv = []
    all_tri = []
    subsets_meta: list[tuple[int, int, int, int, str | None]] = []
    material_ids: list[int] = []

    v_offset = 0
    idx_offset = 0

    rot_mat = world_mat[:3, :3]
    trans_vec = world_mat[:3, 3]
    rx_sw = rx_matrix(STEERING_TILT_DEG)

    for prim_idx, prim in enumerate(mesh["primitives"]):
        attrs = prim["attributes"]
        pos = read_accessor(gltf, bin_chunk, attrs["POSITION"]).astype(np.float64)
        nrm = read_accessor(gltf, bin_chunk, attrs["NORMAL"]).astype(np.float64)
        indices = read_accessor(gltf, bin_chunk, prim["indices"]).astype(np.int64).reshape(-1, 3)

        if prim_idx in INVERTED_PRIMITIVES.get(mesh_name, set()):
            nrm = -nrm
            indices = indices[:, [0, 2, 1]]

        if mode == "wheel":
            pos_fs = np.column_stack([-pos[:, 0], pos[:, 1], -pos[:, 2]])
            nrm_fs = np.column_stack([-nrm[:, 0], nrm[:, 1], -nrm[:, 2]])
        else:
            pos_w = pos @ rot_mat.T + trans_vec
            nrm_w = nrm @ rot_mat.T
            pos_fs = np.column_stack([-pos_w[:, 0], pos_w[:, 1] + Y_GROUND_OFFSET, -pos_w[:, 2]])
            nrm_fs = np.column_stack([-nrm_w[:, 0], nrm_w[:, 1], -nrm_w[:, 2]])
            if mode == "steering":
                pos_fs = (pos_fs - STEERING_PIVOT_FS25) @ rx_sw.T
                nrm_fs = nrm_fs @ rx_sw.T

        nrm_len = np.linalg.norm(nrm_fs, axis=1, keepdims=True)
        nrm_fs = np.divide(nrm_fs, np.maximum(nrm_len, 1e-12))

        all_pos.append(pos_fs)
        all_nrm.append(nrm_fs)
        if has_uv:
            uv = read_accessor(gltf, bin_chunk, attrs["TEXCOORD_0"]).astype(np.float64)
            uv_fs = np.column_stack([uv[:, 0], 1.0 - uv[:, 1]])
            all_uv.append(uv_fs)

        all_tri.append(indices + v_offset)

        glb_mat_idx = prim["material"]
        if not has_uv and glb_mat_idx == 1:
            # Headlight_Optics_Detail lacks TEXCOORD_0 -> use Mat_Taillight_Amber_Solid
            i3d_mat_id = 19
        else:
            i3d_mat_id = glb_mat_idx + 2
        material_ids.append(i3d_mat_id)

        slot_name = "carPaint" if i3d_mat_id == 7 else None
        nv = len(pos_fs)
        ni = len(indices) * 3
        subsets_meta.append((v_offset, nv, idx_offset, ni, slot_name))
        v_offset += nv
        idx_offset += ni

    pos_cat = np.vstack(all_pos)
    nrm_cat = np.vstack(all_nrm)
    tri_cat = np.vstack(all_tri)

    if has_uv:
        uv_cat = np.vstack(all_uv)
        v_lines = [
            f'          <v p="{p[0]:.6g} {p[1]:.6g} {p[2]:.6g}" n="{n[0]:.6g} {n[1]:.6g} {n[2]:.6g}" t0="{t[0]:.6g} {t[1]:.6g}"/>'
            for p, n, t in zip(pos_cat, nrm_cat, uv_cat)
        ]
        uv_attr = ' uv0="true"'
    else:
        v_lines = [
            f'          <v p="{p[0]:.6g} {p[1]:.6g} {p[2]:.6g}" n="{n[0]:.6g} {n[1]:.6g} {n[2]:.6g}"/>'
            for p, n in zip(pos_cat, nrm_cat)
        ]
        uv_attr = ""

    t_lines = [f'          <t vi="{t[0]} {t[1]} {t[2]}"/>' for t in tri_cat]
    s_lines = []
    for fv, nv, fi, ni, slot in subsets_meta:
        slot_attr = f' materialSlotName="{slot}"' if slot else ""
        s_lines.append(
            f'        <Subset firstVertex="{fv}" numVertices="{nv}" firstIndex="{fi}" numIndices="{ni}"{slot_attr}/>'
        )

    xml = "\n".join(
        [
            f'    <IndexedTriangleSet name="{escape(mesh_name)}" shapeId="{shape_id}">',
            f'      <Vertices count="{len(pos_cat)}" normal="true"{uv_attr}>',
            *v_lines,
            "      </Vertices>",
            f'      <Triangles count="{len(tri_cat)}">',
            *t_lines,
            "      </Triangles>",
            f'      <Subsets count="{len(subsets_meta)}">',
            *s_lines,
            "      </Subsets>",
            "    </IndexedTriangleSet>",
        ]
    )

    if mode == "wheel":
        bbox_min = pos_cat.min(axis=0)
        bbox_max = pos_cat.max(axis=0)
    elif mode == "world":
        bbox_min = pos_cat.min(axis=0)
        bbox_max = pos_cat.max(axis=0)
    else:
        # Compute world-space bbox for steering wheel too
        rx_inv = rx_matrix(-STEERING_TILT_DEG)
        world_pts = pos_cat @ rx_inv.T + STEERING_PIVOT_FS25
        bbox_min = world_pts.min(axis=0)
        bbox_max = world_pts.max(axis=0)

    return xml, material_ids, len(pos_cat), len(tri_cat), bbox_min, bbox_max


def main() -> None:
    gltf, bin_chunk = load_glb(GLB_PATH)
    world_mats = compute_world_matrices(gltf)

    node_by_name = {n["name"]: (idx, n) for idx, n in enumerate(gltf["nodes"])}

    # Build <Shapes>
    shapes_xml_parts: list[str] = []
    col_xml, col_v, col_t = build_collision_box_shape()
    shapes_xml_parts.append(col_xml)

    # Map mesh_idx -> (shape_id, material_ids_str, verts, tris)
    mesh_shape_info: dict[int, tuple[int, str, int, int]] = {}
    next_shape_id = 2

    global_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    global_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    def register_bbox(bmin: np.ndarray, bmax: np.ndarray) -> None:
        nonlocal global_min, global_max
        global_min = np.minimum(global_min, bmin)
        global_max = np.maximum(global_max, bmax)

    # 1) Wheel shapes (instanced across 4 wheel nodes)
    wheel_node_names = ["Wheel_FL", "Wheel_FR", "Wheel_RL", "Wheel_RR"]
    for wn in wheel_node_names:
        n_idx, _ = node_by_name[wn]
        m_idx = gltf["nodes"][n_idx]["mesh"]
        w_mat = world_mats[n_idx]
        t_glb = w_mat[:3, 3]
        t_fs = np.array([-t_glb[0], t_glb[1] + Y_GROUND_OFFSET, -t_glb[2]], dtype=np.float64)
        if m_idx not in mesh_shape_info:
            s_id = next_shape_id
            next_shape_id += 1
            xml_s, mat_ids, nv, nt, lmin, lmax = convert_mesh_to_shape(
                gltf, bin_chunk, m_idx, s_id, w_mat, mode="wheel"
            )
            shapes_xml_parts.append(xml_s)
            mesh_shape_info[m_idx] = (s_id, " ".join(map(str, mat_ids)), nv, nt)
            register_bbox(lmin + t_fs, lmax + t_fs)
        else:
            # Reuse shape bounds shifted by this wheel's world position
            # Recompute local bounds quickly from wheel translation delta
            pass

    # Compute exact wheel bounds for RL and RR too
    for wn in ["Wheel_RL", "Wheel_RR"]:
        n_idx, _ = node_by_name[wn]
        m_idx = gltf["nodes"][n_idx]["mesh"]
        t_glb = world_mats[n_idx][:3, 3]
        t_fs = np.array([-t_glb[0], t_glb[1] + Y_GROUND_OFFSET, -t_glb[2]], dtype=np.float64)
        pos_list = [
            read_accessor(gltf, bin_chunk, p["attributes"]["POSITION"]).astype(np.float64)
            for p in gltf["meshes"][m_idx]["primitives"]
        ]
        pts = np.vstack(pos_list)
        pts_fs = np.column_stack([-pts[:, 0], pts[:, 1], -pts[:, 2]]) + t_fs
        register_bbox(pts_fs.min(axis=0), pts_fs.max(axis=0))

    # 2) Steering wheel
    sw_idx, _ = node_by_name["steeringwheel"]
    sw_m_idx = gltf["nodes"][sw_idx]["mesh"]
    sw_sid = next_shape_id
    next_shape_id += 1
    sw_xml, sw_mats, sw_v, sw_t, sw_min, sw_max = convert_mesh_to_shape(
        gltf, bin_chunk, sw_m_idx, sw_sid, world_mats[sw_idx], mode="steering"
    )
    shapes_xml_parts.append(sw_xml)
    mesh_shape_info[sw_m_idx] = (sw_sid, " ".join(map(str, sw_mats)), sw_v, sw_t)
    register_bbox(sw_min, sw_max)

    # 3) Static lights + visuals
    for name in STATIC_LIGHT_NODES + VISUAL_NODES:
        n_idx, _ = node_by_name[name]
        m_idx = gltf["nodes"][n_idx]["mesh"]
        s_id = next_shape_id
        next_shape_id += 1
        xml_s, mat_ids, nv, nt, bmin, bmax = convert_mesh_to_shape(
            gltf, bin_chunk, m_idx, s_id, world_mats[n_idx], mode="world"
        )
        shapes_xml_parts.append(xml_s)
        mesh_shape_info[m_idx] = (s_id, " ".join(map(str, mat_ids)), nv, nt)
        register_bbox(bmin, bmax)

    # Scene builder with deterministic nodeId and path tracking
    scene_lines: list[str] = []
    node_id_counter = 0

    def next_nid() -> int:
        nonlocal node_id_counter
        node_id_counter += 1
        return node_id_counter

    def wheel_shape_attrs(glb_node_name: str) -> tuple[int, str]:
        n_idx, _ = node_by_name[glb_node_name]
        m_idx = gltf["nodes"][n_idx]["mesh"]
        s_id, mats_str, _, _ = mesh_shape_info[m_idx]
        return s_id, mats_str

    fl_sid, fl_mats = wheel_shape_attrs("Wheel_FL")
    fr_sid, fr_mats = wheel_shape_attrs("Wheel_FR")
    rl_sid, rl_mats = wheel_shape_attrs("Wheel_RL")
    rr_sid, rr_mats = wheel_shape_attrs("Wheel_RR")

    # Build <Scene>
    scene_lines.append(
        f'    <Shape name="golG5_main_component1" nodeId="{next_nid()}" shapeId="1" materialIds="1" '
        f'dynamic="true" compound="true" collisionFilterGroup="0x10004" collisionFilterMask="0xfe3ffb83" '
        f'castsShadows="true" receiveShadows="true" nonRenderable="true" clipDistance="300">'
    )

    # 0>0 wheels
    scene_lines.append(f'      <TransformGroup name="wheels" nodeId="{next_nid()}">')
    scene_lines.append(
        f'        <TransformGroup name="wheelFrontLeftRepr" nodeId="{next_nid()}" translation="0.8059 0.3526 1.3431">'
    )
    scene_lines.append(
        f'          <Shape name="wheelFrontLeftDrive" nodeId="{next_nid()}" shapeId="{fl_sid}" materialIds="{fl_mats}" '
        f'castsShadows="true" receiveShadows="true" doubleSided="true" clipDistance="300"/>'
    )
    scene_lines.append("        </TransformGroup>")
    scene_lines.append(
        f'        <TransformGroup name="wheelFrontRightRepr" nodeId="{next_nid()}" translation="-0.8165 0.3526 1.3431">'
    )
    scene_lines.append(
        f'          <Shape name="wheelFrontRightDrive" nodeId="{next_nid()}" shapeId="{fr_sid}" materialIds="{fr_mats}" '
        f'castsShadows="true" receiveShadows="true" doubleSided="true" clipDistance="300"/>'
    )
    scene_lines.append("        </TransformGroup>")
    scene_lines.append(
        f'        <Shape name="wheelBackLeft" nodeId="{next_nid()}" shapeId="{rl_sid}" materialIds="{rl_mats}" '
        f'translation="0.8059 0.3526 -1.4167" castsShadows="true" receiveShadows="true" doubleSided="true" clipDistance="300"/>'
    )
    scene_lines.append(
        f'        <Shape name="wheelBackRight" nodeId="{next_nid()}" shapeId="{rr_sid}" materialIds="{rr_mats}" '
        f'translation="-0.8165 0.3526 -1.4167" castsShadows="true" receiveShadows="true" doubleSided="true" clipDistance="300"/>'
    )
    scene_lines.append("      </TransformGroup>")

    # 0>1 cameras
    scene_lines.append(f'      <TransformGroup name="cameras" nodeId="{next_nid()}">')
    scene_lines.append(
        f'        <TransformGroup name="outdoorCameraTarget" nodeId="{next_nid()}" translation="0 1.25 0" rotation="-18 180 0">'
    )
    scene_lines.append(
        f'          <Camera name="outdoorCamera1" nodeId="{next_nid()}" translation="0 0 5.8" '
        f'fov="54.4322" nearClip="0.3" farClip="5000"/>'
    )
    scene_lines.append("        </TransformGroup>")
    scene_lines.append(
        f'        <Camera name="indoorCamera" nodeId="{next_nid()}" translation="0.40 1.32 -0.05" '
        f'rotation="-12 180 0" fov="65" nearClip="0.1" farClip="5000"/>'
    )
    scene_lines.append(
        f'        <TransformGroup name="cameraRaycastNode1" nodeId="{next_nid()}" translation="0 1.25 2.05"/>'
    )
    scene_lines.append(
        f'        <TransformGroup name="cameraRaycastNode2" nodeId="{next_nid()}" translation="0 1.85 -0.10"/>'
    )
    scene_lines.append(
        f'        <TransformGroup name="cameraRaycastNode3" nodeId="{next_nid()}" translation="0 1.20 -2.05"/>'
    )
    scene_lines.append("      </TransformGroup>")

    # 0>2 lights
    scene_lines.append(f'      <TransformGroup name="lights" nodeId="{next_nid()}">')
    scene_lines.append(f'        <TransformGroup name="realLights" nodeId="{next_nid()}">')
    scene_lines.append(
        f'          <Light name="frontLightLow" nodeId="{next_nid()}" type="spot" translation="0 0.72 2.08" '
        f'rotation="-15 180 0" color="0.95 0.96 1.0" range="25" coneAngle="85" dropOff="2" '
        f'emitDiffuse="true" emitSpecular="true" castShadowMap="false"/>'
    )
    scene_lines.append(
        f'          <Light name="frontLightHigh" nodeId="{next_nid()}" type="spot" translation="0 0.72 2.08" '
        f'rotation="-10 180 0" color="1.0 0.98 0.95" range="55" coneAngle="100" dropOff="2" '
        f'emitDiffuse="true" emitSpecular="true" castShadowMap="false"/>'
    )
    scene_lines.append(
        f'          <Light name="backLightsHigh1" nodeId="{next_nid()}" type="spot" translation="0 0.78 -2.05" '
        f'rotation="-20 0 0" color="0.6 0.02 0.02" range="6" coneAngle="120" dropOff="2" '
        f'emitDiffuse="true" emitSpecular="false" castShadowMap="false"/>'
    )
    scene_lines.append(
        f'          <Light name="reverseLightHigh1" nodeId="{next_nid()}" type="spot" translation="0 0.72 -2.05" '
        f'rotation="-20 0 0" color="0.9 0.92 1.0" range="8" coneAngle="110" dropOff="2" '
        f'emitDiffuse="true" emitSpecular="false" castShadowMap="false"/>'
    )
    scene_lines.append("        </TransformGroup>")
    scene_lines.append(f'        <TransformGroup name="staticLights" nodeId="{next_nid()}">')
    for lname in STATIC_LIGHT_NODES:
        s_id, mats_str = wheel_shape_attrs(lname)
        scene_lines.append(
            f'          <Shape name="{escape(lname)}" nodeId="{next_nid()}" shapeId="{s_id}" '
            f'materialIds="{mats_str}" castsShadows="true" receiveShadows="true" doubleSided="true" clipDistance="300"/>'
        )
    scene_lines.append("        </TransformGroup>")
    scene_lines.append("      </TransformGroup>")

    # 0>3 character
    scene_lines.append(f'      <TransformGroup name="character" nodeId="{next_nid()}">')
    scene_lines.append(
        f'        <TransformGroup name="exitPoint" nodeId="{next_nid()}" translation="1.55 0.05 0.05"/>'
    )
    scene_lines.append(
        f'        <TransformGroup name="playerSkin" nodeId="{next_nid()}" translation="0.40 0.62 -0.10" rotation="0 0 0"/>'
    )
    scene_lines.append(
        f'        <TransformGroup name="playerRightFootTarget" nodeId="{next_nid()}" translation="0.24 0.34 0.42" rotation="0 -10 0"/>'
    )
    scene_lines.append(
        f'        <TransformGroup name="playerLeftFootTarget" nodeId="{next_nid()}" translation="0.56 0.34 0.42" rotation="0 10 0"/>'
    )
    scene_lines.append(
        f'        <TransformGroup name="steeringWheelBase" nodeId="{next_nid()}" '
        f'translation="0.4026 0.9492 0.5397" rotation="-65 0 0">'
    )
    sw_s_id, sw_mats_str = wheel_shape_attrs("steeringwheel")
    scene_lines.append(
        f'          <Shape name="steeringWheel" nodeId="{next_nid()}" shapeId="{sw_s_id}" '
        f'materialIds="{sw_mats_str}" castsShadows="true" receiveShadows="true" doubleSided="true" clipDistance="300">'
    )
    scene_lines.append(
        f'            <TransformGroup name="playerRightHandTarget" nodeId="{next_nid()}" translation="-0.155 0 0.02" rotation="0 0 -90"/>'
    )
    scene_lines.append(
        f'            <TransformGroup name="playerLeftHandTarget" nodeId="{next_nid()}" translation="0.155 0 0.02" rotation="0 0 90"/>'
    )
    scene_lines.append("          </Shape>")
    scene_lines.append("        </TransformGroup>")
    scene_lines.append("      </TransformGroup>")

    # 0>4 attacherJoints
    scene_lines.append(f'      <TransformGroup name="attacherJoints" nodeId="{next_nid()}">')
    scene_lines.append(
        f'        <TransformGroup name="trailerAttacherJointBackLow" nodeId="{next_nid()}" translation="0 0.38 -2.14" rotation="0 90 0"/>'
    )
    scene_lines.append("      </TransformGroup>")

    # 0>5 effects
    scene_lines.append(f'      <TransformGroup name="effects" nodeId="{next_nid()}">')
    scene_lines.append(
        f'        <TransformGroup name="exhaustParticle" nodeId="{next_nid()}" translation="0.48 0.28 -2.08" rotation="-80 0 0"/>'
    )
    scene_lines.append("      </TransformGroup>")

    # 0>6 visuals
    scene_lines.append(f'      <TransformGroup name="visuals" nodeId="{next_nid()}">')
    for vname in VISUAL_NODES:
        s_id, mats_str = wheel_shape_attrs(vname)
        scene_lines.append(
            f'        <Shape name="{escape(vname)}" nodeId="{next_nid()}" shapeId="{s_id}" '
            f'materialIds="{mats_str}" castsShadows="true" receiveShadows="true" doubleSided="true" clipDistance="300"/>'
        )
    scene_lines.append("      </TransformGroup>")
    # 0>7 fuel trigger
    scene_lines.append(
        f'      <Shape name="exactFillRootNodeFuel" nodeId="{next_nid()}" shapeId="1" materialIds="1" '
        f'translation="-0.75 0.78 -1.45" scale="0.25 0.25 0.25" kinematic="true" compound="true" '
        f'collisionFilterGroup="0x40000000" collisionFilterMask="0x20000000" castsShadows="true" '
        f'receiveShadows="true" nonRenderable="true"/>'
    )
    scene_lines.append("    </Shape>")

    materials_lines = build_materials_xml(gltf)

    i3d_text = "\n".join(
        [
            '<?xml version="1.0" encoding="iso-8859-1"?>',
            '<i3D name="golG5" version="1.6" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xsi:noNamespaceSchemaLocation="http://i3d.giants.ch/schema/i3d-1.6.xsd">',
            "  <Asset/>",
            "  <Files>",
            '    <File fileId="1" filename="textures/Tex_Taillight_Amber.dds"/>',
            '    <File fileId="2" filename="textures/Tex_Taillight_Emissive.dds"/>',
            '    <File fileId="3" filename="textures/Tex_Taillight_Ruby.dds"/>',
            '    <File fileId="4" filename="textures/Tex_Headlight_Emissive.dds"/>',
            '    <File fileId="5" filename="textures/Tex_Headlight_Reflector.dds"/>',
            '    <File fileId="6" filename="textures/Tex_Taillight_Crystal.dds"/>',
            '    <File fileId="7" filename="$data/shaders/vehicleShader.xml"/>',
            "  </Files>",
            "  <Materials>",
            *materials_lines,
            "  </Materials>",
            "  <Shapes>",
            *shapes_xml_parts,
            "  </Shapes>",
            "  <Dynamics/>",
            "  <Scene>",
            *scene_lines,
            "  </Scene>",
            "  <Animation/>",
            "  <UserAttributes/>",
            "</i3D>",
            "",
        ]
    )

    I3D_PATH.parent.mkdir(parents=True, exist_ok=True)
    I3D_PATH.write_bytes(i3d_text.encode("iso-8859-1"))

    # Parse back with lxml to verify well-formed XML and extract exact <Scene> paths dynamically
    tree = etree.parse(str(I3D_PATH))
    root = tree.getroot()
    scene_elem = root.find("Scene")
    assert scene_elem is not None

    mapped_paths: dict[str, str] = {}

    def walk_scene(elem: etree._Element, path_str: str) -> None:
        name = elem.get("name")
        if name:
            mapped_paths[name] = path_str
        for child_idx, child in enumerate(elem):
            child_path = f"{path_str}{child_idx}" if path_str.endswith(">") else f"{path_str}|{child_idx}"
            walk_scene(child, child_path)

    for top_idx, top_child in enumerate(scene_elem):
        walk_scene(top_child, f"{top_idx}>")

    # Select the 36 mapped nodes required by Contract Section 2.4
    contract_node_keys = [
        "golG5_main_component1",
        "wheelFrontLeftRepr",
        "wheelFrontLeftDrive",
        "wheelFrontRightRepr",
        "wheelFrontRightDrive",
        "wheelBackLeft",
        "wheelBackRight",
        "outdoorCameraTarget",
        "outdoorCamera1",
        "indoorCamera",
        "cameraRaycastNode1",
        "cameraRaycastNode2",
        "cameraRaycastNode3",
        "frontLightLow",
        "frontLightHigh",
        "backLightsHigh1",
        "reverseLightHigh1",
        "headlight_l",
        "headlight_r",
        "luxfarol",
        "brakelight_l",
        "brakelight_m",
        "brakelight_r",
        "reversinglight_l",
        "reversinglight_r",
        "lanternada",
        "exitPoint",
        "playerSkin",
        "playerRightFootTarget",
        "playerLeftFootTarget",
        "steeringWheelBase",
        "steeringWheel",
        "playerRightHandTarget",
        "playerLeftHandTarget",
        "trailerAttacherJointBackLow",
        "exhaustParticle",
        "exactFillRootNodeFuel",
    ]
    nodes_dict = {k: mapped_paths[k] for k in contract_node_keys}

    # Count scene-instance totals + unique shape totals directly from parsed I3D
    shape_counts: dict[str, tuple[int, int]] = {}
    for its in root.findall("./Shapes/IndexedTriangleSet"):
        sid = its.get("shapeId")
        vc = int(its.find("Vertices").get("count"))
        tc = int(its.find("Triangles").get("count"))
        shape_counts[sid] = (vc, tc)

    scene_shapes = root.findall(".//Scene//Shape")
    total_scene_verts = sum(shape_counts[s.get("shapeId")][0] for s in scene_shapes)
    total_scene_tris = sum(shape_counts[s.get("shapeId")][1] for s in scene_shapes)
    unique_verts = sum(v for v, _ in shape_counts.values())
    unique_tris = sum(t for _, t in shape_counts.values())

    node_map_payload = {
        "nodes": nodes_dict,
        "stats": {
            "source_glb": "source_assets/gol_g5.glb",
            "total_glb_nodes": len(gltf["nodes"]),
            "total_shapes": len(shape_counts),
            "total_materials": len(root.findall("./Materials/Material")),
            "total_textures": len(gltf.get("images", [])),
            "total_vertices": total_scene_verts,
            "total_triangles": total_scene_tris,
            "unique_shape_vertices": unique_verts,
            "unique_shape_triangles": unique_tris,
            "bbox_fs25_min": [round(float(v), 6) for v in global_min],
            "bbox_fs25_max": [round(float(v), 6) for v in global_max],
        },
    }
    NODE_MAP_PATH.write_text(json.dumps(node_map_payload, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {I3D_PATH.relative_to(ROOT_DIR).as_posix()} ({I3D_PATH.stat().st_size:,} bytes)")
    print(f"Wrote {NODE_MAP_PATH.relative_to(ROOT_DIR).as_posix()}")
    print(json.dumps(node_map_payload["stats"], indent=2))


if __name__ == "__main__":
    main()
