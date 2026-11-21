"""Comprehensive 7-point validator for FS25_VW_Gol_G5."""

from __future__ import annotations

import math
import os
import re
import struct
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import lxml.etree as LET
import xmlschema

ROOT_DIR = Path(__file__).resolve().parent.parent
MOD_DIR = ROOT_DIR / "FS25_VW_Gol_G5"
DIST_ZIP = ROOT_DIR / "dist" / "FS25_VW_Gol_G5.zip"
GITIGNORE_PATH = ROOT_DIR / ".gitignore"

MODDESC_XSD_URL = "https://validation.gdn.giants-software.com/xml/fs25/modDesc.xsd"
VEHICLE_XSD_URL = "https://validation.gdn.giants-software.com/xml/fs25/vehicle.xsd"

EXPECTED_DDS = {
    "icon_golG5.dds": (256, 256),
    "store_golG5.dds": (512, 512),
    "brand_vw.dds": (512, 256),
    "vehicles/vwGolG5/textures/Tex_Taillight_Amber.dds": (512, 512),
    "vehicles/vwGolG5/textures/Tex_Taillight_Emissive.dds": (512, 512),
    "vehicles/vwGolG5/textures/Tex_Taillight_Ruby.dds": (512, 512),
    "vehicles/vwGolG5/textures/Tex_Headlight_Emissive.dds": (512, 512),
    "vehicles/vwGolG5/textures/Tex_Headlight_Reflector.dds": (512, 512),
    "vehicles/vwGolG5/textures/Tex_Taillight_Crystal.dds": (512, 512),
}

CARPAINT_SHAPES = {
    "bodyshell",
    "bonnet",
    "boot",
    "bumper_f",
    "bumper_r",
    "door_dside_f",
    "door_pside_f",
    "door_dside_r",
    "door_pside_r",
}

BUILTIN_L10N_KEYS = {"configuration_valueDefault"}
NODE_REF_ATTRS = ("node", "repr", "driveNode", "rotateNode", "linkNode", "shaderNode", "targetNode")


def get_cached_xsd(filename: str, url: str) -> Path:
    cache_dir = Path(tempfile.gettempdir()) / "fs25_xsd_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / filename
    if not target.is_file() or target.stat().st_size == 0:
        print(f"  Downloading {url} -> {target}")
        urllib.request.urlretrieve(url, target)
    return target


def check_1_xsd() -> list[str]:
    errors: list[str] = []
    moddesc_xml = MOD_DIR / "modDesc.xml"
    vehicle_xml = MOD_DIR / "vehicles" / "vwGolG5" / "golG5.xml"

    if not moddesc_xml.is_file():
        return [f"Missing {moddesc_xml}"]
    if not vehicle_xml.is_file():
        return [f"Missing {vehicle_xml}"]

    try:
        moddesc_xsd = get_cached_xsd("modDesc.xsd", MODDESC_XSD_URL)
        vehicle_xsd = get_cached_xsd("vehicle.xsd", VEHICLE_XSD_URL)
    except Exception as exc:
        return [f"Failed to fetch XSD schemas: {exc}"]

    for label, xml_path, xsd_path in [
        ("modDesc.xml", moddesc_xml, moddesc_xsd),
        ("golG5.xml", vehicle_xml, vehicle_xsd),
    ]:
        try:
            lxml_schema = LET.XMLSchema(LET.parse(str(xsd_path)))
            lxml_schema.assertValid(LET.parse(str(xml_path)))
        except Exception as exc:
            errors.append(f"lxml XSD validation failed for {label}: {exc}")

        try:
            xs = xmlschema.XMLSchema(str(xsd_path))
            xs.validate(str(xml_path))
        except Exception as exc:
            errors.append(f"xmlschema validation failed for {label}: {exc}")

    return errors


def check_2_l10n() -> list[str]:
    errors: list[str] = []
    en_path = MOD_DIR / "l10n" / "l10n_en.xml"
    br_path = MOD_DIR / "l10n" / "l10n_br.xml"
    vehicle_xml = MOD_DIR / "vehicles" / "vwGolG5" / "golG5.xml"

    keys_by_lang: dict[str, set[str]] = {}
    for lang, path in [("en", en_path), ("br", br_path)]:
        if not path.is_file():
            errors.append(f"Missing {path}")
            continue
        try:
            tree = LET.parse(str(path))
            keys = set()
            for txt in tree.findall(".//texts/text"):
                name = txt.get("name")
                val = txt.get("text")
                if not name or val is None:
                    errors.append(f"Invalid <text> entry in {path.name}")
                else:
                    keys.add(name)
            keys_by_lang[lang] = keys
        except Exception as exc:
            errors.append(f"Malformed XML in {path.name}: {exc}")

    if len(keys_by_lang) == 2:
        en_keys, br_keys = keys_by_lang["en"], keys_by_lang["br"]
        if en_keys != br_keys:
            only_en = sorted(en_keys - br_keys)
            only_br = sorted(br_keys - en_keys)
            if only_en:
                errors.append(f"Keys only in l10n_en.xml: {only_en}")
            if only_br:
                errors.append(f"Keys only in l10n_br.xml: {only_br}")

        if vehicle_xml.is_file():
            content = vehicle_xml.read_text(encoding="utf-8")
            used_keys = set(re.findall(r"\$l10n_([A-Za-z0-9_]+)", content))
            missing = sorted(k for k in used_keys if k not in BUILTIN_L10N_KEYS and (k not in en_keys or k not in br_keys))
            if missing:
                errors.append(f"Missing $l10n keys used in golG5.xml: {missing}")

    return errors


def build_i3d_scene_index(scene_elem: LET._Element) -> tuple[dict[str, str], dict[str, LET._Element]]:
    """Return mapping of index path ('0>', '0>0', '0>0|0', ...) -> node name and element."""
    paths: dict[str, str] = {}
    elems: dict[str, LET._Element] = {}

    def walk(node: LET._Element, path: str) -> None:
        name = node.get("name", "")
        paths[path] = name
        elems[path] = node
        children = [c for c in node if isinstance(c.tag, str)]
        for idx, child in enumerate(children):
            child_path = f"{path}{idx}" if path.endswith(">") else f"{path}|{idx}"
            walk(child, child_path)

    top_children = [c for c in scene_elem if isinstance(c.tag, str)]
    for top_idx, top_node in enumerate(top_children):
        walk(top_node, f"{top_idx}>")

    return paths, elems


def check_3_i3d() -> tuple[list[str], dict[str, str], set[str]]:
    errors: list[str] = []
    scene_paths: dict[str, str] = {}
    shader_shape_paths: set[str] = set()
    i3d_path = MOD_DIR / "vehicles" / "vwGolG5" / "golG5.i3d"

    if not i3d_path.is_file():
        return [f"Missing {i3d_path}"], scene_paths, shader_shape_paths

    try:
        doc = LET.parse(str(i3d_path))
    except Exception as exc:
        return [f"Malformed I3D XML: {exc}"], scene_paths, shader_shape_paths

    root = doc.getroot()
    if root.tag != "i3D":
        errors.append(f"Unexpected I3D root tag: {root.tag}")
    if root.get("version") != "1.6":
        errors.append(f"Expected I3D version='1.6', got {root.get('version')!r}")

    file_ids: set[str] = set()
    for f in root.findall("./Files/File"):
        fid = f.get("fileId")
        if fid:
            file_ids.add(fid)

    material_ids: set[str] = set()
    shader_material_ids: set[str] = set()
    carpaint_mat_id: str | None = None
    for mat in root.findall("./Materials/Material"):
        mid = mat.get("materialId")
        mname = mat.get("name", "")
        if mid:
            material_ids.add(mid)
        if mname == "Mat_CarPaint_Black":
            carpaint_mat_id = mid
        csid = mat.get("customShaderId")
        if csid:
            if csid not in file_ids:
                errors.append(f"Material {mname} ({mid}) references missing customShaderId={csid}")
            elif mid:
                shader_material_ids.add(mid)
        for tex_tag in ("Texture", "Emissivemap", "Normalmap", "Glossmap"):
            for child in mat.findall(f"./{tex_tag}"):
                tfid = child.get("fileId")
                if tfid and tfid not in file_ids:
                    errors.append(f"Material {mname} ({mid}) <{tex_tag}> references missing fileId={tfid}")

    shapes_by_id: dict[str, LET._Element] = {}
    for its in root.findall("./Shapes/IndexedTriangleSet"):
        sid = its.get("shapeId")
        sname = its.get("name", sid or "<unnamed>")
        if not sid:
            errors.append(f"IndexedTriangleSet {sname} missing shapeId")
            continue
        shapes_by_id[sid] = its

        verts_el = its.find("./Vertices")
        tris_el = its.find("./Triangles")
        subsets_el = its.find("./Subsets")
        if verts_el is None or tris_el is None or subsets_el is None:
            errors.append(f"IndexedTriangleSet {sname} missing Vertices/Triangles/Subsets")
            continue

        v_list = verts_el.findall("./v")
        t_list = tris_el.findall("./t")
        s_list = subsets_el.findall("./Subset")

        v_count_attr = int(verts_el.get("count", -1))
        t_count_attr = int(tris_el.get("count", -1))
        s_count_attr = int(subsets_el.get("count", -1))

        if v_count_attr != len(v_list):
            errors.append(f"{sname}: Vertices count={v_count_attr} != {len(v_list)} <v> tags")
        if t_count_attr != len(t_list):
            errors.append(f"{sname}: Triangles count={t_count_attr} != {len(t_list)} <t> tags")
        if s_count_attr != len(s_list):
            errors.append(f"{sname}: Subsets count={s_count_attr} != {len(s_list)} <Subset> tags")

        if sname == "Headlight_Optics_Detail" and verts_el.get("uv0") is not None:
            errors.append("Headlight_Optics_Detail <Vertices> must NOT have uv0 attribute")

        for sub in s_list:
            fv = int(sub.get("firstVertex", 0))
            nv = int(sub.get("numVertices", 0))
            fi = int(sub.get("firstIndex", 0))
            ni = int(sub.get("numIndices", 0))
            if fv < 0 or nv < 0 or fv + nv > len(v_list):
                errors.append(f"{sname}: subset vertex range [{fv}, {fv + nv}) exceeds {len(v_list)}")
            if fi < 0 or ni < 0 or fi + ni > len(t_list) * 3:
                errors.append(f"{sname}: subset index range [{fi}, {fi + ni}) exceeds {len(t_list) * 3}")

    scene_el = root.find("./Scene")
    if scene_el is None:
        errors.append("Missing <Scene> element in golG5.i3d")
    else:
        scene_paths, scene_elems = build_i3d_scene_index(scene_el)

        for path_str, elem in scene_elems.items():
            if elem.tag != "Shape":
                continue
            shape = elem
            sname = shape.get("name", "<unnamed>")
            sid = shape.get("shapeId")
            mids_raw = shape.get("materialIds", "")
            mids = [m.strip() for m in mids_raw.replace(",", " ").split() if m.strip()]

            if not sid or sid not in shapes_by_id:
                errors.append(f"Shape {sname} references missing shapeId={sid}")
                continue
            if not mids:
                errors.append(f"Shape {sname} has empty materialIds")
            for mid in mids:
                if mid not in material_ids:
                    errors.append(f"Shape {sname} references missing materialId={mid}")

            if any(mid in shader_material_ids for mid in mids):
                shader_shape_paths.add(path_str)

            if sname in CARPAINT_SHAPES:
                if carpaint_mat_id is None:
                    errors.append("Mat_CarPaint_Black not found in <Materials>")
                else:
                    its = shapes_by_id[sid]
                    subsets = its.findall("./Subsets/Subset")
                    found_carpaint_subset = False
                    for idx, mid in enumerate(mids):
                        if mid == carpaint_mat_id:
                            if idx >= len(subsets):
                                errors.append(f"Shape {sname}: materialIds index {idx} has no matching Subset")
                            elif subsets[idx].get("materialSlotName") != "carPaint":
                                errors.append(
                                    f"Shape {sname} subset {idx} (Mat_CarPaint_Black) missing materialSlotName='carPaint'"
                                )
                            else:
                                found_carpaint_subset = True
                    if not found_carpaint_subset:
                        errors.append(f"Car-paint shape {sname} does not use Mat_CarPaint_Black with materialSlotName='carPaint'")

    return errors, scene_paths, shader_shape_paths


def check_4_i3d_mappings(scene_paths: dict[str, str], shader_shape_paths: set[str]) -> list[str]:
    errors: list[str] = []
    vehicle_xml = MOD_DIR / "vehicles" / "vwGolG5" / "golG5.xml"
    if not vehicle_xml.is_file():
        return [f"Missing {vehicle_xml}"]

    doc = LET.parse(str(vehicle_xml))
    root = doc.getroot()

    mappings: dict[str, str] = {}
    for m in root.findall("./i3dMappings/i3dMapping"):
        mid = m.get("id")
        node = m.get("node")
        if not mid or not node:
            errors.append("Invalid <i3dMapping> without id or node")
            continue
        if mid in mappings:
            errors.append(f"Duplicate <i3dMapping id='{mid}'>")
        mappings[mid] = node

    if not mappings:
        errors.append("No <i3dMapping> entries found in golG5.xml")
        return errors

    for mid, node_path in mappings.items():
        if node_path not in scene_paths:
            errors.append(f"i3dMapping id='{mid}' node='{node_path}' does not exist in golG5.i3d <Scene>")

    for elem in root.iter():
        if elem.tag == "i3dMapping":
            continue
        for attr in NODE_REF_ATTRS:
            val = elem.get(attr)
            if val is not None and val not in mappings:
                errors.append(f"<{elem.tag} {attr}='{val}'> is not defined in <i3dMappings>")

    lights_el = root.find("./lights")
    if lights_el is not None:
        for elem in lights_el.iter():
            snode = elem.get("shaderNode")
            if snode is not None and snode in mappings:
                node_path = mappings[snode]
                if node_path not in shader_shape_paths:
                    errors.append(
                        f"<{elem.tag} shaderNode='{snode}'> ({node_path}) does not use a material with customShaderId (vehicleShader.xml)"
                    )

    return errors


def check_5_dds() -> list[str]:
    errors: list[str] = []
    for rel_path, (exp_w, exp_h) in EXPECTED_DDS.items():
        path = MOD_DIR / rel_path
        if not path.is_file():
            errors.append(f"Missing DDS file: {rel_path}")
            continue

        data = path.read_bytes()
        if len(data) < 128:
            errors.append(f"{rel_path}: size {len(data)} < 128 bytes")
            continue

        magic = data[:4]
        if magic != b"DDS ":
            errors.append(f"{rel_path}: invalid magic {magic!r}")
            continue

        dw_size, _dw_flags, height, width, _pitch, _depth, mip_count = struct.unpack_from("<7I", data, 4)
        fourcc = data[84:88]

        if dw_size != 124:
            errors.append(f"{rel_path}: invalid DDS_HEADER dwSize={dw_size} (expected 124)")
        if (width, height) != (exp_w, exp_h):
            errors.append(f"{rel_path}: dimensions {width}x{height} != expected {exp_w}x{exp_h}")
        if (width & (width - 1)) != 0 or (height & (height - 1)) != 0:
            errors.append(f"{rel_path}: dimensions {width}x{height} not power of 2")
        if fourcc != b"DXT5":
            errors.append(f"{rel_path}: FourCC={fourcc!r} != b'DXT5'")

        exp_mips = int(math.floor(math.log2(max(exp_w, exp_h)))) + 1
        if mip_count != exp_mips:
            errors.append(f"{rel_path}: mipMapCount={mip_count} != expected {exp_mips}")

        expected_payload = 0
        w, h = width, height
        for _ in range(mip_count):
            expected_payload += max(1, w // 4) * max(1, h // 4) * 16
            w = max(1, w // 2)
            h = max(1, h // 2)
        expected_total = 128 + expected_payload
        if len(data) != expected_total:
            errors.append(f"{rel_path}: file size {len(data)} != expected {expected_total}")

    return errors


def check_6_zip() -> list[str]:
    errors: list[str] = []
    if not DIST_ZIP.is_file():
        return [f"Missing ZIP file: {DIST_ZIP}"]

    try:
        with zipfile.ZipFile(DIST_ZIP, "r") as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                errors.append(f"Corrupt ZIP entry: {bad_member}")

            names = set(zf.namelist())
            if "modDesc.xml" not in names:
                errors.append("modDesc.xml is not at the root of the ZIP archive")

            # Collect referenced mod-local files from modDesc.xml, golG5.xml, golG5.i3d
            required_files: set[str] = {"modDesc.xml"}

            if "modDesc.xml" in names:
                md_root = LET.fromstring(zf.read("modDesc.xml"))
                icon = md_root.findtext("./iconFilename")
                if icon:
                    required_files.add(icon.strip())
                for brand in md_root.findall("./brands/brand"):
                    img = brand.get("image")
                    if img:
                        required_files.add(img.strip())
                for si in md_root.findall("./storeItems/storeItem"):
                    xf = si.get("xmlFilename")
                    if xf:
                        required_files.add(xf.strip())
                l10n_el = md_root.find("./l10n")
                if l10n_el is not None and l10n_el.get("filenamePrefix"):
                    prefix = l10n_el.get("filenamePrefix").strip()
                    required_files.add(f"{prefix}_en.xml")
                    required_files.add(f"{prefix}_br.xml")

            vehicle_rel = "vehicles/vwGolG5/golG5.xml"
            if vehicle_rel in names:
                v_root = LET.fromstring(zf.read(vehicle_rel))
                img = v_root.findtext("./storeData/image")
                if img and not img.strip().startswith("$data"):
                    required_files.add(img.strip())
                i3d_fn = v_root.findtext("./base/filename")
                if i3d_fn and not i3d_fn.strip().startswith("$data"):
                    required_files.add(i3d_fn.strip())
                for el in v_root.iter():
                    fn = el.get("filename")
                    if fn and not fn.strip().startswith("$data"):
                        required_files.add(fn.strip())

            i3d_rel = "vehicles/vwGolG5/golG5.i3d"
            if i3d_rel in names:
                i3d_root = LET.fromstring(zf.read(i3d_rel))
                for f in i3d_root.findall("./Files/File"):
                    fn = f.get("filename")
                    if fn and not fn.strip().startswith("$data"):
                        resolved = (Path("vehicles/vwGolG5") / fn.strip()).as_posix()
                        required_files.add(resolved)

            for req in sorted(required_files):
                if req not in names:
                    errors.append(f"Referenced file missing from ZIP: {req}")

    except Exception as exc:
        errors.append(f"Failed to inspect ZIP: {exc}")

    return errors


def check_7_gitignore() -> list[str]:
    errors: list[str] = []
    if not GITIGNORE_PATH.is_file():
        return [f"Missing {GITIGNORE_PATH}"]

    lines = {
        line.strip()
        for line in GITIGNORE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    required = {
        "GOLG5_FS25_PROJETO_COMPLETO.md",
        ".agents/",
        "source_assets/",
        "dist/",
        "__pycache__/",
    }
    missing = sorted(required - lines)
    if missing:
        errors.append(f".gitignore missing required entries: {missing}")

    return errors


def main() -> int:
    all_failed = False

    print("Running FS25_VW_Gol_G5 7-point validation...\n")

    err1 = check_1_xsd()
    err2 = check_2_l10n()
    err3, scene_paths, shader_shape_paths = check_3_i3d()
    err4 = check_4_i3d_mappings(scene_paths, shader_shape_paths) if scene_paths else ["Skipped/Failed because I3D scene could not be parsed"]
    err5 = check_5_dds()
    err6 = check_6_zip()
    err7 = check_7_gitignore()

    checks = [
        ("1. Official FS25 XSD (modDesc.xsd & vehicle.xsd)", err1),
        ("2. Localization (l10n_en / l10n_br parity & refs)", err2),
        ("3. I3D v1.6 Integrity (files, materials, shapes, carPaint)", err3),
        ("4. i3dMappings Resolution (golG5.xml <-> golG5.i3d)", err4),
        ("5. Binary DDS Validation (9x DXT5 + full mipmaps)", err5),
        ("6. Mod ZIP Package (dist/FS25_VW_Gol_G5.zip)", err6),
        ("7. Git Ignore Policy (.gitignore)", err7),
    ]

    for title, errs in checks:
        if errs:
            all_failed = True
            print(f"[FAIL] {title}")
            for e in errs:
                print(f"       - {e}")
        else:
            print(f"[PASS] {title}")

    print()
    if all_failed:
        print("RESULT: FAIL")
        return 1
    print("RESULT: ALL 7 CHECKS PASSED (100%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
