import os
import math
import xml.etree.ElementTree as ET

# === GENERAL SETTINGS ===
layer_folder = "C:/Users/hamza/Documents/World_Smith/Terra/Layers" # (change with yours)
planet_radius = 6171.83  # in km (change with yours)

# Namespaces for parsing XML
ns = {
    'gpml': 'http://www.gplates.org/gplates',
    'gml': 'http://www.opengis.net/gml'
}
ET.register_namespace('gpml', ns['gpml'])
ET.register_namespace('gml', ns['gml'])

# === SPHERICAL DISTANCE UTILITIES ===
def haversine(lon1, lat1, lon2, lat2):
    R = planet_radius
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def to_cartesian(lon, lat):
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    x = math.cos(lat_rad) * math.cos(lon_rad)
    y = math.cos(lat_rad) * math.sin(lon_rad)
    z = math.sin(lat_rad)
    return (x, y, z)

def dot(u, v): return sum(ux * vx for ux, vx in zip(u, v))
def norm(v): return math.sqrt(dot(v, v))
def cross(u, v): return (u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2], u[0]*v[1] - u[1]*v[0])
def angle_between(u, v):
    val = dot(u, v) / (norm(u) * norm(v))
    val = max(-1.0, min(1.0, val))
    return math.acos(val)

# Compute the shortest spherical distance between point P and segment AB
def point_segment_distance(P, A, B):
    A3, B3, P3 = to_cartesian(*A), to_cartesian(*B), to_cartesian(*P)
    n_AB = cross(A3, B3)
    if norm(n_AB) == 0:
        return angle_between(P3, A3) * planet_radius
    proj = cross(n_AB, cross(P3, n_AB))
    proj_norm = norm(proj)
    if proj_norm == 0:
        return math.pi * planet_radius
    proj_unit = tuple(x / proj_norm for x in proj)
    angle_AB = angle_between(A3, B3)
    if angle_between(A3, proj_unit) <= angle_AB and angle_between(B3, proj_unit) <= angle_AB:
        return angle_between(P3, proj_unit) * planet_radius
    return min(angle_between(P3, A3), angle_between(P3, B3)) * planet_radius

# === GPML PARSER ===
def read_coord_blocks(gpml_file):
    tree = ET.parse(gpml_file)
    root = tree.getroot()
    blocks = []

    for feature in root.findall('.//gml:featureMember/*', ns):
        plate_elem = feature.find('.//gpml:reconstructionPlateId/gpml:ConstantValue/gpml:value', ns)
        plate_id = int(plate_elem.text) if plate_elem is not None else None

        time_elem = feature.find('.//gpml:geometryImportTime/gml:TimeInstant/gml:timePosition', ns)
        time = float(time_elem.text) if time_elem is not None else None

        name_elem = feature.find('./gml:name', ns)
        name = name_elem.text.strip() if name_elem is not None else None

        type_tag = feature.tag.split('}')[-1]  # e.g. OceanicCrust

        pos = feature.find('.//gml:posList', ns)
        if pos is None or pos.text is None:
            continue
        txt = pos.text.strip()
        vals = list(map(float, txt.split()))
        coords = list(zip(vals[::2], vals[1::2]))

        blocks.append({
            "plate_id": plate_id,
            "time": time,
            "name": name,
            "type": type_tag,
            "pos_elem": pos,
            "coords": coords
        })

    return tree, blocks

# === COORDINATE UTILITIES ===

# Sort points projected along segment AB
def sort_points_along_segment(points, A, B):
    x1, y1, x2, y2 = *A, *B
    dx, dy = x2 - x1, y2 - y1
    def proj(p): return ((p[0] - x1)*dx + (p[1] - y1)*dy) / (dx*dx + dy*dy)
    return sorted(points, key=proj)

# Remove duplicate coordinates (up to 6 decimal places)
def deduplicate(coords):
    seen, output = set(), []
    for lon, lat in coords:
        key = (round(lon, 6), round(lat, 6))
        if key not in seen:
            seen.add(key)
            output.append((lon, lat))
    return output

# === SNAPPING FUNCTION ===
def snap_vertices(reference_coords, modif_coords, tolerance_km):
    new_coords_list = []
    modifications = []
    for lon, lat in modif_coords:
        nearest = min(reference_coords, key=lambda p: haversine(lon, lat, p[0], p[1]))
        dist = haversine(lon, lat, nearest[0], nearest[1])
        if dist <= tolerance_km:
            new_coords_list.append(nearest)
            modifications.append(((lon, lat), nearest, dist))
        else:
            new_coords_list.append((lon, lat))
    return new_coords_list, modifications

# === MODELLING FUNCTION ===
def inject_target_as_vertices(target_coords, shape_coords, corridor_half_width_km):
    resultats = []
    for i in range(len(shape_coords) - 1):
        A, B = shape_coords[i], shape_coords[i + 1]
        hits = [(P, point_segment_distance(P, A, B)) for P in target_coords if point_segment_distance(P, A, B) <= corridor_half_width_km]
        if hits:
            resultats.append({'segment': (A, B), 'vertex_detected': hits})

    new_coords_forme = []
    i_segment = 0
    for i in range(len(shape_coords) - 1):
        A, B = shape_coords[i], shape_coords[i + 1]
        new_coords_forme.append(A)
        if i_segment < len(resultats) and resultats[i_segment]['segment'] == (A, B):
            insertions = [coord for coord, _ in resultats[i_segment]['vertex_detected']]
            for p in sort_points_along_segment(insertions, A, B):
                if p != A and p != B:
                    new_coords_forme.append(p)
            i_segment += 1
    new_coords_forme.append(shape_coords[-1])
    return deduplicate(new_coords_forme), resultats

# === ENTITY SELECTION HELPER ===
def select_entity_interactively(blocks, label=""):
    # Step 1: Plate ID
    ids = sorted(set(str(b["plate_id"]) for b in blocks if b["plate_id"] is not None))
    print(f"\n🧭 Available Plate IDs {label}:")
    print(", ".join(ids))
    chosen_id = input("Your choice: ").strip()

    matching_id = [b for b in blocks if str(b["plate_id"]) == chosen_id]
    if not matching_id:
        print("❌ No entities with that Plate ID.")
        return None

    # Step 2: Time
    times = sorted(set(str(int(b["time"])) for b in matching_id if b["time"] is not None))
    print(f"\n🕒 Available times for Plate ID {chosen_id}:")
    print(", ".join(times))
    chosen_time = input("Your choice (Ma): ").strip()

    matching_time = [b for b in matching_id if str(int(b["time"])) == chosen_time]
    if not matching_time:
        print("❌ No entities with that time.")
        return None

    # Step 3: Name (only if multiple remain)
    if len(matching_time) > 1:
        names = sorted(set(b["name"] for b in matching_time if b["name"]))
        print(f"\n🏷️ Multiple entities found. Available names:")
        print(", ".join(names))
        chosen_name = input("Enter exact name: ").strip()
        matching_name = [b for b in matching_time if b["name"] == chosen_name]
        if not matching_name:
            print("❌ No entity with that name.")
            return None
        return matching_name[0]
    else:
        return matching_time[0]

# === MAIN MENU ===
def main():
    while True:
        print("\n=== GPML TOOL MENU ===")
        print("1. Snap vertices to target")
        print("2. Insert vertices along segments")
        print("Q. Quit")
        choice = input("Choice (1/2/Q): ").strip().lower()

        if choice == "q":
            print("👋 Exiting.")
            break
        elif choice not in ["1", "2"]:
            print("❌ Invalid choice.")
            continue

        # --- Modification file ---
        modif_file = input("Enter file to modify (without extension): ").strip() + ".gpml"
        mod_path = os.path.join(layer_folder, modif_file)
        tree_mod, blocks_mod = read_coord_blocks(mod_path)

        if not blocks_mod:
            print("❌ No blocks found in the file.")
            continue

        print(f"\n🔧 Select the entity to MODIFY in {modif_file}:")
        block_to_modify = select_entity_interactively(blocks_mod)
        if not block_to_modify:
            continue

        # --- Target file ---
        target_file = input("\nEnter target file (without extension): ").strip() + ".gpml"
        target_path = os.path.join(layer_folder, target_file)
        tree_target, blocks_target = read_coord_blocks(target_path)

        if not blocks_target:
            print("❌ No target blocks found.")
            continue

        if target_file == modif_file:
            print(f"\n🎯 Select the TARGET entity in the same file {modif_file}:")
            block_target = select_entity_interactively(blocks_target)
            if not block_target:
                continue
            coords_target = block_target["coords"]
        else:
            coords_target = [pt for b in blocks_target for pt in b["coords"]]

        param = float(input("\nTolerance (km) or corridor half-width: ").strip())

        if choice == "1":
            new_coords, _ = snap_vertices(coords_target, block_to_modify["coords"], param)
        else:
            new_coords, _ = inject_target_as_vertices(coords_target, block_to_modify["coords"], param)

        block_to_modify["pos_elem"].text = ' '.join(f"{x} {y}" for x, y in new_coords)
        tree_mod.write(mod_path, encoding="utf-8", xml_declaration=True)
        print(f"✅ Changes saved to '{modif_file}'.")

if __name__ == "__main__":
    main()

