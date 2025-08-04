import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import math

# === CONFIGURATION ===
gpml_folder = r"C:\Users\hamza\Documents\World_Smith\Gplate_helper\Layers"
planet_radius_km = 6378.14

ns = {
    'gpml': 'http://www.gplates.org/gplates',
    'gml': 'http://www.opengis.net/gml'
}
ET.register_namespace('gpml', ns['gpml'])
ET.register_namespace('gml', ns['gml'])

# === UTILITIES ===
def list_gpml_files(folder):
    return [f for f in os.listdir(folder) if f.endswith(".gpml")]

def user_select_menu(options, prompt):
    print(prompt)
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    while True:
        try:
            choice = int(input("\nEnter number: ")) - 1
            if 0 <= choice < len(options):
                return options[choice]
        except ValueError:
            pass
        print("Invalid choice. Try again.")

def parse_gpml(filepath):
    tree = ET.parse(filepath)
    return tree, tree.getroot()

def extract_entities(root):
    entities = []
    for member in root.findall('.//gml:featureMember', ns):
        for feature in member:
            plate_elem = feature.find('.//gpml:reconstructionPlateId/gpml:ConstantValue/gpml:value', ns)
            name_elem = feature.find('gml:name', ns)
            times = [tp.text.strip() for tp in feature.findall('.//gml:timePosition', ns)]
            if plate_elem is not None:
                plate_id = plate_elem.text.strip()
                name = name_elem.text.strip() if name_elem is not None else "Unnamed"
                entities.append({
                    "element": feature,
                    "plate_id": plate_id,
                    "name": name,
                    "times": times
                })
    return entities

def extract_coords(feature):
    coords = []
    for ring in feature.findall('.//gml:LinearRing', ns):
        for pos_list in ring.findall('.//gml:posList', ns):
            values = list(map(float, pos_list.text.strip().split()))
            for i in range(0, len(values), 2):
                coords.append((values[i], values[i + 1]))
    return coords

def update_coords(feature, new_coords):
    for ring in feature.findall('.//gml:LinearRing', ns):
        for pos_list in ring.findall('.//gml:posList', ns):
            pos_list.text = ' '.join(f"{x} {y}" for x, y in new_coords)

def select_entity(entities):
    unique_ids = sorted(set(e["plate_id"] for e in entities))
    selected_id = user_select_menu(unique_ids, "\nAvailable Plate IDs:")
    filtered_by_id = [e for e in entities if e["plate_id"] == selected_id]

    unique_times = sorted({t for e in filtered_by_id for t in e["times"] if t.isdigit()}, key=float, reverse=True)
    selected_time = user_select_menu(unique_times, f"\nAvailable time positions for ID {selected_id}:")
    filtered_by_time = [e for e in filtered_by_id if selected_time in e["times"]]

    if len(filtered_by_time) > 1:
        names = [e["name"] for e in filtered_by_time]
        selected_name = user_select_menu(names, f"\nAvailable names for ID {selected_id} at time {selected_time}:")
        selected_entity = next(e for e in filtered_by_time if e["name"] == selected_name)
    else:
        selected_entity = filtered_by_time[0]

    return selected_entity, selected_time

# === SNAPPING ===
def haversine(lon1, lat1, lon2, lat2, radius_km):
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return radius_km * 2 * math.asin(math.sqrt(a))

def snap_vertices(source_coords, target_coords, tolerance_km, radius_km):
    snapped_coords = []
    modifications = []
    for lon, lat in source_coords:
        nearest = min(target_coords, key=lambda p: haversine(lon, lat, p[0], p[1], radius_km))
        dist = haversine(lon, lat, nearest[0], nearest[1], radius_km)
        if dist <= tolerance_km:
            snapped_coords.append(nearest)
            modifications.append(((lon, lat), nearest, dist))
        else:
            snapped_coords.append((lon, lat))
    return snapped_coords, modifications

# === SHAPING ===
def to_cartesian(lon, lat):
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    x = math.cos(lat_rad) * math.cos(lon_rad)
    y = math.cos(lat_rad) * math.sin(lon_rad)
    z = math.sin(lat_rad)
    return (x, y, z)

def dot(u, v):
    return sum(ux * vx for ux, vx in zip(u, v))

def norm(v):
    return math.sqrt(dot(v, v))

def cross(u, v):
    return (
        u[1]*v[2] - u[2]*v[1],
        u[2]*v[0] - u[0]*v[2],
        u[0]*v[1] - u[1]*v[0]
    )

def angle_between(u, v):
    dot_product = dot(u, v) / (norm(u) * norm(v))
    dot_product = max(-1.0, min(1.0, dot_product))
    return math.acos(dot_product)

def point_segment_distance(P, A, B):
    A3 = to_cartesian(*A)
    B3 = to_cartesian(*B)
    P3 = to_cartesian(*P)

    n_AB = cross(A3, B3)
    if norm(n_AB) == 0:
        return math.acos(dot(P3, A3)) * planet_radius_km

    proj = cross(n_AB, cross(P3, n_AB))
    proj_norm = norm(proj)
    if proj_norm == 0:
        return math.pi * planet_radius_km

    proj_unit = tuple(x / proj_norm for x in proj)

    angle_AB = angle_between(A3, B3)
    angle_AProj = angle_between(A3, proj_unit)
    angle_BProj = angle_between(B3, proj_unit)

    if angle_AProj <= angle_AB and angle_BProj <= angle_AB:
        return angle_between(P3, proj_unit) * planet_radius_km
    else:
        return min(angle_between(P3, A3), angle_between(P3, B3)) * planet_radius_km

def sort_points_along_segment(points, A, B):
    x1, y1 = A
    x2, y2 = B
    dx = x2 - x1
    dy = y2 - y1
    def scalar_proj(p):
        xp, yp = p
        return ((xp - x1) * dx + (yp - y1) * dy) / (dx * dx + dy * dy)
    return sorted(points, key=scalar_proj)

def deduplicate(coords):
    seen = set()
    output = []
    for lon, lat in coords:
        key = (round(lon, 6), round(lat, 6))
        if key not in seen:
            seen.add(key)
            output.append((lon, lat))
    return output

def shape_entity(source_coords, target_coords, corridor_width_km):
    results = []
    for i in range(len(source_coords) - 1):
        A = source_coords[i]
        B = source_coords[i + 1]
        hits = []
        for P in target_coords:
            d = point_segment_distance(P, A, B)
            if d <= corridor_width_km:
                hits.append((P, d))
        if hits:
            results.append({'segment': (A, B), 'vertex_detected': hits})

    if not results:
        return source_coords, [], 0

    new_coords = []
    i_segment = 0

    for i in range(len(source_coords) - 1):
        A = source_coords[i]
        B = source_coords[i + 1]
        new_coords.append(A)
        if i_segment < len(results):
            seg = results[i_segment]
            if seg['segment'] == (A, B):
                to_insert = [coord for coord, _ in seg['vertex_detected']]
                sorted_points = sort_points_along_segment(to_insert, A, B)
                for p in sorted_points:
                    if p != A and p != B:
                        new_coords.append(p)
                i_segment += 1

    new_coords.append(source_coords[-1])
    deduped = deduplicate(new_coords)
    return deduped, results, len(deduped) - len(source_coords)

# === PLOTTING ===
def plot_with_snapping(original, snapped, target, label_orig, label_target):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Snapping Preview")

    if target:
        xt, yt = zip(*target)
        ax.fill(xt, yt, color='red', alpha=0.6, label=label_target)

    if original:
        xo, yo = zip(*original)
        ax.plot(xo + (xo[0],), yo + (yo[0],), color='blue', linestyle='--', label=label_orig + " (original)")

    if snapped:
        xs, ys = zip(*snapped)
        ax.plot(xs + (xs[0],), ys + (ys[0],), color='green', linestyle='-', label=label_orig + " (snapped)")

    ax.grid(True, linestyle="--", linewidth=0.3)
    ax.legend()
    plt.show()

# === MAIN MENU LOOP ===
def main():
    gpml_files = list_gpml_files(gpml_folder)
    if not gpml_files:
        print("No GPML files found.")
        return

    while True:
        mode = user_select_menu(["Snap vertices", "Shape outline"], "\nChoose operation:")

        print("\n--- Select GPML file for entity to modify ---")
        source_file = user_select_menu(gpml_files, "\nSelect source GPML file:")
        source_path = os.path.join(gpml_folder, source_file)
        source_tree, source_root = parse_gpml(source_path)
        source_entities = extract_entities(source_root)
        if not source_entities:
            print("No valid entities found in source file.")
            continue

        entity1, t1 = select_entity(source_entities)
        coords1 = extract_coords(entity1["element"])

        print("\n--- Select GPML file for target entity ---")
        target_file = user_select_menu(gpml_files, "\nSelect target GPML file:")
        target_path = os.path.join(gpml_folder, target_file)
        target_tree, target_root = parse_gpml(target_path)
        target_entities = extract_entities(target_root)
        if not target_entities:
            print("No valid entities found in target file.")
            continue

        entity2, t2 = select_entity(target_entities)
        coords2 = extract_coords(entity2["element"])

        label1 = f"{entity1['name']} (ID {entity1['plate_id']}, t={t1})"
        label2 = f"{entity2['name']} (ID {entity2['plate_id']}, t={t2})"

        if mode == "Snap vertices":
            try:
                tolerance = float(input("\nEnter snapping tolerance (km): "))
            except ValueError:
                tolerance = 35.0
            new_coords, modifications = snap_vertices(coords1, coords2, tolerance, planet_radius_km)
            for before, after, dist in modifications:
                print(f"{before} → {after} ({dist:.2f} km)")
        else:
            try:
                corridor = float(input("\nEnter corridor half-width (km): "))
            except ValueError:
                corridor = 250.0
            new_coords, results, added = shape_entity(coords1, coords2, corridor)
            if not results:
                print("\n– No vertices detected inside corridor.")
            else:
                print(f"\n✔ {len(results)} segments affected. {added} unique points added.")

        plot_with_snapping(coords1, new_coords, coords2, label1, label2)

        confirm = input("\n✅ Apply changes to source GPML file? (Y/N): ").strip().lower()
        if confirm == "y":
            update_coords(entity1["element"], new_coords)
            source_tree.write(source_path, encoding="utf-8", xml_declaration=True)
            print("✅ Source file updated.")
        else:
            print("❌ Changes discarded.")

        again = input("\n🔁 Do another operation? (Y/N): ").strip().lower()
        if again != "y":
            break

if __name__ == "__main__":
    main()

