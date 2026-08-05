import bpy
import json
import os

BLACK_NOTE_NAMES = [
    "C#2", "D#2", "F#2", "G#2", "A#2",
    "C#3", "D#3", "F#3", "G#3", "A#3",
    "C#4", "D#4", "F#4"
]

WHITE_NOTE_NAMES = [
    "C2", "D2", "E2", "F2", "G2", "A2", "B2",
    "C3", "D3", "E3", "F3", "G3", "A3", "B3",
    "C4", "D4", "E4", "F4", "G4"
]
NOTE_NAMES = BLACK_NOTE_NAMES
def export_pedalboard_json(filepath):
    obj = bpy.context.active_object
    
    if not obj or obj.type != 'MESH':
        raise RuntimeError("Please select a valid Mesh object.")

    # Apply object matrix world to get global 3D coordinates
    matrix_world = obj.matrix_world
    mesh = obj.data

    faces_data = []

    for face in mesh.polygons:
        # Extract 2D world coordinates (X, Y) for each vertex in the face
        world_verts = [matrix_world @ mesh.vertices[v_idx].co for v_idx in face.vertices]
        coords_2d = [[round(.63256-v.x, 4), round(v.y-.0991249, 4)] for v in world_verts]
        
        # Calculate centroid X for spatial sorting along the board
        center_x = sum(v.x for v in world_verts) / len(world_verts)
        
        faces_data.append({
            "center_x": center_x,
            "vertices": coords_2d
        })

    # Sort faces from Left to Right (Top Left X=62.95 -> Top Right X=-63.08)
    # If your keys map in reverse order, set reverse=False
    faces_data.sort(key=lambda f: f["center_x"], reverse=True)

    if len(faces_data) != len(NOTE_NAMES):
        print(f"Warning: Mesh has {len(faces_data)} faces, but expected {len(NOTE_NAMES)} notes.")

    # Pair sorted faces with note names
    output_data = []
    for note, face in zip(NOTE_NAMES, faces_data):
        output_data.append({
            "note": note,
            "vertices": face["vertices"]
        })

    # Save to JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"Successfully exported {len(output_data)} keys to {filepath}")

# Output path (saves in the same directory as your blend file, or user home folder)
blend_filepath = bpy.data.filepath
if blend_filepath:
    output_dir = os.path.dirname(blend_filepath)
else:
    output_dir = os.path.expanduser("~")

export_path = os.path.join(output_dir, "COLOR_keys.json")
export_pedalboard_json(export_path)