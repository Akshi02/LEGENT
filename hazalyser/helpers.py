from typing import Dict, Any
import copy
import json
import os
import trimesh
import importlib.util
from pathlib import Path

from dotenv import load_dotenv
from legent.environment.env import Environment, Action 
from legent.scene_generation.types import Vector3
from legent.action.api import GetSpatialRelations


package_name = "legent"
package_path = importlib.util.find_spec(package_name).submodule_search_locations[0]

# Construct the path to the resource
resource_path = os.path.join(package_path, os.pardir, "hazalyser")
resource_path = os.path.abspath(resource_path)

AGENT_PATH = os.path.join(resource_path, "obsAssets", "agent")
SUBJECT_PATH = os.path.join(resource_path, "obsAssets", "subject")
FRAMEWORKS_PATH = os.path.join(resource_path, "frameworks")
LLM_ANALYSIS_PATH = os.path.join(resource_path, "llm_analysis")
env_path = os.path.join(package_path, os.pardir, ".env")
ENV_VARS_PATH = os.path.abspath(env_path)

load_dotenv()

def get_mesh_size(input_file):
    """Get the bounding box of a mesh file.
    Args:
        input_file: str, the path of the mesh file.
    Returns:
        mesh_size: np.ndarray, the size of the mesh file.
    """
    mesh = trimesh.load(input_file)
    min_vals, max_vals = mesh.bounds[0], mesh.bounds[1]
    return max_vals - min_vals
    
def get_current_scene_state(env: Environment, action: Action = Action()) -> Dict[str, Any]:
    action.text = ""
    obs = env.step(action)
    return obs.game_states

def get_spatial_relations(env: Environment, action: Action = Action()):
    action.text = ""
    action.api_calls = [GetSpatialRelations()]
    obs = env.step(action)
    action.api_calls = []
    object_relations = obs.api_returns.get("object_relations", [])
    return object_relations

def update_position_and_rotation(scene: Dict[str, Any], game_states: Dict[str, Any]) -> None:
    keys = {"instances", "player", "agent"}
    convertibles = {"position", "rotation"}
    for key in keys:
        if key == "instances":
            scene_instances = scene[key]
            obs_instances = game_states[key]
            for obj_id, instance in enumerate(scene_instances):
                if instance["prefab"] in obs_instances[obj_id]["prefab"]:
                    for convertible in convertibles:
                        instance[convertible] = convert_vector(obs_instances[obj_id][convertible])
        else:
            for convertible in convertibles:
                scene[key][convertible] = convert_vector(game_states[key][convertible])                   

def is_structural(odb, inst: Dict[str, Any]) -> bool:
    #inefficient, for single instance checks
    structural_assets = odb.MY_OBJECTS
    wall_prefabs = set(structural_assets.get("wall", []))
    floor_prefabs = set(structural_assets.get("floor", []))
    walls_and_floors = wall_prefabs | floor_prefabs

    return inst["prefab"] in walls_and_floors

def is_small_object_instance(inst: Dict[str, Any]) -> bool:
    # Heuristic: small objects are interactables whose parent is a receptacle or have parent not -1
    return inst["type"] == "interactable"

'''        
def is_subject(inst: Dict[str, Any], subject: str) -> bool:
    return inst["prefab"] == os.path.join(SUBJECT_PATH, subject)
'''
def is_subject(inst: Dict[str, Any], subject: str) -> bool:
    # If no subject configured, nothing should be treated as the subject.
    if not subject:
        return False
    return inst.get("prefab") == os.path.join(SUBJECT_PATH, subject)
   
def matches(odb, inst: Dict[str, Any], name: str) -> bool:
    key = name.lower()
    if inst["prefab"].lower() == key:
        return True
    otype = odb.OBJECT_TO_TYPE.get(inst["prefab"], "").lower()
    return otype == key

def get_env_key(name:str) -> dict:
    raw = os.environ.get(name)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Env var {name} is not valid JSON: {e}") from e

def load_framework_prompt(scene_config) -> dict:
    name = scene_config.framework.strip()
    name = name.lower()
    if not name:
        raise ValueError("framework must be a non-empty string")
    
    path = Path(FRAMEWORKS_PATH)/f"{name}.json"
    if not path.exists():
        raise FileNotFoundError("Framework JSON not found.")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def convert_vector(position_or_rotation: Vector3):
    x = position_or_rotation["x"]
    y = position_or_rotation["y"]
    z = position_or_rotation["z"]
    return (x, y, z)

def deepcopy_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(scene)

def log(*args, **kwargs):
    pass



