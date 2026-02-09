import os
import random
import re
from datetime import datetime
from collections import defaultdict
from dotenv import dotenv_values

from hazalyser.helpers import ENV_VARS_PATH, LLM_ANALYSIS_PATH, get_current_scene_state, get_env_key, get_mesh_size, get_spatial_relations, is_small_object_instance, is_subject, update_position_and_rotation, deepcopy_scene
from hazalyser.smallObjects import add_small_objects
from hazalyser.generator import SceneConfig, SceneBundle, SceneGenerator
from hazalyser.prompter import Prompter
from legent import Environment, ResetInfo, Action
from legent.action.api import HideObject, SaveTopDownView, ShowObject, TakePhoto
from legent.server.rect_placer import RectPlacer

class Controller:
    """
    Controller orchestrates single-room scene generation and interactive control in LEGENT.

    Design goals:
    - Single-room focus with user-chosen room type
    - Optional item dictionary to force specific items (receptacles and/or small objects)
    - Post-selection controls: clutter, spacing, add/remove items
    - Zero edits to LEGENT internals; reuse its generators and assets
    - Chat-controlled workflow: "lock_scene()", "reset_scene()", "quit()"
    """

    def __init__(self, scene_config = SceneConfig()) -> None:
        self.scene_config: SceneConfig = scene_config
        self._locked_scene_bundle: SceneBundle = None
        self._current_scene_bundle: SceneBundle = None
        self._committed_scene_bundle: SceneBundle = None
        self._running: bool = False
        self.marked_scenes = []
        self.command_handlers = {
            "#LOCK": self._lock_scene,
            "#UNLOCK": self._unlock_scene,
            "#RESET": self._reset_scene,
            "#COMMIT": self._commit_scene,
            "#RELOAD": self._reload_scene,
            "#CLUTTER+": self._increase_clutter,
            "#CLUTTER-": self._decrease_clutter,
            "#SPACING+": self._increase_spacing,
            "#SPACING-": self._decrease_spacing,
            "#FWANALYSE": self._fw_analyse,
            "#MTANALYSE": self._mt_analyse #IMPLEMENT the hazard methodologies for analysis.

        }
    
    def start(self, env_path: str = "auto") -> None:
        """
        Start the Controller.
        """
        self._running = True
        env = Environment(env_path=env_path)
        action = Action()
        marked = False

        try:
            self._new_scene(env)
            while self._running:
                obs = env.step()
                cmd = (obs.text or "").strip()

                if cmd in self.command_handlers:
                    self.command_handlers[cmd](env, action)

                elif cmd == "#MARK":
                    if marked:
                        action.text = "Scene already marked."
                        env.step(action)
                    else:
                        action.text = f"Scene {len(self.marked_scenes)} marked."
                        self.marked_scenes.append(self._current_scene_bundle)
                        marked = True
                        env.step(action)

                elif cmd == "#LIST":
                    if self._locked_scene_bundle:
                        action.text = "Scene locked. Cannot list marked scenes."
                        env.step(action)
                    else:
                        if not self.marked_scenes:
                            action.text = "No scenes marked."
                            env.step(action)
                        else:
                            action.text = f"Entering marked scenes, there are {len(self.marked_scenes)} scenes marked."
                            env.step(action)
                            marked = True
                            self._enter_list_view(env, action)
                
                elif cmd == "#CLEAR":
                    self.marked_scenes.clear()
                    action.text = "Marked scenes cleared."
                    marked = False
                    env.step(action)
                
                elif cmd == "#NEW":
                    if not self._locked_scene_bundle:
                        self._new_scene(env)
                        marked = False
                    else:
                        action.text = "Scene already locked. Use #RESET to reset the locked scene or #UNLOCK to unlock the scene."
                        env.step(action)

                elif cmd == "#QUIT":
                    action.text = "Exiting"
                    env.step(action)
                    self._running = False
        finally:
            env.close()

        # ---------- Internals ----------

    def _new_scene(self, env: Environment) -> None:
        scene_generator = SceneGenerator(scene_config=self.scene_config)
        scene_bundle = scene_generator.generate()
        scene = scene_bundle.infos
        self._current_scene_bundle = scene_bundle
        env.reset(ResetInfo(scene=scene))    

    def _show_scene(self, env: Environment, action: Action, index: int) -> None:
        """Helper to show a specific scene."""
        scene_num = index % len(self.marked_scenes)
        scene_bundle = self.marked_scenes[scene_num]
        self._current_scene_bundle = scene_bundle
        scene = scene_bundle.infos
        env.reset(ResetInfo(scene=scene))
    
        action.text = f"You are now in Scene{scene_num}"
        env.step(action)
    
    def _enter_list_view(self, env: Environment, action: Action) -> None:
        """Enter the marked scenes browsing mode."""
        current_index = 0
        self._show_scene(env, action, current_index)

        while True:
            obs = env.step()
            cmd = (obs.text or "").strip()

            if cmd in self.command_handlers:
                self.command_handlers[cmd](env, action)
            
            elif cmd == "#NEXT":
                if self._locked_scene_bundle:
                    action.text = "Scene locked. Cannot go to next."
                    env.step(action)
                else:
                    current_index += 1
                    self._show_scene(env, action, current_index)

            elif cmd == "#PREV":
                if self._locked_scene_bundle:
                    action.text = "Scene locked. Cannot go to previous."
                    env.step(action)
                else:
                    current_index -= 1
                    self._show_scene(env, action, current_index)
            
            elif cmd.startswith("#SCENE"):
                scene_num = re.search(r"\d+", cmd)
                if scene_num:
                    if self._locked_scene_bundle:
                        action.text = "Scene locked. Cannot go to specific scene."
                        env.step(action)
                    else:
                        scene_num = int(scene_num.group())
                        if 0 <= scene_num < len(self.marked_scenes):
                            current_index = scene_num
                            self._show_scene(env, action, current_index)
                        else:
                            action.text = "Scene number out of bounds."
                            env.step(action)
                else:
                    action.text = "Invalid scene number."
                    env.step(action)

            elif cmd == "#REMOVE":
                if self._locked_scene_bundle:
                    action.text = "Scene locked. Cannot remove."
                    env.step(action)
                else:
                    scene_num = current_index % len(self.marked_scenes)
                    self.marked_scenes.pop(scene_num)
                    action.text = "Scene removed."
                    env.step(action)
                    if self.marked_scenes:
                        self._show_scene(env, action, scene_num)
                    else:
                        action.text = "No scenes marked. Exiting marked scenes."
                        env.step(action)
                        return

            elif cmd == "#BACK":
                action.text = "Exiting marked scenes."
                env.step(action)
                return

            elif cmd == "#QUIT":
                action.text = "Exiting"
                env.step(action)
                self._running = False
                return

    def _reset_scene(self, env: Environment, action: Action) -> None:
        env.reset(ResetInfo(scene=self._current_scene_bundle.infos))
        action.text = "Scene reset."
        env.step(action)
        if self._locked_scene_bundle:
            self._locked_scene_bundle = deepcopy_scene(self._current_scene_bundle)
        
    def _lock_scene(self, env: Environment, action: Action) -> None:
        self._locked_scene_bundle = deepcopy_scene(self._current_scene_bundle)
        action.text = "Scene locked."
        env.step(action)

    def _unlock_scene(self, env: Environment, action: Action) -> None:
        self._locked_scene_bundle = None
        self._committed_scene_bundle = None
        action.text = "Scene unlocked."
        env.step(action)

    def _commit_scene(self, env: Environment, action: Action) -> None:
        """Save current scene state of the locked scene"""
        if not self._locked_scene_bundle:
            action.text = "No scene locked. Cannot save."
            env.step(action)
        else:
            current_state = get_current_scene_state(env)
            update_position_and_rotation(self._locked_scene_bundle.infos, current_state)
            self._committed_scene_bundle = deepcopy_scene(self._locked_scene_bundle)

            action.text = "Scene saved."
            env.step(action)
    
    def  _reload_scene(self, env: Environment, action: Action) -> None:
        """Reload the locked scene"""
        if not self._locked_scene_bundle:
            action.text = "No scene locked. Cannot reload."
            env.step(action)
        elif self._committed_scene_bundle:
            self._locked_scene_bundle = deepcopy_scene(self._committed_scene_bundle)
            env.reset(ResetInfo(scene=self._locked_scene_bundle.infos))
            self._locked_scene_bundle._hidden_object_indices.clear()
            action.text = "Scene reloaded."
            env.step(action)
        else:
            self._reset_scene(env, action)

    def _increase_clutter(self, env: Environment, action: Action) -> None:
        """Increase the clutter of the locked scene"""
        if not self._locked_scene_bundle:
            action.text = "No scene locked. Cannot increase clutter."
            env.step(action)
        elif self._locked_scene_bundle._hidden_object_indices:
            hidden = self._locked_scene_bundle._hidden_object_indices
            action.text = ""
            while hidden:
                obj_id = hidden.pop()
                action.api_calls = [ShowObject(obj_id)]
                env.step(action)
            action.api_calls = []
            action.text = "Objects restored"
            env.step(action)
        else:
            room_id = self.scene_config.room_spec.spec.room_id
            gen = self._locked_scene_bundle.generator
            room = gen.rooms[room_id]
            odb = gen.odb

            min_x, min_z, max_x, max_z = gen.placer.bbox
            room_area = (max_x - min_x) * (max_z - min_z)
            hard_cap = max(9, int(room_area / 3))

            current_num_assets = len(room.assets)
            remaining_capacity = max(0, hard_cap - current_num_assets)
            if remaining_capacity == 0:
                action.text = "Room is saturated; cannot add more clutter."
                env.step(action)
                return

            batch_size = min(5, int(0.25 * hard_cap))
            to_place = min(batch_size, remaining_capacity)

            spawnable_asset_groups = self._locked_scene_bundle.spawnable_asset_groups
            spawnable_assets = self._locked_scene_bundle.spawnable_assets
            specified_object_types = self._locked_scene_bundle.specified_object_types

            placed_count = 0
            consecutive_failures = 0
            max_consecutive_failures = 9

            while placed_count < to_place and consecutive_failures < max_consecutive_failures:
                rectangle = room.sample_next_rectangle()
                if rectangle is None:
                    consecutive_failures += 1
                    continue

                x_info, z_info, anchor_delta, anchor_type = room.sample_anchor_location(rectangle)
                asset = gen.sample_and_add_floor_asset(
                    room=room,
                    rectangle=rectangle,
                    anchor_type=anchor_type,
                    anchor_delta=anchor_delta,
                    spawnable_assets=spawnable_assets,
                    spawnable_asset_groups=spawnable_asset_groups,
                    priority_asset_types=[],
                    odb=odb,
                )

                if asset is None:
                    consecutive_failures += 1
                    continue

                room.sample_place_asset_in_rectangle(
                    asset=asset,
                    rectangle=rectangle,
                    anchor_type=anchor_type,
                    x_info=x_info,
                    z_info=z_info,
                    anchor_delta=anchor_delta,
                )

                # Update spawnable pools according to duplicate rules
                added_asset_types = []
                if "assetType" in asset:
                    added_asset_types.append(asset["assetType"])
                else:
                    added_asset_types.extend([o["assetType"] for o in asset["objects"]])
                    if not asset.get("allowDuplicates", True):
                        spawnable_asset_groups = spawnable_asset_groups.query(
                            f"assetGroupName!='{asset['assetGroupName']}'"
                        )

                for asset_type in added_asset_types:
                    allow_dup = odb.PLACEMENT_ANNOTATIONS.loc[asset_type.lower()]["multiplePerRoom"]
                    if not allow_dup:
                        spawnable_asset_groups = spawnable_asset_groups[
                            ~spawnable_asset_groups[f"has{asset_type.lower()}"]
                        ]
                        spawnable_assets = spawnable_assets[
                            spawnable_assets["assetType"] != asset_type
                        ]

                placed_count += 1
                consecutive_failures = 0

            # Persist pools and counts
            self._locked_scene_bundle.spawnable_asset_groups = spawnable_asset_groups
            self._locked_scene_bundle.spawnable_assets = spawnable_assets
            self._locked_scene_bundle.max_floor_objects = min(hard_cap, current_num_assets + placed_count)

            # Materialize the newly added room assets into instances
            object_instances = gen.place_other_objects(specified_object_types, current_num_assets)

            instances = self._locked_scene_bundle.infos["instances"]
            subject = self.scene_config.subject
            structural_assets = odb.MY_OBJECTS
            wall_prefabs = set(structural_assets.get("wall", []))
            floor_prefabs = set(structural_assets.get("floor", []))
            walls_and_floors = wall_prefabs | floor_prefabs

            present_objects = [instance for instance in instances 
                              if instance["prefab"] not in walls_and_floors 
                              and not is_small_object_instance(instance) 
                              and not is_subject(instance, subject)]

            # Add small objects only for the newly added furniture to reduce latency
            try:
                small_object_instances = add_small_objects(
                    present_objects+object_instances,
                    odb,
                    gen.rooms,
                    999,
                    gen.placer.bbox,
                    object_counts={},
                    specified_object_instances={},
                    receptacle_object_counts={},
                )
            except Exception:
                small_object_instances = []

            new_objects = object_instances + small_object_instances
            self._locked_scene_bundle.infos["instances"].extend(new_objects)

            env.reset(ResetInfo(scene = self._locked_scene_bundle.infos))
            action.text = f"Added {len(object_instances)} furniture and {len(small_object_instances)} small objects."
            env.step(action)
            self._locked_scene_bundle.clutter_level += 1
        
    def _decrease_clutter(self, env: Environment, action: Action) -> None:
        """Decrease clutter by hiding a bounded batch of objects."""
        if not self._locked_scene_bundle:
            action.text = "No scene locked. Cannot decrease spacing."
            env.step(action)
        else:
            odb = self._locked_scene_bundle.generator.odb
            instances = self._locked_scene_bundle.infos["instances"]
            object_relations = get_spatial_relations(env, action)
            placement_info = defaultdict(list)
            for obj in object_relations:
                placement_info[obj.get("on_what")].append(obj.get("id"))
            # Identify hideable objects (exclude structural and subject)
            structural_assets = odb.MY_OBJECTS
            wall_prefabs = set(structural_assets.get("wall", []))
            floor_prefabs = set(structural_assets.get("floor", []))
            walls_and_floors = wall_prefabs | floor_prefabs

            hideable_all = {i for i, inst in enumerate(instances)
                            if inst["prefab"] not in walls_and_floors and not is_subject(inst, self.scene_config.subject)}
            max_hidden = int(0.7 * len(hideable_all))
            already_hidden = set(self._locked_scene_bundle._hidden_object_indices)
            if len(already_hidden) >= max_hidden:
                action.text = "Clutter already at minimum."
                env.step(action)
                return

            budget_left = max_hidden - len(already_hidden)
            batch_target = min(int(round(0.2 * max_hidden)), budget_left)

            candidates = list(hideable_all - already_hidden)
            if not candidates:
                action.text = "No hideable objects found."
                env.step(action)
                return

            hidden_now = 0
            random.shuffle(candidates)
            for root in candidates:
                if hidden_now >= batch_target:
                    break
                to_hide = [root]
                # Hide children placed on this object (one level)
                to_hide.extend(placement_info.get(root, []))
                for obj_id in to_hide:
                    self._locked_scene_bundle._hidden_object_indices.append(obj_id)
                    action.api_calls = [HideObject(obj_id)]
                    env.step(action)
                    action.api_calls = []
                    hidden_now += 1
                    already_hidden.add(obj_id)

            action.text = f"Hidden {hidden_now} objects (total {len(already_hidden)}/{max_hidden})."
            env.step(action)
            self._locked_scene_bundle.clutter_level -= 1
                 
    def _increase_spacing(self, env: Environment, action: Action) -> None:
        if not self._locked_scene_bundle:
            action.text = "No locked scene. Cannot increase spacing."
            env.step(action)
            return
        self._resize_room(env, action, delta_cells=(1,1))
        self._locked_scene_bundle.spacing_level += 1

    def _decrease_spacing(self, env: Environment, action: Action) -> None:
        if not self._locked_scene_bundle:
            action.text = "No locked scene. Cannot decrease spacing."
            env.step(action)
            return
        old_x, old_z, new_x, new_z = self._resize_room(env, action, delta_cells=(-1,-1))

        if not (old_x == new_x and old_z == new_z):
            self._locked_scene_bundle.spacing_level -= 1

    def _get_house_dimensions(self, unit_size):
        min_x, min_z, max_x, max_z = self._locked_scene_bundle.generator.placer.bbox
        house_dimensions = (max_x / unit_size, max_z / unit_size)

        return house_dimensions

    def _unhide_objects(self, env: Environment, action: Action):
        hidden_object_indices = self._locked_scene_bundle._hidden_object_indices
        action.text = ""
        while hidden_object_indices:
            action.api_calls = [ShowObject(hidden_object_indices.pop())]
            env.step(action)

    def _resize_room(self, env: Environment, action: Action, delta_cells):
        # Preconditions
        scene_bundle = self._locked_scene_bundle
        info = scene_bundle.infos
        generator = scene_bundle.generator
        odb = generator.odb
        prefabs = odb.PREFABS
        unit_size = generator.unit_size
        room_spec = self.scene_config.room_spec

        # 1) Old/new dims (cells); clamp minimum 3x3
        old_x_cells_f, old_z_cells_f = self._get_house_dimensions(unit_size)
        old_x_cells = max(1, int(round(old_x_cells_f)))
        old_z_cells = max(1, int(round(old_z_cells_f)))
        new_x_cells = max(3, int(old_x_cells + delta_cells[0]))
        new_z_cells = max(3, int(old_z_cells + delta_cells[1]))

        # 2) Structural prefab detection for style consistency
        structural_assets = odb.MY_OBJECTS
        wall_prefabs = set(structural_assets.get("wall", []))
        floor_prefabs = set(structural_assets.get("floor", []))
        walls_and_floors = wall_prefabs | floor_prefabs

        wall_prefab = None
        floor_prefab = None
        seen = set()
        for inst in info["instances"]:
            pf = inst["prefab"]
            if pf in seen:
                continue
            seen.add(pf)
            if wall_prefab is None and pf in wall_prefabs:
                wall_prefab = pf
            elif floor_prefab is None and pf in floor_prefabs:
                floor_prefab = pf
            if wall_prefab and floor_prefab:
                break

        # 3) New structure + structural instances (floors/walls)
        new_structure = generator.generate_structure((new_z_cells, new_x_cells))
        new_x_cells_shape = new_structure.interior.shape[0]
        new_z_cells_shape = new_structure.interior.shape[1]
        new_max_x = new_x_cells_shape * unit_size
        new_max_z = new_z_cells_shape * unit_size

        floor_instances, floors = generator.add_floors_and_walls(
            new_structure, room_spec, odb, prefabs, wall=wall_prefab, floor=floor_prefab
        )

        # 4) Rebuild RectPlacer and rooms
        generator.placer = RectPlacer((0, 0, new_max_x, new_max_z))
        floor_polygons = generator.get_floor_polygons(new_structure.xz_poly_map)
        generator.get_rooms(room_spec=room_spec, floor_polygons=floor_polygons)

        # 5) Helpers
        def footprint(inst_prefab: str, rot_y: float) -> tuple[float, float]:
            # Returns (width_x, depth_z) based on rotation; fallback if prefab unknown
            if inst_prefab in prefabs:
                sz = prefabs[inst_prefab]["size"]
                if rot_y in (0, 180):
                    return sz["x"], sz["z"]
                else:
                    return sz["z"], sz["x"]
            # Fallback for unknowns (e.g., subject glb): approximate as one cell
            return unit_size, unit_size

        def compute_bbox(cx: float, cz: float, w: float, d: float) -> tuple[float, float, float, float]:
            return (cx - w / 2, cz - d / 2, cx + w / 2, cz + d / 2)

        def clamp_center(cx: float, cz: float, w: float, d: float) -> tuple[float, float]:
            return (
                max(w / 2, min(new_max_x - w / 2, cx)),
                max(d / 2, min(new_max_z - d / 2, cz)),
            )

        def jitter_offsets(step: float, rings: int = 3) -> list[tuple[float, float]]:
            offs = [(0.0, 0.0)]
            for r in range(1, rings + 1):
                for dx in (-r, 0, r):
                    for dz in (-r, 0, r):
                        if dx == 0 and dz == 0:
                            continue
                        offs.append((dx * step, dz * step))
            return offs

        # 6) Partition instances
        def is_struct(inst):
            return inst["prefab"] in walls_and_floors

        furniture = []
        small_objs = []
        subject = None
        for inst in info["instances"]:
            if is_struct(inst): 
                continue
            if is_subject(inst, self.scene_config.subject):
                subject = inst
            elif is_small_object_instance(inst):
                small_objs.append(inst)
            else:
                furniture.append(inst)

        # 7) Repack player, agent, subject and furniture
        sx = new_x_cells / old_x_cells
        sz = new_z_cells / old_z_cells
        step = unit_size * 0.5
        offsets = jitter_offsets(step, rings=3)

        # Sort: kinematic (non-interactable) first, larger area first
        def sort_key(inst):
            rot_y = inst.get("rotation", [0, 0, 0])[1]
            w, d = footprint(inst["prefab"], rot_y)
            area = w * d
            is_small = (inst.get("type") == "interactable")
            return (is_small, -area)
        furniture.sort(key=sort_key)

        placed_furniture = []
        dropped_furniture = 0

        # 8) Scale player/agent/subject positions
        player = info.get("player")
        agent = info.get("agent")
        AGENT_HUMAN_SIZE = 1
        for instance in [player, agent, subject]:
            px, py, pz = instance["position"]
            cx, cz = (px * sx, pz * sz)
            instance["position"] = [cx, py, cz]
            if not is_subject(instance, self.scene_config.subject):
                generator.placer.place("agent", cx, cz, AGENT_HUMAN_SIZE, AGENT_HUMAN_SIZE)
            else:
                asset_path = instance["prefab"]
                subject_size = get_mesh_size(asset_path)
                x_size = subject_size[0]
                z_size = subject_size[1]

                generator.placer.place("subject", cx, cz, x_size, z_size)


        for idx, inst in enumerate(furniture):
            rot_y = inst.get("rotation", [0, 0, 0])[1]
            w, d = footprint(inst["prefab"], rot_y)

            # Oversize guard: cannot fit at all
            if w > new_max_x or d > new_max_z:
                dropped_furniture += 1
                continue

            target_x = inst["position"][0] * sx
            target_z = inst["position"][2] * sz

            placed = False
            for dx, dz in offsets:
                cx, cz = clamp_center(target_x + dx, target_z + dz, w, d)
                bbox = compute_bbox(cx, cz, w, d)
                if generator.placer.place_rectangle(f"{inst['prefab']}_{idx}", bbox):
                    # Some instances have tuple positions; rebuild as a list when updating
                    px, py, pz = inst["position"]
                    inst["position"] = [cx, py, cz]
                    placed_furniture.append(inst)
                    placed = True
                    break
            if not placed:
                dropped_furniture += 1

        # 9) Re-add small objects on receptacles after furniture is placed
        #    We place fresh to avoid conflicting with main RectPlacer; small object collisions
        #    are handled inside add_small_objects with its own local placer.
        try:
            small_object_instances = add_small_objects(
                placed_furniture,
                odb,
                generator.rooms,
                scene_bundle.max_object_types_per_room,
                generator.placer.bbox,
                object_counts={},
                specified_object_instances={},
                receptacle_object_counts={}
            )
        except Exception:
            # If small object placement fails, fall back to none to keep scene valid
            small_object_instances = []

        # 10) Update instances and scene metadata
        info["instances"] = floor_instances + [subject] + placed_furniture + small_object_instances
        height = max(12, new_max_z * 1 + 2)
        info["center"] = [new_max_x / 2, height, new_max_z / 2]

        room_polys = []
        for room in generator.rooms.values():
            polygon = list(room.room_polygon.polygon.exterior.coords)
            x_center = sum(x for x, _ in polygon) / len(polygon)
            z_center = sum(z for _, z in polygon) / len(polygon)
            x_size = max(x for x, _ in polygon) - min(x for x, _ in polygon)
            z_size = max(z for _, z in polygon) - min(z for _, z in polygon)
            room_polys.append({
                "room_id": room.room_id,
                "room_type": room.room_type,
                "position": [x_center, 1.5, z_center],
                "size": [x_size, 3, z_size],
                "polygon": polygon
            })
        info["room_polygon"] = room_polys

        # 11) Hidden indices no longer valid; clear them
        scene_bundle._hidden_object_indices = []

        # 12) Reload and report
        env.reset(ResetInfo(scene=info))
        kept = len(placed_furniture)
        action.text = f"Spacing updated: furniture kept {kept}, dropped {dropped_furniture}, small objects {len(small_object_instances)}."
        env.step(action)

        return (old_x_cells, old_z_cells, new_x_cells, new_z_cells)

    def _save_perspectives(
        self,
        env: Environment,
        action: Action = Action(),
        camera_height=1.7,
        width=4096,
        height=4096,
        vertical_field_of_view=60,
        include_top: bool = True,      # NEW
        out_root: str = None,          # NEW
        folder_name: str = None,       # NEW
    ):
        scene_info = self._locked_scene_bundle.infos
        room_type = scene_info["room_polygon"][0]["room_type"]

        # root output dir
        base_root = out_root if out_root else os.path.join(LLM_ANALYSIS_PATH, self.scene_config.analysis_folder)

        # folder naming
        if folder_name is None:
            timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
            folder_name = f"{room_type}_{timestamp}"

        # IMPORTANT: save inside room folder
        save_folder = os.path.join(base_root, room_type, folder_name)
        os.makedirs(save_folder, exist_ok=True)

        action.text = ""
        min_x, min_z, max_x, max_z = self._locked_scene_bundle.generator.placer.bbox
        image_paths = []

        if include_top:
            top_view_path = f"{save_folder}/top.png"
            action.api_calls = [SaveTopDownView(top_view_path)]
            env.step(action)
            image_paths.append(top_view_path)

        rotations = [[0, 0, 0], [0, 90, 0], [0, 180, 0], [0, 270, 0]]
        positions = [
            [(max_x - min_x) / 2, camera_height, 0],
            [0, camera_height, (max_z - min_z) / 2],
            [(max_x - min_x) / 2, camera_height, max_z],
            [max_x, camera_height, (max_z - min_z) / 2],
        ]

        for i, (pos, rot) in enumerate(zip(positions, rotations)):
            side_view_path = f"{save_folder}/side_{i}.png"
            action.api_calls = [TakePhoto(side_view_path, pos, rot, width, height, vertical_field_of_view)]
            env.step(action)
            image_paths.append(side_view_path)

        action.api_calls = []
        return save_folder, image_paths


    '''
    def _save_perspectives(self, env: Environment, action: Action = Action(),
                        camera_height=1.7, width=4096, height=4096, vertical_field_of_view=60):
        """ Take snapshots from different perspectives """
        scene_config = self.scene_config
        analysis_folder = scene_config.analysis_folder
        scene_info = self._locked_scene_bundle.infos
        room_id = scene_info['room_polygon'][0]["room_id"]
        room_type = scene_info['room_polygon'][0]['room_type']
        clutter_level = self._locked_scene_bundle.clutter_level
        spacing_level = self._locked_scene_bundle.spacing_level
        timestamp = datetime.now().strftime("%d%m%Y_%H%M")

        save_folder = os.path.join(LLM_ANALYSIS_PATH, analysis_folder, f"{room_type}_{room_id}_{clutter_level}_{spacing_level}_{timestamp}")
        os.makedirs(save_folder, exist_ok=True)
        

        action.text = ""
        min_x, min_z, max_x, max_z = self._locked_scene_bundle.generator.placer.bbox
        image_paths = []
        top_view_path = f"{save_folder}/top.png"
        api_calls = [SaveTopDownView(top_view_path)]
        action.api_calls = api_calls
        env.step(action)    
        image_paths.append(top_view_path)

        rotations = [[0, 0, 0], [0, 90, 0], [0, 180, 0], [0, 270, 0]]
        positions = [[(max_x - min_x) / 2, camera_height, 0], [0, camera_height, (max_z - min_z) / 2],
                    [(max_x - min_x) / 2, camera_height, max_z], [max_x, camera_height, (max_z - min_z) / 2]]

        for i, (pos, rot) in enumerate(zip(positions, rotations)):
            side_view_path = f"{save_folder}/side_{i}.png"
            api_calls = [TakePhoto(side_view_path, pos, rot, width, height, vertical_field_of_view)]
            action.api_calls = api_calls
            env.step(action)
            image_paths.append(side_view_path)
        
        action.api_calls = []
        images_to_send = self.scene_config.images.strip().lower()
        if images_to_send == "top":
            image_paths = image_paths[:1]

        return save_folder, image_paths
    '''

    #NEW CODES

    def _sample_dims_for_spacing_bin(self, spacing_bin: str):
        """
        spacing_bin: 'low_spacing' | 'medium_spacing' | 'high_spacing'
        Returns (x_cells, z_cells) with minimum 3.
        You can tune ranges below.
        """
        spacing_bin = spacing_bin.lower().strip()

        if spacing_bin == "low_spacing":
            # smallest rooms
            x = random.randint(3, 4)
            z = random.randint(3, 4)
            return (x, z)

        if spacing_bin == "medium_spacing":
            x = random.randint(5, 7)
            z = random.randint(5, 7)
            return (x, z)

        if spacing_bin == "high_spacing":
            x = random.randint(8, 10)
            z = random.randint(8, 10)
            return (x, z)

        # fallback
        return (3, 3)
    
    '''
    def _apply_clutter_category(self, env: Environment, category: str, steps: int = 3):
        """
        Apply a clutter category by calling increase/decrease clutter
        a fixed number of times.

        category: 'high_clutter' | 'medium_clutter' | 'low_clutter'
        """
        category = category.lower().strip()
        action = Action()

        # medium clutter = baseline scene
        if category == "medium_clutter":
            return

        if category == "high_clutter":
            for _ in range(steps):
                try:
                    self._increase_clutter(env, action)
                except Exception:
                    break
            return

        if category == "low_clutter":
            for _ in range(steps):
                try:
                    self._decrease_clutter(env, action)
                except Exception:
                    break
            return
    '''
    def _apply_clutter_category(self, env: Environment, category: str, max_steps: int = 3):
        """
        Early-stopping clutter adjuster.

        - high_clutter: repeatedly call _increase_clutter until instances stop increasing
        - low_clutter: repeatedly call _decrease_clutter until hidden count stops increasing
        - medium_clutter: do nothing
        """
        category = category.lower().strip()
        action = Action()

        if category == "medium_clutter":
            return

        if category == "high_clutter":
            prev_n = -1
            for _ in range(max_steps):
                before_n = len(self._locked_scene_bundle.infos.get("instances", []))
                if before_n <= prev_n:
                    break
                prev_n = before_n

                try:
                    self._increase_clutter(env, action)
                except Exception:
                    break

                after_n = len(self._locked_scene_bundle.infos.get("instances", []))
                if after_n <= before_n:
                    break
            return

        if category == "low_clutter":
            prev_hidden = -1
            for _ in range(max_steps):
                before_hidden = len(getattr(self._locked_scene_bundle, "_hidden_object_indices", []))
                if before_hidden <= prev_hidden:
                    break
                prev_hidden = before_hidden

                try:
                    self._decrease_clutter(env, action)
                except Exception:
                    break

                after_hidden = len(getattr(self._locked_scene_bundle, "_hidden_object_indices", []))
                if after_hidden <= before_hidden:
                    break
            return

    '''
    def generate_4room_dataset(
        self,
        out_root: str,
        per_room_folders: int = 10,
        env_path: str = "auto",
        width: int = 1024,
        height: int = 1024,
        vfov: int = 60,
        camera_height: float = 1.7,
        max_clutter_steps: int = 80,
    ):
        """
        Creates:
        out_root/Bedroom/Bedroom_00/side_0..3.png
        out_root/LivingRoom/LivingRoom_00/side_0..3.png
        out_root/Kitchen/Kitchen_00/side_0..3.png
        out_root/Bathroom/Bathroom_00/side_0..3.png

        10 folders per room => 40 side images per room.
        Top view disabled. Clutter max. Spacing min.
        """

        room_types = ["Bedroom", "LivingRoom", "Kitchen", "Bathroom"]
        env = Environment(env_path=env_path)
        action = Action()

        try:
            for rt in room_types:
                # Force room type
                self.scene_config.room_spec.spec.room_type = rt

                # SPACING MIN: generate smallest room immediately (3x3)
                self.scene_config.dims = (3, 3)

                for i in range(per_room_folders):
                    # new scene + lock
                    self._new_scene(env)
                    self._lock_scene(env, action)
                    
                    # CLUTTER MAX: keep adding until saturated (instance count stops increasing)
                    for _ in range(max_clutter_steps):
                        before_n = len(self._locked_scene_bundle.infos["instances"])
                        self._increase_clutter(env, action)
                        after_n = len(self._locked_scene_bundle.infos["instances"])
                        if after_n <= before_n:
                            break
                             
                    # Save 4 side images only
                    folder_name = f"{rt}_{i:02d}"
                    save_folder, image_paths = self._save_perspectives(
                        env,
                        action=Action(),
                        camera_height=camera_height,
                        width=width,
                        height=height,
                        vertical_field_of_view=vfov,
                        include_top=False,      # DISABLE TOP VIEW
                        out_root=out_root,      # ROOT OUTPUT
                        folder_name=folder_name # EXACT 10 folders
                    )

                    # Save scene.json for each folder (recommended)
                    import json
                    from hazalyser.helpers import get_current_scene_state, update_position_and_rotation

                    current_state = get_current_scene_state(env, Action())
                    update_position_and_rotation(self._locked_scene_bundle.infos, current_state)

                    with open(os.path.join(save_folder, "scene.json"), "w", encoding="utf-8") as f:
                        json.dump(self._locked_scene_bundle.infos, f, ensure_ascii=False, indent=2)

                    # unlock
                    self._unlock_scene(env, action)

                    action.text = f"Saved {rt} [{i+1}/{per_room_folders}] -> {save_folder}"
                    env.step(action)

        finally:
            env.close()
    '''

#------------- BATCH PROCESSING -----------#


    def generate_4room_dataset_by_clutter_and_spacing(
        self,
        out_root: str,
        per_room_per_clutter_per_spacing: int = 100,
        env_path: str = "auto",
        width: int = 1024,
        height: int = 1024,
        vfov: int = 60,
        camera_height: float = 1.7,
        clutter_steps_high: int = 3,
        clutter_steps_low: int = 3,
        batch_size: int = 25,   # NEW: restart env every N scenes
        log_every: int = 1,     # NEW: print progress every N scenes
    ):
        """
        Output:
        out_root/<RoomType>/<ClutterCategory>/<SpacingCategory>/<folder>/
            side_0..3.png
            scene.json

        Added:
        - Batched generation (restart Environment every batch_size scenes)
        - Early stopping (via _apply_clutter_category)
        - Progress logs + ETA
        - Prints current category directory (Bedroom/high_clutter/high_spacing etc.)
        """

        import json
        import time
        from hazalyser.helpers import get_current_scene_state, update_position_and_rotation

        def fmt_time(seconds: float) -> str:
            seconds = int(max(0, seconds))
            h = seconds // 3600
            m = (seconds % 3600) // 60
            s = seconds % 60
            if h > 0:
                return f"{h}h {m}m {s}s"
            if m > 0:
                return f"{m}m {s}s"
            return f"{s}s"

        out_root_abs = os.path.abspath(out_root)
        print("=== DATASET GENERATION START ===")
        print("Working directory (cwd):", os.getcwd())
        print("Output root:", out_root_abs)

        room_types = ["Bedroom", "LivingRoom", "Kitchen", "Bathroom"]
        clutter_categories = ["high_clutter", "medium_clutter", "low_clutter"]
        spacing_categories = ["low_spacing", "medium_spacing", "high_spacing"]

        total = (
            len(room_types)
            * len(clutter_categories)
            * len(spacing_categories)
            * per_room_per_clutter_per_spacing
        )

        done = 0
        start_t = time.time()
        ema_s = None  # exponential moving average seconds per scene

        def restart_env(curr_env: Environment):
            try:
                curr_env.close()
            except Exception:
                pass
            return Environment(env_path=env_path), Action()

        env = Environment(env_path=env_path)
        action = Action()

        try:
            for rt in room_types:
                self.scene_config.room_spec.spec.room_type = rt

                for clutter_cat in clutter_categories:
                    for spacing_cat in spacing_categories:

                        context_str = f"{rt} / {clutter_cat} / {spacing_cat}"
                        category_root = os.path.join(out_root_abs, rt, clutter_cat, spacing_cat)
                        os.makedirs(category_root, exist_ok=True)

                        print("\n=== CATEGORY:", context_str, "===")
                        print("Saving into:", category_root)

                        for i in range(per_room_per_clutter_per_spacing):

                            # Batch restart to avoid Unity/LEGENT hanging
                            if done > 0 and (done % batch_size == 0):
                                print(f"\n[BATCH RESTART] Completed {done}/{total}. Restarting LEGENT env...")
                                env, action = restart_env(env)

                            t0 = time.time()

                            # --- spacing -> dims ---
                            x_cells, z_cells = self._sample_dims_for_spacing_bin(spacing_cat)
                            self.scene_config.dims = (x_cells, z_cells)

                            # --- new scene + lock ---
                            self._new_scene(env)
                            self._lock_scene(env, action)

                            # --- apply clutter (early stop inside) ---
                            if clutter_cat == "high_clutter":
                                self._apply_clutter_category(env, clutter_cat, max_steps=clutter_steps_high)
                            elif clutter_cat == "low_clutter":
                                self._apply_clutter_category(env, clutter_cat, max_steps=clutter_steps_low)
                            else:
                                self._apply_clutter_category(env, clutter_cat, max_steps=0)

                            # --- save 4 side views (no top) ---
                            folder_name = f"{rt}_{clutter_cat}_{spacing_cat}_{i:03d}_x{x_cells}_z{z_cells}"
                            save_folder, _ = self._save_perspectives(
                                env,
                                action=Action(),
                                camera_height=camera_height,
                                width=width,
                                height=height,
                                vertical_field_of_view=vfov,
                                include_top=False,
                                out_root=category_root,   # already room/clutter/spacing
                                folder_name=folder_name
                            )

                            # --- save scene.json ---
                            current_state = get_current_scene_state(env, Action())
                            update_position_and_rotation(self._locked_scene_bundle.infos, current_state)
                            with open(os.path.join(save_folder, "scene.json"), "w", encoding="utf-8") as f:
                                json.dump(self._locked_scene_bundle.infos, f, ensure_ascii=False, indent=2)

                            # --- unlock ---
                            self._unlock_scene(env, action)

                            # --- ETA ---
                            dt = time.time() - t0
                            ema_s = dt if ema_s is None else (0.95 * ema_s + 0.05 * dt)

                            done += 1
                            elapsed = time.time() - start_t
                            remaining = total - done
                            eta = remaining * (ema_s if ema_s else 0.0)

                            # --- logging ---
                            if (done % log_every) == 0:
                                print(
                                    f"[{done}/{total}] {context_str} | "
                                    f"sample={i+1}/{per_room_per_clutter_per_spacing} | "
                                    f"dt={dt:.2f}s | elapsed={fmt_time(elapsed)} | ETA={fmt_time(eta)} | "
                                    f"saved={save_folder}"
                                )

        finally:
            try:
                env.close()
            except Exception:
                pass
            print("\n=== DATASET GENERATION END ===")



    def _fw_analyse(self, env: Environment, action: Action) -> None:
        # save all images for reference
        if not self._locked_scene_bundle:
            action.text = "No scene locked. Cannot analyse."
            env.step(action)
        elif not self.scene_config.framework:
            action.text = "You need to specify the framework in the scene configuration"
            env.step(action)
        else:
            object_db = self._locked_scene_bundle.generator.odb
            scene_info = self._locked_scene_bundle.infos
            current_state = get_current_scene_state(env, action)
            update_position_and_rotation(scene_info, current_state)
            object_relations = get_spatial_relations(env, action)
            save_path, image_paths = self._save_perspectives(env, action)

            llm_key = self.scene_config.llm_key.strip().lower()
            llm_keys = []
            if llm_key == "all":
                env_vars = dotenv_values(ENV_VARS_PATH)
                for llm_key in env_vars.keys():
                    llm_keys.append(llm_key)
            else:
                llm_keys.append(llm_key)

            for llm_key in llm_keys:
                api_info = get_env_key(llm_key)
                api_key = api_info.get("api_key", None)
                base_url = api_info.get("base_url", None)
                model_name = api_info.get("model_name", "")
                vision_support = api_info.get("vision_support", False)
                temperature = self.scene_config.temperature
                max_tokens = self.scene_config.max_tokens

                analyser = Prompter(self.scene_config, odb=object_db, api_key=api_key, base_url=base_url, model_name=model_name, vision_support=vision_support)
                analyser.analyse(save_path, scene_info, object_relations, image_paths, temperature, max_tokens)

                action.text = f"Analysis saved to {save_path}"
                env.step(action)

    def _mt_analyse(self, env: Environment, action: Action) -> None:
        if not self._locked_scene_bundle:
            action.text = "No scene locked. Cannot analyse."
        elif not self.scene_config.method:
            raise RuntimeError("No method specified.")
        else:
            pass