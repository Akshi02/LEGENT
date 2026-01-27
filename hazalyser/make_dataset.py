from hazalyser import Controller, SceneConfig

if __name__ == "__main__":
    cfg = SceneConfig()
    cfg.images = "all"  # doesn't matter because we're disabling top in _save_perspectives
    controller = Controller(scene_config=cfg)

    controller.generate_4room_dataset(
        out_root="dataset_by_room",
        per_room_folders=10,
        width=1024,
        height=1024,
        vfov=60,
        camera_height=1.7,
        max_clutter_steps=80
    )
