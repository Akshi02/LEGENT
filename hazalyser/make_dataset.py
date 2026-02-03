from hazalyser import Controller, SceneConfig

if __name__ == "__main__":
    cfg = SceneConfig()
    cfg.images = "all"  # doesn't matter because we're disabling top in _save_perspectives
    controller = Controller(scene_config=cfg)

    controller.generate_4room_dataset_by_clutter_and_spacing(
        out_root="dataset_by_room",
        per_room_per_clutter_per_spacing=100,
        width=1024,
        height=1024,
        vfov=60,
        camera_height=1.7,
        clutter_steps_high=3,
        clutter_steps_low=3,
    )
