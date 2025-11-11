from hazalyser import SceneConfig, Controller, HazardRoom, HazardRoomSpec

room_spec = HazardRoomSpec(room_spec_id="ESHA-SingleRoom", 
                           spec=HazardRoom(room_id=1, room_type="Bedroom"))
scene_config = SceneConfig(room_spec = room_spec, dims = (4, 5) , subject="oldman_ernest.glb", subject_info="The subject is an elderly man with walking difficulties and poor vision", 
                           task="The agent needs to hand some medicines to the subject", framework="esha", llm_key="all", images="all", analysis_folder="LivingRoom")

controller = Controller(scene_config)

controller.start()
