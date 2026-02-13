class CameraHandoff:
    """
    Maintains lost-object search state across cameras.
    """

    def __init__(self):
        self.active_search = False
        self.query_embedding = None
        self.source_camera = None

    def register_loss(self, embedding, camera_id):
        self.query_embedding = embedding
        self.source_camera = camera_id
        self.active_search = True

        print(f"🔎 ReID search started from {camera_id}")

    def clear(self):
        self.active_search = False
        self.query_embedding = None
        self.source_camera = None

    def is_active(self):
        return self.active_search
