class ObjectRegistrar:
    """
    Locks onto the first detected personal object and prevents
    the system from registering new background/static blobs.

    This converts the pipeline from 'scene analysis' to
    'target tracking'.
    """

    def __init__(self):
        self.registered_id = None
        self.initialized = False

    def select(self, tracks):
        """
        Choose which track becomes the object-of-interest.
        Called only once.
        """
        if self.initialized or not tracks:
            return

        # Choose the largest / most confident track (simple heuristic)
        self.registered_id = tracks[0].id
        self.initialized = True

        print(f"🎯 Locked onto object ID {self.registered_id}")

    def filter_active(self, tracks):
        """
        After registration, only allow that object to exist.
        """
        if not self.initialized:
            return tracks

        return [t for t in tracks if t.id == self.registered_id]

    def check_lost(self, lost_tracks):
        """
        Detect disappearance of the registered object.
        """
        if not self.initialized:
            return None

        for t in lost_tracks:
            if t.id == self.registered_id:
                return t

        return None
