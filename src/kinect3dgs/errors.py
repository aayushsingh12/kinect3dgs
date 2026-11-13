class KinectError(RuntimeError):
    """Base class for Kinect failures."""


class KinectNotFoundError(KinectError):
    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


class KinectBusyError(KinectError):
    """A kernel driver (gspca_kinect) or another process holds the device."""


class KinectTimeoutError(KinectError):
    """The device is present but produced no frames in time."""
