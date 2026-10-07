class PlateflowError(Exception):
    pass


class InvalidImageError(PlateflowError):
    pass


class UnreadableFileError(PlateflowError):
    pass


class ModelNotFoundError(PlateflowError):
    pass


class EmptyDetectionError(PlateflowError):
    pass


class OcrError(PlateflowError):
    pass


class StreamDisconnectionError(PlateflowError):
    pass


class DatabaseError(PlateflowError):
    pass
