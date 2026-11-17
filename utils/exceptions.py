"""Custom exception types used across the AI layer for clean, specific error handling."""


class AIBaseException(Exception):
    """Base exception for all AI layer errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ModelNotLoadedError(AIBaseException):
    """Raised when an inference call is made before a required model is loaded."""


class InvalidInputError(AIBaseException):
    """Raised when input data fails validation beyond what Pydantic enforces."""


class ConfigError(AIBaseException):
    """Raised when a configuration file is missing or malformed."""


class ImageProcessingError(AIBaseException):
    """Raised when an uploaded image cannot be decoded or processed."""


class InferenceError(AIBaseException):
    """Raised when a model prediction call fails unexpectedly."""
