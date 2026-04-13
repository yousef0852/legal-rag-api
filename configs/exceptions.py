class LegalRAGException(Exception):
    def __init__(self, message: str, error_code: str = "UNKNOWN_ERROR", details: dict = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict:
        return {"error": self.error_code, "message": self.message, "details": self.details}


class IndexError(LegalRAGException):
    def __init__(self, reason: str):
        super().__init__(message=f"Indexing failed: {reason}", error_code="INDEX_ERROR", details={"reason": reason})


class RetrievalError(LegalRAGException):
    def __init__(self, reason: str):
        super().__init__(message=f"Retrieval failed: {reason}", error_code="RETRIEVAL_ERROR", details={"reason": reason})


class LLMError(LegalRAGException):
    def __init__(self, reason: str, error_code: str = "LLM_ERROR"):
        super().__init__(message=f"LLM call failed: {reason}", error_code=error_code, details={"reason": reason})
