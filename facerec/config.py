from dataclasses import dataclass
from typing import List

# pretty self explanatory but essentially this is the config for the face engine
@dataclass
class EngineConfig:
    model_name: str = "buffalo_l"
    det_size: int = 640
    provider: str = "cpu"
    min_face_score: float = 0.5
# above is ^ the model and what it decides to use for detection 
    def provider_chain(self) -> List[str]:
        provider = self.provider.strip().lower()
        if provider == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def ctx_id(self) -> int:
        return 0 if self.provider.strip().lower() == "cuda" else -1
