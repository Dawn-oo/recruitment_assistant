from abc import ABC, abstractmethod
from typing import Sequence

class RerankerProvider(ABC):

    @abstractmethod
    def score_pairs(self,pairs: Sequence[tuple[str, str]]) -> list[float]:
        pass