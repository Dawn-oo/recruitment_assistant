from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentParseResult:

    markdown: str

    content_list: list | dict | None = None

    document_hash: str | None = None

    parser_task_id: str | None = None

    raw_result_path: str | None = None

    from_cache: bool = False


class DocumentParser(ABC):

    @abstractmethod
    def parse(
        self,
        file_path: str | Path,
    ) -> DocumentParseResult:
        """
        文档 -> 标准化文档解析结果
        """
        pass