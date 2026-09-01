from __future__ import annotations

import json
import os
import time
import zipfile
import requests
import logging
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from .base import DocumentParseResult, DocumentParser
from app.tools.hash_fun import calculate_file_hash


_ = load_dotenv()
logger = logging.getLogger(__name__)

path = Path(__file__).resolve().parents[0]
CACHE_DIR = path / "resume_parser_cache"


class MinerUError(Exception):
    """document_parser API 异常。"""


class MinerUParser(DocumentParser):

    BASE_URL = "https://mineru.net/api/v4"

    def __init__(
        self,
        token: str | None = None,
        model_version: str = "vlm",
        poll_interval: int = 3,
        timeout: int = 300,
        cache_dir: str | Path = CACHE_DIR,
    ):
        self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.token = token or os.getenv("MINERU_API_TOKEN")

        if not self.token:
            raise ValueError(
                "没有找到 MINERU_API_TOKEN，请在 .env 中配置 Token"
            )

        self.model_version = model_version
        self.poll_interval = poll_interval
        self.timeout = timeout

        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def apply_upload_url(
        self,
        file_path: str | Path,
        *,
        data_id: str | None = None,
        is_ocr: bool = False,
        enable_table: bool = True,
        enable_formula: bool = False,
        language: str = "ch",
    ) -> tuple[str, str]:
        """
        向 document_parser 申请本地文件上传 URL。

        Returns:
            (batch_id, upload_url)
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        if not path.is_file():
            raise ValueError(f"不是有效文件: {path}")

        file_info: dict[str, Any] = {
            "name": path.name,
            "is_ocr": is_ocr,
        }

        if data_id is not None:
            file_info["data_id"] = data_id

        payload = {
            "files": [file_info],
            "model_version": self.model_version,
            "language": language,
            "enable_table": enable_table,
            "enable_formula": enable_formula,
        }

        response = requests.post(
            f"{self.BASE_URL}/file-urls/batch",
            headers=self.headers,
            json=payload,
            timeout=30,
        )

        response.raise_for_status()

        result = response.json()

        if result.get("code") != 0:
            raise MinerUError(
                f"申请上传 URL 失败: "
                f"code={result.get('code')}, "
                f"msg={result.get('msg')}"
            )

        data = result["data"]

        batch_id = data["batch_id"]
        file_urls = data["file_urls"]

        if not file_urls:
            raise MinerUError("document_parser 未返回文件上传 URL")

        return batch_id, file_urls[0]

    def parse(
        self,
        file_path: str | Path,
        cache_dir: str | Path = None,
        *,
        data_id: str | None = None,
        is_ocr: bool = False,
    ) -> DocumentParseResult:

        if cache_dir is None:
            cache_dir = self.cache_dir

        # 传入的是原始文件的路径
        path = Path(file_path)
        file_hash = calculate_file_hash(path)
        cache_path = Path(cache_dir) / file_hash
        zip_path = cache_path / f"{file_hash}.zip"
        result_dir = cache_path / "result"

        if zip_path.exists() and zip_path.is_file():

            logger.debug(f"发现已有解析结果 {path.name}")
            if not result_dir.exists():
                self.extract_zip(zip_path, result_dir)

            markdown = self.find_markdown(result_dir)
            content_list = self.find_content_list(result_dir)

            return DocumentParseResult(
                markdown=markdown,
                content_list=content_list,
                document_hash=file_hash,
                raw_result_path=str(cache_path),
                from_cache=True
            )

        else:
            logger.debug("未发现已有解析结果")
            logger.debug(f"开始解析: {path.name}")

            try:
                result = self._parse_with_mineru(
                    file_path, data_id=data_id,is_ocr=is_ocr,
                )
            except Exception as e:
                raise Exception(f"使用mineru解析失败 {e}")
            return result


    @staticmethod
    def upload_file(file_path: str | Path,upload_url: str) -> None:
        """
        将本地文件 PUT 到 document_parser 返回的签名 URL。
        """

        path = Path(file_path)

        with path.open("rb") as file:
            response = requests.put(
                upload_url,
                data=file,
                timeout=120,
            )

        if response.status_code != 200:
            raise MinerUError(
                f"上传文件失败: "
                f"HTTP {response.status_code}, "
                f"{response.text[:500]}"
            )

    def get_batch_result(self,batch_id: str) -> dict[str, Any]:
        """
        查询解析任务。
        """

        response = requests.get(
            f"{self.BASE_URL}/extract-results/batch/{batch_id}",
            headers=self.headers,
            timeout=30,
        )

        response.raise_for_status()

        result = response.json()

        if result.get("code") != 0:
            raise MinerUError(
                f"查询任务失败: "
                f"code={result.get('code')}, "
                f"msg={result.get('msg')}"
            )

        return result["data"]

    def wait_until_done(self,batch_id: str) -> str:
        """
        轮询 document_parser，直到解析完成。
        """

        start_time = time.monotonic()

        while True:
            if time.monotonic() - start_time > self.timeout:
                raise TimeoutError(
                    f"document_parser 解析超时，batch_id={batch_id}"
                )

            data = self.get_batch_result(batch_id)

            extract_results = data.get("extract_result", [])

            if not extract_results:
                time.sleep(self.poll_interval)
                continue

            result = extract_results[0]

            state = result.get("state")

            if state == "running":
                progress = result.get("extract_progress", {})

                extracted_pages = progress.get("extracted_pages")
                total_pages = progress.get("total_pages")

                if (
                    extracted_pages is not None
                    and total_pages is not None
                ):
                    print(
                        f"document_parser 正在解析: "
                        f"{extracted_pages}/{total_pages} 页"
                    )
                else:
                    print("文件正在解析...")

            elif state == "pending":
                print("document_parser 任务排队中...")

            elif state == "waiting-file":
                print("document_parser 等待文件上传...")

            elif state == "converting":
                print("document_parser 正在转换结果...")

            elif state == "done":
                full_zip_url = result.get("full_zip_url")

                if not full_zip_url:
                    raise MinerUError(
                        "任务完成，但没有返回 full_zip_url"
                    )

                return full_zip_url

            elif state == "failed":
                raise MinerUError(
                    f"document_parser 解析失败: "
                    f"{result.get('err_msg', 'unknown error')}"
                )

            else:
                print(f"未知 document_parser 状态: {state}")

            time.sleep(self.poll_interval)

    @staticmethod
    def download_zip(zip_url: str,output_path: str | Path) -> Path:
        """
        下载 document_parser 返回的解析结果 ZIP。
        """

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with requests.get(
            zip_url,
            stream=True,
            timeout=120,
        ) as response:
            response.raise_for_status()

            with output_path.open("wb") as file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        file.write(chunk)

        return output_path

    @staticmethod
    def extract_zip(zip_path: str | Path,output_dir: str | Path) -> Path:
        """
        解压 document_parser 解析结果。
        """

        zip_path = Path(zip_path)
        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(output_dir)

        return output_dir

    @staticmethod
    def find_markdown(result_dir: str | Path) -> str:
        """
        从 document_parser 结果目录中读取 full.md。
        """

        result_dir = Path(result_dir)

        markdown_files = list(
            result_dir.rglob("full.md")
        )

        if not markdown_files:
            raise FileNotFoundError(
                f"没有在 {result_dir} 中找到 full.md"
            )

        return markdown_files[0].read_text(
            encoding="utf-8"
        )

    @staticmethod
    def find_content_list(result_dir: str | Path) -> list | dict | None:
        """
        查找 document_parser 的 content_list.json。
        """

        result_dir = Path(result_dir)

        files = list(
            result_dir.rglob("*_content_list.json")
        )

        if not files:
            files = list(
                result_dir.rglob("content_list.json")
            )

        if not files:
            return None

        with files[0].open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def _parse_with_mineru(self,file_path: Path,output_dir: str | Path = None,*,data_id: str | None = None,
        is_ocr: bool = False,) -> DocumentParseResult:
        """
        完整流程：

        本地文件
        -> 获取上传 URL
        -> 上传
        -> 等待解析
        -> 下载 ZIP
        -> 解压
        -> 返回 Markdown / content_list
        """
        # 先计算原始文件的哈希值，作为结果zip的文件名
        path = Path(file_path)
        file_hash_key = calculate_file_hash(path)
        logger.debug(f"文件哈希值计算完成: {file_hash_key}")

        if output_dir is None:
            output_dir = self.cache_dir

        start = time.perf_counter()
        # 1. 获取上传地址
        batch_id, upload_url = self.apply_upload_url(
            path,
            data_id=data_id,
            is_ocr=is_ocr,
        )

        logger.debug(f"文件上传地址获取成功, 文件id为: {batch_id}")

        # 2. 上传 PDF

        self.upload_file(path,upload_url)

        logger.debug(f"文件{path.name}上传MinerU平台成功，等待解析完成...,batch_id={batch_id}")

    # 3. 等待解析完成
        zip_url = self.wait_until_done(batch_id)

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(f"MinerU 解析{path.name}文件完成，batch_id={batch_id},耗时：{elapsed_ms}毫秒")

        # 每个任务单独保存
        task_dir = Path(output_dir) / file_hash_key

        zip_path = task_dir / f"{file_hash_key}.zip"
        extract_dir = task_dir / "result"

        # 4. 下载 ZIP
        try:
            self.download_zip(zip_url,zip_path)
            logger.debug(f"文件{path.name}下载完成，结果保存到{zip_path}")
        except Exception as e:
            raise Exception(f"文件{path.name}下载失败: {e}，注意检查网络连接，不要使用代理下载")

        # 5. 解压
        self.extract_zip(zip_path,extract_dir)
        logger.debug(f"文件{path.name}解压完成，结果保存到{extract_dir}")

        # 6. 获取 Markdown
        markdown = self.find_markdown(extract_dir)
        logger.debug(f"文件{path.name}解析完成")

        # 7. 获取 content_list
        content_list = self.find_content_list(extract_dir)

        return DocumentParseResult(
            markdown=markdown,
            content_list=content_list,
            parser_task_id=batch_id,
            raw_result_path=str(extract_dir),
            document_hash=file_hash_key
        )

