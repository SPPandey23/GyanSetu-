import os
import hashlib
import pickle
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter

from config import constants
from config.settings import settings
from utils.logging import logger


class DocProcessor:
    SUPPORTED_TYPES = ('.pdf', '.docx', '.txt', '.md')

    def __init__(self):
        self.headers = [("#", "Header 1"), ("##", "Header 2")]
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def validate_files(self, files: List) -> None:
        total_size = sum(f.size for f in files)
        if total_size > constants.MAX_TOTAL_SIZE:
            raise ValueError(
                f"Upload too large! Max allowed: {constants.MAX_TOTAL_SIZE // 1024 // 1024} MB"
            )

    def process(self, files: List) -> List:
        self.validate_files(files)
        all_chunks = []
        seen = set()

        for file in files:
            try:
                chunks = self._get_chunks(file)
                for chunk in chunks:
                    content = chunk.page_content.strip()
                    if content not in seen:
                        seen.add(content)
                        all_chunks.append(chunk)
                logger.info(f"Processed {file.name}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to process {file.name}: {str(e)}")
                continue

        logger.info(f"Total unique chunks across files: {len(all_chunks)}")
        return all_chunks

    def _get_chunks(self, file) -> List:
        file_hash = self._hash_file(file)
        cache_path = self.cache_dir / f"{file_hash}.pkl"

        if self._is_cache_valid(cache_path):
            logger.info(f"[CACHE HIT] Skipping OCR for: {file.name}")
            return self._load_cache(cache_path)

        logger.info(f"[OCR] Processing: {file.name}")
        chunks = self._process_file(file)
        self._save_cache(chunks, cache_path)
        return chunks

    def _process_file(self, file) -> List:
        if not file.name.endswith(self.SUPPORTED_TYPES):
            logger.warning(f"Skipping unsupported file: {file.name}")
            return []

        suffix = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            converter = DocumentConverter()
            markdown = converter.convert(tmp_path).document.export_to_markdown()
            splitter = MarkdownHeaderTextSplitter(self.headers)
            return splitter.split_text(markdown)
        finally:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass

    

    def _hash_file(self, file) -> str:
        file.seek(0)
        content = file.read()
        file.seek(0)
        return hashlib.sha256(content).hexdigest()

   
    def _is_cache_valid(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return age < timedelta(days=settings.CACHE_EXPIRE_DAYS)

    def _save_cache(self, chunks: List, cache_path: Path) -> None:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"timestamp": datetime.now().timestamp(), "chunks": chunks}, f)
        except Exception as e:
            logger.error(f"Failed to save cache {cache_path}: {str(e)}")

    def _load_cache(self, cache_path: Path) -> List:
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)["chunks"]
        except Exception as e:
            logger.error(f"Failed to load cache {cache_path}: {str(e)}")
            return []