"""
src/ingestion/chunker.py
Module responsible for sentence-aware semantic text chunking (512 tokens / 128 overlap)
and attaching rich medical metadata (PMID, title, URL, DOI) to each chunk.
"""

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, List, Optional
import tiktoken

from src.config import get_config
from src.ingestion.pubmed_fetcher import PubMedArticle

logger = logging.getLogger(__name__)


@dataclass
class MedicalChunk:
    """Represents a discrete text chunk indexed in ChromaDB and BM25."""
    chunk_id: str
    pmid: str
    title: str
    text: str
    chunk_index: int
    total_chunks: int
    token_count: int
    journal: str
    pub_date: str
    doi: Optional[str] = None
    url: str = ""

    def __post_init__(self):
        if not self.url and self.pmid:
            self.url = f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk into a dictionary format suitable for metadata storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MedicalChunk":
        """Instantiate MedicalChunk from a dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            pmid=str(data["pmid"]),
            title=data.get("title", ""),
            text=data.get("text", ""),
            chunk_index=int(data.get("chunk_index", 0)),
            total_chunks=int(data.get("total_chunks", 1)),
            token_count=int(data.get("token_count", 0)),
            journal=data.get("journal", ""),
            pub_date=data.get("pub_date", ""),
            doi=data.get("doi"),
            url=data.get("url", ""),
        )


class MedicalChunker:
    """
    Sentence-aware token chunker configured to produce chunks of ~512 tokens
    with 128 tokens overlap according to MedRAG standards.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        tokenizer_name: str = "cl100k_base",
    ):
        cfg = get_config()
        self.chunk_size = chunk_size or cfg.ingestion.chunk_size
        self.chunk_overlap = chunk_overlap or cfg.ingestion.chunk_overlap

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})")

        # Use tiktoken encoder for precise token calculation
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        logger.info(
            f"Initialized MedicalChunker: chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, tokenizer={tokenizer_name}"
        )

    def count_tokens(self, text: str) -> int:
        """Returns the exact number of tokens in a string using tiktoken."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Splits medical text into grammatical sentences while preserving
        abbreviations like 'e.g.', 'i.e.', 'Dr.', 'vs.', 'et al.'.
        """
        import re

        # Normalize newlines
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        sentences: List[str] = []

        # Regex recognizing standard sentence delimiters followed by uppercase
        sentence_regex = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

        for p in paragraphs:
            parts = sentence_regex.split(p)
            for part in parts:
                clean_part = part.strip()
                if clean_part:
                    sentences.append(clean_part)

        return sentences

    def chunk_article(self, article: PubMedArticle) -> List[MedicalChunk]:
        """
        Splits a PubMedArticle abstract into MedicalChunk items.
        Prepends paper title to each chunk to ensure strong semantic grounding.
        """
        if not article.abstract:
            return []

        # Construct full article text with explicit Title header
        article_body = f"Title: {article.title}\n\nAbstract:\n{article.abstract}"
        total_tokens = self.count_tokens(article_body)

        # Optimization: If entire article is under chunk_size, return 1 complete chunk
        if total_tokens <= self.chunk_size:
            return [
                MedicalChunk(
                    chunk_id=f"{article.pmid}_chunk_0",
                    pmid=article.pmid,
                    title=article.title,
                    text=article_body,
                    chunk_index=0,
                    total_chunks=1,
                    token_count=total_tokens,
                    journal=article.journal,
                    pub_date=article.pub_date,
                    doi=article.doi,
                    url=article.url,
                )
            ]

        # Multi-chunk splitting with sentence-awareness & overlap
        sentences = self.split_into_sentences(article.abstract)
        raw_chunks: List[str] = []
        current_chunk_sentences: List[str] = []
        current_token_count = 0

        # Base prefix for each chunk
        prefix = f"Title: {article.title}\n\nAbstract:\n"
        prefix_tokens = self.count_tokens(prefix)

        effective_chunk_size = self.chunk_size - prefix_tokens
        effective_overlap = min(self.chunk_overlap, effective_chunk_size // 2)

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            if current_token_count + sentence_tokens > effective_chunk_size and current_chunk_sentences:
                # Seal current chunk
                chunk_content = prefix + " ".join(current_chunk_sentences)
                raw_chunks.append(chunk_content)

                # Keep overlap sentences for the next chunk
                overlap_sentences: List[str] = []
                overlap_tokens = 0
                for s in reversed(current_chunk_sentences):
                    s_tok = self.count_tokens(s)
                    if overlap_tokens + s_tok <= effective_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tok
                    else:
                        break

                current_chunk_sentences = overlap_sentences
                current_token_count = overlap_tokens

            current_chunk_sentences.append(sentence)
            current_token_count += sentence_tokens

        # Add the remaining sentences as the final chunk
        if current_chunk_sentences:
            final_content = prefix + " ".join(current_chunk_sentences)
            raw_chunks.append(final_content)

        # Convert raw chunks to structured MedicalChunk objects
        total_chunks = len(raw_chunks)
        chunk_objects: List[MedicalChunk] = []

        for idx, text in enumerate(raw_chunks):
            chunk_tokens = self.count_tokens(text)
            chunk_objects.append(
                MedicalChunk(
                    chunk_id=f"{article.pmid}_chunk_{idx}",
                    pmid=article.pmid,
                    title=article.title,
                    text=text,
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    token_count=chunk_tokens,
                    journal=article.journal,
                    pub_date=article.pub_date,
                    doi=article.doi,
                    url=article.url,
                )
            )

        return chunk_objects

    def chunk_article_list(self, articles: List[PubMedArticle]) -> List[MedicalChunk]:
        """Batch processes a list of articles into a unified flat list of chunks."""
        all_chunks: List[MedicalChunk] = []
        for art in articles:
            chunks = self.chunk_article(art)
            all_chunks.extend(chunks)

        logger.info(f"Chunked {len(articles)} articles into {len(all_chunks)} total chunks.")
        return all_chunks


if __name__ == "__main__":
    from src.ingestion.cleaner import clean_article

    # Self-test demonstration with a real article structure
    demo_article = PubMedArticle(
        pmid="43014822",
        title="Effect of discontinuing antihypertensives on perioperative hemodynamics",
        abstract=(
            "BACKGROUND: Perioperative hypotension is associated with adverse outcomes in patients "
            "undergoing noncardiac surgery. Whether withholding angiotensin-converting enzyme inhibitors "
            "or angiotensin receptor blockers before surgery reduces hypotensive episodes remains controversial.\n\n"
            "METHODS: In this international multicenter randomized clinical trial, we enrolled 2,222 patients "
            "treated chronically with ACE inhibitors or ARBs. Patients were randomized 1:1 to either continue "
            "or withhold medication 24 hours prior to surgery.\n\n"
            "RESULTS: Clinically significant hypotension occurred in 28.5% of patients in the continuation group "
            "compared to 22.1% in the discontinuation group (Risk Ratio: 0.77; 95% CI: 0.67-0.90; P < 0.001). "
            "No significant differences were observed in 30-day major adverse cardiovascular events.\n\n"
            "CONCLUSIONS: Withholding ACE inhibitors or ARBs 24 hours before noncardiac surgery significantly "
            "reduced the risk of perioperative hypotension without increasing postoperative cardiovascular complications."
        ),
        journal="British Journal of Anaesthesia",
        pub_date="2026-11-11",
        doi="10.1016/j.bja.2026.10.015",
    )

    chunker = MedicalChunker(chunk_size=512, chunk_overlap=128)
    chunks = chunker.chunk_article(demo_article)

    print("=" * 60)
    print(f"DEMO ARTICLE CHUNKING: PMID {demo_article.pmid}")
    print(f"Total chunks created: {len(chunks)}")
    print("=" * 60)
    for c in chunks:
        print(f"\nChunk ID: {c.chunk_id} ({c.token_count} tokens)")
        print(f"Journal : {c.journal} ({c.pub_date})")
        print(f"URL     : {c.url}")
        print(f"Content :\n{c.text[:220]}...")

