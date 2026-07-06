"""
=============================================================
  Face Physiognomy Project — RAG Builder  (v2)
=============================================================

CHANGES FROM v1:
  - OCR-based extraction (EasyOCR) instead of PyMuPDF text layer
    because the book is a scanned PDF with no embedded text.
  - Chapter/page mapping instead of keyword detection
    because we have the book's table of contents as ground truth.

WHAT THIS FILE DOES:
  One-time script. Run it once to build the knowledge base
  from the physiognomy book PDF. Saves the index to disk.

PIPELINE:
  PDF page
    → render as image        (PyMuPDF)
    → OCR                    (EasyOCR)
    → clean text
    → assign region+chapter  (CHAPTER_MAPPING — page-based)
    → embed                  (sentence-transformers)
    → FAISS index
    → save to disk

HOW TO RUN ON KAGGLE:
  !pip install pymupdf easyocr sentence-transformers faiss-cpu -q
  !git clone https://github.com/YOUR_USERNAME/face-physiognomy-project.git
  import sys; sys.path.insert(0, 'face-physiognomy-project/src')

  from rag_builder import RAGBuilder
  builder = RAGBuilder(
      pdf_path   = "/kaggle/input/your-book/book.pdf",
      output_dir = "/kaggle/working/rag_index",
  )
  builder.build()

OUTPUT (save as Kaggle Dataset):
  rag_index/
  ├── index.faiss       ← FAISS search index
  ├── chunks.pkl        ← TextChunk objects (text + metadata)
  └── build_info.json   ← build stats for debugging

Install:
  pip install pymupdf easyocr sentence-transformers faiss-cpu
"""

import os
import json
import pickle
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
from collections import Counter

import numpy as np


# =============================================================
#  Chapter / Page Mapping
# =============================================================
#
#  CONCEPT: Why page-based mapping instead of keyword detection?
#
#  We have the book's table of contents — that is ground truth.
#  Keyword matching tries to GUESS what we already KNOW.
#  Page-based mapping is:
#    - 100% accurate (no guessing)
#    - Deterministic (same page always → same region)
#    - Extensible (add chapter granularity for free)
#
#  Structure per region:
#    "region_name": {
#        "pages"   : () tuble of pages 
#        "chapters": { page_num: "chapter_name", ... }
#    }
#
#  HOW IT WORKS AT RUNTIME:
#    chunk on page 29 → lookup → region="nose", chapter="nose_ridge"
#    chunk on page 17 → lookup → region="eyes", chapter="eyes_spacing"
#    chunk on page 5  → lookup → region="general", chapter="introduction"

CHAPTER_MAPPING: Dict[str, Dict] = {
    "forehead": {
        "pages": (11, 12, 61, 62),
        "chapters": {
            11: "forehead_shapes",
            12: "forehead_shapes",
            61: "lines",
            62: "lines",
        }
    },
    "eyebrows": {
        "pages": (13, 14,15,16),
        "chapters": {
            13: "eyebrows_basic_shapes",
            14: "eyebrows_position",
            15: "eyebrows_specific_types",
            16: "eyebrows_specific_types",
        }
    },
    "eyes": {
        "pages": (17, 18,19,20,21,22,23,24,25,26),
        "chapters": {
            17: "eyes_spacing",
            18: "eyes_angle",
            19: "eyes_depth",
            20: "eyes_corner_indents_and_eyes_iris_size",   
            21: "eyes_pupil_response",
            22: "eyes_stress_signs",
            23: "eyelids_top",
            24: "eyelids_bottom",
            25: "eyelashes",
            26: "eye_puffs",
        }
    },
    "nose": {
        "pages": (27, 28,29,30,31,32,33,34,35),
        "chapters": {
            27: "nose_size_shape",
            28: "nose_size_shape",
            29: "nose_ridge",
            30: "nose_width",
            31: "nose_tip_angle",
            32: "nose_tip_size_shape",
            33: "nose_tip_size_shape",
            34: "nostrils_size_shape",
            35: "nostrils_size_shape",
        }
    },
    "ears": {
        "pages": (36, 37,38,39,40,41),
        "chapters": {
            36: "ears_size",
            37: "ears_cups_ridges",
            38: "ears_placement",
            39: "ears_placement",
            40: "ears_height",
            #41: "ear_eyebrow_combinations",
        }
    },
    "mouth": {
        "pages": (44, 45,46,47,48,49),
        "chapters": {
            44: "mouth_size",
            45: "mouth_angle",
            46: "lips_size_shape",
            47: "lips_size_shape",
            48: "teeth",
            49: "smiles",
        }
    },

    "jaw_chin": {
        "pages": (42, 43, 50, 51, 52, 53, 54, 59, 60),
        "chapters": {
          42: "cheeks",
          43: "cheeks",
          50: "jaws",
          51: "chins",
          52: "chins",
          53: "chins",
          54: "chins",
          59: "dimples_clefts",
          60: "dimples_clefts",
        }
    },
    "face_overview": {
        "pages": (41, 55, 56, 57, 58, 63, 64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80),
        "chapters": {
            41: "ear_eyebrow_combinations",
            55: "chin_eyebrow_combinations",
            56: "chin_eyebrow_combinations",
            57: "chin_eyebrow_combinations",
            58: "chin_eyebrow_combinations",
            63: "lines",
            64: "lines",
            65: "lines",
            66: "lines",
            67: "facial_hair",
            68: "facial_hair",
            69: "face_shape",
            70: "face_shape",
            71: "face_shape",
            72: "face_types",
            73: "combination_face_types",
            74: "facial_dominance",
            75: "facial_dominance",
            76: "profile_types",
            77: "profile_types",
            78: "profile_combinations",
            79: "head_types",
            80: "head_types",
          
        }
    },
}


def get_region_and_chapter(page_num: int) -> Tuple[str, str]:
    """
    Look up which region and chapter a page belongs to.

    This replaces the old keyword detection entirely.
    One page → one region → one chapter. No ambiguity.

    Args:
        page_num : book page number (human-readable, 1-indexed)

    Returns:
        (region, chapter) tuple
        e.g. (29) → ("nose", "nose_ridge")
             (17) → ("eyes", "eyes_spacing")
             (5)  → ("general", "introduction")

    HOW IT WORKS:
        Iterates CHAPTER_MAPPING and checks if page_num falls
        within the (start, end) range of each region.
        Returns the chapter label for that specific page.
    """
  
    for region, data in CHAPTER_MAPPING.items():
        pages = data["pages"]               # ← نفس السطر
        if page_num in pages:              # ← indent صح
            chapter = data["chapters"].get(page_num, region)
            return region, chapter
    return "general", "introduction"

    # Page is outside all mapped ranges (intro, appendix, etc.)
    return "general", "introduction"


# =============================================================
#  Data Classes
# =============================================================

@dataclass
class TextChunk:
    """
    One searchable unit from the book.

    WHY store region + chapter separately?
      region  → broad filter  ("search only nose chunks")
      chapter → fine filter   ("search only nose_ridge chunks")
      The Agentic AI can decide which granularity to use.

    Example:
      chunk_id = "p29_3"
      page     = 29
      region   = "nose"
      chapter  = "nose_ridge"
      content  = "The nose ridge, or nasal bridge..."
    """
    chunk_id : str
    page     : int
    region   : str
    chapter  : str
    content  : str
    char_len : int = 0

    def __post_init__(self):
        self.char_len = len(self.content)


@dataclass
class BuildStats:
    """Collected during build — for debugging and logging."""
    pdf_path        : str  = ""
    pages_processed : int  = 0
    pages_empty     : int  = 0
    total_chunks    : int  = 0
    embedding_dim   : int  = 0
    region_dist     : dict = field(default_factory=dict)
    chapter_dist    : dict = field(default_factory=dict)
    embed_model     : str  = ""


# =============================================================
#  RAG Builder Class
# =============================================================

class RAGBuilder:
    """
    Builds the RAG knowledge base from a scanned PDF book.

    4-step pipeline:
      extract() → chunk() → embed() → save()
    All steps called automatically by build().

    Usage:
        builder = RAGBuilder(
            pdf_path   = "/path/to/book.pdf",
            output_dir = "rag_index",
            start_page = 11,   # first meaningful page (book-numbered)
            end_page   = 128,  # last page to process
        )
        builder.build()
    """

    def __init__(
        self,
        pdf_path         : str,
        output_dir       : str  = "rag_index",
        start_page       : int  = 11,    # book page 11 = first content page
        end_page         : int  = 81, # only the feature chapters in the book 
        min_chunk_len    : int  = 60,    # skip paragraphs shorter than this
        embed_model_name : str  = "all-MiniLM-L6-v2",
        ocr_gpu          : bool = False, # set True if Kaggle GPU is enabled
        render_dpi       : int  = 200,   # higher = better OCR, slower
    ):
        self.pdf_path         = pdf_path
        self.output_dir       = output_dir
        self.start_page       = start_page
        self.end_page         = end_page
        self.min_chunk_len    = min_chunk_len
        self.embed_model_name = embed_model_name
        self.ocr_gpu          = ocr_gpu
        self.render_dpi       = render_dpi

        # render_dpi → fitz zoom factor (72 DPI is fitz default)
        self.zoom = render_dpi / 72

        self.chunks     : List[TextChunk]      = []
        self.embeddings : Optional[np.ndarray] = None
        self.stats      = BuildStats(
            pdf_path    = pdf_path,
            embed_model = embed_model_name
        )

    # ----------------------------------------------------------
    #  Step 1: Extract Text via OCR
    # ----------------------------------------------------------

    def extract(self) -> List[Dict]:
        """
        Convert each PDF page to an image, then run EasyOCR.

        WHY render as image first?
          The PDF is scanned — pages are stored as images inside
          the PDF container. get_text() returns nothing because
          there is no text layer. We must:
            1. Render the page to a pixel image (PyMuPDF)
            2. Pass that image to an OCR engine (EasyOCR)

        WHY EasyOCR over Tesseract?
          - No system install needed (pure pip)
          - Better accuracy on clean scanned text
          - Simple API: reader.readtext(image)
          - Works well on Kaggle out of the box

        WHY render_dpi=200?
          Higher DPI = larger image = better OCR accuracy.
          200 DPI is a good balance between quality and speed.
          For very small text, try 250 or 300.

        Returns:
            List of {"page": int, "content": str}
            page is the BOOK page number (human-readable)
        """
        import fitz      # PyMuPDF — for rendering pages as images
        import easyocr   # OCR engine

        print(f"[1/4] Extracting text via OCR")
        print(f"      PDF   : {self.pdf_path}")
        print(f"      Pages : {self.start_page} → {self.end_page}")
        print(f"      DPI   : {self.render_dpi}  (zoom={self.zoom:.1f}x)")
        print(f"      GPU   : {self.ocr_gpu}")

        # Initialize EasyOCR — loads the model once, reused per page
        # lang_list=['en'] → English only (faster, more accurate for this book)
        print("      Loading EasyOCR model (first time may take ~30 sec)...")
        reader = easyocr.Reader(
            lang_list          = ['en'],
            gpu                = self.ocr_gpu,
            verbose            = False,
        )

        doc    = fitz.open(self.pdf_path)
        pages  = []
        matrix = fitz.Matrix(self.zoom, self.zoom)  # zoom matrix for rendering

        # PDF page index is 0-based, book page numbers are 1-based.
        # We assume book page N = PDF page index N-1.
        # Adjust pdf_page_offset if your PDF has extra pages at the start.
        pdf_page_offset = 14   # change if book page 11 ≠ PDF index 10

        for book_page in range(self.start_page, self.end_page + 1):
            pdf_idx = book_page - 1 + pdf_page_offset

            if pdf_idx >= len(doc):
                print(f"      WARNING: book page {book_page} → "
                      f"PDF index {pdf_idx} out of range. Stopping.")
                break

            # ── Render page as image ───────────────────────────
            #
            # get_pixmap() renders the PDF page to a pixel grid.
            # matrix controls the zoom (resolution).
            # clip=None → render the full page.
            #
            pix = doc[pdf_idx].get_pixmap(matrix=matrix, alpha=False)

            # Convert pixmap to numpy array for EasyOCR
            # pix.samples = raw bytes (RGB)
            # reshape to (height, width, channels)
            img = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img.reshape(pix.h, pix.w, pix.n)

            # ── Run OCR ───────────────────────────────────────
            #
            # readtext() returns a list of:
            #   [bounding_box, text, confidence]
            #
            # detail=0 → return only the text strings (faster)
            # paragraph=True → group nearby text into paragraphs
            #
            ocr_results = reader.readtext(img, detail=0, paragraph=True)

            # Join all detected text blocks into one string
            raw_text = "\n".join(ocr_results).strip()

            if raw_text:
                pages.append({
                    "page"    : book_page,
                    "content" : raw_text,
                })
                print(f"      p{book_page:>3} ✓  ({len(raw_text):>5} chars)")
            else:
                self.stats.pages_empty += 1
                print(f"      p{book_page:>3} ✗  (empty — check PDF offset)")

        doc.close()
        self.stats.pages_processed = len(pages)
        print(f"\n      Done: {len(pages)} pages extracted, "
              f"{self.stats.pages_empty} empty")
        return pages

    # ----------------------------------------------------------
    #  Step 2: Chunk the Text
    # ----------------------------------------------------------

    def chunk(self, pages: List[Dict]) -> List[TextChunk]:
        """
        Split each page's text into paragraph-level chunks.
        Assign region + chapter from CHAPTER_MAPPING (page-based).

        WHY paragraph-level chunks?
          - Fits within the embedding model's context window (~512 tokens)
          - More precise search results (paragraph vs full page)
          - Each chunk focuses on one idea

        WHY NOT overlap-based chunking?
          Our chunks are naturally bounded by paragraphs.
          Overlap (e.g. LangChain's RecursiveCharacterTextSplitter)
          is better for dense technical text with no clear breaks.
          For a face-reading book with short paragraphs, paragraph
          splitting is cleaner and more interpretable.

        HOW REGION IS ASSIGNED:
          chunk on page 29 → get_region_and_chapter(29)
                           → ("nose", "nose_ridge")
          No keyword guessing. Deterministic. Always correct.
        """
        print(f"\n[2/4] Chunking text")

        chunks  = []
        counter = 0

        for page_data in pages:
            book_page = page_data["page"]
            region, chapter = get_region_and_chapter(book_page)

            # Split on double newline (paragraph boundary)
            # EasyOCR with paragraph=True already groups text,
            # so each item in the OCR output is roughly a paragraph.
            # We split again here in case multiple paragraphs were merged.
            raw_paragraphs = page_data["content"].split("\n\n")

            # If no double newlines, treat each line as a chunk
            if len(raw_paragraphs) == 1:
                raw_paragraphs = page_data["content"].split("\n")

            for para in raw_paragraphs:
                clean = " ".join(para.split())   # normalize whitespace

                # Skip chunks that are too short
                # (OCR noise, headers, page numbers, etc.)
                if len(clean) < self.min_chunk_len:
                    continue

                chunk = TextChunk(
                    chunk_id = f"p{book_page}_{counter}",
                    page     = book_page,
                    region   = region,
                    chapter  = chapter,
                    content  = clean,
                )
                chunks.append(chunk)
                counter += 1

        # Stats
        self.stats.total_chunks = len(chunks)
        self.stats.region_dist  = dict(Counter(c.region  for c in chunks))
        self.stats.chapter_dist = dict(Counter(c.chapter for c in chunks))

        print(f"      Total chunks: {len(chunks)}")
        print(f"\n      Region distribution:")
        for region, count in sorted(self.stats.region_dist.items(),key=lambda x: -x[1]):
            bar = "█" * (count // 2)
            print(f" {region:<20} {count:>4}  {bar}")

        self.chunks = chunks
        return chunks

    # ----------------------------------------------------------
    #  Step 3: Embed the Chunks
    # ----------------------------------------------------------

    def embed(self, chunks: List[TextChunk]) -> np.ndarray:
        """
        Convert each chunk's text into a vector (embedding).

        CONCEPT: What is an embedding?
          A list of numbers that represents the MEANING of text.
          Similar meanings → similar vectors (close in space).

          "wide nose with flared nostrils" → [0.2, -0.5, 0.8, ...]
          "broad nasal with open nostrils" → [0.19, -0.48, 0.79, ...]
          "blue sky today"                 → [-0.9,  0.3, -0.1, ...]

          The first two are CLOSE. The third is FAR.
          This is how RAG finds relevant results without exact keywords.

        MODEL: all-MiniLM-L6-v2
          - 384-dimensional vectors
          - Fast, accurate, free, English-optimized
          - Max input: ~256 words per chunk (our chunks are shorter)

        Returns:
            numpy float32 array, shape = (num_chunks, 384)
        """
        from sentence_transformers import SentenceTransformer

        print(f"\n[3/4] Embedding {len(chunks)} chunks")
        print(f"      Model: {self.embed_model_name}")

        model = SentenceTransformer(self.embed_model_name)
         # Embed the main region and the sub sregion "chabter" to the chunk in order to make the retriveral easy to detect the area 
        texts      = texts = [f""" Region: {c.region} Chapter: {c.chapter} {c.content}"""for c in chunks]
        embeddings = model.encode(
                                  texts,
                                  show_progress_bar = True,
                                  batch_size = 32,
                                  convert_to_numpy  = True,
                                  ).astype("float32")

        self.stats.embedding_dim = embeddings.shape[1]
        print(f" Shape: {embeddings.shape}  (chunks × vector_dim)")

        self.embeddings = embeddings
        return embeddings

    # ----------------------------------------------------------
    #  Step 4: Build FAISS Index and Save
    # ----------------------------------------------------------

    def save(self, embeddings: np.ndarray):
        """
        Build FAISS index and save everything to disk.

        CONCEPT: What is FAISS?
          Facebook AI Similarity Search.
          Stores all vectors and finds the NEAREST ones to a query
          very fast — even with millions of vectors.

          IndexFlatL2:
            "Flat"  = no compression (exact search, not approximate)
            "L2"    = Euclidean distance between vectors
            Best for our size (~500-1000 chunks). Exact and fast.

        WHAT GETS SAVED:
          index.faiss   → the vectors + FAISS search structure
          chunks.pkl    → original TextChunk objects (text + metadata)
                          FAISS returns indices — we look up text here
          build_info.json → stats for debugging

        WHY TWO SEPARATE FILES?
          FAISS only stores numbers (vectors), not text.
          When search returns "index 42 is closest", we need chunks.pkl
          to find out what chunk 42 actually says.
          The index in chunks[42] matches the FAISS vector index 42.
        """
        import faiss

        print(f"\n[4/4] Building FAISS index")
        os.makedirs(self.output_dir, exist_ok=True)

        # Build index
        dim   = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        print(f" Vectors: {index.ntotal}, dim={dim}")

        # Save FAISS index
        faiss_path = os.path.join(self.output_dir, "index.faiss")
        faiss.write_index(index, faiss_path)
        print(f" Saved : {faiss_path}")

        # Save chunks
        chunks_path = os.path.join(self.output_dir, "chunks.pkl")
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        print(f" Saved : {chunks_path}")

        # Save build info
        info_path = os.path.join(self.output_dir, "build_info.json")
        with open(info_path, "w") as f:
            json.dump(asdict(self.stats), f, indent=2)
        print(f" Saved : {info_path}")

        print(f"\n✓ RAG index ready in '{self.output_dir}/'")
        print(f"  Next step: upload this folder as a Kaggle Dataset")

    # ----------------------------------------------------------
    #  Main Entry Point
    # ----------------------------------------------------------

    def build(self):
        """
        Run the full pipeline: extract → chunk → embed → save.

        Runtime on Kaggle CPU: ~15-30 min (OCR is the slow step).
        Runtime on Kaggle GPU: ~5-10 min (EasyOCR uses GPU).
        """
        print("=" * 55)
        print("  RAG Builder v2 — Face Physiognomy Project")
        print("=" * 55)

        pages      = self.extract()
        chunks     = self.chunk(pages)
        embeddings = self.embed(chunks)
        self.save(embeddings)

        print("\n" + "=" * 55)
        print(f"  Total chunks  : {self.stats.total_chunks}")
        print(f"  Embedding dim : {self.stats.embedding_dim}")
        print(f"  Regions found : {len(self.stats.region_dist)}")
        print("=" * 55)

        return self.stats
