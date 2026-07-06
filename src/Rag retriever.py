"""
=============================================================
  Face Physiognomy Project — RAG Retriever (V2)
=============================================================

CHANGE FROM V1:
  search_all_parts() now accepts SectionDescription objects
  (from FaceDescriber V2) instead of the old DescriptionResult format.

  SectionDescription.features_json structure:
    { "region": { "feature": {"value": "...", "confidence": 0.0} } }

  Query building extracts only features with value != null
  and confidence >= MIN_CONFIDENCE, converts them to natural
  language, and searches the book by region.
"""

import pickle
from typing import Dict, List, Optional, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from face_describer import SectionDescription


# =============================================================
#  Configuration
# =============================================================

MIN_CONFIDENCE = 0.5   # features below this are excluded from RAG query


# =============================================================
#  Query Builder
# =============================================================

def build_query_from_section(
    region        : str,
    region_features : Dict[str, Any],
) -> str:
    """
    Convert region features JSON to a natural language RAG query.

    Input (V2 format):
      {
        "tip_shape"    : {"value": "small_ball", "confidence": 0.9},
        "nostril_shape": {"value": "round",      "confidence": 0.8},
        "nose_ridge"   : {"value": "high_ridge", "confidence": 0.7},
        "bump_on_bridge": {"value": true,        "confidence": 0.6},
      }

    Output:
      "small ball tip nose with round nostrils and high ridge"

    WHY natural language?
      The book chunks were written in natural English sentences.
      A natural language query produces closer embedding matches
      than a list of keywords.

    Rules:
      - Only include features with confidence >= MIN_CONFIDENCE
      - Skip null / false / absent values
      - Replace underscores with spaces
      - True/false features: include feature name only if True
    """
    parts = []
    region_clean = region.replace("_", " ")

    for feature_name, feature_data in region_features.items():
        if not isinstance(feature_data, dict):
            continue

        value      = feature_data.get("value")
        confidence = feature_data.get("confidence", 1.0)

        # Skip absent or uncertain features
        if value is None or value == "null" or value is False:
            continue
        if isinstance(confidence, (int, float)) and confidence < MIN_CONFIDENCE:
            continue

        clean_name = feature_name.replace("_", " ")

        if value is True:
            # Boolean feature — include the feature name as descriptor
            parts.append(clean_name)
        elif isinstance(value, str):
            clean_value = value.replace("_", " ")
            parts.append(f"{clean_value} {clean_name}")

    if not parts:
        return region_clean

    if len(parts) == 1:
        return f"{parts[0]} {region_clean}"
    elif len(parts) == 2:
        return f"{parts[0]} {region_clean} with {parts[1]}"
    else:
        rest = " and ".join(parts[1:])
        return f"{parts[0]} {region_clean} with {rest}"


def extract_regions_from_sections(
    descriptions: Dict[str, SectionDescription],
) -> Dict[str, Dict[str, Any]]:
    """
    Flatten 3 SectionDescription objects into a per-region dict.

    Input:
      {
        "section_1": SectionDescription(features_json={
          "forehead": {...}, "eyebrows": {...}, "eyes": {...}
        }),
        "section_2": SectionDescription(features_json={
          "eyes": {...}, "nose": {...}, "cheeks": {...}
        }),
        "section_3": SectionDescription(features_json={
          "mouth": {...}, "jaw": {...}, "chin": {...}
        }),
      }

    Output:
      {
        "forehead": {...},
        "eyebrows": {...},
        "eyes"    : {...},   # merged from section_1 + section_2
        "nose"    : {...},
        "cheeks"  : {...},
        "mouth"   : {...},
        "jaw"     : {...},
        "chin"    : {...},
      }

    WHY merge eyes?
      Eyes appear in section_1 (lids, lashes, puffs) and
      section_2 (white showing). Merging gives the RAG
      the full picture for eye-related retrieval.
    """
    merged: Dict[str, Dict] = {}

    for section_id in ["section_1", "section_2", "section_3"]:
        desc = descriptions.get(section_id)
        if not desc or not desc.success or not desc.features_json:
            continue

        for region, features in desc.features_json.items():
            if not isinstance(features, dict):
                continue
            if region not in merged:
                merged[region] = {}
            merged[region].update(features)

    return merged


# =============================================================
#  RAG Retriever
# =============================================================

class PhysiognomyRetriever:
    """
    Semantic search over the physiognomy book knowledge base.

    Accepts V2 SectionDescription objects and searches
    the FAISS index per region.

    Usage:
        retriever = PhysiognomyRetriever(
            index_path  = "/path/to/index.faiss",
            chunks_path = "/path/to/chunks.pkl",
        )
        evidence = retriever.search_all_parts(descriptions, top_k=3)
    """

    def __init__(
        self,
        index_path  : str,
        chunks_path : str,
        model_name  : str = "all-MiniLM-L6-v2",
    ):
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        self.model = SentenceTransformer(model_name)
        print(f"RAG Retriever ready — {len(self.chunks)} chunks")

    # ----------------------------------------------------------
    #  Core Search
    # ----------------------------------------------------------

    def search(
        self,
        query   : str,
        region  : Optional[str] = None,
        chapter : Optional[str] = None,
        top_k   : int           = 3,
    ) -> List[Dict]:
        """
        Semantic search over the book.

        Args:
            query   : natural language description
            region  : optional filter (e.g. "nose")
            chapter : optional fine filter (e.g. "nose_ridge")
            top_k   : number of results to return

        Returns:
            List of dicts with: rank, score, page, region,
                                chapter, content, query_used
        """
        query_vec = self.model.encode(
            [query], convert_to_numpy=True
        ).astype("float32")

        search_k           = top_k * 5 if (region or chapter) else top_k
        distances, indices = self.index.search(query_vec, search_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]

            if region and chunk.region != region:
                continue
            if chapter and chunk.chapter != chapter:
                continue

            # L2 distance → similarity score (0–1)
            score = float(1 / (1 + dist))

            results.append({
                "rank"      : len(results) + 1,
                "score"     : round(score, 4),
                "page"      : chunk.page,
                "region"    : chunk.region,
                "chapter"   : chunk.chapter,
                "content"   : chunk.content,
                "query_used": query,
            })

            if len(results) >= top_k:
                break

        return results

    # ----------------------------------------------------------
    #  Search from Section Descriptions (V2 main method)
    # ----------------------------------------------------------

    def search_all_parts(
        self,
        descriptions : Dict[str, SectionDescription],
        top_k        : int = 3,
    ) -> Dict[str, Dict]:
        """
        Search the book for every region extracted by the VLM.

        Steps:
          1. Flatten 3 SectionDescriptions into per-region features
          2. Build natural language query per region
          3. Search FAISS with region filter
          4. Return evidence dict

        Args:
            descriptions : output of FaceDescriber.describe()
            top_k        : passages per region

        Returns:
            {
              "nose": {
                "query"  : "small ball tip nose with round nostrils",
                "region" : "nose",
                "results": [{"page": 32, "score": 0.81, ...}, ...]
              },
              ...
            }
        """
        # Step 1: Flatten sections → regions
        regions = extract_regions_from_sections(descriptions)

        evidence = {}
        for region, features in regions.items():
            # Step 2: Build query
            query = build_query_from_section(region, features)

            print(f"  Searching: {region} → '{query[:60]}...'")

            # Step 3: Search
            results = self.search(
                query  = query,
                region = region,
                top_k  = top_k,
            )

            evidence[region] = {
                "query"  : query,
                "region" : region,
                "results": results,
            }

        return evidence
