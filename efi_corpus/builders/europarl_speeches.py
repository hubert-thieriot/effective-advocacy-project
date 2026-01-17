"""
EuroparlSpeechesCorpusBuilder - Corpus builder for European Parliament speeches.

Uses the European Parliament Open Data API v2:
https://data.europarl.europa.eu/api/v2/speeches
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple
import time
import xml.etree.ElementTree as ET

import requests

from .base import BaseCorpusBuilder
from ..rate_limiter import RateLimiter, RateLimitConfig
from ..types import BuilderParams, DiscoveryItem
from ..utils import ensure_date


@dataclass
class EuroparlSpeech:
    """Parsed speech payload ready for storage."""

    stable_id: str
    url: str
    title: Optional[str]
    published_at: Optional[str]
    language: Optional[str]
    text: str
    raw_xml: Optional[str]
    authors: List[str]
    meta_extra: Dict[str, Any]


class EuroparlSpeechesCorpusBuilder(BaseCorpusBuilder):
    """
    Corpus builder that collects European Parliament speeches via the EP Open Data API.

    Default behavior:
    - Uses /speeches with include-output=xml_fragment to retrieve speech text.
    - Filters to English (language=en, search-language=en) unless overridden.
    - Filters to PLENARY_DEBATE_SPEECH unless overridden.
    """

    API_BASE_URL = "https://data.europarl.europa.eu/api/v2"
    SPEECHES_ENDPOINT = "/speeches"
    DEFAULT_PAGE_SIZE = 50

    def __init__(
        self,
        corpus_dir: Path,
        fetcher=None,
        cache_root: Path = None,
        rate_limit_config: RateLimitConfig | None = None,
        session: Optional[requests.Session] = None,
    ):
        super().__init__(corpus_dir, fetcher, cache_root)
        self.session = session or requests.Session()
        if rate_limit_config is None:
            rate_limit_config = RateLimitConfig(requests_per_minute=60, min_interval=1.0)
        self.rate_limiter = RateLimiter(rate_limit_config)

    def _request(self, params: Dict[str, Any], user_agent: str) -> Dict[str, Any]:
        url = f"{self.API_BASE_URL}{self.SPEECHES_ENDPOINT}"
        headers = {
            "Accept": "application/ld+json",
            "User-Agent": user_agent,
        }
        self.rate_limiter.wait_if_needed()
        response = self.session.get(url, params=params, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _normalize_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    @staticmethod
    def _local_name(tag: str) -> str:
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag

    @staticmethod
    def _build_text_query(keywords: List[str]) -> str:
        terms: List[str] = []
        for keyword in keywords:
            if keyword is None:
                continue
            token = str(keyword).strip()
            if not token:
                continue
            if " " in token and not (token.startswith('"') and token.endswith('"')):
                token = f'"{token}"'
            terms.append(token)
        if not terms:
            return ""
        if len(terms) == 1:
            return terms[0]
        return " OR ".join(terms)

    @staticmethod
    def _stable_id_from_payload(payload: Dict[str, Any]) -> str:
        activity_id = payload.get("activity_id") or payload.get("id") or ""
        if not activity_id:
            activity_id = str(payload)
        return sha1(activity_id.encode("utf-8")).hexdigest()

    def _build_query_params(self, params: BuilderParams, offset: int, limit: int) -> Dict[str, Any]:
        extra = params.extra or {}
        query: Dict[str, Any] = {}

        text_query = extra.get("text_query") or extra.get("text")
        if not text_query:
            text_query = self._build_text_query(params.keywords)
        if text_query:
            query["text"] = text_query

        title_query = extra.get("title")
        if title_query:
            query["title"] = title_query

        language = extra.get("language") or "en"
        search_language = extra.get("search_language") or extra.get("search-language") or language
        if language:
            query["language"] = language
        if search_language:
            query["search-language"] = search_language

        # Date range in the API is sitting-date / sitting-date-end
        sitting_date = extra.get("sitting_date") or extra.get("sitting-date")
        sitting_date_end = extra.get("sitting_date_end") or extra.get("sitting-date-end")
        if not sitting_date and params.date_from:
            sitting_date = ensure_date(params.date_from, "date_from").isoformat()
        if not sitting_date_end and params.date_to:
            sitting_date_end = ensure_date(params.date_to, "date_to").isoformat()
        if sitting_date:
            query["sitting-date"] = sitting_date
        if sitting_date_end:
            query["sitting-date-end"] = sitting_date_end

        activity_types = extra.get("activity_types") or extra.get("activity_type") or extra.get("activity-type")
        if activity_types is None:
            activity_types = ["PLENARY_DEBATE_SPEECH"]
        activity_types = self._normalize_list(activity_types)
        if activity_types:
            query["activity-type"] = activity_types

        parliamentary_terms = extra.get("parliamentary_terms") or extra.get("parliamentary_term") or extra.get("parliamentary-term")
        parliamentary_terms = self._normalize_list(parliamentary_terms)
        if parliamentary_terms:
            query["parliamentary-term"] = parliamentary_terms

        person_ids = extra.get("person_ids") or extra.get("person_id") or extra.get("person-id")
        person_ids = self._normalize_list(person_ids)
        if person_ids:
            query["person-id"] = person_ids

        include_output = extra.get("include_output") or extra.get("include-output") or ["xml_fragment"]
        include_output = self._normalize_list(include_output)
        if include_output:
            query["include-output"] = include_output

        sort_by = extra.get("sort_by") or extra.get("sort-by")
        sort_by = self._normalize_list(sort_by)
        if sort_by:
            query["sort-by"] = sort_by

        query["format"] = extra.get("format", "application/ld+json")
        query["offset"] = offset
        query["limit"] = limit
        return query

    def _apply_rate_limit_overrides(self, params: BuilderParams) -> None:
        extra = params.extra or {}
        requests_per_minute = extra.get("requests_per_minute")
        min_interval = extra.get("min_interval")
        enabled = extra.get("rate_limit_enabled")
        if requests_per_minute is not None:
            self.rate_limiter.config.requests_per_minute = int(requests_per_minute)
        if min_interval is not None:
            self.rate_limiter.config.min_interval = float(min_interval)
        if enabled is not None:
            self.rate_limiter.config.enabled = bool(enabled)

    def _select_fragment(self, payload: Dict[str, Any], language: Optional[str]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        records = payload.get("recorded_in_a_realization_of")
        records = self._normalize_list(records)
        for record in records:
            fragment = record.get("api:xmlFragment") or record.get("api:xmlfragment")
            fragment_text = self._select_language_fragment(fragment, language)
            if fragment_text:
                return fragment_text, record

        fragment = payload.get("api:xmlFragment") or payload.get("api:xmlfragment")
        fragment_text = self._select_language_fragment(fragment, language)
        if fragment_text:
            return fragment_text, None

        return None, None

    @staticmethod
    def _select_language_fragment(fragment: Any, language: Optional[str]) -> Optional[str]:
        if not fragment:
            return None
        if isinstance(fragment, str):
            return fragment
        if isinstance(fragment, dict):
            if language:
                for key in (language, language.lower(), language.upper()):
                    if key in fragment and fragment[key]:
                        return fragment[key]
            if "en" in fragment and fragment["en"]:
                return fragment["en"]
            for value in fragment.values():
                if value:
                    return value
        return None

    def _parse_xml_fragment(self, xml_fragment: str) -> Tuple[str, Dict[str, str]]:
        if not xml_fragment:
            return "", {}
        try:
            root = ET.fromstring(xml_fragment)
        except ET.ParseError:
            try:
                root = ET.fromstring(f"<root>{xml_fragment}</root>")
            except ET.ParseError:
                return xml_fragment.strip(), {}

        speaker_meta: Dict[str, str] = {}
        for elem in root.iter():
            if self._local_name(elem.tag) == "from":
                speaker_line = " ".join(t.strip() for t in elem.itertext() if t.strip())
                if speaker_line:
                    speaker_meta["speaker_line"] = speaker_line
                for child in elem.iter():
                    local = self._local_name(child.tag)
                    if local == "person":
                        speaker_name = " ".join(t.strip() for t in child.itertext() if t.strip())
                        if speaker_name:
                            speaker_meta["speaker"] = speaker_name
                    elif local == "role":
                        role = " ".join(t.strip() for t in child.itertext() if t.strip())
                        if role:
                            speaker_meta["speaker_role"] = role
                break

        paragraphs: List[str] = []
        for elem in root.iter():
            if self._local_name(elem.tag) == "p":
                text = " ".join(t.strip() for t in elem.itertext() if t.strip())
                if text:
                    paragraphs.append(text)

        if paragraphs:
            return "\n\n".join(paragraphs).strip(), speaker_meta

        text = " ".join(t.strip() for t in root.itertext() if t.strip())
        return text.strip(), speaker_meta

    @staticmethod
    def _extract_language_codes(values: List[str]) -> List[str]:
        codes: List[str] = []
        for value in values:
            if not value:
                continue
            if "/" in value:
                value = value.rsplit("/", 1)[-1]
            codes.append(str(value))
        return codes

    def _to_speech(self, payload: Dict[str, Any], params: BuilderParams) -> Optional[EuroparlSpeech]:
        extra = params.extra or {}
        language = extra.get("language") or "en"
        fragment, record = self._select_fragment(payload, language)
        if not fragment:
            return None

        text, speaker_meta = self._parse_xml_fragment(fragment)
        if not text:
            return None

        activity_id = payload.get("activity_id") or payload.get("id") or ""
        if not activity_id:
            activity_id = str(payload)
        stable_id = self._stable_id_from_payload(payload)

        item_id = payload.get("id") or ""
        url = f"https://data.europarl.europa.eu/{item_id}" if item_id else ""

        activity_label = payload.get("activity_label") or {}
        title = None
        if isinstance(activity_label, dict):
            title = activity_label.get(language) or activity_label.get("en")
            if title is None and activity_label:
                title = next(iter(activity_label.values()), None)

        published_at = payload.get("activity_start_date") or payload.get("activity_date")

        authors: List[str] = []
        participation = self._normalize_list(payload.get("had_participation"))
        person_ids: List[str] = []
        participation_roles: List[str] = []
        for entry in participation:
            if not isinstance(entry, dict):
                continue
            person_ids.extend(self._normalize_list(entry.get("had_participant_person")))
            role = entry.get("participation_role")
            if role:
                participation_roles.append(role)
        if speaker_meta.get("speaker"):
            authors.append(speaker_meta["speaker"])
        elif person_ids:
            authors.extend(person_ids)

        record_meta: Dict[str, Any] = {}
        if isinstance(record, dict):
            record_meta = {
                "record_id": record.get("id"),
                "record_identifier": record.get("identifier"),
                "record_number": record.get("number"),
                "speech_notation_id": record.get("notation_speechId"),
                "original_languages": self._extract_language_codes(
                    self._normalize_list(record.get("originalLanguage"))
                ),
            }

        meta_extra = {
            "activity_id": payload.get("activity_id"),
            "activity_type": payload.get("had_activity_type"),
            "activity_end_date": payload.get("activity_end_date"),
            "activity_label": activity_label,
            "person_ids": person_ids,
            "participation_roles": participation_roles,
            **speaker_meta,
            **record_meta,
        }

        return EuroparlSpeech(
            stable_id=stable_id,
            url=url,
            title=title,
            published_at=published_at,
            language=language,
            text=text,
            raw_xml=fragment,
            authors=authors,
            meta_extra=meta_extra,
        )

    def _get_previous_failed_ids(self, manifest: Dict[str, Any]) -> set[str]:
        failed_ids: set[str] = set()
        for entry in manifest.get("history", []):
            for failed in entry.get("failed_details", []):
                stable_id = failed.get("stable_id")
                if stable_id:
                    failed_ids.add(stable_id)
        return failed_ids

    def discover(self, params: BuilderParams) -> Iterable[DiscoveryItem]:
        """Yield discovery items with minimal metadata (without text)."""
        self._apply_rate_limit_overrides(params)
        extra = params.extra or {}
        limit = int(extra.get("page_size") or self.DEFAULT_PAGE_SIZE)
        offset = int(extra.get("offset") or 0)
        user_agent = extra.get("user_agent") or "EFI-Corpus/1.0 (europarl_speeches)"
        max_items = extra.get("max_items")
        if max_items is not None:
            max_items = int(max_items)

        discovered = 0
        total = None
        while True:
            query = self._build_query_params(params, offset=offset, limit=limit)
            payload = self._request(query, user_agent)
            items = payload.get("data", []) or []
            if total is None:
                total = payload.get("meta", {}).get("total")

            if not items:
                break

            for item in items:
                discovered += 1
                activity_id = item.get("activity_id") or item.get("id") or ""
                url = f"https://data.europarl.europa.eu/{item.get('id')}" if item.get("id") else ""
                label = item.get("activity_label") or {}
                title = None
                language = extra.get("language") or "en"
                if isinstance(label, dict):
                    title = label.get(language) or label.get("en")
                    if title is None and label:
                        title = next(iter(label.values()), None)
                yield DiscoveryItem(
                    url=url or activity_id,
                    canonical_url=url or activity_id,
                    published_at=item.get("activity_start_date") or item.get("activity_date"),
                    title=title,
                    language=extra.get("language") or "en",
                    authors=None,
                    extra={"activity_id": activity_id},
                )

                if max_items and discovered >= max_items:
                    return

            offset += limit
            if total is not None and offset >= total:
                return

    def run(self, *, params: BuilderParams | None = None, override: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Run the corpus builder.
        """
        manifest = self.corpus.load_manifest()
        if params is None:
            if not manifest:
                raise ValueError("No manifest found; first run must provide params.")
            params = BuilderParams(**manifest["params"])

        if override:
            merged = {**params.__dict__, **override}
            params = BuilderParams(**merged)

        manifest.setdefault("name", self.corpus.corpus_path.name)
        manifest.setdefault("source", "europarl_speeches")
        manifest["params"] = params.__dict__
        manifest.setdefault("history", [])

        self._apply_rate_limit_overrides(params)
        extra = params.extra or {}
        limit = int(extra.get("page_size") or self.DEFAULT_PAGE_SIZE)
        offset = int(extra.get("offset") or 0)
        max_items = extra.get("max_items")
        if max_items is not None:
            max_items = int(max_items)

        user_agent = extra.get("user_agent") or "EFI-Corpus/1.0 (europarl_speeches)"
        request_interval = extra.get("request_interval")
        if request_interval is not None:
            request_interval = float(request_interval)

        skip_previously_failed = params.skip_previously_failed
        previously_failed = self._get_previous_failed_ids(manifest) if skip_previously_failed else set()
        if skip_previously_failed and previously_failed:
            print(f"Skipping {len(previously_failed)} items that failed in previous runs")

        discovered = 0
        added = 0
        skipped_duplicate = 0
        skipped_empty = 0
        skipped_failed = 0
        failed: List[Dict[str, Any]] = []

        total = None
        while True:
            query = self._build_query_params(params, offset=offset, limit=limit)
            payload = self._request(query, user_agent)
            items = payload.get("data", []) or []
            if total is None:
                total = payload.get("meta", {}).get("total")
                if total is not None:
                    print(f"API reports {total} matching speeches")

            if not items:
                break

            for item in items:
                discovered += 1
                try:
                    speech = self._to_speech(item, params)
                    if speech is None:
                        skipped_empty += 1
                        continue

                    if self.corpus.has_doc(speech.stable_id):
                        skipped_duplicate += 1
                        continue

                    if speech.stable_id in previously_failed:
                        skipped_failed += 1
                        continue

                    meta = {
                        "doc_id": speech.stable_id,
                        "uri": speech.url,
                        "title": speech.title,
                        "published_at": speech.published_at,
                        "language": speech.language,
                        "authors": speech.authors,
                        "source": manifest["source"],
                        "extra": speech.meta_extra,
                    }

                    raw_bytes = (speech.raw_xml or speech.text).encode("utf-8")
                    raw_ext = "xml" if speech.raw_xml else "txt"
                    fetch_info = {
                        "fetched_at": time.time(),
                        "source": "europarl_api",
                        "endpoint": "speeches",
                        "request_params": query,
                    }

                    self.corpus.write_document(
                        stable_id=speech.stable_id,
                        meta=meta,
                        text=speech.text,
                        raw_bytes=raw_bytes,
                        raw_ext=raw_ext,
                        fetch_info=fetch_info,
                    )

                    self.corpus.append_index({
                        "id": speech.stable_id,
                        "url": speech.url,
                        "title": speech.title,
                        "published_at": speech.published_at,
                        "language": speech.language,
                        "speaker": speech.meta_extra.get("speaker"),
                        "activity_id": speech.meta_extra.get("activity_id"),
                        "speech_notation_id": speech.meta_extra.get("speech_notation_id"),
                    })

                    added += 1
                except Exception as exc:
                    stable_id = self._stable_id_from_payload(item)
                    failed.append({
                        "stable_id": stable_id,
                        "url": item.get("id"),
                        "error": str(exc),
                    })
                    continue

                if max_items and discovered >= max_items:
                    break

            offset += limit
            if max_items and discovered >= max_items:
                break
            if total is not None and offset >= total:
                break
            if request_interval:
                time.sleep(request_interval)

        manifest["history"].append({
            "run_at": time.time(),
            "discovered": discovered,
            "added": added,
            "skipped_duplicate": skipped_duplicate,
            "skipped_empty": skipped_empty,
            "skipped_failed": skipped_failed,
            "failed": len(failed),
            "failed_details": failed,
            "date_from": params.date_from,
            "date_to": params.date_to,
            "query": self._build_query_params(params, offset=0, limit=limit),
        })
        manifest["doc_count"] = self.corpus.get_document_count()
        self.corpus.save_manifest(manifest)

        print("\nBuild Summary:")
        print(f"  Discovered: {discovered}")
        print(f"  Added: {added}")
        print(f"  Skipped (empty/no text): {skipped_empty}")
        print(f"  Skipped (duplicate): {skipped_duplicate}")
        print(f"  Skipped (previous failures): {skipped_failed}")
        print(f"  Failed: {len(failed)}")
        print(f"  Total docs in corpus: {manifest['doc_count']}")

        return {
            "discovered": discovered,
            "added": added,
            "skipped_empty": skipped_empty,
            "skipped_duplicate": skipped_duplicate,
            "skipped_failed": skipped_failed,
            "failed": len(failed),
            "total_docs": manifest["doc_count"],
            "failed_details": failed,
        }
