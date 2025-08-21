# youtube_client.py
import os
import re
import time
import uuid
import json
import random
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from urllib.parse import urlparse, parse_qs

import requests
import pycountry
from dotenv import load_dotenv
from googleapiclient.discovery import build
from isodate import parse_duration

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled
)
from youtube_transcript_api.formatters import TextFormatter


class YouTubeClient:
    """
    One-stop client for:
      - YouTube Data API v3 (channel lookup, stats, video metadata)
      - Transcript fetching: captions (manual/auto/translated) with Whisper fallback
      - Batch: latest N videos from a channel (by name or channel ID)

    Quick start:
        client = YouTubeClient(env_path='config/.env')
        vids = client.get_channel_videos_data("GoogleDevelopers", n=5, fetch_transcripts=True)
        one = client.fetch_transcript("https://youtu.be/dQw4w9WgXcQ", lang="en")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        env_path: str = 'config/.env',
        whisper_model_path: str = '/home/emil/whisper.cpp/models/ggml-medium.bin',
        whisper_binary: str = '/home/emil/whisper.cpp/build/bin/whisper-cli'
    ):
        if api_key:
            self.api_key = api_key
        else:
            load_dotenv(env_path)
            self.api_key = os.getenv('API_KEY')

        if not self.api_key:
            raise ValueError('YouTube API key not provided. Set API_KEY in env or pass it directly.')

        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        self.whisper_model_path = whisper_model_path
        self.whisper_binary = whisper_binary

    # ---------------------------
    # Helpers
    # ---------------------------

    _ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

    def _normalize_video_id(self, video_or_url: str) -> str:
        """
        Accepts a raw video ID or a YouTube URL and returns the 11-char video ID when possible.
        Supports:
          - https://www.youtube.com/watch?v=VIDEOID
          - https://youtu.be/VIDEOID
          - https://www.youtube.com/shorts/VIDEOID
          - https://www.youtube.com/embed/VIDEOID
          - https://www.youtube.com/live/VIDEOID
        Falls back to the input string if nothing matches.
        """
        s = video_or_url.strip()

        # Already looks like an ID
        if self._ID_RE.match(s):
            return s

        try:
            parsed = urlparse(s)
        except Exception:
            return s

        # Not a URL? return as-is
        if not parsed.scheme and not parsed.netloc:
            return s

        # Try query param ?v=...
        qs = parse_qs(parsed.query or "")
        v_param = qs.get("v", [])
        if v_param:
            cand = v_param[0]
            if self._ID_RE.match(cand):
                return cand

        # Try path-based formats
        segs = [seg for seg in (parsed.path or "").split("/") if seg]

        # youtu.be/<id>
        if parsed.netloc.endswith("youtu.be") and segs:
            cand = segs[0]
            return cand if self._ID_RE.match(cand) else s

        # youtube.com/<kind>/<id>
        if parsed.netloc.endswith("youtube.com") and segs:
            if segs[0] in {"shorts", "embed", "live", "v"} and len(segs) >= 2:
                cand = segs[1]
                return cand if self._ID_RE.match(cand) else s

        return s

    def _safe_int(self, v) -> int:
        try:
            return int(v)
        except Exception:
            return 0

    # ---------------------------
    # Channel helpers
    # ---------------------------

    def get_channel_id(self, channel_name: str) -> Optional[str]:
        resp = (
            self.youtube.search()
            .list(q=channel_name, type='channel', part='snippet', maxResults=1)
            .execute()
        )
        items = resp.get('items', [])
        if not items:
            return None
        return items[0].get('id', {}).get('channelId') or items[0]['snippet']['channelId']

    def get_channel_stats(self, channel_id: str) -> Dict[str, Any]:
        resp = self.youtube.channels().list(part="snippet,statistics", id=channel_id).execute()
        items = resp.get("items", [])
        if not items:
            raise ValueError(f"No channel found for ID {channel_id}")

        item = items[0]
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})

        published_at = snippet.get("publishedAt")
        registration_date = (
            datetime.fromisoformat(published_at.replace("Z", "+00:00")).date().isoformat()
            if published_at else None
        )

        country_code = snippet.get("country")
        country = pycountry.countries.get(alpha_2=country_code).name if country_code else 'Unknown'

        return {
            "channel_id": channel_id,
            "channel_name": snippet.get("title"),
            "country": country,
            "registration_date": registration_date,
            "subscriber_count": self._safe_int(stats.get("subscriberCount")),
            "view_count": self._safe_int(stats.get("viewCount")),
            "video_count": self._safe_int(stats.get("videoCount")),
        }

    # ---------------------------
    # Transcript retrieval
    # ---------------------------

    def _fetch_transcript_api(
        self,
        video_id: str,
        lang: str = 'en',
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        jitter_range: Tuple[float, float] = (0.0, 0.3),
    ) -> dict:
        time.sleep(random.uniform(*jitter_range))
        attempt = 0
        while True:
            try:
                transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
                transcript = None
                source = None

                try:
                    transcript = transcripts.find_manually_created_transcript([lang]); source = 'manual'
                except NoTranscriptFound:
                    pass

                if transcript is None:
                    try:
                        transcript = transcripts.find_generated_transcript([lang]); source = 'auto'
                    except NoTranscriptFound:
                        pass

                if transcript is None:
                    for t in transcripts:
                        if t.language_code != lang and not t.is_generated:
                            transcript, source = t, 'manual'
                            break

                if transcript is None:
                    for t in transcripts:
                        if t.language_code != lang and t.is_generated:
                            transcript, source = t, 'auto'
                            break

                if transcript is None:
                    for t in transcripts:
                        if t.is_translatable:
                            transcript = t.translate(lang); source = 'translated'
                            break

                if transcript is None:
                    raise NoTranscriptFound(f"No transcript (or translatable) for {video_id!r}")

                raw = transcript.fetch()
                text = TextFormatter().format_transcript(raw)

                return {
                    "video_id": video_id,
                    "language": transcript.language_code,
                    "source": source,
                    "timed_transcript": raw,
                    "clean_transcript": text
                }

            except (TranscriptsDisabled, NoTranscriptFound) as e:
                return {"video_id": video_id, "error": str(e)}

            except (requests.exceptions.RequestException, Exception) as e:
                attempt += 1
                if attempt >= max_retries:
                    return {"video_id": video_id, "error": f"Failed after {max_retries} attempts: {type(e).__name__}: {e}"}
                wait = backoff_factor * (2 ** (attempt - 1)) + random.uniform(*jitter_range)
                time.sleep(wait)

    def _transcribe_video_whisper(self, video: str) -> dict:
        base = str(uuid.uuid4())[:8]
        audio_filename = f"{base}.wav"
        json_file = f"{audio_filename}.json"

        # Use the URL directly if given; otherwise build from ID
        url = f"https://www.youtube.com/watch?v={video}" if "http" not in video else video

        subprocess.run(["yt-dlp", "-x", "--audio-format", "wav", "-o", audio_filename, url], check=True)
        subprocess.run(
            [self.whisper_binary, "-m", self.whisper_model_path, "-f", audio_filename, "--output-txt", "--output-json"],
            check=True, capture_output=True, text=True
        )

        if not os.path.exists(json_file):
            # Normalize ID for consistency even on error
            vid_norm = self._normalize_video_id(video)
            return {"video_id": vid_norm, "error": f"Whisper output not found: {json_file}"}

        with open(json_file, 'r', encoding='utf-8') as jf:
            data = json.load(jf)

        detected_language = (
            data.get('params', {}).get('language')
            or data.get('info', {}).get('language')
            or 'unknown'
        )

        timed_transcript: List[dict] = data.get('transcription') or data.get('segments') or []
        text_segments = [seg.get('text', '') for seg in timed_transcript]

        vid_norm = self._normalize_video_id(video)
        return {
            "video_id": vid_norm,
            "language": detected_language,
            "source": "whisper",
            "clean_transcript": "".join(text_segments).strip(),
            "timed_transcript": timed_transcript
        }

    def fetch_transcript(self, video: str, lang: str = 'en') -> dict:
        """
        Fetch transcript by **video ID or any YouTube URL**.
        Tries official captions first; falls back to local Whisper if needed.
        """
        vid = self._normalize_video_id(video)
        api_result = self._fetch_transcript_api(vid, lang)
        if 'error' not in api_result:
            return api_result
        return self._transcribe_video_whisper(video)

    # ---------------------------
    # Video metadata (+ optional transcript)
    # ---------------------------

    def _build_video_dict(self, item: dict) -> Dict[str, Any]:
        snip = item.get("snippet", {})
        stats = item.get("statistics", {})
        cd = item.get("contentDetails", {})

        video_id = item.get("id")
        channel_id = snip.get('channelId')
        channel_title = snip.get('channelTitle')

        try:
            duration = cd.get('duration')
            duration_sec = parse_duration(duration).total_seconds() if duration else None
        except Exception:
            duration_sec = None

        pub = snip.get("publishedAt")
        published_at = datetime.fromisoformat(pub.replace("Z", "+00:00")).date().isoformat() if pub else None

        return {
            "channel_id": channel_id,
            "channel_title": channel_title,
            "video_id": video_id,
            "video_title": snip.get("title"),
            "video_description": snip.get("description"),
            "published_at": published_at,
            "duration_sec": duration_sec,
            "view_count": self._safe_int(stats.get("viewCount")),
            "like_count": self._safe_int(stats.get("likeCount")),
            "comment_count": self._safe_int(stats.get("commentCount")),
            "dislike_count": None,  # Not in API
        }

    def get_video_data(self, video_id: str, fetch_transcript_flag: bool = False, transcript_lang: str = 'en') -> dict:
        # Accept ID or URL here too (quietly normalize)
        vid = self._normalize_video_id(video_id)

        resp = self.youtube.videos().list(part="snippet,statistics,contentDetails", id=vid).execute()
        items = resp.get("items", [])
        if not items:
            raise ValueError(f"No video found for ID {vid}")
        base = self._build_video_dict(items[0])

        if fetch_transcript_flag:
            t = self.fetch_transcript(vid, transcript_lang)
            if 'error' not in t:
                base.update({
                    "transcript_language": t.get("language"),
                    "video_clean_transcript": t.get("clean_transcript"),
                    "video_timed_transcript": t.get("timed_transcript"),
                })
            else:
                base.update({
                    "transcript_language": None,
                    "video_clean_transcript": None,
                    "video_timed_transcript": None,
                    "transcript_error": t.get("error"),
                })
        else:
            base.update({
                "transcript_language": None,
                "video_clean_transcript": None,
                "video_timed_transcript": None,
            })
        return base

    # ---------------------------
    # Batch helpers: latest N videos from a channel
    # ---------------------------

    def _get_uploads_playlist_id(self, channel_id: str) -> str:
        resp = self.youtube.channels().list(part="contentDetails", id=channel_id).execute()
        items = resp.get("items", [])
        if not items:
            raise ValueError(f"No channel found for ID {channel_id}")
        return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    def _iter_playlist_video_ids(self, playlist_id: str, n: int) -> List[str]:
        ids: List[str] = []
        page_token = None
        while len(ids) < n:
            resp = self.youtube.playlistItems().list(
                part="contentDetails",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=page_token
            ).execute()
            for it in resp.get("items", []):
                vid = it.get("contentDetails", {}).get("videoId")
                if vid:
                    ids.append(vid)
                    if len(ids) >= n:
                        break
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return ids

    def get_channel_video_ids(self, channel: str, n: int = 10) -> List[str]:
        """
        Get up to N latest video IDs from a channel (accepts channel ID like 'UC...' or display name).
        Uses the channel's Uploads playlist (newest→oldest).
        """
        channel_id = channel if channel.startswith("UC") else self.get_channel_id(channel)
        if not channel_id:
            raise ValueError(f"Channel not found for '{channel}'")
        uploads = self._get_uploads_playlist_id(channel_id)
        return self._iter_playlist_video_ids(uploads, n)

    def get_videos_data(
        self,
        video_ids: List[str],
        fetch_transcripts: bool = False,
        transcript_lang: str = "en"
    ) -> List[dict]:
        results: List[dict] = []
        # Normalize everything (in case URLs slipped in)
        video_ids = [self._normalize_video_id(v) for v in video_ids]

        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i+50]
            resp = self.youtube.videos().list(
                part="snippet,statistics,contentDetails",
                id=",".join(batch)
            ).execute()
            items = resp.get("items", [])
            built = {it["id"]: self._build_video_dict(it) for it in items}

            for vid in batch:
                base = built.get(vid)
                if not base:
                    continue
                if fetch_transcripts:
                    t = self.fetch_transcript(vid, transcript_lang)
                    if 'error' not in t:
                        base.update({
                            "transcript_language": t.get("language"),
                            "video_clean_transcript": t.get("clean_transcript"),
                            "video_timed_transcript": t.get("timed_transcript"),
                        })
                    else:
                        base.update({
                            "transcript_language": None,
                            "video_clean_transcript": None,
                            "video_timed_transcript": None,
                            "transcript_error": t.get("error"),
                        })
                else:
                    base.update({
                        "transcript_language": None,
                        "video_clean_transcript": None,
                        "video_timed_transcript": None,
                    })
                results.append(base)
        return results

    def get_channel_videos_data(
        self,
        channel: str,
        n: int = 10,
        fetch_transcripts: bool = False,
        transcript_lang: str = "en"
    ) -> List[dict]:
        ids = self.get_channel_video_ids(channel, n=n)
        if not ids:
            return []
        return self.get_videos_data(ids, fetch_transcripts=fetch_transcripts, transcript_lang=transcript_lang)
    def sentiment_dynamics(
        self,
    transcript,                          # FetchedTranscript, list[dict], or our fetch_transcript(...) result
    analyzer: str = "vader",             # "vader" | "hf"
    hf_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
    step_s: float = 2.0,                 # bin size in seconds
    smooth_ewm_alpha: float = 0.30,      # EWMA alpha (0..1), None to disable
    min_delta_for_shift: float = 0.20,   # flag changes where |Δ smoothed| >= this
) -> dict:
    """
    Convert a timed transcript into a per-time-bin sentiment series.
    Returns:
      {
        "bins": [
          {"t_start": float, "t_end": float, "t_center": float,
           "score_raw": float | None, "score_smooth": float | None}
        ],
        "snippets": [
          {"text": str, "start": float, "end": float, "duration": float, "score": float}
        ],
        "shifts": [ {"t_center": float, "delta": float} ]    # large sentiment jumps (post-smoothing)
      }
    Notes:
      - VADER 'compound' ∈ [-1,1]. HF signed score maps POS→+prob, NEG→-prob.
      - Overlap-weighted averaging per time bin for robust dynamics.
    """
    # ---------- 1) Normalize transcript into rows (text, start, end, duration) ----------
    def _to_rows(tr):
        # Accept our own fetch_transcript(...) dict
        if isinstance(tr, dict) and "timed_transcript" in tr:
            tr = tr["timed_transcript"]

        rows = []
        # Case A: list of dicts from youtube-transcript-api ({"text","start","duration"})
        if isinstance(tr, list):
            for s in tr:
                txt = s.get("text", "")
                st = float(s.get("start", 0.0))
                dur = float(s.get("duration", 0.0))
                rows.append({"text": txt, "start": st, "end": st + max(dur, 0.0), "duration": max(dur, 0.0)})
            return rows

        # Case B: FetchedTranscript(snippets=[FetchedTranscriptSnippet(...), ...])
        # We duck-type it: has attribute 'snippets', each with .text/.start/.duration
        snippets = getattr(tr, "snippets", None)
        if snippets is not None:
            for s in snippets:
                txt = getattr(s, "text", "")
                st = float(getattr(s, "start", 0.0))
                dur = float(getattr(s, "duration", 0.0))
                rows.append({"text": txt, "start": st, "end": st + max(dur, 0.0), "duration": max(dur, 0.0)})
            return rows

        # Last resort: try to parse repr-like strings (not recommended)
        raise TypeError("Unsupported transcript type. Pass list[{'text','start','duration'}], "
                        "a fetch_transcript(...) result, or a FetchedTranscript object.")

    rows = _to_rows(transcript)
    if not rows:
        return {"bins": [], "snippets": [], "shifts": []}

    # ---------- 2) Build sentiment scorer ----------
    def _make_vader():
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
        except Exception as e:
            raise ImportError("NLTK not installed. Install via `pip install nltk`.") from e
        # Ensure lexicon
        try:
            sia = SentimentIntensityAnalyzer()
        except Exception:
            import nltk
            nltk.download("vader_lexicon")
            from nltk.sentiment import SentimentIntensityAnalyzer as _SIA
            sia = _SIA()
        return lambda text: float(sia.polarity_scores(text or "")["compound"])

    def _make_hf(model_name: str):
        try:
            from transformers import pipeline
        except Exception as e:
            raise ImportError("Transformers not installed. Install via `pip install transformers`.") from e
        nlp = pipeline("sentiment-analysis", model=model_name)
        def _score(text: str) -> float:
            if not text:
                return 0.0
            out = nlp(text[:512])[0]  # keep it cheap
            label = (out.get("label") or "").upper()
            prob = float(out.get("score") or 0.0)
            # Map to signed score
            if "NEG" in label:
                return -prob
            if "POS" in label:
                return prob
            # Neutral/unknown → 0
            return 0.0
        return _score

    if analyzer.lower() == "vader":
        score_fn = _make_vader()
    elif analyzer.lower() == "hf":
        score_fn = _make_hf(hf_model)
    else:
        raise ValueError("analyzer must be 'vader' or 'hf'.")

    # ---------- 3) Score each snippet ----------
    for r in rows:
        r["score"] = score_fn(r["text"])

    # ---------- 4) Overlap-weighted binning ----------
    import math
    t_max = max(r["end"] for r in rows)
    step = float(step_s)
    if step <= 0:
        raise ValueError("step_s must be > 0")

    n_bins = int(math.ceil(t_max / step)) or 1
    sums = [0.0] * n_bins
    weights = [0.0] * n_bins

    for r in rows:
        s, e, sc = r["start"], r["end"], r["score"]
        if e <= s:
            continue
        b0 = int(math.floor(s / step))
        b1 = int(math.floor((max(e - 1e-9, s)) / step))
        for b in range(max(0, b0), min(n_bins - 1, b1) + 1):
            bin_s = b * step
            bin_e = (b + 1) * step
            overlap = max(0.0, min(e, bin_e) - max(s, bin_s))
            if overlap > 0:
                sums[b] += sc * overlap
                weights[b] += overlap

    # Raw averages per bin
    score_raw = [ (sums[i] / weights[i]) if weights[i] > 0 else None for i in range(n_bins) ]

    # ---------- 5) Optional EWMA smoothing ----------
    score_smooth = [None] * n_bins
    if smooth_ewm_alpha is not None:
        alpha = float(smooth_ewm_alpha)
        prev = None
        for i in range(n_bins):
            x = score_raw[i]
            if x is None:
                # Hold previous smoothed value (or None if none yet)
                score_smooth[i] = prev
            else:
                prev = x if prev is None else (alpha * x + (1 - alpha) * prev)
                score_smooth[i] = prev

    # ---------- 6) Change-point flags on smoothed deltas ----------
    shifts = []
    if score_smooth.count(None) != len(score_smooth):
        prev = None
        for i, val in enumerate(score_smooth):
            if val is None:
                continue
            if prev is not None:
                delta = val - prev
                if abs(delta) >= float(min_delta_for_shift):
                    t_center = (i + 0.5) * step
                    shifts.append({"t_center": t_center, "delta": float(delta)})
            prev = val

    # ---------- 7) Build bin timeline ----------
    bins = []
    for i in range(n_bins):
        t_s = i * step
        t_e = (i + 1) * step
        bins.append({
            "t_start": round(t_s, 3),
            "t_end": round(t_e, 3),
            "t_center": round((t_s + t_e) / 2.0, 3),
            "score_raw": None if score_raw[i] is None else float(score_raw[i]),
            "score_smooth": None if score_smooth[i] is None else float(score_smooth[i]),
        })

    return {
        "bins": bins,
        "snippets": rows,
        "shifts": shifts
    }
