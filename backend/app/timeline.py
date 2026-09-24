# Timeline for Concise answers in segmented analysis.
#
# A Concise answer is one word or number per chunk ("Yes", "No", "3"), so a long video
# yields a long list of them - a van parked for five minutes is twenty "Yes" rows. This
# joins identical answers in back-to-back chunks into spans ("Yes 4:30-9:30"), which is
# the whole-video answer: when something was true, for how long, and how often it
# changed. It is plain code over the chunks' final answers - the model never sees
# another chunk's output, so memory stays flat however long the video is.
import re
from typing import Any, Dict, List, Optional

# Concise answers are a word or a number; cap generation so a chunk can't ramble.
CONCISE_MAX_NEW_TOKENS = 32

# Longer than this isn't a concise answer (the model ignored the instruction), so it
# can't be compared with its neighbours and is left out of the timeline.
_MAX_ANSWER_WORDS = 4


def normalize_answer(text: str) -> Optional[str]:
    """"**Yes.**" -> "Yes", "3 people" -> "3 people". None if not a short answer."""
    if not text or "**[" in text:  # empty, or an in-band error message
        return None
    t = re.sub(r"[*_`#>]", "", text).strip().strip(".!,;:\"'").strip()
    if not t or len(t.split()) > _MAX_ANSWER_WORDS:
        return None
    return t[:1].upper() + t[1:].lower()


class Timeline:
    """Joins identical answers in back-to-back chunks into spans.

    A chunk skipped for having no motion extends the span before it: nothing in the
    picture changed, so the answer hasn't either - it is carried over and counted
    separately, so it's visible which part of a span the model actually looked at. A
    chunk with no usable answer (an error, or a long reply) ends the span, since we
    don't know what was true there.
    """

    def __init__(self) -> None:
        self.spans: List[Dict[str, Any]] = []
        self._current: Optional[Dict[str, Any]] = None

    def add(self, index: int, start: float, end: float,
            answer: Optional[str], skipped: bool = False) -> Optional[str]:
        """Record one chunk. Returns the answer that now applies to it (for a skipped
        chunk, the one carried over), or None."""
        cur = self._current
        contiguous = cur is not None and cur["last_chunk"] == index - 1
        if skipped:
            if not contiguous:
                return None
            cur["end"], cur["last_chunk"] = end, index
            cur["chunks"] += 1
            cur["carried"] += 1
            return cur["answer"]
        if answer is None:
            self._current = None
            return None
        if contiguous and cur["answer"] == answer:
            cur["end"], cur["last_chunk"] = end, index
            cur["chunks"] += 1
            return answer
        self._current = {
            "answer": answer, "start": start, "end": end,
            "first_chunk": index, "last_chunk": index, "chunks": 1, "carried": 0,
        }
        self.spans.append(self._current)
        return answer

    def as_list(self) -> List[Dict[str, Any]]:
        return [dict(s) for s in self.spans]
