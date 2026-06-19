from typing import Literal, Optional
from pydantic import BaseModel, Field

SortChoice = Literal["relevance", "hot", "top", "new", "comments"]
TimeChoice = Literal["hour", "day", "week", "month", "year", "all"]


class ScrapeRequest(BaseModel):
    query: str = Field(min_length=2)
    sort: SortChoice = "relevance"
    time_filter: TimeChoice = "week"
    limit: int = Field(default=50, ge=1, le=200)
    deep_search: bool = False
    report: bool = False
    min_score: int = Field(default=5, ge=0)
    max_threads: int = Field(default=15, ge=1, le=50)
    instructions: str = ""
    model: str = "gemma4:31b-cloud"


class ThreadScore(BaseModel):
    """One AI relevance verdict for a single thread. The model returns these via
    Ollama structured output (JSON schema), so the shape is guaranteed."""
    url: str
    relevance: int = Field(ge=0, le=100, description="0-100 relevance to query + instructions")
    reason: str = Field(default="", description="One-line justification; emitted in threads_selected for optional UI display")


class ThreadScoreBatch(BaseModel):
    """Wrapper so Ollama returns a JSON object (not a bare array) — required by the
    structured-output `format` field."""
    scores: list[ThreadScore]


class JobResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    posts_count: int
    report: Optional[str] = None
    error: Optional[str] = None
