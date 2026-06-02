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
    max_threads: int = Field(default=15, ge=1, le=30)
    instructions: str = ""
    model: str = "gemma4:31b-cloud"


class JobResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    posts_count: int
    report: Optional[str] = None
    error: Optional[str] = None
