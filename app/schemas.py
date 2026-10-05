from typing import Annotated

from pydantic import BaseModel, Field


MetadataKey = Annotated[str, Field(min_length=1, max_length=128)]
MetadataValue = Annotated[str, Field(max_length=512)]
MetadataMap = dict[MetadataKey, MetadataValue]


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=5_000_000)
    source: str | None = Field(default=None, max_length=2048)
    collection: str = Field(default="default", pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    metadata: MetadataMap = Field(default_factory=dict, max_length=32)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=5, ge=1, le=20)
    collection: str = Field(default="default", pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    filters: MetadataMap = Field(default_factory=dict, max_length=32)


class SearchResult(BaseModel):
    document_id: str
    title: str
    chunk_id: str
    content: str
    score: float
    metadata: dict[str, object] = Field(default_factory=dict)


class AskRequest(SearchRequest):
    model: str | None = Field(default=None, max_length=128)


class AskResponse(BaseModel):
    answer: str
    sources: list[SearchResult]


class EvaluationCaseInput(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    expected_answer: str = Field(default="", max_length=10000)
    expected_document_id: str | None = Field(default=None, max_length=100)
    metadata: MetadataMap = Field(default_factory=dict, max_length=32)


class EvaluationDatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4000)
    collection: str = Field(default="default", pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    cases: list[EvaluationCaseInput] = Field(min_length=1, max_length=100)
