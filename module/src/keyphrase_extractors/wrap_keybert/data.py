from typing import Literal, Self

from pydantic import BaseModel, Field, computed_field, confloat, model_validator


class EmbeddingPrompts(BaseModel):
    passage: str
    query: str


class EmbeddingModel(BaseModel):
    name: str
    prompts: EmbeddingPrompts | None = None
    device: str = Field(default="mps", examples=["cpu", "mps", "cuda", "npu"])
    trust_remote_code: bool = True
    # model_kwargs = {"torch_dtype": torch.float16} # torch.float16, torch.bfloat16, torch.float


class Parameters(BaseModel):
    diversity_mode: Literal["normal", "use_maxsum", "use_mmr"]
    top_n_phrases: int

    max_filtered_phrases: int
    max_filtered_sentences: int
    cutoff_ratio_phrases: confloat(ge=0.0, le=1.0, strict=False) | None
    cutoff_ratio_sentences: confloat(ge=0.0, le=1.0, strict=False) | None

    threshold: confloat(ge=0.0, le=1.0, strict=False) | None
    nr_candidates: int
    nr_candidates_ratio: confloat(ge=0.0, le=1.0, strict=False) | None
    diversity: confloat(ge=0.0, le=1.0)

    @computed_field
    @property
    def use_maxsum(self) -> bool:
        return self.diversity_mode == "use_maxsum"

    @computed_field
    @property
    def use_mmr(self) -> bool:
        return self.diversity_mode == "use_mmr"

    @model_validator(mode="after")
    def validate_nr_candidates(self) -> Self:
        if (
            self.nr_candidates < self.max_filtered_phrases
            or self.nr_candidates < self.max_filtered_sentences
        ):
            raise ValueError(
                f"`nr_candidates` ({self.nr_candidates}) must be greater than or equal to both "
                f"`max_filtered_phrases` ({self.max_filtered_phrases}) and "
                f"`max_filtered_sentences` ({self.max_filtered_sentences})."
            )
        else:
            return self

    @model_validator(mode="after")
    def validate_nr_candidates_ratio(self) -> Self:
        if (
            self.nr_candidates_ratio is None
            and self.cutoff_ratio_phrases is None
            and self.cutoff_ratio_sentences is None
        ):
            return self
        elif (
            self.nr_candidates_ratio is None
            or self.cutoff_ratio_phrases is None
            or self.cutoff_ratio_sentences is None
        ):
            raise ValueError(
                """All values for `cutoff_ratio_phrases`, `cutoff_ratio_sentences`, and `nr_candidates_ratio`
                must be provided or all must be None."""
            )
        elif (
            self.nr_candidates_ratio < self.cutoff_ratio_phrases
            or self.nr_candidates_ratio < self.cutoff_ratio_sentences
        ):
            raise ValueError(
                f"`nr_candidates_ratio` ({self.nr_candidates_ratio}) must be greater than or equal to both "
                f"`cutoff_ratio_phrases` ({self.cutoff_ratio_phrases}) and "
                f"`cutoff_ratio_sentences` ({self.cutoff_ratio_sentences})."
            )
        else:
            return self
