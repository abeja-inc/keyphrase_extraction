import re
from typing import Literal

import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from spacy import Language

from ..base_extractor import BaseExtractor
from .data import EmbeddingModel, Parameters
from .model import JapaneseKeyBERTModel


class KeyBERTBasedExtractor(BaseExtractor):
    def __init__(
        self,
        model_config: EmbeddingModel,
        batchsize: int = 32,
        use_prompt: bool = False,
        stop_words: list[str] | None = None,
        show_progress_bar: bool = True,
        pos_list: list[
            Literal[
                "NOUN",
                "PROPN",
                "VERB",
                "ADJ",
                "ADV",
                "INTJ",
                "PRON",
                "NUM",
                "AUX",
                "CONJ",
                "SCONJ",
                "DET",
                "ADP",
                "PART",
                "PUNCT",
                "SYM",
                "X",
            ]
        ] = ["NOUN", "PROPN", "ADJ", "NUM"],
    ):
        super().__init__(stop_words)
        # Initialize an embedding model
        use_prompt = False if model_config.prompts is None else True
        prompts = model_config.prompts.model_dump() if use_prompt else None
        model = SentenceTransformer(
            model_name_or_path=model_config.name,
            prompts=prompts,
            **model_config.model_dump(exclude=["name", "prompts"]),
        )

        # Initialize an extractor
        self.stop_words = (
            stop_words if stop_words is not None else self._get_stopword_list()
        )
        self.text_processor: Language = spacy.load("ja_ginza")
        self.kw_model = JapaneseKeyBERTModel(
            model=model,
            text_processor=self.text_processor,
            batchsize=batchsize,
            use_prompt=use_prompt,
            stop_words=stop_words,
            pos_list=pos_list,
            show_progress_bar=show_progress_bar,
        )

    def _split_text_into_sentences(
        self, text: str, minimux_strings: int = 10
    ) -> list[str]:
        # 正規表現で改行、句点、ピリオドで分割
        sentences = re.split(r"(?<=[\n。．])|(?<=\. )", text)
        # 空の文字列を除外し、リストを返す
        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip() and (len(sentence.strip()) >= minimux_strings)
        ]

    def _normalize_keyphrase(self, text: str, pred_keyphrases: list[str]) -> list[str]:
        doc = self.text_processor(text)
        word_list = [
            {"word": token.text, "lower_case": token.text.lower()}
            for token in doc
            if not token.is_space
        ]
        word_df = pd.DataFrame(word_list).drop_duplicates()

        norm_list = []
        for word in pred_keyphrases:
            if " " in word:
                splited_words = word.split(" ")
                norm_word = "".join(
                    self._get_norm_word(s, word_df) for s in splited_words
                )
            else:
                norm_word = self._get_norm_word(word, word_df)
            norm_list.append(norm_word)
        return norm_list

    def get_keyphrase(
        self,
        input_text: str,
        max_characters: int | None = None,
        diversity_mode: Literal["normal", "use_maxsum", "use_mmr"] = "normal",
        top_n_phrases: int = 10,
        max_filtered_phrases: int = 10,
        max_filtered_sentences: int = 10,
        cutoff_ratio_phrases: float | None = None,
        cutoff_ratio_sentences: float | None = None,
        threshold: float | None = None,
        nr_candidates: int = 20,
        nr_candidates_ratio: float | None = None,
        diversity: float = 0.7,
        filter_sentences: bool = True,
        phrasing: bool = True,
    ) -> list[list[str]]:
        params = Parameters(
            diversity_mode=diversity_mode,
            top_n_phrases=top_n_phrases,
            max_filtered_phrases=max_filtered_phrases,
            max_filtered_sentences=max_filtered_sentences,
            cutoff_ratio_phrases=cutoff_ratio_phrases,
            cutoff_ratio_sentences=cutoff_ratio_sentences,
            threshold=threshold,
            nr_candidates=nr_candidates,
            nr_candidates_ratio=nr_candidates_ratio,
            diversity=diversity,
        )

        docs: list[str]
        if isinstance(input_text, str):
            if max_characters is not None:
                docs = self._chunk(
                    text=input_text,
                    max_characters=max_characters,
                )
            else:
                docs = [input_text]
        elif isinstance(input_text, list):
            docs = input_text
        else:
            raise ValueError(
                f"The type of input_text must be str or list[str]; {type(input_text)=}"
            )

        sentences: list[list[str]] = []
        for _doc in docs:
            sentences.append(self._split_text_into_sentences(text=_doc))

        keyphrases_list: list[list[tuple[str, float]]] = (
            self.kw_model.extract_keyphrases(
                docs=docs,
                sentences=sentences,
                params=params,
                filter_sentences=filter_sentences,
                phrasing=phrasing,
            )
        )

        keyphrases: list[list[str]] = [
            [t[0].replace(" ", "").strip() for t in _words]
            for _doc, _words in zip(docs, keyphrases_list, strict=True)
        ]
        return keyphrases
