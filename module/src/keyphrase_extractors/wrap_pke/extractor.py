import pandas as pd
import pke
from pke.base import LoadFile

from ..base_extractor import BaseExtractor


class PKEBasedExtractor(BaseExtractor):
    def __init__(
        self,
        extractor: LoadFile = pke.unsupervised.MultipartiteRank(),
        stop_words: list[str] | None = None,
    ):
        super().__init__(stop_words)

        self.extractor = extractor
        self.pos_list: list[str] = ["NOUN", "PROPN", "ADJ", "NUM"]

        self.stop_words = (
            stop_words if stop_words is not None else self._get_stopword_list()
        )

    def _normalize_keyphrase(self, text: str, pred_keyphrases: list[str]) -> list[str]:
        parser = pke.readers.RawTextReader(language="ja")
        parsed_sentences = parser.read(text=text, spacy_model=None)
        word_list = [sentence.words for sentence in parsed_sentences]
        word_list = sum(word_list, [])
        word_list = [{"word": w, "lower_case": w.lower()} for w in word_list]
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
        input_text: str | list[str],
        top_n_phrases: int = 10,
        max_characters: int | None = None,
    ) -> list[list[str]]:
        if isinstance(input_text, str):
            if max_characters is not None:
                chunked_inputs = self._chunk(
                    text=input_text,
                    max_characters=max_characters,
                )
            else:
                chunked_inputs = [input_text]
        elif isinstance(input_text, list):
            chunked_inputs = input_text
        else:
            raise ValueError(
                f"The type of input_text must be str or list[str]; {type(input_text)=}"
            )

        result_keyphrases: list[list[str]] = []
        for i in range(len(chunked_inputs)):
            doc: str = chunked_inputs[i]

            self.extractor.load_document(
                input=doc,
                language="ja",
                stoplist=self.stop_words,
                normalization=None,
            )
            self.extractor.candidate_filtering(pos_blacklist=self.stop_words)
            self.extractor.candidate_selection(pos={*self.pos_list})
            self.extractor.candidate_weighting(
                threshold=0.7, method="average", alpha=1.1
            )

            # get top-k keyphrases
            keyphrases_and_importances: list[tuple[str, float]] = (
                self.extractor.get_n_best(top_n_phrases)
            )
            keyphrases_chunk = [t[0] for t in keyphrases_and_importances]
            keyphrases_chunk = self._normalize_keyphrase(doc, keyphrases_chunk)

            result_keyphrases.append(keyphrases_chunk)
        return result_keyphrases
