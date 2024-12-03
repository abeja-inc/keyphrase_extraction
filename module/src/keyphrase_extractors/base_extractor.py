import re
import urllib

import pandas as pd

class BaseExtractor:
    def __init__(self, stop_words: list[str] | None = None):
        # Japanese Stopwors
        self.stop_words = (
            stop_words if stop_words is not None else self._get_stopword_list()
        )

    def _get_stopword_list(self) -> list[str]:
        """
        Download Japanese stop words
        """
        slothlib_url: str = (
            "http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt"
        )

        slothlib_file = urllib.request.urlopen(slothlib_url)
        stop_words = [line.decode("utf-8").strip() for line in slothlib_file]
        return stop_words

    def _get_norm_word(self, word: str, word_df: pd.DataFrame) -> str:
        norm_values = word_df[word_df["lower_case"] == word]["word"].to_list()
        return norm_values[0] if len(norm_values) != 0 else word

    def _normalize_keyphrase(self, text: str, pred_keyphrases: list[str]) -> list[str]:
        raise NotImplementedError("The function `_normalize_keyphrase` is not implemented.")

    def _chunk(self, text: str, max_characters: int = 3000) -> list[str]:
        """
        Chunk text to fit within a given maximum character count
        """
        sentence_split_marks: str = r"\. |\? |! |。|！|？|\n"
        phrase_split_marks: str = r", |、 |\s"

        chunked_texts: list[str] = []

        while len(text) > max_characters:
            split_position = None
            for match in re.finditer(sentence_split_marks, text[:max_characters]):
                split_position = match

            if split_position is not None:
                end = split_position.end()
            else:
                space_position = None
                for match in re.finditer(phrase_split_marks, text[:max_characters]):
                    space_position = match

                if space_position is not None:
                    end = space_position.end()
                else:
                    end = self.max_characters - 1

            chunked_texts.append(text[: end + 1])
            text = text[end + 1 :].lstrip()

        chunked_texts.append(text)

        return chunked_texts
