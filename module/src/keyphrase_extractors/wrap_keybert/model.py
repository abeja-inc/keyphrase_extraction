from itertools import chain
from typing import Literal

import numpy as np
from keybert._maxsum import max_sum_distance
from keybert._mmr import mmr
from nltk import RegexpParser
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from spacy import Language
from spacy.tokens.doc import Doc

from .data import Parameters


class JapaneseKeyBERTModel:
    def __init__(
        self,
        model: SentenceTransformer,
        text_processor: Language,
        batchsize: int = 32,
        stop_words: list[str] | None = None,
        use_prompt: bool = False,
        rrf_k: int = 60,
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
        # Embedding model
        self.model = model
        self.batchsize = batchsize
        self.use_prompt = use_prompt
        self.use_prompt = use_prompt

        # Initialize a tokenizer
        self.text_processor = text_processor
        self.pos_list = pos_list
        self.stop_words = stop_words

        # Parameters
        self.show_progress_bar = show_progress_bar
        self.rrf_k = rrf_k

    def _words_to_phrases(
        self, words: list[str], words_pos: list[str], grammar: str = None
    ) -> tuple[list[str]]:
        # Initialize default grammar if none is provided
        if grammar is None:
            grammar = r"""
                    NBAR:
                        {<NOUN|PROPN|ADJ>*<NOUN|PROPN>}
                    
                    NP:
                        {<NBAR>}
                        {<NBAR><ADP><NBAR>}
                """

        # Parse sentence using the chunker
        grammar_parser = RegexpParser(grammar)
        tuples = [(str(i), words_pos[i]) for i in range(len(words))]
        tree = grammar_parser.parse(tuples)

        # Extract phrases and their POS tags
        candidates = set()
        for subtree in tree.subtrees():
            if subtree.label() == "NP":
                leaves = subtree.leaves()
                first = int(leaves[0][0])
                last = int(leaves[-1][0])
                phrase = " ".join(words[first : last + 1])
                candidates.add(phrase)

        # Add individual words if they are not part of NP and match the POS list
        for word, pos in zip(words, words_pos, strict=True):
            if pos in self.pos_list and word not in candidates:
                candidates.add(word)

        return list(candidates)

    def _tokenize_text(self, text: str, phrasing: bool = True) -> list[str]:
        doc: Doc = self.text_processor(text)

        words = []
        words_pos = []
        for token in doc:
            words.append(token.text)
            words_pos.append(token.pos_)
        if phrasing:
            tokens = self._words_to_phrases(words=words, words_pos=words_pos)
        else:
            tokens = list(
                set(
                    [
                        _word
                        for _word, _pos in zip(words, words_pos, strict=True)
                        if _pos in self.pos_list
                    ]
                )
            )

        return tokens

    def _extract_sentences(
        self, docs: list[str], sentences: list[list[str]]
    ) -> list[list[tuple[str, float]]]:
        # Calculate sentence-level similarity to document
        doc_embeddings: np.ndarray
        sentence_embeddings: list[np.ndarray]
        if self.use_prompt:
            doc_embeddings = self.model.encode(
                sentences=docs,
                prompt="passage",
                batch_size=1,
                show_progress_bar=self.show_progress_bar,
            )

            sentence_embeddings = [
                self.model.encode(
                    sentences=_sentences,
                    prompt="query",
                    batch_size=32,
                    show_progress_bar=self.show_progress_bar,
                )
                for _sentences in sentences
            ]
        else:
            doc_embeddings = self.model.encode(
                sentences=docs, batch_size=1, show_progress_bar=self.show_progress_bar
            )

            sentence_embeddings = [
                self.model.encode(
                    sentences=_sentences,
                    batch_size=32,
                    show_progress_bar=self.show_progress_bar,
                )
                for _sentences in sentences
            ]

        key_sentences: list[list[tuple[str, float]]] = []
        for chunk_idx, (_sent_embeds, _sentences) in enumerate(
            zip(sentence_embeddings, sentences, strict=True)
        ):
            _doc_embed: np.ndarray = doc_embeddings[chunk_idx].reshape(1, -1)
            if self.params.cutoff_ratio_sentences is None:
                _top_n = self.params.max_filtered_sentences
                _nr_candidates = self.params.nr_candidates
            else:
                _top_n = max(
                    int(len(_sentences) * self.params.cutoff_ratio_sentences), 1
                )
                _nr_candidates = max(
                    int(len(_sentences) * self.params.nr_candidates_ratio), _top_n
                )
            try:
                # Maximal Marginal Relevance (MMR)
                if self.params.use_mmr:
                    _key_sentences = mmr(
                        _doc_embed,
                        _sent_embeds,
                        _sentences,
                        _top_n,
                        self.params.diversity,
                    )

                # Max Sum Distance
                elif self.params.use_maxsum:
                    _key_sentences = max_sum_distance(
                        _doc_embed,
                        _sent_embeds,
                        _sentences,
                        _top_n,
                        _nr_candidates,
                    )

                # Cosine-based keyphrase extraction
                else:
                    distances = cosine_similarity(_doc_embed, _sent_embeds)
                    _key_sentences = [
                        (_sentences[i], round(float(distances[0][i]), 4))
                        for i in distances.argsort()[0][-_top_n:]
                    ][::-1]

                if self.params.threshold is not None:
                    _key_sentences = [
                        key for key in _key_sentences if key[1] >= self.params.threshold
                    ]

            # Capturing empty keyphrases
            except ValueError:
                _key_sentences = []

            key_sentences.append(_key_sentences)

        return key_sentences

    def _extract_phrases(
        self, sentences: list[list[str]], phrases: list[list[list[str]]]
    ) -> list[list[list[tuple[str, float]]]]:
        sentence_embeddings: list[np.ndarray]
        phrase_embeddings: list[list[np.ndarray]]

        if self.use_prompt:
            sentence_embeddings = [
                self.model.encode(
                    sentences=_sentences,
                    prompt="passage",
                    batch_size=32,
                    show_progress_bar=self.show_progress_bar,
                )
                for _sentences in sentences
            ]

            phrase_embeddings = [
                [
                    self.model.encode(
                        sentences=_phrases_one_sentence,
                        prompt="query",
                        batch_size=32,
                        show_progress_bar=self.show_progress_bar,
                    )
                    for _phrases_one_sentence in _phrases
                ]
                for _phrases in phrases
            ]

        else:
            sentence_embeddings = [
                self.model.encode(
                    sentences=_sentences,
                    batch_size=32,
                    show_progress_bar=self.show_progress_bar,
                )
                for _sentences in sentences
            ]

            phrase_embeddings = [
                [
                    self.model.encode(
                        sentences=_phrases_one_sentence,
                        batch_size=32,
                        show_progress_bar=self.show_progress_bar,
                    )
                    for _phrases_one_sentence in _phrases
                ]
                for _phrases in phrases
            ]

        key_phrase: list[list[list[tuple[str, float]]]] = []
        for _sentence_embeds, _phrase_embeds_list, _phrases_list in zip(
            sentence_embeddings, phrase_embeddings, phrases, strict=True
        ):
            _key_phrase_chunk = []
            for sent_idx, (_phrase_embeds, _phrases) in enumerate(
                zip(_phrase_embeds_list, _phrases_list, strict=True)
            ):
                _sentence_embed: np.ndarray = _sentence_embeds[sent_idx].reshape(1, -1)
                if self.params.cutoff_ratio_phrases is None:
                    _top_n = self.params.max_filtered_phrases
                    _nr_candidates = self.params.nr_candidates
                else:
                    _top_n = max(
                        int(len(_phrases) * self.params.cutoff_ratio_phrases), 1
                    )
                    _nr_candidates = max(
                        int(len(_phrases) * self.params.nr_candidates_ratio), _top_n
                    )
                try:
                    # Maximal Marginal Relevance (MMR)
                    if self.params.use_mmr:
                        _key_phrases = mmr(
                            _sentence_embed,
                            _phrase_embeds,
                            _phrases,
                            _top_n,
                            self.params.diversity,
                        )

                    # Max Sum Distance
                    elif self.params.use_maxsum:
                        _key_phrases = max_sum_distance(
                            _sentence_embed,
                            _phrase_embeds,
                            _phrases,
                            _top_n,
                            max(self.params.nr_candidates, _top_n),
                        )

                    # Cosine-based keyphrase extraction
                    else:
                        distances = cosine_similarity(_sentence_embed, _phrase_embeds)
                        _key_phrases = [
                            (_phrases[i], round(float(distances[0][i]), 4))
                            for i in distances.argsort()[0][-_top_n:]
                        ][::-1]

                    if self.params.threshold is not None:
                        _key_phrases = [
                            key
                            for key in _key_phrases
                            if key[1] >= self.params.threshold
                        ]

                # Capturing empty keyphrases
                except ValueError:
                    _key_phrases = []

                _key_phrase_chunk.append(_key_phrases)
            key_phrase.append(_key_phrase_chunk)

        return key_phrase

    def _reciprocal_rank_fusion(
        self,
        sentence_similarities: list[list[float]],
        key_phrases: list[list[list[tuple[str, float]]]],
    ) -> list[list[tuple[str, float]]]:
        # Calcurate the ranks of sentences
        sentence_ranks: list[list[int]] = [
            (np.array(_similarities).argsort()[::-1] + 1).tolist()
            for _similarities in sentence_similarities
        ]

        # Calcurate the ranks of phrases
        phrases_flatten: list[list[tuple[str, float]]] = []
        rrf_scores: list[list[float]] = []
        for _sent_ranks, phrases_list in zip(sentence_ranks, key_phrases, strict=True):
            _phrases: list[str] = []
            _scores: list[float] = []
            for _sent_rank, _phrases_and_scores in zip(
                _sent_ranks, phrases_list, strict=True
            ):
                _phrases += [
                    _phrase_and_score[0] for _phrase_and_score in _phrases_and_scores
                ]
                _phrase_ranks: list[int] = (
                    np.array(
                        [
                            _phrase_and_score[1]
                            for _phrase_and_score in _phrases_and_scores
                        ]
                    ).argsort()[::-1]
                    + 1
                )
                _scores += [
                    (1 / (_sent_rank + self.rrf_k)) + (1 / (_phrase_rank + self.rrf_k))
                    for _phrase_rank in _phrase_ranks
                ]

            rrf_scores.append(_scores)
            phrases_flatten.append(
                [
                    (_phrase, _score)
                    for _phrase, _score in zip(_phrases, _scores, strict=True)
                ]
            )

        sorted_indices = [np.argsort(_scores)[::-1] for _scores in rrf_scores]
        sorted_phrase_scores: list[list[tuple[str, float]]] = [
            [_phrases[_idx] for _idx in _indices]
            for _phrases, _indices in zip(phrases_flatten, sorted_indices, strict=True)
        ]

        return sorted_phrase_scores

    def extract_keyphrases(
        self,
        docs: list[str],
        sentences: list[list[str]],
        params: Parameters,
        filter_sentences: bool = True,
        phrasing: bool = True,
    ) -> list[list[tuple[str, float]]]:
        self.params = params

        if filter_sentences:
            # Extract the key sentences
            key_sentences: list[list[tuple[str, float]]] = self._extract_sentences(
                docs=docs, sentences=sentences
            )

            # Get phrases
            sentences: list[list[str]] = []
            phrases: list[list[list[str]]] = []
            for _sentences in key_sentences:
                phrases.append(
                    [
                        self._tokenize_text(text=_sent[0], phrasing=phrasing)
                        for _sent in _sentences
                    ]
                )
                sentences.append([_sent[0] for _sent in _sentences])

            # Extract the key phrases
            key_phrases: list[list[list[tuple[str, float]]]] = self._extract_phrases(
                sentences=sentences, phrases=phrases
            )

            # Reciprocal Rank Fusion for final ranking
            sentence_similarities: list[list[float]] = [
                [_sent[1] for _sent in _sentences] for _sentences in key_sentences
            ]
            sorted_keyphrases: list[list[tuple[str, float]]] = (
                self._reciprocal_rank_fusion(
                    sentence_similarities=sentence_similarities,
                    key_phrases=key_phrases,
                )
            )
        else:
            docs: list[list[str]] = [[_doc] for _doc in docs]

            phrases: list[list[list[str]]] = []
            for _sentences in sentences:
                _phrases: list[str] = []
                for _sent in _sentences:
                    _phrases += self._tokenize_text(text=_sent, phrasing=phrasing)
                phrases.append([list(set(_phrases))])

            # Extract the key phrases
            key_phrases: list[list[list[tuple[str, float]]]] = self._extract_phrases(
                sentences=docs, phrases=phrases
            )

            sorted_keyphrases: list[list[tuple[str, float]]] = [
                _phrases[0] for _phrases in key_phrases
            ]

        # Get top-n phrases
        result_keyphrases: list[list[tuple[str, float]]] = []
        for _keyphrases in sorted_keyphrases:
            unique_keyphrases: dict[str, float] = {}
            for _keyphrase in _keyphrases:
                if _keyphrase[0] not in unique_keyphrases:
                    unique_keyphrases[_keyphrase[0]] = _keyphrase[1]
            top_n_keyphrases = sorted(
                unique_keyphrases.items(), key=lambda x: x[1], reverse=True
            )[: self.params.top_n_phrases]
            result_keyphrases.append(top_n_keyphrases)

        return result_keyphrases
