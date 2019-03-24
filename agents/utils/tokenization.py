import spacy
from spacy.attrs import ORTH, LEMMA, POS
from spacy.lang.en import STOP_WORDS


class SpacyVectorizer:
    def __init__(self):
        self.stop_words = {"a", "the"}
        with open("vocab.txt") as f:
            self.vocab = [line.rstrip("\n") for line in f]
        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.pad_idx = self.token_to_idx["<PAD>"]
        self.unk_idx = self.token_to_idx["<UNK>"]
        self.join_symbol = "<|>"
        self.join_symbol_idx = self.token_to_idx[self.join_symbol]
        self.tokenizer = spacy.load("en_core_web_sm").tokenizer
        for exception in [self.join_symbol, "<s>", "</s>"]:
            tok_exceptions = [
                exception,
                [{ORTH: exception, LEMMA: exception, POS: "VERB"}],
            ]
            self.tokenizer.add_special_case(*tok_exceptions)

    def __call__(self, s: str):
        raw_tokens = self.tokenizer(s.lower())
        final_tokens = []
        bad_symbols = {"_", "|", "\|", ","}
        for token in raw_tokens:
            if not token.is_space and not token.pos_ in ["PUNCT", "SYM"]:
                cleaned_token = token.orth_.strip()
                if (
                    cleaned_token
                    and cleaned_token not in bad_symbols
                    and "$" not in cleaned_token
                    and cleaned_token not in self.stop_words
                ):
                    final_tokens.append(cleaned_token)
        indices = [self.token_to_idx.get(t, self.unk_idx) for t in final_tokens]
        return indices




if __name__ == "__main__":
    pass