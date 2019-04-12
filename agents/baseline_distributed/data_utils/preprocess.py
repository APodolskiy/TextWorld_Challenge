from typing import Optional, List, Callable

import spacy

INFO_TYPES = ["description", "feedback", "inventory", "recipe", "command"]
SEP_TOKEN = "<SEP>"
ITM_TOKEN = "<ITM>"


def preprocess(text: Optional[str], info_type: str, tokenizer: Callable, lower_case: bool = True) -> List[str]:
    """
    Clean and tokenize text
    :param text: text string
    :param info_type: text data type
    :param tokenizer: function for tokenization
    :param lower_case: whether to lowercase text
    :return: list of tokens
    """
    assert info_type in INFO_TYPES
    text = clean_text(text, info_type)
    tokens = tokenize_text(text, info_type, tokenizer=tokenizer, lower_case=lower_case)
    return tokens


def clean_text(text: str, text_type: str) -> str:
    """
    Clean text according to the provided text type.
    :param text: text string
    :param text_type: type of the text data
    :return: cleansed text
    """
    assert text_type in INFO_TYPES
    if text_type == "feedback":
        if "$$$$$$$" in text:
            text = ""
        if "-=" in text:
            text = text.split("-=")[0]
    text = text.strip()
    if not text:
        text = "nothing"
    return text


def tokenize_text(text: str, text_type: str, tokenizer: Callable, lower_case: bool = True) -> List[str]:
    tokens = []
    if text_type == "inventory":
        if lower_case:
            text = text.lower()
        text_parts = [it.strip() for it in text.split("\n") if len(it) > 0]
        if len(text_parts) > 1:
            text_parts = [it.strip() for it in text_parts[1:]]
            tokens = _tokenize_item_text_parts(text_parts, tokenizer)
        else:
            tokens = [t.text for t in tokenizer(text_parts[0])]
    elif text_type == "recipe":
        ingr_pos = text.find("Ingredients:") + len("Ingredients:")
        dirc_pos = text.find("Directions:")
        if lower_case:
            text = text.lower()
        ingredients = [it.strip() for it in text[ingr_pos:dirc_pos].split("\n") if it]
        directions = [it.strip() for it in text[dirc_pos + len("Directions:"):].split("\n") if it]
        ingr_tokens = _tokenize_item_text_parts(ingredients, tokenizer)
        dirc_tokens = _tokenize_item_text_parts(directions, tokenizer)
        tokens = ingr_tokens + [SEP_TOKEN] + dirc_tokens
    else:
        text = text.replace('\n', ' ').strip()
        tokens = [t.text for t in tokenizer(text) if t.text != ' ']
    return tokens


def _tokenize_item_text_parts(text_parts: List[str], tokenizer: Callable, itm_token: str = ITM_TOKEN):
    tokens = []
    for text_part in text_parts:
        text_tokens = [t.text for t in tokenizer(text_part)]
        tokens += [itm_token] + text_tokens
    return tokens


if __name__ == "__main__":
    nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])
    feedback = "You are hungry! Let's cook a delicious meal. Check the cookbook in the kitchen\n" \
               "for the recipe. Once done, enjoy your meal!\n" \
               "\n" \
               "-= Kitchen =-\n" \
               "Guess what, you are in a place we're calling a kitchen.\n" \
               "\n" \
               "You make out a fridge. The fridge contains a block of cheese. You can make out\n" \
               "an oven. You make out a table. The table is massive. But the thing is empty. You\n" \
               "see a counter. The counter is vast. On the counter you can see a cookbook and a\n" \
               "knife. You can make out a stove. But there isn't a thing on it."
    print(clean_text(feedback, "feedback"))
    print(preprocess(feedback, "feedback", tokenizer=nlp))
    inventory = "You are carrying:\n" \
                "  a roasted white onion\n" \
                "  a red onion\n"
    print(inventory)
    print(clean_text(inventory, "inventory"))
    print(preprocess(inventory, "inventory", tokenizer=nlp))
    description = "-= Kitchen =-\n" \
                  "Guess what, you are in a place we're calling a kitchen.\n" \
                  "\n" \
                  "You make out a fridge. The fridge contains a block of cheese. You can make out\n" \
                  "an oven. You make out a table. The table is massive. But the thing is empty. You\n" \
                  "see a counter. The counter is vast. On the counter you can see a cookbook and a\n" \
                  "knife. You can make out a stove. But there isn't a thing on it.\n"
    print(clean_text(description, "description"))
    print(preprocess(description, "description", nlp))
    recipe = "Ingredients:\n" \
             "  white onion\n" \
             "  yellow potato\n" \
             "\n" \
             "Directions:\n" \
             "  slice the white onion\n" \
             "  roast the white onion\n" \
             "  roast the yellow potato\n" \
             "  prepare meal"
    print(recipe)
    print(preprocess(recipe, "recipe", nlp))
