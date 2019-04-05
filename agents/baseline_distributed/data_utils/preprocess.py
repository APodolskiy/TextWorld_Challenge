from typing import Optional, List, Callable

INFO_TYPES = ["description", "feedback", "inventory", "recipe"]
SEP_TOKEN = "<SEP>"
ITM_TOKEN = "<ITM>"


def preprocess(text: Optional[str], info_type, tokenizer=None, lower_case=True) -> List[str]:
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
    if lower_case:
        text = text.lower()
    tokens = tokenize_text(text, info_type, tokenizer=tokenizer)
    return tokens


def clean_text(text: str, text_type: str) -> str:
    """
    Clean text according to the provided text type.
    :param text: text string
    :param text_type: type of the text data
    :param sep_token: token used for text parts separation
    :return: cleansed text
    """
    assert type in INFO_TYPES
    if text_type == "feedback":
        if "$$$$$$$" in text:
            text = ""
        if "-=" in text:
            text = text.split("-=")[0]
    elif text_type == "inventory":
        text_parts = [it.strip() for it in text.split("\n") if len(it) > 0]
        text_parts = text_parts[1:] if len(text_parts) > 1 else text_parts
        text = sep_token.join(text_parts)
    text = text.strip()
    if not text:
        text = "nothing"
    return text


def tokenize_text(text: str, text_type: str, tokenizer: Callable) -> List[str]:
    if text_type == "inventory":
        pass
    elif text_type == "recipe":
        pass
    tokens = [t.text for t in tokenizer(text)]
    return tokens


if __name__ == "__main__":
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
    inventory = "You are carrying:\n" \
                "  a roasted white onion\n" \
                "  a red onion\n"
    print(inventory)
    print(clean_text(inventory, "inventory"))
    description = "-= Kitchen =-\n" \
                  "Guess what, you are in a place we're calling a kitchen.\n" \
                  "\n" \
                  "You make out a fridge. The fridge contains a block of cheese. You can make out\n" \
                  "an oven. You make out a table. The table is massive. But the thing is empty. You\n" \
                  "see a counter. The counter is vast. On the counter you can see a cookbook and a\n" \
                  "knife. You can make out a stove. But there isn't a thing on it.\n"
    print(clean_text(description, "description"))
    inventory = [it.strip() for it in inventory.split("\n") if len(it) > 0]
    print(inventory)
