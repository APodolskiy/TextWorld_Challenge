from typing import Optional


INFO_TYPES = ["description", "feedback", "inventory", "recipe"]


def preprocess(text: Optional[str], info_type, tokenizer=None, lower_case=True):
    assert info_type in INFO_TYPES
    text = clean_text(text, info_type)


def clean_text(text: str, text_type: str) -> str:
    assert type in INFO_TYPES
    if text_type == "feedback":
        if "$$$$$$$" in text:
            text = ""
        if "-=" in text:
            text = text.split("-=")[0]
    elif text_type == "inventory":
        text = [it.strip() for it in text.split("\n") if len(it) > 0]
        if len(text) > 0:
            pass
    elif text_type == "recipe":
        pass
    else:
        pass
    return text


def tokenize_text(text: str, text_type: str):
    pass


def clean_text(s, type_):
    assert type_ in ["inventory", "feedback", "description"]
    if type_ == "inventory":
        result = s.split("carrying")[1]
        if result[0] == ":":
            result = result[1:]
    elif type_ == "feedback":
        result = s
        if "$$$$$$$" in result:
            result = ""
        if "-=" in result:
            result = result.split("-=")[0]
    else:
        result = s
    result = result.strip()
    if not result:
        result = "nothing"
    return result


def preproc(s, str_type='None', tokenizer=None, lower_case=True):
    if s is None:
        return ["nothing "]
    s = s.replace("\n", ' ')
    if s.strip() == "":
        return ["nothing"]
    if str_type == 'feedback':
        if "$$$$$$$" in s:
            s = ""
        if "-=" in s:
            s = s.split("-=")[0]
    s = s.strip()
    if len(s) == 0:
        return ["nothing"]
    tokens = [t.text for t in tokenizer(s)]
    if lower_case:
        tokens = [t.lower() for t in tokens]
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
