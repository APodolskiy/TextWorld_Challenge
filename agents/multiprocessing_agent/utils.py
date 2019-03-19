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
