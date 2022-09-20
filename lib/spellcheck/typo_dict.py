pattern_dict = {
    "patients father:": "",
    "patients mother:": "",
    "patients guardian:": "",
    "patient:": "",
    "doctors:": "",
    "doctor:": "",
    "ptafather:": "",
    "patient's husband" : "",
    "patients husband" : "",
}
typo_dict = {
    "paitent's": "patients",
    "Pateints": "patients",
    "pateint's": "patients",
    "patient’s": "patients",
    "Parients": "patients",
    "pateints": "patients",
    "paitents": "patients",
    "docto": "doctor",
    "studiy": "study",
    "conversaion": "conversation",
    "convsersation": "conversation",
    "ecord": "record",
    "conerns": "concerns",
    "permisso": "permission",
    "paitent'": "patients",
    "gardurdain": "guardian",
    "parebts": "parents",
    "recrding": "recording",
    "gardurdain's": "guardians",
    "gurdains": "guardians",
    "patietnt's": "patients",
    "sfather": "father",
    "fathter": "father",
    "fatherr": "father",
    "uncle": "guardian",
    "procced": "proceed",
    "isue": "issue",
    "pateint": "patients",
    "childerns": "children",
    "brusied": "bruised",
    "x-ray": "xray",
    "illeness": "illness",
    "check-up": "checkup",
    "that’s": "that is",
    "he’ll": "he will",
    "i’ll": "i will",
    "unhygenic": "unhygienic",
    "didn’t": "did not",
    "wasn’t": "was not",
    "wasnt": "was not",
    "won’t": "wont",
    "whats": "what is",
    "it’s": "it is",
    "haven’t": "have not",
    "doesn’t": "does not",
    "doesnt": "does not",
    "don’t": "do not",
    "diaherria": "diarrhea",
    "dr": "doctor",
    "seziures": "seizures",
    "concers": "concerns",
    "probelm": "problem",
    "medcine": "medicine",
    "gurdian": "guardian",

}

name_list = ["tabassum", "wania", "hina", "emaan", "basit", "sharjeel", "zarish", "husnain", "fareeqa", "waqas",
             "mujtaba",
             "raqeeb", "sania", "ehtesham", "hira", "jahangir", "halima", "anusha", "minahil", "sajid", "qaiser",
             "hashim",
             "husna", "malaika", "subhan", "anaya", "hashir", "nisar", "fahad", "fahd", "zeeshan", "abiha", "maaz",
             "qurat",
             "azaan", "azhaan", "jasim", "umaima", "beenish", "laiba", "nisaar", "sundas", "sikandar", "ayaz",
             "hoorain",
             "isra", "umer", "hunain", "aamir", "fiza", "hasaam", "sughra", "sughara", "inshal", "tabeer", "hammad",
             "bakhsh", "humayun", "kashif", "saima", "shayan", "saboor", "nabila", "abdur", "inaya", "nasir", "ahil",
             "ayaan", "rafay", "daniyal", "irum", "dania", "wajid", "daud", "razzaq", "hasnain", "zahid", "rubina",
             "hajra", "mehrab", "anoushy", "jawad", "khagista", "malika", "wahajh", "huzaifa", "hania", "jalil",
             "um-e-hania", "tabish", "abeera", "ammara", "wahab", "haseena", "nauman", "raikot", "pindi", "iqra",
             "moeez", "adiba", "njab", "hasham", "umer"]

medicine_list = ["calpol", "burofen", "pulmonol", "amoxil", "augmentin", "brufen", "amoxilin", "cerelac", "bf1", "bflg"
    , "ors"]

common_phrase_list = ["inshallah", "alekum", "assalam", "insha'allah", ""]

pattern_dict = {
    ":": "",
    "patients mother:": "",
    "patients guardian:": "",
    "patient:": "",
    "doctors:": "",
    "doctor:": "",
    "ptafather:": "",
}

numwords = {}


def init_numwords():
    # create our default word-lists
    # singles
    units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
    ]

    # tens
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    # larger scales
    scales = ["hundred", "thousand", "million", "billion", "trillion"]

    # divisors
    numwords["and"] = (1, 0)

    # perform our loops and start the swap
    for idx, word in enumerate(units):    numwords[word] = (1, idx)
    for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
    for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)


def parse_int(textnum):
    # initialize the word list if it is empty
    if not numwords:
        init_numwords()

    # primary loop
    current = result = 0
    # loop while splitting to break into individual words
    stringToReplace = ""
    returnTextNum = textnum
    for word in textnum.replace("-", " ").split():
        # if problem then fail-safe
        if word not in numwords:
            # print("Illegal word: " + word)
            if stringToReplace != "":
                # print("current number in cache:" + str(current+result))
                returnTextNum = returnTextNum.replace(stringToReplace.strip(), str(current + result), 1)
                stringToReplace = ""
                result = 0
                current = 0
            continue

        if word == "and" and stringToReplace == "":
            continue

        # use the index by the multiplier
        scale, increment = numwords[word]
        current = current * scale + increment
        # print(scale)
        stringToReplace += word + " "
        # if larger than 100 then push for a round 2
        if scale > 100:
            result += current
            current = 0

    if stringToReplace != "":
        # print("Final current number in cache:" + str(current+result))
        returnTextNum = returnTextNum.replace(stringToReplace.strip(), str(current + result), 1)

    # return the result plus the current
    return returnTextNum
