from transformers import AutoTokenizer;
from matplotlib import pyplot as mp;

def gptTwoTokinazation(txt: str) -> list:
    tokenizer = AutoTokenizer.from_pretrained('gpt2');
    gpt2_tokens = tokenizer.encode(txt);
    for token in gpt2_tokens:
        decodedToken = tokenizer.decode([token]);
        print(f"{token}: {decodedToken}", end = ",");
    return list(gpt2_tokens)

def CharacterBasedTokens(txt: str) -> list:
    txtCharsLookuptable = {};
    i = 0;
    for c in txt:
        txtCharsLookuptable[c] = i;
        i += 1;
    tokenizedTxt = []
    for c in txt:
        tokenizedTxt.append(txtCharsLookuptable[c]);
    return tokenizedTxt;

def WordBasedTokens(txt: str) -> list:
    wordListLookupTable = {};
    i = 0;
    for w in set(sorted(txt.split(" "))):
        wordListLookupTable[w] = i;
        i += 1;
    tokenizedTxt = [];
    for w in txt.split(" "):
        tokenizedTxt.append(wordListLookupTable[w]);
    return tokenizedTxt;

def plotgraph(data: list[dict], xlabel: str, ylabel: str, titles: list[str]) -> None:
    _, ax = mp.subplots(1, len(data));
    for i in range(len(data)):
        ax[i].bar(list(data[i].keys()), list(data[i].values()));
        ax[i].set_title(titles[i]);
    # mp.bar(list(data.keys()), list(data.values()));
    # mp.xticks(list(data.keys()));
    mp.xlabel(xlabel);
    mp.ylabel(ylabel);

# txt = "this is a test text generated to understand the workings of the tokenization concept."
txt = "The way you do anything is the way you do everything.";

gpttokens = gptTwoTokinazation(txt);
characterTokens = CharacterBasedTokens(txt);
wordTokens = WordBasedTokens(txt);

compareTotalArray = {"GPT-2" : len(gpttokens), "Character": len(characterTokens), "Word": len(wordTokens)};
compareUniqueArray = {"GPT-2": len(set(gpttokens)),"Character":len(set(characterTokens)), "Word": len(set(wordTokens))};

print(f"\n gpt token count = {len(set(gpttokens))}, \n character token count = {len(set(characterTokens))}, \n word token count = {len(set(wordTokens))}.")

# print(list(compareTotalArray.keys()))
plotgraph([compareTotalArray, compareUniqueArray], "", "Count", ["Total", "unique"]);
# plotgraph(compareTotalArray, "", "Count");
# plotgraph(compareUniqueArray, "", "Count");

mp.show();
