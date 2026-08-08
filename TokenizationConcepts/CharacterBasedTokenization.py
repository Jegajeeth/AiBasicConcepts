import matplotlib.pyplot as matplotlib


txt = "this is a test text generated to understand the workings of the tokenization concept."
# txt = "The way you do anything is the way you do everything.";


# toknization using dictanary O(n)
char_vocab = {};
i = 0;

for c in txt:
    if(char_vocab.get(c) == None):
        char_vocab[c] = i;
        i += 1;
print("Token :", end=" ");
for cvk in char_vocab.keys():
    print(f" \"{cvk}\" |", end="");
print();
print("-" * (i * 8));
print("Index:", end=" ");
for cvi in char_vocab.values():
    print(f" \"{cvi}\" |", end="");
print();

tokenizedTxt = [char_vocab[char] for char in txt];
print (tokenizedTxt);

# tokenization using list and sorting O(n log n)
# characters = [char for char in txt];
# char_vocab = list(sorted(set(characters)));

# print(char_vocab);

# tokenizedTxt = [char_vocab.index(char) for char in txt];
# print(tokenizedTxt);

_,ax = matplotlib.subplots();
ax.plot(tokenizedTxt, 'o', markersize=10, markerfacecolor=[.0,.0,.0], linestyle="--");
# ax.set(xlable="Txt tokens");
ax.grid(visible=True, axis="y");
matplotlib.yticks(range(len(char_vocab)));

ax2 = ax.twinx();  # create a second plot within the same graph
ax2.plot(tokenizedTxt,alpha=0); # alpha - changes the oppacity of the plots

# matplotlib.plot(xpoints, ypoints);
matplotlib.xlabel("vocab_text");
matplotlib.ylabel("txt_tokens");
matplotlib.yticks(range(len(char_vocab)), labels=list(char_vocab.keys()));
matplotlib.show();


