from matplotlib import pyplot as mp;
from transformers import AutoTokenizer;
import requests

baseurl = 'https://www.gutenberg.org/cache/epub'

bookurls = [
    # code       title
    ['84',    'Frankenstein'    ],
    ['64317', 'GreatGatsby'     ],
    ['11',    'AliceWonderland' ],
    ['1513',  'RomeoJuliet'     ],
    ['76',    'HuckFinn'        ],
    ['219',   'HeartDarkness'   ],
    ['2591',  'GrimmsTales'     ],
    ['2148',  'EdgarAllenPoe'   ],
    ['36',    'WarOfTheWorlds'  ],
    ['829',   'GulliversTravels']
]

tokenLenData = [];
tokenLenUniqueData = [];
tokenizer = AutoTokenizer.from_pretrained('gpt2');

for [code, title] in bookurls:
    url = f"{baseurl}/{code}/pg{code}.txt"
    bookContent = requests.get(url).text;
    gptTwoToken = tokenizer.encode(bookContent);
    bookWords = bookContent.split(" ");
    tokenLenData.append([title, len(bookContent), len(bookWords), len(gptTwoToken)])


print(f'  {"Book title":14} |  {"Chars":>7}  |  {"Words":>7}  |  {"Tokens":>7} ')
print('-----------------+-----------+-----------+-------------')
for [title, bookContentlen, bookWordslen, gptTwoTokenlen] in tokenLenData:
    print(f"{title:16} |  {bookContentlen:>7,d}  |  {bookWordslen:>7d}  |  {gptTwoTokenlen:>7d}");

# _, ax = mp.subplots(1, 2);
_, ax = mp.subplots();

characterPlotData = [];
characterPlotData.append(list(contentLendata[1] for contentLendata in tokenLenData));
characterPlotData.append(list(contentLendata[2] for contentLendata in tokenLenData));
characterPlotData.append(list(contentLendata[3] for contentLendata in tokenLenData));
ax.plot(characterPlotData, "h");
# ax[0].plot(1, (contentLendata[2] for contentLendata in tokenLenData), "s", xticks="Words");
# ax[0].plot(2, (contentLendata[3] for contentLendata in tokenLenData), "o", xticks="GPT-2");

mp.show();