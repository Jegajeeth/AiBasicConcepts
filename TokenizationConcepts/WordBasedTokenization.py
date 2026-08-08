
import matplotlib.pyplot as mp;


class WordBasedTokenization:
    def __init__(self):
        # txt = "this is a test text generated to understand the workings of the tokenization concept.";
        txt = "The way you do anything is the way you do everything."

        wordLookTable = {};
        txtWords = txt.split(" ");
        i = 0;
        
        for word in sorted(txtWords):
            wordLookTable[word] = i;
            i += 1;

        # for i in range(len(txtArray)):
        #     txtArray[i] = self.removeSpecialChar(txtArray[i]);
        #     # print(self.removeSpecialChar(txtArray[i]))

        print(f"The list of words are: \n {txtWords}");
        print(f"The word corpus of the txt are: {wordLookTable}");
        
        tokenizedTxt = [wordLookTable[word] for word in txtWords];
        print(f"Tokenized text is: {tokenizedTxt}");
        
        graph = GraphBuilder();
        graph.buildGraphWithSquarePlots(data = tokenizedTxt, xlable="tokenized text", ylable="tokens", showTwin=True, twinYLable="token words", twinPlot=[i for i in wordLookTable.keys()]);
        
    
class GraphBuilder:
    def buildGraphWithSquarePlots(self, data: list, xlable: str, ylable: str = "", showTwin: bool = False, twinYLable: str = "", twinPlot: list = []) -> None:
        _, ax = mp.subplots();
        ax.plot(data, 's');
        ax.grid(visible=True, axis="y");
        mp.yticks(range(len(data)))
        mp.xlabel(xlable);
        mp.ylabel(ylable);
        
        if(showTwin):
            ax2 = ax.twinx();
            ax2.plot(twinPlot, alpha = 0.1); # alpha - changes the oppacity of the plots
            mp.ylabel(twinYLable);
        
        mp.show();

    # def removeSpecialChar(self, s: str) -> str:
    #     rs = [];
    #     for i in s:
    #         if(not i.isalnum()):
    #             continue;
    #         rs.append(i);
        
    #     return "".join(rs);

tokenization = WordBasedTokenization();