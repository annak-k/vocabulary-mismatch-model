import numpy as np

class Agent(object):
    def __init__(self, dialect):
        lexfile = np.genfromtxt("{}_lexicon.csv".format(dialect), delimiter=',', dtype="unicode")
        self.objects = [str(o) for o in lexfile[0,1:]]
        self.expressions = [str(e) for e in lexfile[1:,0]]
        self.lexicon = np.array(lexfile[1:,1:], dtype=np.float64)
        self.params = {"speaker weight": 0.5, "addressee weight": 0.5, "lambda": 0.9}

    def produce(self, context, addressee):
        self_literal = self.literal_listen(context)
        addressee_literal = addressee.literal_listen(context)

        probs = {}

        for word in set(self_literal).union(set(addressee_literal)):
            utility = (self.params["speaker weight"]*self_literal.get(word, 0) +
                        self.params["addressee weight"]*addressee_literal.get(word, 0))
            probs[word] = np.exp(self.params["lambda"]*utility)

        probs = {k: v/np.sum(list(probs.values())) for k, v in probs.items()}

        return probs
    
    def literal_listen(self, context):
        probs = {}
        c = self.objects.index(context[0])
        for w, word in enumerate(self.expressions):
            if self.lexicon[w][c] > 0:
                probs[word] = np.log(self.lexicon[w][c] * 1/len(context))

        return probs
    
    
if __name__ == "__main__":
    agent1 = Agent("CAN")
    agent2 = Agent("US")
    context1 = ["LAMP", "PEN", "BANANA"]
    context2 = ["SODAPOP", "PEN", "BANANA"]
    context3 = ["TOQUE", "CAP", "PEN"]

    print(agent1.produce(context3, agent2))