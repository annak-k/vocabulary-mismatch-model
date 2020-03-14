import numpy as np

class Agent(object):
    def __init__(self, dialect, lambda_param=0.9):
        lexfile = np.genfromtxt("{}_lexicon.csv".format(dialect), delimiter=',', dtype="unicode")
        self.objects = [str(o) for o in lexfile[0,1:]]
        self.expressions = [str(e) for e in lexfile[1:,0]]
        self.lexicon = np.array(lexfile[1:,1:], dtype=np.float64)
        self.params = {"speaker weight": 0.5, "addressee weight": 0.5, "lambda": lambda_param}

    def produce(self, context, addressee):
        self_literal = self.literal_listen(context)
        addressee_literal = addressee.literal_listen(context)

        probs = {}

        for word in set(self_literal).union(set(addressee_literal)):
            # TODO: we shouldn't use .get(word, 0), because, since it's been logged,
            # it should really be -infinity
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
    
    
    ### EXPERIMENTAL: ADDITIONAL LAYER OF RECURSION FOR PRAGMATIC SPEAKER ### 
    
    def literal_listen_matrix(self, context):
        probs = np.zeros_like(self.lexicon)
        for o, curr_object in enumerate(self.objects):
            for w, word in enumerate(self.expressions):
                if curr_object in context and self.lexicon[w][o] > 0:
                    probs[w, o] = np.log(self.lexicon[w][o] * 1/len(context))
        return probs
    
    
    def pragmatic_listen_matrix(self, context, speaker):
        probs = np.zeros_like(self.lexicon)
        for i in range(len(context)):
            curr_context = [context[i]] + context[:i] + context[i+1:]
            curr_word_probs = speaker.produce_matrix(curr_context, self, recurse=False)
            o = self.objects.index(context[i])
            for word, prob in curr_word_probs.items():
                w = self.expressions.index(word)
                probs[w, o] = prob
        probs = np.array([row / np.sum(row) if np.sum(row) > 0 else row
                          for row in probs])
        return probs
    
    
    def produce_matrix(self, context, addressee, recurse=False):
        if recurse:
            self_speaker = self.pragmatic_listen_matrix(context, self)
            addressee_speaker = addressee.pragmatic_listen_matrix(context, self)
        else:
            self_speaker = self.literal_listen_matrix(context)
            addressee_speaker = addressee.literal_listen_matrix(context)

        probs = {}

        c = self.objects.index(context[0])
        for w, word in enumerate(self.expressions):
            if self_speaker[w, c] != 0 or addressee_speaker[w, c] != 0:
                utility = (self.params["speaker weight"]*self_speaker[w, c] +
                           self.params["addressee weight"]*addressee_speaker[w, c])
                probs[word] = np.exp(self.params["lambda"]*utility)

        probs = {k: v/np.sum(list(probs.values())) for k, v in probs.items()}
        return probs
            

    
    
if __name__ == "__main__":
    agent1 = Agent("CAN")
    agent2 = Agent("US")
    context1 = ["LAMP", "PEN", "BANANA"]
    context2 = ["SODAPOP", "PEN", "BANANA"]
    context3 = ["TOQUE", "CAP", "PEN"]

    print(agent1.produce(context3, agent2))