import numpy as np


class Agent(object):
    def __init__(
            self, dialect, lambda_param=1, speaker_weight=0.5,
            length_cost_smoothing=1, preferences=True,
            pragmatic_weight=1, pref_weight=1):
        lexfile = np.genfromtxt("{}_lexicon.csv".format(dialect), delimiter=',', dtype="unicode")
        self.objects = [str(o) for o in lexfile[0,1:]]
        self.expressions = [str(e) for e in lexfile[1:,0]]
        self.preferences = np.array(lexfile[1:,1:], dtype=np.float64)
        
        self.lexicon = np.array(lexfile[1:,1:], dtype=np.float64)
        self.lexicon[self.lexicon > 0] = 1
        self.params = {
            "speaker weight": speaker_weight,
            "lambda": lambda_param,
            "length cost smoothing": length_cost_smoothing,
            "pragmatic_weight": pragmatic_weight,
            "pref_weight": pref_weight
        }
        
    def literal_listen_matrix(self, context):
        probs = np.zeros_like(self.preferences)
        for o, curr_object in enumerate(self.objects):
            for w, word in enumerate(self.expressions):
                if curr_object in context and self.lexicon[w][o] > 0:
                    probs[w, o] = self.lexicon[w, o] * 1/len(context)
                else:
                    probs[w, o] = 0
        probs = np.array([row / np.sum(row) if np.sum(row) > 0 else row
                          for row in probs])
        return probs

    def produce_matrix_plain(self, context):
        listener = self.literal_listen_matrix(context)
        probs = np.zeros((len(self.expressions), len(context)))
        for c, curr_object in enumerate(context):
            for w, _ in enumerate(self.expressions):
                o = self.objects.index(curr_object)
                utility = np.where(listener[w, o] > 0.00001, np.log(listener[w, o]), np.NaN)
                if listener[w, o] > 0:
                    probs[w, c] = np.where(not np.isnan(utility), np.exp(self.params["lambda"]*utility), 0)
        probs = np.array([row / np.sum(row) if np.sum(row) != 0 else row
                          for row in probs.T]).T
        return {expression: prob for expression, prob in zip(self.expressions, probs[:, 0])
                if prob > 0}

    def literal_listen_matrix_mutant(self, context, other):
        mutant_lexicon = self.lexicon + other.lexicon
        mutant_lexicon[mutant_lexicon > 1] = 1
        probs = np.zeros_like(mutant_lexicon)
        for o, curr_object in enumerate(self.objects):
            for w, word in enumerate(self.expressions):
                if curr_object in context and mutant_lexicon[w][o] > 0:
                    probs[w, o] = mutant_lexicon[w][o] * 1/len(context)
                else:
                    probs[w, o] = 0
        probs = np.array([row / np.sum(row) if np.sum(row) > 0 else row
                          for row in probs])
        return probs 

    def produce_matrix_mutant(
            self, context, addressee, recurse=False, length_cost=False):
        combined_listener = self.literal_listen_matrix_mutant(
            context, addressee)
        
        probs = np.zeros((len(self.expressions), len(context)))

        for c, curr_object in enumerate(context):
            for w, word in enumerate(self.expressions):
                o = self.objects.index(curr_object)

                utility_combined = np.where(
                    combined_listener[w, o] > 0.00001, 
                    np.log(combined_listener[w, o]), np.NaN)
                
                utility_prefer = (
                    (1 - self.params["speaker weight"]) * addressee.preferences[w, o] +
                    self.params["speaker weight"] * self.preferences[w, o])
                utility_prefer = np.where(
                    utility_prefer > 0.00001, 
                    np.log(utility_prefer), 
                    np.NaN)
        
                utility_vmm = (self.params["pragmatic_weight"] * utility_combined +
                               self.params["pref_weight"] * utility_prefer)
                if length_cost:
                    utility_vmm += (self.params["length cost smoothing"] *
                                    np.log(1 / len(word.split("_"))))

                probs[w, c] = np.where(
                    not np.isnan(utility_vmm),
                    np.exp(self.params["lambda"] * utility_vmm), 0)

        probs = np.array([row / np.sum(row) if np.sum(row) != 0 else row
                          for row in probs.T]).T
        return {expression: prob
                for expression, prob in zip(self.expressions, probs[:, 0])
                if prob > 0}
    