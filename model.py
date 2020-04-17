import numpy as np

class Agent(object):
    def __init__(
            self, dialect, lambda_param=0.9, speaker_weight=0.5,
            length_cost_smoothing=0.1, preferences=True,
            pragmatic_weight=1, pref_weight=1):
        lexfile = np.genfromtxt("{}_lexicon.csv".format(dialect), delimiter=',', dtype="unicode")
        self.objects = [str(o) for o in lexfile[0,1:]]
        self.expressions = [str(e) for e in lexfile[1:,0]]
        self.preferences = np.array(lexfile[1:,1:], dtype=np.float64)
        
        # Made lexicon binary and added preferences file.
        self.lexicon = np.array(lexfile[1:,1:], dtype=np.float64)
        self.lexicon[self.lexicon > 0] = 1
        self.params = {
            "speaker weight": speaker_weight,
            "lambda": lambda_param,
            "length cost smoothing": length_cost_smoothing,
            "pragmatic_weight": pragmatic_weight,
            "pref_weight": pref_weight
        }

#     def produce(self, context, addressee):
#         self_literal = self.literal_listen(context)
#         addressee_literal = addressee.literal_listen(context)

#         probs = {}

#         for word in set(self_literal).union(set(addressee_literal)):
#             # TODO: we shouldn't use .get(word, 0), because, since it's been logged,
#             # it should really be -infinity
#             utility = (self.params["speaker weight"]*self_literal.get(word, 0) +
#                         self.params["addressee weight"]*addressee_literal.get(word, 0))
#             probs[word] = np.exp(self.params["lambda"]*utility)

#         probs = {k: v/np.sum(list(probs.values())) for k, v in probs.items()}
#         return probs
    
#     def literal_listen(self, context):
#         probs = {}
#         c = self.objects.index(context[0])
#         for w, word in enumerate(self.expressions):
#             if self.lexicon[w][c] > 0:
#                 probs[word] = np.log(self.lexicon[w][c] * 1/len(context))
#         return probs
    
    
    ### EXPERIMENTAL: ADDITIONAL LAYER OF RECURSION FOR PRAGMATIC SPEAKER ### 
    
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
        # probs = np.log(probs)
        return probs
    
    
    def pragmatic_listen_matrix(self, context, speaker):
        probs = np.zeros_like(self.preferences)
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
    
    
    def produce_matrix(self, context, addressee, recurse=False, length_cost=False):
        if recurse:
            self_listener = self.pragmatic_listen_matrix(context, self)
            addressee_listener = addressee.pragmatic_listen_matrix(context, self)
        else:
            self_listener = self.literal_listen_matrix(context)
            addressee_listener = addressee.literal_listen_matrix(context)
            
        # print(self_listener)
        # print(addressee_listener)

        probs = np.zeros((len(self.expressions), len(context)))

        for c, curr_object in enumerate(context):
            for w, word in enumerate(self.expressions):
                o = self.objects.index(curr_object)
                if self_listener[w, o] >= 0 or addressee_listener[w, o] >= 0:
                    utility = (self.params["speaker weight"]*self_listener[w, o] +
                               (1 - self.params["speaker weight"])*addressee_listener[w, o])
                    if length_cost:
                        utility *= self.params["length cost weight"] / len(word.split("_"))
                    # probs[w, c] = np.exp(self.params["lambda"]*utility)
                    probs[w, c] = self.params["lambda"]*utility

        # print(probs)
        # probs = {k: v/np.sum(list(probs.values())) for k, v in probs.items()}
        probs = np.array([row / np.sum(row) if np.sum(row) != 0 else row
                          for row in probs.T]).T
        return {expression: prob for expression, prob in zip(self.expressions, probs[:, 0])
                if prob > 0}

    def produce_matrix_plain(self, context):
        listener = self.literal_listen_matrix(context)
        probs = np.zeros((len(self.expressions), len(context)))
        # print(listener)
        for c, curr_object in enumerate(context):
            for w, _ in enumerate(self.expressions):
                o = self.objects.index(curr_object)
                utility = np.where(listener[w, o] > 0.00001, np.log(listener[w, o]), np.NaN)
                if listener[w, o] > 0:
                    probs[w, c] = np.where(not np.isnan(utility), np.exp(self.params["lambda"]*utility), 0)
        # print(probs)
        probs = np.array([row / np.sum(row) if np.sum(row) != 0 else row
                          for row in probs.T]).T
        # print(probs)
        return {expression: prob for expression, prob in zip(self.expressions, probs[:, 0])
                if prob > 0}

    ### EXPERIMENTAL: ALTERNATIVE VERSION WITH COMBINED VOCABULARY
    
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


#     def produce_matrix_mutant(
#             self, context, addressee, recurse=False, length_cost=False):
#         self_listener = self.literal_listen_matrix(context)
#         addressee_listener = self.literal_listen_matrix_mutant(
#             context, addressee)
        
#         probs = np.empty((len(self.expressions), len(context)))

#         for c, curr_object in enumerate(context):
#             for w, word in enumerate(self.expressions):
#                 o = self.objects.index(curr_object)
#                 if (self_listener[w, o] != np.NaN or
#                     addressee_listener[w, o] != np.NaN):
#                     utility = (self.params["addressee weight"] +
#                                addressee_listener[w, o] +
#                                np.mean(
#                                    [addressee.preferences[w, o],
#                                     self.preferences[w, o]]))
#                     if length_cost:
#                         utility += self.params["length cost weight"] - len(word.split("_"))
#                     probs[w, c] = np.where(not np.isnan(utility), np.exp(self.params["lambda"]*utility), 0)
#         probs = np.array([row / np.sum(row) if np.sum(row) != 0 else row
#                           for row in probs.T]).T
#         return {expression: prob
#                 for expression, prob in zip(self.expressions, probs[:, 0])
#                 if prob > 0}
    
    
    def produce_matrix_mutant(
            self, context, addressee, recurse=False, length_cost=False):
        # self_listener = self.literal_listen_matrix(context)
        combined_listener = self.literal_listen_matrix_mutant(
            context, addressee)
        
        probs = np.zeros((len(self.expressions), len(context)))

        for c, curr_object in enumerate(context):
            for w, word in enumerate(self.expressions):
                o = self.objects.index(curr_object)

                utility_combined = np.where(
                    combined_listener[w, o] > 0.00001, 
                    np.log(combined_listener[w, o]), np.NaN)

                utility_prefer = np.where(
                    addressee.preferences[w, o] > 0.00001 and self.preferences[w, o] > 0.00001, 
                    np.log((1 - self.params["speaker weight"]) * addressee.preferences[w, o] +
                           self.params["speaker weight"] * self.preferences[w, o]), 
                    np.NaN)
#                 utility_prefer = (
#                     (1 - self.params["speaker weight"]) * addressee.preferences[w, o] +
#                     self.params["speaker weight"] * self.preferences[w, o])

                utility_vmm = (self.params["pragmatic_weight"] * utility_combined +
                               self.params["pref_weight"] * utility_prefer)
                if length_cost:
#                     utility_vmm += np.log(
#                         (1 + self.params["length cost smoothing"]) / 
#                         (len(word.split("_")) + self.params["length cost smoothing"]))
#                     utility_vmm += (
#                         (1 + self.params["length cost smoothing"]) / 
#                         (len(word.split("_")) + self.params["length cost smoothing"]))

                    utility_vmm += (self.params["length cost smoothing"] *
                                    np.log(1 / len(word.split("_"))))
#                     utility_vmm += (
#                         self.params["length cost smoothing"] / len(word.split("_")))

                probs[w, c] = np.where(
                    not np.isnan(utility_vmm),
                    np.exp(self.params["lambda"] * utility_vmm), 0)

        probs = np.array([row / np.sum(row) if np.sum(row) != 0 else row
                          for row in probs.T]).T
        return {expression: prob
                for expression, prob in zip(self.expressions, probs[:, 0])
                if prob > 0}
    