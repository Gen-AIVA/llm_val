import os


class CHRF():
    def __init__(self,
                 lowercase: bool=True,
                 char_order: int=6,
                 word_order: int=0,
                 beta: int=2,
                 whitespace: bool=False):
        self.lowercase = lowercase
        self.char_order = char_order
        self.word_order = word_order
        self.beta = beta
        self.whitespace = whitespace
        self.chrf = os.environ["chrf"]

    def evaluate_response(self, 
                          response: str, 
                          reference: str) -> float:
        response = response.lower()
        reference = reference.lower()
        reference_list = reference.split()
        reference_list = [w for w in reference_list if len(w) > 2]
        if not any(word in response for word in reference_list):
            return 0

        scores = self.chrf.compute(predictions=[response],
                                   references=[reference],
                                   lowercase=self.lowercase,
                                   char_order=self.char_order,
                                   word_order=self.word_order,
                                   beta=self.beta,
                                   whitespace=self.whitespace)
        return scores['score']


