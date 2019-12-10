from collections import defaultdict

symbols = ['O', 'B-positive', 'I-positive', 'B-neutral', 'I-neutral', 'B-negative', 'I-negative']

def get_symbol_word_counts(training_file):

    # initialize dictionary of counts
    symbol_word_counts = {}
    for symbol in symbols:
        symbol_word_counts[symbol] = defaultdict(int)

    with open(training_file, encoding="utf8") as f:
        for line in f:
            if line.isspace():
                continue
            word = line.split(' ')[0].strip()
            symbol = line.split(' ')[-1].strip()
            symbol_word_counts[symbol][word] += 1

    # find total symbol counts
    symbol_counts = {}
    for symbol in symbol_word_counts:
        symbol_counts[symbol] = sum(symbol_word_counts[symbol].values())

    return symbol_word_counts, symbol_counts

def estimate_emission_params(symbol_word_counts, symbol_counts):

    emission_probab = {}
    for symbol in symbol_word_counts:
        emission_probab[symbol] = {}
        for word in symbol_word_counts[symbol]:
            emission_prob[symbol][word] = float(symbol_word_counts[symbol][word])/(symbol_counts[symbol] + 1)

    return emission_probabilities

def get_emission_prob(training_file):
 
    symbol_word_counts, symbol_counts = get_symbol_word_counts(training_file)
    return estimate_emission_params(symbol_word_counts, symbol_counts)

def emission_prob(symbol, word, emission_prob, symbol_counts):


    unseen_word = True

    for sym in emission_prob:
        if word in emission_prob[sym]:
            unseen_word = False

    if unseen_word:
        return 1/(1 + symbol_counts[symbol])
    else:
        if word in emission_prob[symbol]:
            return emission_prob[symbol][word]
        else:
            return 0

def find_symbol_estimate(dev_file, emission_probabilities, symbol_counts):
    predicted_word_symbol_sequence = []
    with open(dev_file, encoding="utf8") as f:
        for line in f:
            if not line.isspace():
                word = line.strip()
                scores_and_symbols = [(emission_prob(symbol, word, emission_prob, symbol_counts), symbol) for symbol in symbols]
                argmax = max(scores_and_symbols, key=lambda score_and_symbol: score_and_symbol[0])[1]
                predicted_word_symbol_sequence.append((word, argmax))
            else:
                predicted_word_symbol_sequence.append(('',''))

    return predicted_word_symbol_sequence

def write_part_2_dev_out(filename, predicted_word_symbol_sequence):
    result_file = open(filename, "w", encoding="utf8")

    for word_and_symbol in predicted_word_symbol_sequence:
        result_file.write(' '.join(word_and_symbol) + "\n")
        

    
if __name__=="__main__":
    
    