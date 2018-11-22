import copy
import pickle
from functools import reduce
from scipy.misc import logsumexp
import numpy as np

START_STATE = '*START*'
START_WORD = '*START*'
END_STATE = '*END*'
END_WORD = '*END*'
RARE_WORD = '*RARE_WORD*'


def load_data(data_path='PoS_data.pickle',
              words_path='all_words.pickle',
              pos_path='all_PoS.pickle'):
    with open('PoS_data.pickle', 'rb') as f:
        data = pickle.load(f)
    with open('all_words.pickle', 'rb') as f:
        words = pickle.load(f)
    with open('all_PoS.pickle', 'rb') as f:
        pos = pickle.load(f)

    return data, words, pos


def data_example(data_path='PoS_data.pickle',
                 words_path='all_words.pickle',
                 pos_path='all_PoS.pickle'):
    """
    An example function for loading and printing the Parts-of-Speech data for
    this exercise.
    Note that these do not contain the "rare" values and you will need to
    insert them yourself.

    :param data_path: the path of the PoS_data file.
    :param words_path: the path of the all_words file.
    :param pos_path: the path of the all_PoS file.
    """

    with open('PoS_data.pickle', 'rb') as f:
        data = pickle.load(f)
    with open('all_words.pickle', 'rb') as f:
        words = pickle.load(f)
    with open('all_PoS.pickle', 'rb') as f:
        pos = pickle.load(f)

    print("The number of sentences in the data set is: " + str(len(data)))
    print("\nThe tenth sentence in the data set, along with its PoS is:")
    print(data[10][1])
    print(data[10][0])

    print("\nThe number of words in the data set is: " + str(len(words)))
    print("The number of parts of speech in the data set is: " + str(len(pos)))

    print("one of the words is: " + words[34467])
    print("one of the parts of speech is: " + pos[17])

    print(pos)


class Baseline(object):
    '''
    The baseline model.
    '''

    def __init__(self, pos_tags, words, training_set):
        '''
        The init function of the baseline Model.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        '''

        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.training_set = training_set
        self.pos2i = {pos: i for (i, pos) in enumerate(pos_tags)}
        self.word2i = {word: i for (i, word) in enumerate(words)}
        # self.pos_word_prob = self.get_pos_word_probability(training_set)
        self.tag_word_data = self.data_to_taples(training_set)
        self.e = self.baseline_mle()
        # TODO: YOUR CODE HERE

    def MAP(self, sentences):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
        '''
        tags = []
        for sentence in sentences:
            sentence_tags = []
            for word in sentence:
                sentence_tags.append(self.find_max_tag(word))
            tags.append(sentence_tags)
        return tags

    def find_max_tag(self, word):
        """
        find tag that that discribes the word in the most sences
        """
        count = []
        for tag in self.pos_tags:
            count.append(self.tag_word_data.count((tag, word)))
        max_index = np.argmax(np.asarray(count))
        return self.pos_tags[max_index]

    def data_to_taples(self, data):
        tag_word_data = []
        for sentence in data:
            for i in range(len(sentence[0])):
                tag_word_data.append((sentence[0][i], sentence[1][i]))
        return tag_word_data

    def baseline_mle(self):
        """
        a function for calculating the Maximum Likelihood estimation of the
        multinomial and emission probabilities for the baseline model.

        :param training_set: an iterable sequence of sentences, each containing
                both the words and the PoS tags of the sentence (as in the "data_example" function).
        :param model: an initial baseline model with the pos2i and word2i mappings among other things.
        :return: a mapping of the multinomial and emission probabilities. You may implement
                the probabilities in |PoS| and |PoS|x|Words| sized matrices, or
                any other data structure you prefer.
        """
        tag_word_tapples = self.data_to_taples(self.training_set)
        e = np.zeros((self.pos_size, self.words_size))
        for tag, word in tag_word_tapples:
            e[self.pos2i[tag], self.word2i[word]] += 1
        sum = e.sum(axis=1)[..., np.newaxis]
        e = e / sum
        return e


class HMM(object):
    '''
    The basic HMM_Model with multinomial transition functions.
    '''

    def __init__(self, pos_tags, words, training_set):
        '''
        The init function of the basic HMM Model.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        '''

        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos: i for (i, pos) in enumerate(pos_tags)}
        self.word2i = {word: i for (i, word) in enumerate(words)}
        self.training_set = training_set
        self.e = []
        self.t = {}
        self.q = {}
        self.hmm_mle()

        # TODO: YOUR CODE HERE

    def sample_word(self, last_word):
        """
        sample a werd from the HMM given the last imeshin was word = last_word
        """
        return

    def sample(self, n):
        '''
        Sample n sequences of words from the HMM.
        :return: A list of word sequences.
        '''
        old_tag="START"
        word = old_tag
        sentances = []

        for i in range(n):
            sentance = []
            old_tag = "START"
            while word != "END" and word != ".":
                if sentance.count(",") ==3 and sentance[-1] == ",":
                    sentance[-1] = "END"
                    break
                trancemition_probabileteys = [self.t[new_tag, old_tag] for new_tag in self.pos_tags]
                print(np.sum(trancemition_probabileteys))
                print("the old_tag iis: "+old_tag)
                old_tag = np.random.choice(self.pos_tags, p=trancemition_probabileteys)
                emition_probabileteys = self.e[self.pos2i[old_tag],:]
                print(np.sum(emition_probabileteys))

                word = np.random.choice(self.words, p=emition_probabileteys)
                while word == "RARE":
                    word = np.random.choice(self.words, p=emition_probabileteys)
                sentance.append(word)
            sentances.append(sentance)
        return sentances

    # TODO: YOUR CODE HERE

    def one_virerbi(self, sentence):
        if(sentence[0]=="START"):
            sentence.remove("START")
        if(sentence[-1]=="END"):
            sentence.remove("END")
        sentence2i = {word: i for (i, word) in enumerate(sentence)}
        pos2i = self.pos2i
        K = self.pos_size
        T = len(sentence)
        T1 = np.zeros((K, T))
        T2 = np.zeros((K, T))
        for tag in self.pos_tags:
            T1[pos2i[tag], 0] = self.e[self.pos2i[tag],self.word2i[sentence[0]]] * self.t[tag, "START"]
            T2[pos2i[tag], 0] = self.q[tag]

        for index in range(1, len(sentence)):
            for tag in self.pos_tags:
                t= T1[:, index - 1]
                Akj_Bjy = [ self.t[tag, temp_tag] * self.e[self.pos2i[temp_tag]][self.word2i[sentence[index]]] for
                     temp_tag in self.pos_tags]
                alfa = np.multiply(t,Akj_Bjy)
                max_tag_index = np.argmax(alfa)
                T1[pos2i[tag], index] = alfa[max_tag_index]
                T2[pos2i[tag], index] = max_tag_index
        max_index = np.argmax(T1[:, -1])

        X = []
        # print("adding tag: "+self.pos_tags[max_index])
        from_index = max_index
        for index in range(T-1, -1,-1):
            from_index = T2[int(from_index), index]
            # print("adding tag: "+ self.pos_tags[int(from_index)])
            X.append(self.pos_tags[int(from_index)])

        X.insert(0,"END")
        X.append("START")
        return X[::-1]

    def viterbi(self, sentences):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
        '''
        # TODO: YOUR CODE HERE
        X = []
        for sentance in sentences:
            X.append(self.one_virerbi(sentance))
        return X
    def hmm_mle(self):
        """
        a function for calculating the Maximum Likelihood estimation of the
        transition and emission probabilities for the standard multinomial HMM.

        :param training_set: an iterable sequence of sentences, each containing
                both the words and the PoS tags of the sentence (as in the "data_example" function).
        :param model: an initial HMM with the pos2i and word2i mappings among other things.
        :return: a mapping of the transition and emission probabilities. You may implement
                the probabilities in |PoS|x|PoS| and |PoS|x|Words| sized matrices, or
                any other data structure you prefer.
        """

        b = Baseline(self.pos_tags, self.words, self.training_set)
        self.e = b.e
        self.q = {tag: 0 for tag in self.pos_tags}
        first_tag = [sentence[0][1] for sentence in self.training_set]
        for tag in first_tag:
            self.q[tag] += 1
        for tag in self.pos_tags:
            self.q[tag] /= len(first_tag)

        t = {}
        tag_count = {}
        # initialize all to zero
        for tag1 in self.pos_tags:
            tag_count[tag1] = 0
            for tag2 in self.pos_tags:
                t[tag1, tag2] = 0

        # count i before j
        for sentence in self.training_set:
            sentence_tags = sentence[0]
            for i in range(len(sentence[0])-1):
                t[sentence_tags[i+1], sentence_tags[i]] += 1
                tag_count[sentence_tags[i]] += 1
            # tag_count[sentence_tags[-1]]+=1
        self.t = t
        # divide by some of tag1 count
        for tag1 in self.pos_tags:
            for tag2 in self.pos_tags:
                if tag1 == "END" or tag2 =="START":
                    continue
                t[tag2, tag1] /= tag_count[tag1]

        return self.q, self.e, self.t


class MEMM(object):
    '''
    The base Maximum Entropy Markov Model with log-linear transition functions.
    '''

    def __init__(self, pos_tags, words, training_set, phi):
        '''
        The init function of the MEMM.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        :param phi: the feature mapping function, which accepts two PoS tags
                    and a word, and returns a list of indices that have a "1" in
                    the binary feature vector.
        '''

        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos: i for (i, pos) in enumerate(pos_tags)}
        self.word2i = {word: i for (i, word) in enumerate(words)}
        self.phi = phi


        # TODO: YOUR CODE HERE


    def one_virerbi(self, sentence,w):
        if(sentence[0]=="START"):
            sentence.remove("START")
        if(sentence[-1]=="END"):
            sentence.remove("END")
        # sentence2i = {word: i for (i, word) in enumerate(sentence)}
        # pos2i = self.pos2i
        # K = self.pos_size
        # T = len(sentence)
        # T1 = np.zeros((K, T))
        # T2 = np.zeros((K, T))
        # for tag in self.pos_tags:
        #     T1[pos2i[tag], 0] = self.e[self.pos2i[tag],self.word2i[sentence[0]]] * self.t[tag, "START"]
        #     T2[pos2i[tag], 0] = self.t[tag, "START"]
        #
        # for index in range(1, len(sentence)):
        #     for tag_new in self.pos_tags:
        #         t= T1[:, index - 1]
        #         p = [self.features(tag_new,self.pos_tags[i], sentence, i) for i in range(self.pos_size)]
        #         where= np.where(p==1)
        #         sum =np.sum(w[where])
        #         expo = np.exp(sum)
        #
        #         # alfa/=np.sum(alfa)
        #
        #
        #
        # #         max_tag_index = np.argmax(alfa)
        # #         T1[pos2i[tag], index] = alfa[max_tag_index]
        # #         T2[pos2i[tag], index] = max_tag_index
        # # max_index = np.argmax(T1[:, -1])
        #
        # X = []
        # # print("adding tag: "+self.pos_tags[max_index])
        # from_index = max_index
        # for index in range(T-1, -1,-1):
        #     from_index = T2[int(from_index), index]
        #     print("adding tag: "+ self.pos_tags[int(from_index)])
        #     X.append(self.pos_tags[int(from_index)])
        #
        # X.insert(0,"END")
        # X.append("START")
        # return X[::-1]

    def features(self, tag_new, param, sentence, i):
        pass


    def viterbi(self, sentences, w):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of POS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :param w: a dictionary that maps a feature index to it's weight.
        :return: iterable sequence of POS tag sequences.
        '''




        # TODO: YOUR CODE HERE


def perceptron(self,training_set, initial_model, w0, eta=0.1, epochs=1):
    """
    learn the weight vector of a log-linear model according to the training set.
    :param training_set: iterable sequence of sentences and their parts-of-speech.
    :param initial_model: an initial MEMM object, containing among other things
            the phi feature mapping function.
    :param w0: an initial weights vector.
    :param eta: the learning rate for the perceptron algorithm.
    :param epochs: the amount of times to go over the entire training data (default is 1).
    :return: w, the learned weights vector for the MEMM.
    """
    pass
    # TODO: YOUR CODE HERE


def add_START_to_data(seqenses):
    seqenses = copy.deepcopy(seqenses)
    for seqense in seqenses:
        seqense[0].insert(0, "START")
        seqense[1].insert(0, "START")
    return seqenses

def parce_data(data,words,tags, threshold):
    words.append("RARE")
    words.append("START")
    words.append("END")
    tags.append("START")
    tags.append("END")
    Words_count = {word: 0 for word in words}
    for seqense in data:
        for word in seqense[1]:
            Words_count[word]+=1

    for seqense in data:
        for i in range(len(seqense[0])):
            if Words_count[seqense[1][i]]< threshold:
                seqense[1][i] = "RARE"

    for seqense in data:
        seqense[0].insert(0, "START")
        seqense[1].insert(0, "START")
        seqense[0].append("END")
        seqense[1].append("END")
    return words, tags


if __name__ == '__main__':
    data, words, pos = load_data()
    test, train = data[:20], data[20:]
    words, pos = parce_data(data, words, pos, 10)
    # todo train test split of data
    hmm = HMM(pos, words, train)
    # seq = hmm.sample(1)
    # print(seq)
    # tags = hmm.viterbi(np.array(data)[:len(data)*8//10,1])
    tags = hmm.viterbi([data[0][1]])
    print(data[0][1])
    print(data[0][0])
    print(tags)
