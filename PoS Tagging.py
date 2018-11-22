import copy
import pickle
from functools import reduce
from scipy.misc import logsumexp
import numpy as np
from scipy.sparse import csr_matrix

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
        self.e = np.array([])
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
                old_tag = np.random.choice(self.pos_tags, p=trancemition_probabileteys)
                emition_probabileteys = self.e[self.pos2i[old_tag],:]
                word = np.random.choice(self.words, p=emition_probabileteys)
                while word == "RARE":
                    word = np.random.choice(self.words, p=emition_probabileteys)
                sentance.append(word)
            sentances.append(sentance)
        return sentances

    # TODO: YOUR CODE HERE

    def one_virerbi(self, sentence):

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
            if (sentance[0] == "START"):
                sentance.remove("START")
            if (sentance[-1] == "END"):
                sentance.remove("END")
            s =self.one_virerbi(sentance)
            s.insert(0, "END")
            s.append("START")
            X.append(s)
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

    def __init__(self, pos_tags, words, training_set):
        '''
        The init function of the MEMM.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        :param phi: the feature mapping function, which accepts two PoS tags
                    and a word, and returns a list of indices that have a "1" in
                    the binary feature vector.
        '''

        self.feachers = [self.e_featcher, self.t_featcher]
        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos: i for (i, pos) in enumerate(pos_tags)}
        self.word2i = {word: i for (i, word) in enumerate(words)}
        self.hmm = HMM(pos_tags, words, training_set)
        # self.phi_vec = self.map_phi()
        self.flat_e = self.hmm.e.flatten()


        # TODO: YOUR CODE HERE


    def one_virerbi(self, sentence,w):

        sentence2i = {word: i for (i, word) in enumerate(sentence)}
        pos2i = self.pos2i
        K = self.pos_size
        T = len(sentence)
        T1 = np.zeros((K, T))
        T2 = np.zeros((K, T))
        for tag in self.pos_tags:
            T1[pos2i[tag], 0] = self.e[self.pos2i[tag], self.word2i[sentence[0]]] * self.t[tag, "START"]
            T2[pos2i[tag], 0] = 0

        for index in range(1, len(sentence)):
            for tag in self.pos_tags:
                t = T1[:, index - 1]
                f= self.features_w_exp(sentence, tag, sentence[index])
                t_features_w_exp = np.multiply(t,f)
                max_tag_index = np.argmax(t_features_w_exp)
                value = np.divide(t_features_w_exp[max_tag_index], np.sum(t_features_w_exp))
                T1[pos2i[tag], index] = value
                T2[pos2i[tag], index] = max_tag_index
        max_index = np.argmax(T1[:, -1])

        X = []
        # print("adding tag: "+self.pos_tags[max_index])
        from_index = max_index
        for index in range(T - 1, -1, -1):
            from_index = T2[int(from_index), index]
            # print("adding tag: "+ self.pos_tags[int(from_index)])
            X.append(self.pos_tags[int(from_index)])
        return X[::-1]


    def map_phi(self):
        e= self.hmm.e
        t= self.hmm.t
        vec = []
        for tag  in self.pos_tags:
            vec.extend([t[old_t, tag] for old_t in self.pos_tags])
        vec.extend(e.flatten())

        vec = np.asarray(vec)
        thrushold_indexes = vec>0
        vec[thrushold_indexes] = 1
        return vec

    def t_featcher(self, las_tag, tag, word):
        if self.hmm.t[tag, las_tag]> 0 :
            return self.pos2i[tag]*self.pos_size+self.pos2i[las_tag], len(self.hmm.t)
        return None,len(self.hmm.t)

    def e_featcher(self,las_tag, tag, word):
        if self.hmm.e[tag, word]> 0 :
            return self.pos2i[pos]*self.words_size + self.word2i[word], len(self.flat_e)
        return None, len(self.flat_e)


    def phi_endeces(self, las_tag, tag, word):
        """:returns list of indexes that are ones in the phi vector"""
        start_index = 0
        indeses = []
        for feacher in self.feachers:
            indese, size = feacher(las_tag, tag, word)
            indese+=start_index
            start_index += size
            indeses.append(indese)
        return indeses

    def phi_vec(self, las_tag, tag, word):
        start_index = 0
        vec = []
        for feacher in self.feachers:
            indese, size = feacher(las_tag, tag, word)
            vec.extend([0]*size)
            indese+=start_index
            start_index += size
            vec[indese] = 1
        return np.asarray(vec)

    def features_w_exp(self, tag, word, w):
        phi = np.apply_along_axis(self.phi_endeces, arr= np.asarray(self.pos_tags), tag = tag, word =  word, axis=0)
        # x= np.apply_along_axis(lambda i: np.exp(csr_matrix(i) @ w),  arr = phi,axis=0)
        x= [np.exp(np.sum(w[i])) for i in phi]
        return x


    def viterbi(self, sentences, w):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of POS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :param w: a dictionary that maps a feature index to it's weight.
        :return: iterable sequence of POS tag sequences.
        '''
        X = []
        for sentance in sentences:
            if (sentance[0] == "START"):
                sentance.remove("START")
            if (sentance[-1] == "END"):
                sentance.remove("END")
            s =self.one_virerbi(sentance)
            s.insert(0, "END")
            s.append("START")
            X.append(s)
        return X



        # TODO: YOUR CODE HERE


    def perceptron(self,training_set,  w0, eta=0.1, epochs=1):
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
        w = [w0]
        for sequence in training_set:
            words = np.asarray(sequence[1])
            tags = sequence[0]
            y = self.one_virerbi(words,w[-1])
            first = sum([self.phi_vec(tags[i - 1], tags[i], words[i]) for i in range(1, len(tags) - 1)])
            second = sum([self.phi_vec(y[i - 1], y[i], words[i]) for i in range(1, len(tags) - 1)])
            w_i= w[-1] + eta*(first+second)
            w.append(w_i)
        return np.sum(np.asarray(w), axis=1)



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
    hmm = MEMM(pos, words, train)
    # seq = hmm.sample(1)
    # print(seq)
    # tags = hmm.viterbi(np.array(data)[:len(data)*8//10,1])
    tags = hmm.viterbi([data[0][1]], np.random.rand(len(pos)**2+len(pos)*len(words)))
    print(data[0][1])
    print(data[0][0])
    print(tags)
