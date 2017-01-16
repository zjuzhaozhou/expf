import sys
import nltk
import cPickle
import operator

import random
from deepwalk import graph
from deepwalk import walks as serialized_walks

import numpy as np
from collections import OrderedDict


unkown_token = "UNKOWN_TOKEN"

class LstmRank(object):
    def __init__(self, name, path_len=5,
                       valid_portion=0.1, train_portion=0.8):
        self.name = name
        self.vocab_size = 8000
        self.n_users = 50000
        self.path_len = path_len
        self.map_users()
        self.get_num_qs()
        print 'Number of users: %d.' % self.num_u
        print 'Number of questions: %d.' % self.num_q

        # partition dataset by questions
        sidx = np.random.permutation(self.num_q)
        n_valid = int(np.round(self.num_q * valid_portion))
        self.valid, data = (sidx[:n_valid], sidx[n_valid:])
        print 'Number of questions in valid set: %d.' % len(self.valid)

        pidx = np.random.permutation(len(data))
        n_train = int(np.round(len(data) * train_portion))
        self.train, self.test = (data[pidx[:n_train]], data[pidx[n_train:]])
        print 'Number of questions in train set: %d.' % len(self.train)
        print 'Number of questions in test set: %d.' % len(self.test)

        self.index_to_train = np.zeros(self.num_q, dtype='int32')
        self.index_to_train[self.train] = 1

        self.index_to_test = np.zeros(self.num_q, dtype='int32')
        self.index_to_test[self.test] = 1
        self.gen_qa_pairs()

    def get_num_qs(self):
        with open('%s_i2q'%self.name) as f:
            self.num_q = len(f.readlines())
    
    def gen_qa_pairs(self):
        best_qa = {}
        all_qa =  {}
        with open('%s_q2u2s'%self.name) as f:
            for line in f:
                q, u, s = line.strip('\n\r').split()
                q, u, s = int(q), int(u), int(s)
                u_l = best_qa.get(q, [])
                if not u_l or u_l[0][1] < s:
                    best_qa.update({q: [(u, s)]})
                elif u_l[0][1] == s:
                    u_l.append((u, s))
                    best_qa.update({q: u_l})
                all_u = all_qa.get(q, {})
                all_u.update({u:s})
                all_qa.update({q: all_u})

        self.all_qa = all_qa
        self.best_qa = best_qa
        # return (all_qa, best_qa)

    def map_users(self):
        with open('%s_i2t'%self.name) as f:
            self.u2t = [ l.split()[1] for l in f ]
        self.t2u = dict([(t, u) for u, t in enumerate(self.u2t)])
        self.num_u = len(self.u2t)
    
    def _gen_paths(self, num_walks, walk_length):
        print "Number of nodes(train graph): %d."%(len(self.G.nodes()))
        walks = graph.build_deepwalk_corpus(self.G, num_walks, walk_length,
                                            alpha=0, rand=random.Random(0))
        fo = open('%s_paths.%d'%(self.name, self.path_len), 'w+')
        for path in walks:
            labels = ''
            for node in path:
                if node < self.num_u:
                    labels += 'u%d '%node
                else:
                    labels += 'q%d '%(node-self.num_u)
            fo.write('%s\n'%labels)
        fo.close()
        return walks
    
    def create_graph(self):
        twi_links = []
        with open('%s_twitter_links' % self.name) as f:
            for line in f:
                u, vl = line.split(':')
                if u in self.t2u:
                    twi_links.append([(u, v) for v in vl.split()
                                             if v in self.t2u])
        
        fo = open('%s_graph.%d'%(self.name, self.path_len), 'w+')
        for link_l in twi_links:
            for link in link_l:
                u, v = link
                fo.write('%s %s\n'%(self.t2u[u], self.t2u[v]))
        for q, u_dict in self.all_qa.items():
            if not self.index_to_train[q]:
                continue
            u_list = u_dict.keys()
            for u in u_list:
                fo.write('%s %s\n'%(self.num_u + int(q), u))
                fo.write('%s %s\n'%(u, self.num_u + int(q)))
        fo.close()
        # Dreicted graph
        self.G = graph.load_edgelist('%s_graph.%d'%(self.name, self.path_len),
                                     undirected=False)

    def prepare_triplet(self, num_walks, walk_length):
        self.create_graph()
        paths = self._gen_paths(num_walks, walk_length)
        triplets = []
        for path in paths:
            q_list = []
            u_list = []
            for i, node in enumerate(path):
                if int(node) < self.num_u:
                    u_list.append((node, i))
                else:
                    q_list.append((node-self.num_u, i))
            path_map = np.zeros([len(q_list), len(u_list)])
            for i, q in enumerate(q_list):
                dists = [abs(q[1]-u[1]) for u in u_list]
                rank_by_dist = sorted(range(len(u_list)), key=lambda x:dists[x])
                for j, k in zip(rank_by_dist[:-1], rank_by_dist[1:]):
                    rj = rank_by_dist[j]
                    rk = rank_by_dist[k]
                    if rj == rk:
                        sj = self.all_qa[q[0]][u_list[j][0]] 
                        sk = self.all_qa[q[0]][u_list[k][0]] 
                        if sj > sk:
                            triplets.append((q[0], u_list[j][0], u_list[k][0]))
                        else:
                            triplets.append((q[0], u_list[k][0], u_list[j][0]))
                    else:
                        triplets.append((q[0], u_list[j][0], u_list[k][0]))
        return triplets

        # old fashion
        # import operator
        # self.data_y = [[]] * self.num_q
        # for q, u_list in all_qa.items():
        #     ranks = sorted(u_list, key=operator.itemgetter(1))
        #     ranks = [x[0] for x in ranks]
        #     self.data_y[q] = zip(ranks[1:], ranks[:-1])

    
    def prepare_q(self):
        import itertools
        with open('%s_i2q'%self.name) as f:
            sentences = [ x.decode('utf-8').split('|')[1] for x in f ]
        tked_sentences = [ nltk.word_tokenize(sent) for sent in sentences ]
        word_freq = nltk.FreqDist(itertools.chain(*tked_sentences))
        self.vocab = word_freq.most_common(self.vocab_size - 1)
        index_to_word = [ x[0] for x in self.vocab ]
        index_to_word.append(unkown_token)
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

        print "Using vocabulary size %d." % self.vocab_size
        print ("The least frequent word in our vocabulary is '%s' "
               "and appeared %d times.") % (self.vocab[-1][0], self.vocab[-1][1])

        maxlen = 0
        for i, sent in enumerate(tked_sentences):
            tked_sentences[i] = [ w if w in word_to_index else unkown_token for w in sent ]
            if len(tked_sentences[i]) > maxlen:
                maxlen = len(tked_sentences[i])
        print 'Maximum length of questions: %d.' % maxlen

        self.data_x = np.asarray([[word_to_index[w] for w in sent] for sent in tked_sentences])
    
    def unpack_triplet(self, triplets):
        train_q = []
        train_u_good = []
        train_u_bad = []
        for idx, triplet in enumerate(triplets):
            q, ui, uj = triplet
            train_q.append(self.data_x[q])
            train_u_good.append(ui)
            train_u_bad.append(uj)
            self.index_to_train[q] = idx+1
        print 'Number of triplets: %d.' % len(train_q)
        return (train_q, train_u_good, train_u_bad)

    def build_validset(self, indexs):
        valid_x = []
        valid_u = []
        valid_y = []
        for q in indexs:
            valid_x.append(self.data_x[q])
            valid_u.append(self.all_qa[q].keys())
            index_to_user = dict([(u, i) for i, u in enumerate(self.all_qa[q].keys())])
            valid_y.append([index_to_user[x[0]] for x in self.best_qa[q]])
        print 'build set size:', len(valid_x)
        return (valid_x, valid_u, valid_y)
    
    def run(self):
        self.prepare_q()
        triplets = self.prepare_triplet(3, self.path_len)
        main_train = self.unpack_triplet(triplets)
        train = self.build_validset(self.train)
        valid = self.build_validset(self.valid)
        test = self.build_validset(self.test)
        import sys
        print sys.getsizeof(main_train)
        print sys.getsizeof(train)
        print sys.getsizeof(valid)
        print sys.getsizeof(test)
        cPickle.dump((main_train, train, valid, test), open('%s%d_expf.pkl'%(self.name, self.path_len), 'w+'))


if __name__ == '__main__':
    fname = sys.argv[1]
    path_len = int(sys.argv[2])
    lm = LstmRank(fname, path_len)
    lm.run()
