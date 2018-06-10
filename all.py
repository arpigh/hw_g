
import gensim 
import networkx as nx
import matplotlib.pyplot as plt 

m = 'ruscorpora_upos_skipgram_300_5_2018.vec.gz' ##загружаем модель word2vec
if m.endswith('.vec.gz'):
    model = gensim.models.KeyedVectors.load_word2vec_format(m, binary=False)
elif m.endswith('.bin.gz'):
    model = gensim.models.KeyedVectors.load_word2vec_format(m, binary=True)
else:
    model = gensim.models.KeyedVectors.load(m)


model.init_sims(replace=True) #нормализация 

words = ['свет_NOUN', 'вспышка_NOUN', 'молния_NOUN', 'сиять_VERB', 'сверкать_VERB', 'светлый_ADJ', 'ярко_ADV', 'ослеплять_VERB',         
        'ослепительный_ADJ']##семантическое поле 

def init_g(words):  ##создаем и добавляем вершины в граф
    G = nx.Graph() 
    for i in range(len(words)):
        G.add_node(i+1, label=words[i])
    return G

def init_edges(words, G): ##добавление ребер
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            if model.similarity(words[i], words[j]) > 0.5:   ##если кос сходство болше 0.5,       
                G.add_edge(i+1,j+1) ##добавляем ребро между вершинами
                print(model.similarity(words[i], words[j]))

G = init_g(words)
init_edges(words, G)
labels = {} #словарь для подписи графа
for i in range(len(words)):
    labels[i+1]  = words[i]

# Центральность узлов (важность узлов)
print('\nЦентральность узлов')
deg = nx.degree_centrality(G)
max_cent = deg[sorted(deg, key=deg.get, reverse=True)[0]] #находим максимальное значение
for nodeid in sorted(deg, key=deg.get, reverse=True):
    if deg[nodeid] == max_cent: ##выводим самые центральные слова (с максимальным)
        print(words[nodeid-1], deg[nodeid])

#кластреный коэффициент
print('\nКластреный коэффициент')
print(nx.average_clustering(G))
print(nx.transitivity(G))

##построение графа
pos=nx.spring_layout(G)

plt.figure(figsize = (20, 15))
nx.draw_networkx_nodes(G, pos, node_color='blue', node_size=50)
nx.draw_networkx_edges(G, pos, edge_color='red') 
plt.axis('off') 
nx.draw_networkx_labels(G, pos, font_size=10, font_family='Arial', labels = labels)
plt.show() 

