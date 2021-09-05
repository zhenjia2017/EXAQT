import networkx as nx
from heapq import heappush,heappop,heapify
import math
import numpy as np

verbose=0
MAX_MATCH=1
Distribute_Node_wt_flag=0

stop_list=set(['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can\'t', 'cannot', 'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most', 'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours	ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves'])

def get_cost(t,G):
    c=0.0
    for (n1,n2) in t.edges():
        if (n1,n2) in G.edges():
            data=G.get_edge_data(n1,n2)
        else:
            data=G.get_edge_data(n2,n1)
        if 'wlist' not in data:
            print("edge of n1, n2 has no wlist")
            print (n1,n2)
            break
        for d in data['wlist']:
            c+=1-d

    if c<0:
        print("\n\n ==== =========== ERROR NEG COST ===== \n\n",c)

    if Distribute_Node_wt_flag==1:
        for n in t.nodes():
            node_weight=G.node[n]['weight']
            c+=1-node_weight
        if c<0:
            print("\n\n ==== =========== ERROR NEG COST Node===== \n\n",c)

    return c

def grow_graph(g1,v,u):
    g=nx.Graph()
    for n in g1.nodes():
        g.add_node(n)
    for (n1,n2) in g1.edges():
        g.add_edge(n1,n2)
    if u not in g.nodes():
        g.add_node(u)
        g.add_edge(v,u) #to keep it a tree #access weight from G if needed
    return g

def merge_graph(g1,g2):
    g=nx.Graph()
    for n in g1.nodes():
        g.add_node(n)
    for (n1,n2) in g1.edges():
        g.add_edge(n1,n2)
    for n in g2.nodes():
        if n not in g.nodes():
            g.add_node(n)
    for (n1,n2) in g2.edges():
        if (n1,n2) not in g.edges() and (n2,n1) not in g.edges():
            g.add_edge(n1,n2)
    return g

def update_queue(Q,cg,u,p):
    for i in range(0,len(Q)):
        if Q[i][1]==u and Q[i][2]==p:
            Q[i]=(cg,u,p)
            heapify(Q)
            break
    return	Q

def check_history(pop_hist,v,p):
    if v in pop_hist:
        if p in pop_hist[v]:
            return 0
    return 1

def save_GST(GST_set,v,g1,corner,G1):
    g=nx.Graph()
    for n in g1.nodes():
        g.add_node(n)
        nn = n.split('::')
        if nn[1] == 'Predicate' and n in corner:  # Add neighbour entities from SPO which are not corner stones
            for nb in G1.neighbors(n):
                if nb.split('::')[1] == 'Entity' and nb not in g1.neighbors(n) and nb not in corner:
                    g.add_node(nb)
                    g.add_edge(n, nb)

    for (n1,n2) in g1.edges():
        g.add_edge(n1,n2)
    pot_ans_flag=0
    for n in g.nodes():
        if n.split('::')[1] == 'Entity':
            pot_ans_flag=1
    flag=1
    for (v2,g2) in GST_set:
        if set(g.edges())==set(g2.edges()):
            flag=0
            break

    if flag==1 and pot_ans_flag==1:
        GST_set.append((v,g))
    return GST_set,flag,pot_ans_flag

def get_GST(Q,T,P,G,G1,no_GST,corner):
    pop_hist={}
    ite=0
    pop_cov=-1
    merge_cov=-1
    min_cost=999999
    GST_count=0
    GST_set=[]
    final_GST_cost=-99999
    final_GST_flag=0
    leave_loop=0

    while len(Q)>0:
        x=heappop(Q)
        cc=x[0]
        v=x[1]
        p=x[2]
        ite+=1
        if len(p)>pop_cov:
            pop_cov=len(p)

        check=check_history(pop_hist,v,p)
        if check==0:
            continue #To avoid executing same pop
        if v not in pop_hist:
            pop_hist[v]=set()
        pop_hist[v].add(p)

        if p==P:
            current_cost=get_cost(T[v][p],G)
            if final_GST_cost>-99999 and current_cost>final_GST_cost: #No more GST with same cost as final GST
                leave_loop=1
                break

            GST_set,New_GST_flag,pot_ans_flag=save_GST(GST_set,v,T[v][p],corner,G1)
            if New_GST_flag==1 and pot_ans_flag==1:
                GST_count+=1
            if GST_count==no_GST and final_GST_flag==0:
                final_GST_cost=get_cost(T[v][p],G)
                final_GST_flag=1

        if leave_loop==1:
            break
        #GROW
        for u in G.neighbors(v):

            g=grow_graph(T[v][p],v,u) #Crete temporary merge tree/graph

            cg=get_cost(g,G)
            flag=0
            if u in T and p in T[u]:
                cu=get_cost(T[u][p],G)
                if cg<cu: #If T[u][p] already exists, check if cost(merged_graph)<T[u][p]; in that case update T[u][p]
                    T[u][p]=g
                    flag=1
                    Q=update_queue(Q,cg,u,p)

            else: #If T[u][p] does not exist, create T[u][p]=merged graph
                if u not in T:
                    T[u]={}
                T[u][p]=g
                flag=1
                heappush(Q,(cg,u,p))
        #MERGE
        p1=p

        all_p2=set()
        for p2 in T[v]:
            all_p2.add(p2)

        for p2 in all_p2: #because T[v] changes during iteration
            if len(p1.intersection(p2))==0:
                g=merge_graph(T[v][p1],T[v][p2])
                cg=get_cost(g,G)
                p=frozenset(p1.union(p2))
                if p in T[v]:#all_p2:
                    cp=get_cost(T[v][p],G)
                    if cg<cp:
                        T[v][p]=g
                        Q=update_queue(Q,cg,v,p)
                else:
                    T[v][p]=g
                    heappush(Q,(cg,v,p))
                    if len(p)>merge_cov:
                        merge_cov=len(p)
                        if len(p)==len(P):
                            if cg<min_cost:
                                min_cost=cg

    return 	GST_set

def initialize_queue(G,corner):
    Q=[]
    T={}
    for v in corner:
        T[v]={}
        g = nx.Graph()
        g.add_node(v) #access weight from G if needed
        p=frozenset([corner[v]]) #Query term
        T[v][p]=g
        c=get_cost(g,G)
        heappush(Q,(c,v,p))

    for v in T:
        for p in T[v]:
            if verbose:
                print("Query and tree --->",p,T[v][p].nodes(),T[v][p].edges())
    return T,Q

def get_cornerstone_distance(T, G, corner, n0):
    w=0.0
    for n in T.nodes():
        if n in corner:
            w+=len(nx.shortest_path(T,n0,n))-1
    return w    
  
def get_cornerstone_distance_wt(T, G, corner, n0):
    w=0.0
    for n in T.nodes():
        if n in corner:
            path=nx.shortest_path(T,n0,n)
            c=0.0
            for i in range(0,len(path)-1):
                n1=path[i]
                n2=path[i+1]
                if (n1,n2) in G.edges():
                    data=G.get_edge_data(n1,n2)
                else:
                    data=G.get_edge_data(n2,n1)
                for d in data['wlist']:	#Use if sum of cost is needed
                    c+=1-d #cost=1-weight
            w+=c
    return w      
        
def get_cornerstone_weight(T, G, corner):
    w=0.0
    for n in T.nodes():
        if n in corner:
            w+=G.node[n]['weight']
    return w

def issubseq(n1,n2):
    nw1=(n1.split('::'))[0].split()
    nw2=(n2.split('::'))[0].split()
    if len(nw1)==0:
        return 0
    i=0
    flag=0
    for j in range(0,len(nw2)):
        if nw1[i].lower()==nw2[j].lower():
            i+=1
            if i==len(nw1):
                flag=1
                break

    if flag==1:
        return 1
    else:
        return 0

def cosine_similarity(a,b):
	s1=0.0
	s2=0.0
	s3=0.0
	if len(a)!=len(b):
		return 0.0
	for i in range(0,len(a)):
		s1+=a[i]*b[i]
		s2+=a[i]*a[i]
		s3+=b[i]*b[i]
	if s2>0 and s3>0:
		val=(s1/(math.sqrt(s2)*math.sqrt(s3)))
		val_norm=(val+1.0)/2.0
		return val_norm
	else:
		return 0

def cosine_similarity_MAX_MATCH(a, b, gdict):
    a = a.lower()
    aw1 = a.replace('-', ' ').split()
    b = b.lower()
    bw1 = b.replace('-', ' ').split()

    max_match = -1
    for el1 in aw1:
        if el1 in gdict and el1 not in stop_list:
            avec = gdict[el1]
            for el2 in bw1:
                if el2 in gdict and el2 not in stop_list:
                    bvec = gdict[el2]
                    val = cosine_similarity(avec, bvec)
                    if val > max_match:
                        max_match = val
    return max_match

def get_type_simi(n0, G, ans_type, gdict):
    veclen = 300
    term_types = set()
    for (n1, n2) in G.edges():
        if n0 == n2 and n1.split(':')[1] == 'Type':
            term_types.add(n1.split(':')[0])
        else:
            if n0 == n1 and n2.split(':')[1] == 'Type':
                term_types.add(n2.split(':')[0])

    if len(term_types) == 0 or len(
            ans_type) == 0:  # if answer does not have type or node n0 does not have type, keep n0
        return 1.0

    if MAX_MATCH == 1:
        maxval = 0.0

        for n1 in term_types:
            for n2 in ans_type:
                val = cosine_similarity_MAX_MATCH(n1, n2, gdict)
                if val > maxval:
                    maxval = val

    else:
        t_dict = {}
        a_dict = {}

        for n in term_types:
            nw1 = n.split()
            avec = np.zeros(veclen)
            c = 0.0
            for el in nw1:
                if el in gdict and el.lower() not in stop_list:
                    avec = np.add(avec, np.array(gdict[el]))
                    c += 1.0
            if c > 0:
                avec = np.divide(avec, c)

            t_dict[n] = avec.tolist()

        for n in ans_type:
            nw1 = n.split()
            avec = np.zeros(veclen)
            c = 0.0
            for el in nw1:
                if el in gdict and el.lower() not in stop_list:
                    avec = np.add(avec, np.array(gdict[el]))
                    c += 1.0
            if c > 0:
                avec = np.divide(avec, c)

            a_dict[n] = avec.tolist()

        maxval = 0.0

        for n1 in t_dict:
            for n2 in a_dict:
                val = cosine_similarity(t_dict[n1], a_dict[n2])
                if val > maxval:
                    maxval = val
    return maxval

def call_main_rawGST(QKG, corner1, no_GST):
    corner_nodes = set()
    print("\n\nSize of the cornerstone ", len(corner1))
    G = QKG.copy()
    G1 = QKG.copy()

    corner = {}
    for e in corner1:
        if e in G.nodes():
            corner[e] = corner1[e]
            corner_nodes.add(e)

    print("\n\nSize of the graph_prune ", len(G.nodes()), len(G.edges()))

    print("Running GST Algorithm...")

    T, Q=initialize_queue(G,corner)
    P=set() #Entire query
    for v in corner:
        P.add(corner[v])

    count={}
    for v in corner:
        if corner[v] not in count:
            count[corner[v]]=set()
        count[corner[v]].add(v)

    GST_set = get_GST(Q,T,P,G,G1,no_GST,corner)

    rawGST_list = []
    for ii in range(0,len(GST_set)):
        (v,T)=GST_set[ii]
        rawGST_list.append(T)
    print ('\nlength of GST: ', len(rawGST_list))
    if len(rawGST_list) > 0:
        unionGST = nx.compose_all(rawGST_list)

    return unionGST





