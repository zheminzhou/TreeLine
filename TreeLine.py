# Map Events Over History (MEOH)
import numpy as np, csv, sys, os, json
try :
    import cPickle as pickle
except :
    import pickle

from dendropy import Tree, Node, Taxon
from tempfile import mkdtemp
from copy import deepcopy
#from scipy.optimize import minimize

def isnum(v) :
    try :
        float(v)
        return True
    except :
        return False

class MetaTree :
    def __init__(self, tre) :
        self.tre = tre
        self.traits = {}

class TreeLine :
    data = dict(
        n_node = 0,
        taxa=None,
        trees=None,
        superTree=None,
    )
    
    param = dict(
        workspace = '.',
        proj_barcode = '',
        proj_name = '',
        
        file_trees = None,
        file_traits = None,
        file_treeline = None,

        trees_burnIn=0,
        trees_sampleFreq=1,
        trees_maxNum=10,
        superTree_method='consensus', # sumtrees file ASTRID
        
        traits_ignoreMissing=True,
        traits_ASR=None,
        traits_rescale=1.,
        traits_EMNum=10,
        traits_inference = 'marginal',

        treeline_subTree=None,
        treeline_dataX = 's:branch.length',
        treeline_dataY = 'd:pangene/s:branch.length',
        treeline_bootstrap = 1000,
        treeline_histBin = 50,
        treeline_direct = 'tip-to-root',
    )

    def __init__ (self, **kwargs) :
        if 'workspace' in kwargs :
            self.param['workspace'] = kwargs['workspace']
        if 'proj_barcode' not in kwargs :
            self.proj_folder = mkdtemp(prefix='prj_', dir=self.param['workspace'])
            self.param['proj_barcode'] = self.proj_folder.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
        else :
            self.param['proj_barcode'] = kwargs['proj_barcode']
            self.proj_folder = os.path.join( self.param['workspace'], self.param['proj_barcode'] )
            
        dmp_file = os.path.join(self.proj_folder, 'paramFile')
        if os.path.isfile(dmp_file) :
            self.param.update( json.load(open(dmp_file,'rb')) )
        data_file = os.path.join(self.proj_folder, 'dataFile')
        if os.path.isfile(data_file) :
            self.data = pickle.load(open(data_file, 'rb'))

        for k, v in kwargs.iteritems() :
            if k in self.param :
                if isinstance(v, dict) :
                    self.param[k].update(v)
                else :
                    self.param[k] = v

        json.dump(self.param, open(dmp_file, 'wb'), sort_keys=True, indent=2)
        if 'file_trees' in kwargs :
            self.data['taxa'], self.data['trees'], self.data['n_node'] = self.read_trees(**self.param)
        if 'superTree_method' in kwargs :
            self.data['superTree'] = self.get_super_tree(**self.param)
        if 'file_traits' in kwargs :
            self.read_metadata(**self.param)
        if 'traits_ASR' in kwargs :
            print self.infer_ancestral(**self.param)
        if 'treeline_dataY' in kwargs :
            self.infer_treeline(**self.param)

        data_file = os.path.join(self.proj_folder, 'dataFile')
        pickle.dump(self.data, open(data_file, 'wb'))
        print self.param['proj_barcode']
        return None
    
    
    def read_trees(self, file_trees, trees_burnin=0, trees_sampleFreq=1, trees_maxNum=10, traits_ignoreMissing=True, **args) :
        # read trees (including traits when possible)
        data_trees = []
        schema = 'newick'
        with open(file_trees) as fin :
            if fin.readline().upper().startswith('#NEXUS') :
                schema = 'nexus'
        for id, tre in enumerate(Tree.yield_from_files([file_trees], schema=schema)) :
            if id >= trees_burnin :
                if not tre.label :
                    tre.label = str(id)
                if (id - trees_burnin) % trees_sampleFreq == 0 :
                    data_trees.append(tre)
                if len(data_trees) >= trees_maxNum : break
        
        taxa = {}
        for tre in data_trees :
            for taxon in tre.taxon_namespace :
                taxa[taxon.label] = 1
        for id, taxon in enumerate(sorted(taxa.keys())) :
            taxa[taxon] = id
        
        n_taxa, n_tree, n_node = len(taxa), len(data_trees), 0
        digit_code = np.power(2, np.arange(n_taxa, dtype='object'))
        trait_categories = {'branch.length':[1, 'continuous', None], 'branch.age':[1, 'continuous', None]}
        for tre in data_trees :
            internal_id = n_taxa
            for node in tre.postorder_node_iter() :
                for annotation in node.annotations :
                    n, v = annotation.name, annotation.value
                    if annotation.name not in trait_categories :
                        if isinstance(v, list) :
                            trait_categories[n] = [len(v), 'continuous', None] if isnum(v[0]) else [len(v), 'discrete', {}]
                        else :
                            trait_categories[n] = [1, 'continuous', None] if isnum(v) else [1, 'discrete', {}]
                    if trait_categories[n][1] == 'discrete' :
                        if isinstance(v, list) :
                            for vv in v :
                                trait_categories[n][2][vv] = 1
                        else :
                            trait_categories[n][2][v] = 1
                if node.is_leaf() :
                    node.id = taxa[node.taxon.label]
                    node.barcode = digit_code[node.id]
                else :
                    node.id = internal_id
                    internal_id += 1
                    node.barcode = 0L
                    for c in node.child_nodes() :
                        node.barcode += c.barcode
            if internal_id > n_node : n_node = internal_id
            tre.seed_node.age = tre.seed_node.distance_from_tip()
            for node in tre.preorder_node_iter() :
                if node.parent_node :
                    node.age = node.parent_node.age - node.edge_length
                

        for cc, tc in trait_categories.iteritems() :
            if tc[1] == 'discrete' :
                if traits_ignoreMissing :
                    tc[2].update({"-":-1, "":-1, "0":-1})
                tc[2].update(dict([[k, id] for id, k in enumerate(sorted([k for k, v in tc[2].iteritems() if v > 0]))]))
        # read all traits' values
        trees = []
        for tre in data_trees :
            for node in tre.nodes() :
                if node.edge_length < 1e-8 and node.parent_node is not None :
                    if node.is_leaf() :
                        node.edge_length = 1e-8
                    else :
                        parent = node.parent_node
                        for child in node.child_nodes() :
                            child._set_parent_node(parent)
                        parent.remove_child(node)
                        parent.set_child_nodes(parent.child_nodes() + node.child_nodes())
            trees.append(MetaTree(tre))
            mt = trees[-1]
            mt.traits['branch'] = [np.empty([2, n_node, 1]), [['length', 0], ['age', 1]], None]
            for node in tre.postorder_node_iter() :
                mt.traits['branch'][0][:, node.id, 0] = [node.edge_length, node.age]
                for annotation in node.annotations :
                    k, v = annotation.name, annotation.value
                    
                    tck = trait_categories[k]
                    if tck[1] == 'continuous' :
                        if k not in mt.traits :
                            mt.traits[k] = [np.empty([1, n_node, tck[0]]), [[k +':0', 0]], None]
                            mt.traits[k][0].fill(np.nan)
                        mt.traits[k][0][0, node.id, :] = v
                    else :
                        if k not in mt.traits :
                            mt.traits[k] = [np.zeros([tck[0], n_node], dtype=int), [['{0}:{1}'.format(k, id), id] for id in np.arange(tck[0])], tck[2] ]
                            mt.traits[k][0].fill(-1)
                        
                        mt.traits[k][0][:, node.id] = np.vectorize(tck[2].get)(v)
                node.annotations.clear()
                node.annotations.add_new('id', node.id)
            for k, v in mt.traits.iteritems() :
                if v[2] is not None :
                    ids = np.lexsort(v[0].T)
                    d = v[0][ids]
                    uniq_ids = np.concatenate([[1], np.sum(d[:-1] != d[1:], 1) ])
                    d = d[uniq_ids > 0]

                    data = np.ones([d.shape[0], d.shape[1], np.max(v[2].values())+1], dtype=float)
                    data.fill(np.nan)
                    axis = np.where(d>=0)
                    data[axis[0], axis[1], :] = 0
                    data[axis[0], axis[1], d[d>=0]] = 1
                    v[0] = data
                    dd = []
                    mat_id = -1
                    for i in uniq_ids :
                        if i > 0 :
                            mat_id += 1
                        dd.append(mat_id)
                    v[1] = [[name, i] for (name, oi), i in zip(v[1], np.array(dd)[ids])]
        return taxa, trees, n_node


    def get_super_tree(self, superTree_method, **args) :
        def consensus(self, **args) :
            n_tree, n_branch = 0, {}
            for mt_id, mt in enumerate(self.data['trees']) :
                if len(mt.tre.leaf_nodes()) == len(self.data['taxa']) :
                    n_tree += 1
                    for node in mt.tre.preorder_node_iter() :
                        if node.barcode not in n_branch :
                            n_branch[node.barcode] = [[1., mt_id, node]]
                        else :
                            n_branch[node.barcode].append([1., mt_id, node])
            n_branch = sorted([[len(v)/float(n_tree), k, v] for k, v in n_branch.iteritems()], reverse=True)
            consensus_tree = []
            for posterior, branch, nodes in n_branch :
                for cbr, _, _ in consensus_tree :
                    b1, b2 = (branch, cbr) if branch < cbr else (cbr, branch)
                    if not (( (b1 & b2) == b1 ) or ( (b1 & (~b2)) == b1 )) :
                        branch = 0
                        break
                if branch :
                    consensus_tree.append([branch, posterior, nodes])
            return sorted(consensus_tree, reverse=True)
        def MCC(self, **args) :
            n_tree, n_branch = 0, {}
            for mt_id, mt in enumerate(self.data['trees']) :
                if len(mt.tre.leaf_nodes()) == len(self.data['taxa']) :
                    n_tree += 1
                    for node in mt.tre.preorder_node_iter() :
                        if node.barcode not in n_branch :
                            n_branch[node.barcode] = [[1., mt_id, node]]
                        else :
                            n_branch[node.barcode].append([1., mt_id, node])
            for mt_id, mt in enumerate(self.data['trees']) :
                if len(mt.tre.leaf_nodes()) == len(self.data['taxa']) :
                    mt.score = np.sum([len(n_branch[node.barcode]) for node in mt.tre.preorder_node_iter() ])
            tre = max(self.data['trees'], key=lambda x:x.score).tre
            return [[n.barcode, len(n_branch[n.barcode])/float(n_tree), n_branch[n.barcode]] for n in tre.preorder_node_iter()]
        def ASTRID(self, **args) :
            pass
        branches = locals()[superTree_method](self, **args)
        supertree = Tree()
        sn = supertree.seed_node
        sn.barcode, sn.posterior = branches[0][0], branches[0][1]
        sn.age = np.sum([n[2].age*n[0] for n in branches[0][2]])/np.sum([n[0] for n in branches[0][2]])
        for cbr, posterior, nodes in branches[1:] :
            while (sn.barcode & cbr) != cbr :
                sn = sn.parent_node
            new_node = Node() if len(nodes) == 0 or (not nodes[0][2].taxon) else Node(taxon=Taxon(label=nodes[0][2].taxon.label))
            sn.add_child(new_node)
            sn = new_node
            sn.barcode, sn.posterior = cbr, posterior
            sn.edge_length = 0.0 if len(nodes) == 0 else np.sum([n[2].edge_length*n[0] for n in nodes])/np.sum([n[0] for n in nodes])
            sn.age = sn.parent_node.age if len(nodes) == 0 else np.sum([n[2].age*n[0] for n in nodes])/np.sum([n[0] for n in nodes])
        internal_id = len(self.data['taxa'])
        for node in supertree.postorder_node_iter() :
            if node.is_leaf() :
                node.id = self.data['taxa'][node.taxon.label]
            else :
                node.id = internal_id
                internal_id += 1
        return supertree
    
    def read_metadata(self, file_traits, traits_ignoreMissing=True, **args) :
        strains, matrices = [], {}
        with open(file_traits, 'r') as fin :
            header = fin.readline().strip().split('\t')
            ids = []
            headers, cat_col = np.array(['#tag', '#category', '#tree']), np.array([-1, -1, -1])
            for id, head in enumerate(header) :
                if head.startswith('#') :
                    cat_col[ headers == head.lower() ] = id
                else :
                    try :
                        strains.append(self.data['taxa'][head])
                        ids.append(id)
                    except :
                        pass
                    
            ids = np.array(ids)
            for line in csv.reader(fin, delimiter='\t') :
                line = np.array(line)
                if not line.size : continue
                h = line[cat_col]
                h[cat_col<0] = 'default'
                tag, category, tree = h
                if category not in matrices :
                    matrices[category] = [[line[ids]], [tag], {}, [tree]]
                else :
                    matrices[category][0].append(line[ids])
                    matrices[category][1].append(tag)
                    matrices[category][3].append(tree)
        
        for category, (mat, tags, types, trees) in matrices.items() :
            types = np.unique(mat)
            if traits_ignoreMissing :
                types = types[(types != '') & (types != '-') & (types != '0')]
            types = {t:id for id, t in enumerate(sorted(types.tolist()))}
            if traits_ignoreMissing :
                types.update({'0':-1, '-':-1, '':-1})
            mat = np.vectorize(types.get)(mat)
            m = np.zeros(shape=[mat.shape[0], self.data['n_node']], dtype=int)
            m.fill(-1)
            m[:, strains] = mat

            ids = np.lexsort(m.T)
            d = m[ids]
            uniq_ids = np.concatenate([[1], np.sum(d[:-1] != d[1:], 1) ])
            d = d[uniq_ids > 0]
        
            data = np.ones([d.shape[0], d.shape[1], np.max(types.values())+1], dtype=float)
            data.fill(np.nan)
            axis = np.where(d>=0)
            data[axis[0], axis[1], :] = 0
            data[axis[0], axis[1], d[d>=0]] = 1
            
            indices = []
            mat_id = -1
            for i in uniq_ids :
                if i > 0 :
                    mat_id += 1
                indices.append(mat_id)
            indices = np.array(indices)[ids]
            tags, trees = np.array(tags), np.array(trees)
            for mt in self.data['trees'] :
                meta_in = ((trees == 'default') | (trees == mt.tre.label))
                mt.traits[category] = [deepcopy(data), zip(tags[meta_in], indices[meta_in]), types]
        return strains, matrices


    def __get_scaling(self, category) :
        self.model = []        
        def __get_scaling2(rescales) :
            overall_lk = 0.0
            rescale = np.exp(rescales[0]-10)
            for mt in self.data['trees'] :
                if category not in mt.traits :
                    continue
                trait = mt.traits[category]
                n_state = np.max(trait[2].values()) + 1
                transitions = np.zeros(shape=[self.data['n_node'], n_state, n_state])
                for branch in mt.tre.preorder_node_iter() :
                    if branch.edge_length is not None :
                        tr = transitions[branch.id]
                        br_mut = np.exp(-branch.edge_length*rescale)
                        tr.fill((1.0-br_mut)/n_state)
                        np.fill_diagonal(tr, (1.0+(n_state-1.)*br_mut)/n_state)
    
                weights = np.bincount([i for (t, i) in trait[1]], minlength=trait[0].shape[0])
                mat = deepcopy(trait[0][weights>0])
                mat[np.isnan(mat)] = 1./n_state
                weights = weights[weights>0]
                
                for alpha, weight in zip(mat, weights) :
                    beta = np.ones(alpha.shape[0])
                    tot_lk = 0
                    for branch in mt.tre.postorder_node_iter() : 
                        id = branch.id
                        alpha_sum = sum(alpha[id])
                        tot_lk += np.log(alpha_sum)
                        alpha[id] = alpha[id]/alpha_sum
                        if branch.parent_node is not None :
                            beta = np.dot(alpha[id], transitions[id])
                            alpha[branch.parent_node.id] *= beta
                    overall_lk += tot_lk * weight
            print rescale, rescales[0], overall_lk
            self.model.append([rescale, overall_lk])
            return -overall_lk
        return __get_scaling2

    def gs(self, category) :
        def gs2(rescale) :
            overall_lk = 0.0
            for mt in self.data['trees'] :
                if category not in mt.traits :
                    continue
                trait = mt.traits[category]
                n_state = np.max(trait[2].values()) + 1
                transitions = np.zeros(shape=[self.data['n_node'], n_state, n_state])
                for branch in mt.tre.preorder_node_iter() :
                    if branch.edge_length is not None :
                        tr = transitions[branch.id]
                        br_mut = np.exp(-branch.edge_length*rescale)
                        tr.fill((1.0-br_mut)/n_state)
                        np.fill_diagonal(tr, (1.0+(n_state-1.)*br_mut)/n_state)
    
                weights = np.bincount([i for (t, i) in trait[1]], minlength=trait[0].shape[0])
                mat = deepcopy(trait[0][weights>0])
                mat[np.isnan(mat)] = 1./n_state
                weights = weights[weights>0]
                
                for alpha, weight in zip(mat, weights) :
                    beta = np.ones(alpha.shape[0])
                    tot_lk = 0
                    for branch in mt.tre.postorder_node_iter() : 
                        id = branch.id
                        alpha_sum = sum(alpha[id])
                        tot_lk += np.log(alpha_sum)
                        alpha[id] = alpha[id]/alpha_sum
                        if branch.parent_node is not None :
                            beta = np.dot(alpha[id], transitions[id])
                            alpha[branch.parent_node.id] *= beta
                    overall_lk += tot_lk * weight
            return -overall_lk
        return gs2

    def infer_ancestral(self, traits_ASR, traits_rescale=1.0, traits_infer='marigin', **args) :
        retvalue = []
        for category in traits_ASR.split(',') :
            try:
                rescale = float(traits_rescale)
            except :
                self.model = [[1., self.gs(category=category)(1.)]]
                next_direction = 1
                for ite in range(30) :
                    min_id = min(enumerate(self.model), key=lambda x:x[1][1])[0]
                    if next_direction == 1 :
                        new_scale = self.model[min_id][0]*10. if min_id == len(self.model)-1 else np.sqrt(self.model[min_id][0]*self.model[min_id+1][0])
                        value = self.gs(category=category)(new_scale)
                        next_direction = 1 if value < self.model[min_id][1] else -1
                            
                        self.model[min_id+1:min_id+1] = [[new_scale, value]]
                    else :
                        new_scale = self.model[min_id][0]/10. if min_id == 0 else np.sqrt(self.model[min_id][0]*self.model[min_id-1][0])
                        value = self.gs(category=category)(new_scale)
                        next_direction = -1 if value < self.model[min_id][1] else 1
                        
                        self.model[min_id:min_id] = [[new_scale, value]]
                rescale = min(self.model, key=lambda x:x[1])[0]
                #from mystic.solvers import diffev2
                #diffev2(self.__get_scaling(category=category), x0=[(5., 15.)], bounds=[(1., 19)], maxiter=10, npop=10, gtol=6, ftol=0.1)
                #rescale = max(self.model, key=lambda x:x[1])[0]
            retvalue.append([category, rescale])
        
            for mt in self.data['trees'] :
                if category not in mt.traits :
                    continue
                trait = mt.traits[category]
                n_state = np.max(trait[2].values()) + 1
                transitions = np.zeros(shape=[self.data['n_node'], n_state, n_state])
                for branch in mt.tre.preorder_node_iter() :
                    if branch.edge_length is not None :
                        tr = transitions[branch.id]
                        br_mut = np.exp(-branch.edge_length*rescale)
                        tr.fill((1.0-br_mut)/n_state)
                        np.fill_diagonal(tr, (1.0+(n_state-1.)*br_mut)/n_state)
    
                weights = np.bincount([i for (t, i) in trait[1]], minlength=trait[0].shape[0])
                mat = trait[0][weights>0]
                mat[np.isnan(mat)] = 1./n_state
                weights = weights[weights>0]

                for alpha in mat :
                    beta = np.ones(alpha.shape)
                    
                    for branch in mt.tre.postorder_node_iter() : 
                        id = branch.id
                        alpha[id] = alpha[id]/sum(alpha[id])
                        if branch.parent_node is not None :
                            beta[id] = np.dot(alpha[id], transitions[id])
                            alpha[branch.parent_node.id] *= beta[id]
                    if traits_infer == 'marigin' :
                        for branch in mt.tre.preorder_node_iter() :
                            id = branch.id
                            if branch.parent_node is not None :
                                alpha[id] *= np.dot(alpha[branch.parent_node.id]/beta[id], transitions[id])
                    else :
                        for branch in mt.tre.preorder_node_iter() :
                            id = branch.id
                            if branch.parent_node is not None :
                                alpha[id] *= np.dot(alpha[branch.parent_node.id], transitions[id])
                                m_id = np.argmax(alpha[id])
                                alpha[id].fill(0.0)
                                alpha[id][m_id] = 1.0
        return retvalue
    
    def infer_treeline(self, treeline_subTree, treeline_bootstrap=1000, treeline_histBin=50, treeline_direct='tip-to-root', \
                       treeline_dataX='s:branch#length', treeline_dataY='d:pangene#all#0#-,d:plasmid#0/s:branch#length', \
                       **args) :
    
        data_trees = self.data['trees']
        subtrees = [mt.tre.mrca(taxon_labels=treeline_subTree.split(',')) for mt in data_trees] if treeline_subTree is not None else [mt.tre.seed_node for mt in data_trees]
        branches = []
        for subtre in subtrees :
            tips, br = [], {}
            for node in subtre.preorder_iter() :
                if node == subtre :
                    br[node.id] = []
                else :
                    br[node.id] = [node.id] + br.get(node.parent_node.id, []) if treeline_direct == 'tip-to-root' else br.get(node.parent_node.id, []) + [node.id]
                if node.is_leaf() :
                    tips.append(np.array(br[node.id]))
            branches.append(tips)
        all_dist = [ np.sum( [(np.sum(x_list[0][:, tid, br]) + 1e-60)/(np.sum(x_list[1][:, tid, br]) + 1e-30) for br in path] ) \
                     for tid, branch in enumerate(branches) for path in branches ]
        x_bin = float(np.sum(all_dist)/len(all_dist))/n_bin
    
        x_list, y_list = [], []
        for axis, lst in ( ([x.split(',') for x in treeline_Xaxis.split('/')][:2], x_list), \
                           ([y.split(',') for y in treeline_Yaxis.split('/')][:2], y_list)  ) :
            for a in axis :
                lst.append([])
                for group in a :
                    groups = group.split('#')
    
                    category = groups[0]
                    data = data_traits[category][0]
                    for id, c in enumerate(groups[1:]) :
                        if c.lower() != 'all' :
                            if c == '+' :
                                data = data * (data > 0)
                            elif c == '-' :
                                data = data * (data < 0)
                            elif id == 0 :
                                data = data[ data_traits[category][2] == c, :, :, : ]
                            elif id == 1 :
                                data = data[ :, :, :, [ data_traits[category][1][c] ] ]
                    lst[-1].append(np.sum(data, 3))
    
            lst[0] = np.vstack(lst[0])
            if len(lst) > 1 :
                lst[1] = np.vstack(lst[1])
            else :
                lst.append(np.ones(shape=[1, lst[0].shape[1], lst[0].shape[2]]))
    
        curves = np.zeros(shape=[treeline_sampleNum, n_bin])
        for id in np.arange(treeline_sampleNum) :
            print id
            tid = np.random.randint(len(branches)) if id > 0 else 0
            branch = branches[tid]
    
            if np.ceil(id*x_sample) > np.ceil((id-1)*x_sample) :
                if id > 0 :
                    x_set = [np.random.randint(0, len(x_list[0]), len(x_list[0])), \
                             np.random.randint(0, len(x_list[1]), len(x_list[1])) ]
                else :
                    x_set = [np.arange(x_list[0].shape[0]), np.arange(x_list[1].shape[0])]
            if np.ceil(id*y_sample) > np.ceil((id-1)*y_sample) :
                if id > 0 :
                    y_set = [np.random.randint(0, len(y_list[0]), len(y_list[0])), \
                             np.random.randint(0, len(y_list[1]), len(y_list[1])) ]
                else :
                    y_set = [np.arange(y_list[0].shape[0]), np.arange(y_list[1].shape[0])]
    
            save = np.zeros(shape=[len(branch), n_bin])
            for pid, path in enumerate(branch) :
    
                x = (np.sum(x_list[0][x_set[0], tid, path], [0, 1]) + 1e-60)/(np.sum(x_list[1][x_set[1], tid, path], [0, 1]) + 1e-30)
                y = (np.sum(y_list[0][y_set[0], tid, path], [0, 1]) + 1e-60)/(np.sum(y_list[1][y_set[1], tid, path], [0, 1]) + 1e-30)
    
                acc = [0., 0., 0., 0.]
                curve = []
                for m, n in zip(x, y) :
                    acc[2], acc[3] = m, m*n
                    while acc[0] + acc[2] >= x_bin :
                        d1 = x_bin-acc[0]
                        d2 = d1/acc[2]*acc[3]
                        curve.append((acc[1]+d2)/x_bin)
                        acc[2] -= d1
                        acc[3] -= d2
                        acc[0] = acc[1] = 0.
                    acc = [acc[0]+acc[2], acc[1]+acc[3], 0., 0.]
                if acc[0] >= 0.5 * x_bin :
                    curve.append(acc[1]/acc[0])
                curve = curve[:n_bin]
                save[pid, :len(curve)] = curve
    
            curves[id] = np.sum(save, 0)/np.sum(save>0, 0)
        for cid, c in enumerate(curves.T) :
            c = np.sort(c[c > 0])
            curve = c[np.array([int(c.size*0.025), int(c.size*0.5), int(c.size*0.975)])]
            print x_bin*cid, curve[0], curve[1], curve[2]
    
    def infer_breakpoints() :
        pass




def update_traits(data_trees, data_traits, **args) :
    new_traits = {}
    for category, (mat, types, tags) in data_traits.iteritems() :
        new_traits['s:' + category] = [mat, types, tags]
        new_traits['d:' + category] = [np.zeros(mat.shape), types, tags]
        dmat = new_traits['d:' + category][0]
        for tid, tre in enumerate(data_trees) :
            for node in  tre.preorder_node_iter() :
                if node.parent_node is not None :
                    id, pid = node.id, node.parent_node.id
                    dmat[:, tid, id] = mat[:, tid, id] - mat[:, tid, pid]
    return new_traits

# stage 2: generate plot
    

def main(*argv) :
    parameters = dict( arg.split('=', 1) for arg in argv[1:] )
    treeline = TreeLine(**parameters)

if __name__ == '__main__' :
    main(*sys.argv)