# TreeLine: infer frequencies of genetic events using trees
# 1. add tree (create dataset)
# 2. add trait (update dataset)
# 3. draw line
# 4. infer category

import numpy as np, csv, sys, os, json, re
from dendropy import Tree, Node, Taxon
from tempfile import mkdtemp
from copy import deepcopy
import cPickle as pickle

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
    proj_folder = None
    data = dict(
        n_node = 0,
        taxa = None,
        trees = None,
        superTree = None,
        binSize = None,
        curves = None,
    )

    param = dict(
        tree = dict(
            treefile = None,
            burnIn = 0,
            sampleFreq = 1,
            maxNum = 10,
            ignoreMissing = False,
            superTree = 'consensus', # MCC ASTRID id filename
        ),
        trait = dict(
            traitfile = None,
            ignoreMissing = True,
            rescale = 1.,
            maxEM = 10,
            inference = 'marginal',
            focus = None,
        ),
        line = dict(
            subTree = None,
            stem = False,
            dataX = '{s[branch][length]}',
            dataY = '({d[pangene][*][*][+]} + {d[plasmid][*][*][+]})/{s[branch][length]}',
            sampleNum = 1000,
            binNum = 100,
            direct = 'tip-to-root',
        ),
        group = dict(
            maxNum = 5,
            binStatus = 'constant', # linear
        ),
        view = dict(
            tree = None,
            traits = None,
        )
    )

    def __init__ (self, method='tree', **kwargs) :
        assert method in ('tree', 'trait', 'line', 'group'), 'method is not recognized.'
        if 'in' in kwargs :
            self.data = pickle.load(open(kwargs.pop('in'), 'rb'))
        fout = sys.stdout if 'out' not in kwargs else open(kwargs.pop('out'), 'wb')

        self.param[method].update(kwargs)
        eval('self.func_' + method)(**self.param[method])

        pickle.dump(self.data, fout)
        fout.close()

    def func_tree(self, superTree, **args) :
        taxa, trees, n_node = self.load_tree(**args)
        self.data = dict(
            taxa=taxa,
            trees=trees,
            n_node=n_node,
        )
        self.data['superTree'] = self.get_super_tree(superTree)
        return self.data


    def load_tree(self, treefile, burnIn = 0, sampleFreq = 1, maxNum = 10, ignoreMissing=False, **args) : # sumtrees file ASTRID
        # read trees (including traits when possible)
        data_trees = []
        with open(treefile) as fin :
            schema = 'nexus' if fin.readline().upper().startswith('#NEXUS') else 'newick'

        for id, tre in enumerate(Tree.yield_from_files([treefile], schema=schema)) :
            if maxNum > 0 and id > maxNum : break
            if id >= burnIn :
                if not tre.label :
                    tre.label = str(id)
                if (id - burnIn) % sampleFreq == 0 :
                    data_trees.append(tre)
        # find all tips
        taxa = {}
        for tre in data_trees :
            for taxon in tre.taxon_namespace :
                taxa[taxon.label] = 1
        for id, taxon in enumerate(sorted(taxa.keys())) :
            taxa[taxon] = id
        # load in metadata trait types
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
                    node.id, internal_id = internal_id, internal_id+1
                    node.barcode = sum([c.barcode for c in node.child_nodes()])
            if internal_id > n_node : n_node = internal_id
            tre.seed_node.age = tre.seed_node.distance_from_tip()
            for node in tre.preorder_node_iter() :
                if node.parent_node :
                    node.age = node.parent_node.age - node.edge_length

        # convert traits into discrete characters
        for cc, tc in trait_categories.iteritems() :
            if tc[1] == 'discrete' :
                if ignoreMissing :
                    tc[2].update({"-":-1, "":-1, "0":-1})
                tc[2].update(dict([[k, id] for id, k in enumerate(sorted([k for k, v in tc[2].iteritems() if v > 0]))]))
        # read traits' values
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
        def parse_trees(**args) :
            n_tree, n_branch = float(len(self.data['trees'])), {}
            for mt_id, mt in enumerate(self.data['trees']) :
                w = (float(len(mt.tre.leaf_nodes())) / len(self.data['taxa']))**2
                for node in mt.tre.preorder_node_iter() :
                    if node.barcode not in n_branch :
                        n_branch[node.barcode] = [[w, mt_id, node]]
                    else :
                        n_branch[node.barcode].append([w, mt_id, node])
            return n_tree, n_branch
        def consensus(self, **args) :
            n_tree, n_branch = parse_trees(**args)
            n_branch = sorted([[len(v)/n_tree, k, v] for k, v in n_branch.iteritems()], reverse=True)
            consensus_tree = []
            for posterior, branch, nodes in n_branch :
                for cbr, _, _ in consensus_tree :
                    b1, b2 = sorted([branch, cbr])
                    if not (( (b1 & b2) == b1 ) or ( (b1 & (~b2)) == b1 )) :
                        branch = 0
                        break
                if branch :
                    consensus_tree.append([branch, posterior, nodes])
            return sorted(consensus_tree, reverse=True)
        def MCC(self, **args) :
            n_tree, n_branch = parse_trees(**args)
            for mt_id, mt in enumerate(self.data['trees']) :
                if len(mt.tre.leaf_nodes()) == len(self.data['taxa']) :
                    mt.score = np.sum([len(n_branch[node.barcode]) for node in mt.tre.preorder_node_iter() ])
            tre = max(self.data['trees'], key=lambda x:x.score).tre
            return [[n.barcode, len(n_branch[n.barcode])/n_tree, n_branch[n.barcode]] for n in tre.preorder_node_iter()]
        def load_subtree(self, treeLabel, **args) :
            n_tree, n_branch = parse_trees(**args)
            for mt_id, mt in enumerate(self.data['trees']) :
                if mt.tre.label == treeLabel :
                    tre = mt.tre
                    break
            return [[n.barcode, len(n_branch[n.barcode])/n_tree, n_branch[n.barcode], n.age, n.edge_length] for n in tre.preorder_node_iter()]

        #def ASTRID(self, **args) :
            #from dendropy import PhylogeneticDistanceMatrix

        def load_tree(self, consFile=None, **args) :
            n_tree, n_branch = parse_trees(**args)

            with open(consFile) as fin :
                schema = 'nexus' if fin.readline().upper().startswith('#NEXUS') else 'newick'
            for tre in Tree.yield_from_files([consFile], schema=schema) :
                break

            internal_id = n_taxa = len(self.data['taxa'])
            digit_code = np.power(2, np.arange(n_taxa, dtype='object'))

            for node in tre.postorder_node_iter() :
                if node.is_leaf() :
                    node.id = self.data['taxa'][node.taxon.label]
                    node.barcode = digit_code[node.id]
                else :
                    node.id, internal_id = internal_id, internal_id+1
                    node.barcode = sum([c.barcode for c in node.child_nodes()])

            tre.seed_node.age = tre.seed_node.distance_from_tip()
            for node in tre.preorder_node_iter() :
                if node.parent_node :
                    node.age = node.parent_node.age - node.edge_length
            return [[n.barcode, len(n_branch.get(n.barcode, []))/n_tree, n_branch.get(n.barcode, []), n.age, n.edge_length] for n in tre.preorder_node_iter()]

        if superTree_method in ('MCC', 'ASTRID', 'consensus') :
            branches = locals()[superTree_method](self, **args)
        elif os.path.isfile(superTree_method) :
            branches = load_tree(self, consFile=superTree_method, **args)
        else :
            branches = load_subtree(self, treeLabel=superTree_method, **args)
        supertree = Tree()
        sn = supertree.seed_node
        sn.barcode, sn.posterior = branches[0][0], branches[0][1]
        sn.age = branches[0][3] if len(branches[0])> 3 else np.sum([n[2].age*n[0] for n in branches[0][2]])/np.sum([n[0] for n in branches[0][2]])
        sn.contain = [ [b[0], b[1], b[2].id] for b in branches[0][2] ]
        for br in branches[1:] :
            cbr, posterior, nodes = br[:3]
            while (sn.barcode & cbr) != cbr :
                sn = sn.parent_node
            new_node = Node() if len(nodes) == 0 or (not nodes[0][2].taxon) else Node(taxon=Taxon(label=nodes[0][2].taxon.label))
            sn.add_child(new_node)
            sn = new_node
            sn.barcode, sn.posterior = cbr, posterior
            sn.contain = [ [b[0], b[1], b[2].id] for b in nodes ]
            if len(br) <= 3 :
                sn.edge_length = 0.0 if len(nodes) == 0 else np.sum([n[2].edge_length*n[0] for n in nodes])/np.sum([n[0] for n in nodes])
                sn.age = sn.parent_node.age if len(nodes) == 0 else np.sum([n[2].age*n[0] for n in nodes])/np.sum([n[0] for n in nodes])
            else :
                sn.age, sn.edge_length = br[3:]
        internal_id = len(self.data['taxa'])
        for node in supertree.postorder_node_iter() :
            if node.is_leaf() :
                node.id = self.data['taxa'][node.taxon.label]
            else :
                node.id = internal_id
                internal_id += 1
        return MetaTree(supertree)

    def func_trait(self, **args) :
        if args['traitfile'] is not None :
            strains, matrices = self.read_metadata(**args)
        if args['focus'] is None :
            args['focus'] = ','.join(matrices.keys())
        retvalue = self.infer_ancestral(**args)
        #print retvalue

    def read_metadata(self, traitfile, ignoreMissing=True, **args) :
        strains, matrices = [], {}
        with open(traitfile, 'r') as fin :
            header = fin.readline().strip().split('\t')
            ids = []
            headers = []
            for id, head in enumerate(header) :
                if head.startswith('#') :
                    headers.append(id)
                else :
                    try :
                        strains.append(self.data['taxa'][head])
                        ids.append(id)
                    except :
                        pass

            ids = np.array(ids)
            for id, line in enumerate(csv.reader(fin, delimiter='\t')) :
                line = np.array(line)
                if not line.size : continue
                category = line[headers[0]] if len(headers) > 0 else 'default'
                tag = line[headers[1]] if len(headers) >= 2 else str(id)

                if category not in matrices :
                    matrices[category] = [[line[ids]], [tag], {}]
                else :
                    matrices[category][0].append(line[ids])
                    matrices[category][1].append(tag)

        for category, (mat, tags, types) in matrices.items() :
            types = np.unique(mat)
            if ignoreMissing :
                types = types[(types != '') & (types != '-') & (types != '0')]
            types = {t:id for id, t in enumerate(sorted(types.tolist()))}
            if ignoreMissing :
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
            tags = np.array(tags)
            for mt in self.data['trees'] :
                mt.traits[category] = [deepcopy(data), zip(tags, indices), types]
        return strains, matrices

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

    def infer_ancestral(self, focus, rescale=1.0, infer='marigin', **args) :
        retvalue = []
        for category in focus.split(',') :
            try:
                scale = float(rescale)
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
                scale = min(self.model, key=lambda x:x[1])[0]
            retvalue.append([category, scale])

            for mt in self.data['trees'] :
                if category not in mt.traits :
                    continue
                trait = mt.traits[category]
                n_state = np.max(trait[2].values()) + 1
                transitions = np.zeros(shape=[self.data['n_node'], n_state, n_state])
                for branch in mt.tre.preorder_node_iter() :
                    if branch.edge_length is not None :
                        tr = transitions[branch.id]
                        br_mut = np.exp(-branch.edge_length*scale)
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
                    if infer == 'marigin' :
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
                trait[0][weights>0] = mat
        return retvalue

    def func_line(self, **args) :
        self.data['binSize'], self.data['curves'] = self.infer_treeline(**args)
    def infer_treeline(self, subTree, sampleNum=1000, binNum=50, direct='tip-to-root', stem=True, \
                       dataX='{s[branch][length]}', dataY='({d[pangene][*][0][-]} + {d[plasmid][*]})/{s[branch][length]}', \
                       **args) :

        data_trees = self.data['trees']
        data, tags, axis = [None, None], [None, None], [None, None]
        for data_id, source in enumerate( (dataX, dataY) ) :
            items = re.findall(r'\{[^\}]+\}', source)
            data[data_id] = [ [ None for j in np.arange(len(items)) ] for i in np.arange(len(data_trees)) ]
            tags[data_id] = [ [ None for j in np.arange(len(items)) ] for i in np.arange(len(data_trees)) ]
            axis[data_id] = re.sub(r'\{[^\}]+\}', lambda x: {t:'{'+str(i)+'}' for i, t in enumerate(items)}[x.group()], source)
            for item_id, item in enumerate(items) :
                filter = (['.*' if f == "*" else f for f in re.split(r'[\]\[]+', item[1:-1])[:-1]] + ['.*', '.*', '.*'])[:5]
                for mt_id, mt in enumerate(data_trees) :
                    d = mt.traits[filter[1]]
                    tag_indices = {}
                    for tag, id in d[1] :
                        if re.match(filter[2], tag) is not None :
                            if tag not in tag_indices :
                                tag_indices[tag] = []
                            tag_indices[tag].append(id)
                    #tag_indices = [ [tag, id] for tag, id in d[1] if re.match(filter[2], tag) is not None ]  # change to weighting
                    tags[data_id][mt_id][item_id] = tag_indices.values()
                    if d[2] is not None :
                        state_indices = np.unique([ id for tag, id in d[2].iteritems() if id >= 0 and re.match(filter[3], tag) is not None ])
                        d = d[0][:, :, state_indices]
                    else :
                        d = deepcopy(d[0])
                    if filter[0] == 'd' :
                        for node in mt.tre.postorder_node_iter() :
                            d[:, node.id, :] = d[:, node.id, :] - d[:, node.parent_node.id, :] if node.parent_node else 0.
                    if filter[4] == '+' :
                        d = d * (d > 0)
                    elif filter[4] == '-' :
                        d = d * (d < 0)
                    d = np.nan_to_num(np.sum(d, 2)) + 1e-100
                    data[data_id][mt_id][item_id] = d

        subtrees = [mt.tre.mrca(taxon_labels=subTree.split(',')) for mt in data_trees] if subTree is not None else [mt.tre.seed_node for mt in data_trees]
        branches = []
        for subtre in subtrees :
            tips, br = [], {}
            for node in subtre.preorder_iter() :
                if node == subtre :
                    br[node.id] = [node.id] if stem else []
                else :
                    br[node.id] = [node.id] + br.get(node.parent_node.id, []) if direct == 'tip-to-root' else br.get(node.parent_node.id, []) + [node.id]
                if node.is_leaf() :
                    tips.append(np.array(br[node.id]))
            branches.append(tips)

        span = []
        for treeId, treeBranch in enumerate(branches) :
            xdata = np.array([ np.sum(item[np.array([tt for t in tag for tt in t])], 0) for item, tag in zip(data[0][treeId], tags[0][treeId]) ])
            span.extend([np.sum([eval(axis[0].format(*xdata.T[brId])) for brId in tipPath ])for tipPath in treeBranch])

        x_bin = np.mean(span)/float(binNum)

        curves = np.empty([sampleNum, len(self.data['taxa']), binNum])
        curves[:] = np.nan

        for ite, treeId in enumerate(np.random.randint(0, len(branches), sampleNum)) :
            xdata = np.array([ np.sum(item[np.array([ tt for id in np.random.randint(0, len(tag), len(tag)) for tt in tag[id] ])], 0) for item,tag in zip(data[0][treeId], tags[0][treeId]) ])
            ydata = np.array([ np.sum(item[np.array([ tt for id in np.random.randint(0, len(tag), len(tag)) for tt in tag[id] ])], 0) for item,tag in zip(data[1][treeId], tags[1][treeId]) ])

            treeData = [np.array([[eval(axis[0].format(*xdata.T[brId])), eval(axis[1].format(*ydata.T[brId]))] for brId in tipPath ])for tipPath in branches[treeId]]
            treeData = np.random.choice(treeData, curves.shape[1])

            for d, c in zip(treeData, curves[ite]) :
                d = deepcopy(d)
                bins, x_bin_x = [0.], x_bin
                dId = 0
                while dId < len(d) :
                    dd = d[dId]
                    if dd[0] <= x_bin_x :
                        x_bin_x -= dd[0]
                        bins[-1] += dd[0]*dd[1]
                        dId += 1
                    elif dd[0] > x_bin_x :
                        dd[0] -= x_bin_x
                        bins[-1] += x_bin_x * dd[1]
                        x_bin_x = x_bin
                        bins.append(0.)
                bins = np.array(bins)
                bins[-1] /= (x_bin - x_bin_x)
                bins[:-1] /= x_bin
                bins = bins[:binNum]
                c[:bins.size] = bins[:]

        for cid, c in enumerate(np.nanmean(curves, 1).T) :
            c = np.sort(c[c > 0])
            curve = c[np.array([int(c.size*0.025), int(c.size*0.5), int(c.size*0.975)])]
            print x_bin*cid, curve[0], curve[1], curve[2]
        return x_bin, curves

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