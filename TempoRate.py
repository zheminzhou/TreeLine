# Map Events Over History (MEOH)
import numpy as np, csv, sys
from dendropy import Tree

parameters = {
    'control_debug': 0,
    'control_treeFile': None,
    'control_matFile': None,
    'control_dumpFile': None,
    'control_outFile': None,
    
    'data_trees': None,
    'data_consTree': None,
    'data_traits': None,
    'data_plot': None,
    
    'tree_burnin': 0, 
    'tree_sampleFreq': 1,
    'tree_maxNum': 10,
    'tree_subtree':None,

    'trait_ignoreMissing': True,
    'trait_rescale': 1.,
    'trait_EMNum': 10,
    'trait_infer': 'marigin',

    'treeline_Xaxis': 's:branch.length',
    'treeline_Yaxis': 'd:pangene#all#+/s:branch.length',
    'treeline_sampleNum': 1000, 
    'treeline_XsampleFreq': 1., 
    'treeline_YsampleFreq': 1., 
    'treeline_binNum': 50, 
    'treeline_direct':'tip-to-root', 
}

def read_trees(control_treeFile, tree_burnin=0, tree_sampleFreq=1, tree_maxNum=10, trait_ignoreMissing=True, **args) :
    data_trees = []
    schema = 'newick'
    with open(control_treeFile) as fin :
        if fin.readline().upper().startswith('#NEXUS') :
            schema = 'nexus'
    for id, tre in enumerate(Tree.yield_from_files([control_treeFile], schema=schema)) :
        if id >= tree_burnin :
            if (id - tree_burnin) % tree_sampleFreq == 0 :
                data_trees.append(tre)
            if len(data_trees) >= tree_maxNum : break
        
    n_tree, n_node = len(data_trees), len(data_trees[0].nodes())
    data_traits = {'branch.length':[np.zeros(shape=[ 1, n_tree, n_node, 1]), None, ['branch.length']]}

    for node in data_trees[0].preorder_node_iter() :
        for annotation in node.annotations :
            if annotation.name not in data_traits :
                if isinstance(annotation.value, list) :
                    try :
                        if isinstance(float(annotation.value[0]), float) :
                            data_traits[annotation.name] = [np.zeros(shape=[ 1, n_tree, n_node, len(annotation.value)]), None, [annotation.name]]
                    except :
                        data_traits[annotation.name] = [np.zeros(shape=[ 1, n_tree, n_node, len(annotation.value)], dtype=int), \
                                                        {'0':-1, '-':-1, '':-1} if trait_ignoreMissing else {'':-1}, [annotation.name]]
                else :
                    try :
                        if isinstance(float(annotation.value), float) :
                            data_traits[annotation.name] = [np.zeros(shape=[ 1, n_tree, n_node, 1]), None, [annotation.name]]
                    except :
                        data_traits[annotation.name] = [np.zeros(shape=[ 1, n_tree, n_node, 1], dtype=int), \
                                                        {'0':-1, '-':-1, '':-1} if trait_ignoreMissing else {'':-1}, [annotation.name]]
    for tid, tre in enumerate(data_trees) :
        for nid, node in enumerate(tre.nodes()) :
            node.id = nid
            if node.edge_length < 1e-8 and node.parent_node is not None :
                if node.is_leaf() :
                    node.edge_length = 1e-8
                else :
                    parent = node.parent_node
                    for child in node.child_nodes() :
                        child._set_parent_node(parent)
                    parent.remove_child(node)
                    parent.set_child_nodes(parent.child_nodes() + node.child_nodes())
                    continue
            data_traits['branch.length'][0][0, tid, nid, 0] = node.edge_length if node.parent_node else 0.0
            
            for annotation in node.annotations :
                if annotation.name in data_traits :
                    k, v = annotation.name, annotation.value
                    if isinstance(v, basestring) : v = [v]

                    if data_traits[k][1] is None :
                        data_traits[k][0][0, tid, nid] = [float(vv) for vv in v]
                    else :
                        for vv in v :
                            if vv not in data_traits[k][1] :
                                data_traits[k][1][vv] = max(data_traits[k][1].values()) + 1
                        data_traits[k][0][0, tid, nid] = [data_traits[k][1][vv] for vv in v]
            node.annotations.clear()
            node.annotations.add_new('id', node.id)
    for key, states in data_traits.iteritems() :
        if states[1] is not None :
            s, categories, tags = states
            new_state = np.zeros([s.shape[0], s.shape[1], s.shape[2], np.max(s)+1], dtype=int)
            axis = np.where(s>=0)[:3]
            new_state[axis[0], axis[1], axis[2], s[s>=0]] = 1
            data_traits[key][0] = new_state
    return data_trees, data_traits

def read_metadata(control_matFile, trait_ignoreMissing=True, **args) :
    strains, matrices = [], {}
    with open(control_matFile, 'r') as fin :
        header = fin.readline().strip().split('\t')
        ids, cat_col = [], []
        for id, head in enumerate(header) :
            if head.startswith('#') :
                cat_col.append(id)
            else :
                ids.append(id)
                strains.append(head)
        ids, cat_col = np.array(ids), np.array(cat_col[:2])
        for line in csv.reader(fin, delimiter='\t') :
            line = np.array(line)
            if not line.size : continue
            category = line[cat_col][0]
            tag = line[cat_col][1]
            mat = line[ids]
            if category not in matrices :
                matrices[category] = [[mat], {}, [tag]]
            else :
                matrices[category][0].append(mat)
                matrices[category][2].append(tag)
    
    for category, (mat, types, tags) in matrices.items() :
        types = np.unique(mat)
        if trait_ignoreMissing :
            types = types[(types != '') & (types != '-') & (types != '0')]
        types = {t:id for id, t in enumerate(sorted(types.tolist()))}
        if trait_ignoreMissing :
            types.update({'0':-1, '-':-1, '':-1})
        mat = np.vectorize(types.get)(mat)
        #index = np.apply_along_axis(lambda x:len(np.unique(x[x>=0]))>1, 1, mat)
        #mat, tags = mat.T[index], np.array(tags)[index]
        matrices[category] = [mat.T, types, np.array(tags)]
    return strains, matrices
    


def get_scaling(data_trees, mat, trait_EMNum) :
    scaling = 1.0
    for iter in xrange(trait_EMNum) :
        n_state = np.max(types.values()) + 1
        n_branch = max([n.id for n in data_trees[0].nodes()]) + 1
        alpha = np.ones(shape=[mat.shape[1], len(data_trees), n_branch, n_state])
        for tid, tre in enumerate(data_trees) :
            tip_ids = {node.taxon.label:node.id for node in tre.leaf_nodes()}
            name2tip = np.array([ tip_ids.get(n, -1) for n in names ])
            if np.sum(name2tip < 0) > 0 :
                print >> sys.stderr, 'WARNING: Some strains in metadata table are not in the trees'
            assert len(tip_ids) == np.sum(name2tip >= 0), 'FATAL ERROR: Some tips in the trees are not in the metadata matrix'
            tip_in_use = name2tip[name2tip >= 0]
    
            transitions = np.zeros(shape=[n_branch, n_state, n_state])
            for branch in tre.preorder_node_iter() :
                if branch.edge_length is not None :
                    tr = transitions[branch.id]
                    br_mut = np.exp(-branch.edge_length*scaling)
                    tr.fill((1.0-br_mut)/n_state)
                    np.fill_diagonal(tr, (1.0+(n_state-1.)*br_mut)/n_state)
        '''...'''
    return scaling


def mat_nr(mat) :
    character_order = np.lexsort(mat)
    mat2 = mat[:, character_order]
    character_nr = np.concatenate([[True], np.sum(mat2[:, :-1] != mat2[:, 1:], 0)>0])
    mat_type = mat[:, character_order[character_nr]]
    mat_index = np.vstack([character_order, character_nr]).T
    id = -1
    for m in mat_index :
        if m[1] :
            id += 1
        m[1] = id
    return mat_type, mat_index[np.argsort(mat_index.T[0]), :]
    
def AncestralReconstruction(data_trees, names, matrices, rescale=1.0, trait_EMNum=10, trait_infer='marigin', **args) :
    new_traits = {}
    for category, (mat, types, tags) in matrices.iteritems() :
        mat_type, mat_index = mat_nr(mat)
        if rescale is None :
            scaling = get_scaling(data_trees, mat, trait_EMNum)
        else :
            scaling = rescale

        n_state = np.max(types.values()) + 1
        n_branch = max([n.id for n in data_trees[0].nodes()]) + 1

        trait_type = np.ones(shape=[mat_type.shape[1], len(data_trees), n_branch, n_state])
        for tid, tre in enumerate(data_trees) :
            tip_ids = {node.taxon.label:node.id for node in tre.leaf_nodes()}
            name2tip = np.array([ tip_ids.get(n, -1) for n in names ])
            if np.sum(name2tip < 0) > 0 :
                print >> sys.stderr, 'WARNING: Some strains in metadata table are not in the trees'
            assert len(tip_ids) == np.sum(name2tip >= 0), 'FATAL ERROR: Some tips in the trees are not in the metadata matrix'
            tip_in_use = name2tip[name2tip >= 0]
    
            transitions = np.zeros(shape=[n_branch, n_state, n_state])
            for branch in tre.preorder_node_iter() :
                if branch.edge_length is not None :
                    tr = transitions[branch.id]
                    br_mut = np.exp(-branch.edge_length*scaling)
                    tr.fill((1.0-br_mut)/n_state)
                    np.fill_diagonal(tr, (1.0+(n_state-1.)*br_mut)/n_state)

            for sid, m in enumerate(mat_type.T) :
                alpha = trait_type[sid, tid, :, :]
                beta = np.ones(shape=[n_branch, n_state])
                
                alpha[tip_in_use] = 0.0
                alpha[tip_in_use, m[name2tip >= 0]] = 1.0
                
                for branch in tre.postorder_node_iter() : 
                    id = branch.id
                    alpha[id] = alpha[id]/sum(alpha[id])
                    if branch.parent_node is not None :
                        beta[id] = np.dot(alpha[id], transitions[id])
                        alpha[branch.parent_node.id] *= beta[id]
                if trait_infer == 'marigin' :
                    for branch in tre.preorder_node_iter() :
                        id = branch.id
                        if branch.parent_node is not None :
                            alpha[id] *= np.dot(alpha[branch.parent_node.id]/beta[id], transitions[id])
                else :
                    for branch in tre.preorder_node_iter() :
                        id = branch.id
                        if branch.parent_node is not None :
                            alpha[id] *= np.dot(alpha[branch.parent_node.id], transitions[id])
                            m_id = np.argmax(alpha[id])
                            alpha[id].fill(0.0)
                            alpha[id][m_id] = 1.0
        new_traits[category] = [trait_type[mat_index.T[1], :, :, :], types, tags]
    return new_traits

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
def TL_plot(data_trees, data_traits, tree_subtree='FA1062AA,HA1701AA', \
            treeline_Xaxis='s:branch.length', treeline_Yaxis='d:pangene#all#0#-,d:plasmid#0/s:branch.length', \
            treeline_sampleNum=1000, treeline_XsampleFreq=1., treeline_YsampleFreq=1., treeline_binNum=50, treeline_direct='tip-to-root', **args) :
    
    subtrees = [tre.mrca(taxon_labels=tree_subtree.split(',')) for tre in data_trees] if tree_subtree is not None else [tre.seed_node() for tre in data_trees]
    branches = []
    for subtre in subtrees :
        tips, br = [], {}
        for node in tre.preorder_iter() :
            if node == subtre :
                br[node.id] = []
            else :
                br[node.id] = [node.id] + br.get(node.parent_node.id, []) if treeline_direct == 'tip-to-root' else br.get(node.parent_node.id, []) + [node.id]
            if node.is_leaf() :
                tips.append(np.array(br[nid]))
        branches.append(tips)
    all_dist = [ np.sum( [(np.sum(x_list[0][:, tid, br]) + 1e-60)/(np.sum(x_list[1][:, tid, br]) + 1e-30) for br in path] ) \
                 for tid, branch in enumerate(branches) for path in branch ]
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
            # x = [(np.sum(x_list[0][x_set[0], tid, br]) + 1e-60)/(np.sum(x_list[1][x_set[1], tid, br]) + 1e-30) for br in path ]
            # y = [(np.sum(y_list[0][y_set[0], tid, br]) + 1e-60)/(np.sum(y_list[1][y_set[1], tid, br]) + 1e-30) for br in path ]

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
        # curves[id] = np.apply_along_axis(lambda x:np.sum(x)/np.sum(x>0) if np.sum(x>0) > 0 else 0., 1, save.T)
    for cid, c in enumerate(curves.T) :
        c = np.sort(c[c > 0])
        curve = c[np.array([int(c.size*0.025), int(c.size*0.5), int(c.size*0.975)])]
        print x_bin*cid, curve[0], curve[1], curve[2]
    

def main(*argv) :
    global parameters
    parameters.update(dict( arg.split('=', 1) for arg in argv[1:] ))
    if not parameters['control_debug'] :
        parameters['data_trees'], parameters['data_traits'] = read_trees(**parameters)
        names, matrices = read_metadata(**parameters)
        new_traits = AncestralReconstruction(names=names, matrices=matrices, **parameters)
        parameters['data_traits'].update(**parameters)
        import pickle
        pickle.dump(parameters, open('test.dmp', 'w'))
    else :
        import pickle
        parameters = pickle.load(open('test.dmp'))
    traits = update_traits(**parameters)
    output = TL_plot(parameters['data_trees'], traits)

if __name__ == '__main__' :
    main(*sys.argv)