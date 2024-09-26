import gzip
import os
import sys
import pickle
from textwrap import dedent

import numpy as np
from tree_sitter import Tree
from tree_sitter_languages import get_parser
from tqdm import tqdm


def _supported_file_extensions():
    return {
        '.rb': 'ruby',
        '.go': 'go',
        '.rs': 'rust',
        '.java': 'java',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.py': 'python',
        '.c': 'c',
        '.h': 'c',
        '.cpp': 'cpp',
        '.hpp': 'cpp',
        '.kt': 'kotlin',
        '.kts': 'kotlin',
        '.ktm': 'kotlin',
        '.php': 'php',
    }


def _traverse_tree(tree: Tree):
    cursor = tree.walk()
    reached_root = False
    while reached_root is False:
        yield cursor.node
        if cursor.goto_first_child():
            continue
        if cursor.goto_next_sibling():
            continue
        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_root = True
            if cursor.goto_next_sibling():
                retracing = False


def _extract_functions(nodes, fp, file_content, relevant_node_types):
    out = []
    for n in nodes:
        if n.type in relevant_node_types:
            node_text = dedent('\n'.join(file_content.split('\n')[
                               n.start_point[0]:n.end_point[0]+1]))
            out.append(
                {'file': fp, 'line': n.start_point[0], 'text': node_text})
    return out


def _get_repo_functions(fp_list, supported_file_extensions, relevant_node_types):
    functions = []
    for fp in fp_list:
        if not os.path.isfile(fp):
            continue
        try:
            with open(fp, 'r') as f:
                lang = supported_file_extensions.get(fp[fp.rfind('.'):])
                if lang:
                    parser = get_parser(lang)
                    file_content = f.read()
                    tree = parser.parse(bytes(file_content, 'utf8'))
                    all_nodes = list(_traverse_tree(tree.root_node))
                    functions.extend(_extract_functions(
                        all_nodes, fp, file_content, relevant_node_types))
        except:
            with open(fp, 'r', encoding='cp1252') as f:
                lang = supported_file_extensions.get(fp[fp.rfind('.'):])
                if lang:
                    parser = get_parser(lang)
                    file_content = f.read()
                    tree = parser.parse(bytes(file_content, 'cp1252'))
                    all_nodes = list(_traverse_tree(tree.root_node))
                    functions.extend(_extract_functions(
                        all_nodes, fp, file_content, relevant_node_types))
    return functions


def do_embed(args, model):
    root=args.path_to_repo
    nodes_to_extract = ['function_definition', 'method_definition',
                        'function_declaration', 'method_declaration']
    fp_list = tqdm([root + '/' + f for f in os.popen('git -C {} ls-files'.format(root)).read().split('\n')])

    functions = _get_repo_functions(
        fp_list, _supported_file_extensions(), nodes_to_extract)
    
    if not functions:
        print('No supported languages found in {}. Exiting'.format(args.path_to_repo))
        sys.exit(1)

    print('Embedding {} functions in {} batches. This is done once and cached in .embeddings'.format(
        len(functions), int(np.ceil(len(functions)/args.batch_size))))
    corpus_embeddings = model.encode(
        [f['text'] for f in functions], convert_to_tensor=True, show_progress_bar=True, batch_size=args.batch_size)

    dataset = {'functions': functions,
               'embeddings': corpus_embeddings.to('cpu'), 'model_name': args.model_name_or_path}
    with gzip.open(args.path_to_repo + '/' + '.embeddings', 'w') as f:
        f.write(pickle.dumps(dataset))

    current_commit = os.popen('git rev-parse HEAD').read().strip()
    with open(os.path.join(args.path_to_repo, '.embeddings.meta'), "w") as f:
        f.write(current_commit)
    return dataset

def update_embed(args, model ):
    with gzip.open(args.path_to_repo + '/' + '.embeddings', 'r') as f:
        dataset = pickle.loads(f.read())
        if args.gpu:
            dataset['embeddings'] = dataset['embeddings'].to("cuda:0")

    if not os.path.isfile(os.path.join(args.path_to_repo, '.embeddings.meta')):
        print('Warning: Aborting attempt to update embeddings cache because .embeddings is present but there is no corresponding .embeddings.meta file. As a result, sem might be out of sync with your current repo. To fix this, regenerate the embeddings by deleting .embeddings file.')
        return dataset
    
    nodes_to_extract = ['function_definition', 'method_definition',
                    'function_declaration', 'method_declaration']

    current_commit = os.popen('git rev-parse HEAD').read().strip()
    with open(os.path.join(args.path_to_repo, '.embeddings.meta'), "r") as f:
        old_commit = f.read().strip()
        if old_commit != current_commit:
            delta = os.popen(f"git diff --name-status {old_commit} {current_commit}").read().strip().split('\n')
        else:
            return
    
    print("The current repo seems to out of date the cache. The following files are different:")
    to_delete = set()
    to_modify = set()
    for p in [i.split('\t') for i in delta]:
        status = p[0]
        fp = p[1]
        fp = os.path.join(args.path_to_repo, fp)
        if status == "A":
            to_modify.add(fp)
        if status == "D": 
            to_delete.add(fp)
        if status == "M":
            to_modify.add(fp)
        print(f"{status}\t{fp}")
    
    import torch
    for idx, tpl  in enumerate(zip(dataset["functions"], dataset["embeddings"])):
        
        function = tpl[0]
        embeddings = tpl[1]
        if function['file'] in to_delete or function['file'] in to_modify:
            dataset["functions"].remove(function)
            dataset["embeddings"] = torch.cat((dataset['embeddings'][:idx], dataset['embeddings'][idx+1:]), 0)
            continue

    functions = []
    for fp in to_modify:
        functions.extend(_get_repo_functions([fp], _supported_file_extensions(), nodes_to_extract))

    print('Embedding {} functions in {} batches. This is done once and cached in .embeddings'.format(
        len(functions), int(np.ceil(len(functions)/args.batch_size))))
    corpus_embeddings = model.encode(
        [f['text'] for f in functions], convert_to_tensor=True, show_progress_bar=True, batch_size=args.batch_size)

    dataset['functions'].extend(functions)
    dataset['embeddings'] = torch.cat((dataset['embeddings'].to('cpu'), corpus_embeddings.to('cpu')), 0)

    with gzip.open(args.path_to_repo + '/' + '.embeddings', 'w') as f:
        f.write(pickle.dumps(dataset))
    
    with open(os.path.join(args.path_to_repo, '.embeddings.meta'), "w") as f:
        f.write(current_commit)
    return dataset
