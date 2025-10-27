from sklearn.feature_extraction import FeatureHasher
from torch_geometric.data import *
from tqdm import tqdm

import hashlib
import numpy as np
import logging
import os
import re
import torch

from config import *
from kairos_utils import *

# Setting for logging
logger = logging.getLogger("embedding_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + 'embedding.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def path2higlist(p):
    l=[]
    spl=p.strip().split('/')
    for i in spl:
        if len(l)!=0:
            l.append(l[-1]+'/'+i)
        else:
            l.append(i)
    return l

def ip2higlist(p):
    l=[]
    spl=p.strip().split('.')
    for i in spl:
        if len(l)!=0:
            l.append(l[-1]+'.'+i)
        else:
            l.append(i)
    return l

def list2str(l):
    s=''
    for i in l:
        s+=i
    return s

TOKEN_SPLIT_PATTERN = re.compile(r"[^A-Za-z0-9]+")


def _tokenize_context(parts):
    tokens = []
    for part in parts:
        if part is None:
            continue
        if not isinstance(part, str):
            part = str(part)
        for token in TOKEN_SPLIT_PATTERN.split(part):
            if token:
                tokens.append(token.lower())
    if not tokens:
        tokens.append("<unk>")
    return tokens


def _hash_token(token):
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest()
    return (int(digest, 16) % (context_vocab_size - 1)) + 1 if context_vocab_size > 1 else 0


def _build_context_tensors(nodeid2msg, node_idx):
    context = nodeid2msg.get(node_idx, {})
    parts = []
    if isinstance(context, dict) and context:
        for key, value in context.items():
            parts.append(key)
            parts.append(value)
    elif context is not None:
        parts.append(context)
    tokens = _tokenize_context(parts)

    token_tensor = torch.zeros(context_max_seq_len, dtype=torch.long)
    mask_tensor = torch.zeros(context_max_seq_len, dtype=torch.bool)

    for idx, token in enumerate(tokens[:context_max_seq_len]):
        token_tensor[idx] = _hash_token(token)
        mask_tensor[idx] = True

    return token_tensor, mask_tensor


def gen_feature(cur, nodeid2msg=None):
    # Firstly obtain all node labels
    if nodeid2msg is None:
        nodeid2msg = gen_nodeid2msg(cur=cur)

    # Construct the hierarchical representation for each node label
    node_msg_dic_list = []
    for i in tqdm(nodeid2msg.keys()):
        if type(i) == int:
            if 'netflow' in nodeid2msg[i].keys():
                higlist = ['netflow']
                higlist += ip2higlist(nodeid2msg[i]['netflow'])

            if 'file' in nodeid2msg[i].keys():
                higlist = ['file']
                higlist += path2higlist(nodeid2msg[i]['file'])

            if 'subject' in nodeid2msg[i].keys():
                higlist = ['subject']
                higlist += path2higlist(nodeid2msg[i]['subject'])
            node_msg_dic_list.append(list2str(higlist))

    # Featurize the hierarchical node labels
    FH_string = FeatureHasher(n_features=node_embedding_dim, input_type="string")
    node2higvec=[]
    for i in tqdm(node_msg_dic_list):
        vec=FH_string.transform([i]).toarray()
        node2higvec.append(vec)
    node2higvec = np.array(node2higvec).reshape([-1, node_embedding_dim])
    torch.save(node2higvec, artifact_dir + "node2higvec")
    return node2higvec

def gen_relation_onehot():
    relvec=torch.nn.functional.one_hot(torch.arange(0, len(rel2id.keys())//2), num_classes=len(rel2id.keys())//2)
    rel2vec={}
    for i in rel2id.keys():
        if type(i) is not int:
            rel2vec[i]= relvec[rel2id[i]-1]
            rel2vec[relvec[rel2id[i]-1]]=i
    torch.save(rel2vec, artifact_dir + "rel2vec")
    return rel2vec

def gen_vectorized_graphs(cur, node2higvec, rel2vec, nodeid2msg, logger):
    for day in tqdm(range(2, 14)):
        start_timestamp = datetime_to_ns_time_US('2018-04-' + str(day) + ' 00:00:00')
        end_timestamp = datetime_to_ns_time_US('2018-04-' + str(day + 1) + ' 00:00:00')
        sql = """
        select * from event_table
        where
              timestamp_rec>'%s' and timestamp_rec<'%s'
               ORDER BY timestamp_rec;
        """ % (start_timestamp, end_timestamp)
        cur.execute(sql)
        events = cur.fetchall()
        logger.info(f'2018-04-{day}, events count: {len(events)}')
        edge_list = []
        for e in events:
            edge_temp = [int(e[1]), int(e[4]), e[2], e[5]]
            if e[2] in include_edge_type:
                edge_list.append(edge_temp)
        logger.info(f'2018-04-{day}, edge list len: {len(edge_list)}')
        if not edge_list:
            logger.info(f'2018-04-{day}, no edges selected, skipping dataset serialization.')
            continue
        dataset = TemporalData()
        src = []
        dst = []
        msg = []
        t = []
        src_context_tokens = []
        src_context_mask = []
        dst_context_tokens = []
        dst_context_mask = []

        for i in edge_list:
            src_idx = int(i[0])
            dst_idx = int(i[1])
            src.append(src_idx)
            dst.append(dst_idx)
            msg.append(
                torch.cat([torch.from_numpy(node2higvec[src_idx]), rel2vec[i[2]], torch.from_numpy(node2higvec[dst_idx])]))
            t.append(int(i[3]))

            src_tokens, src_mask = _build_context_tensors(nodeid2msg, src_idx)
            dst_tokens, dst_mask = _build_context_tensors(nodeid2msg, dst_idx)
            src_context_tokens.append(src_tokens)
            src_context_mask.append(src_mask)
            dst_context_tokens.append(dst_tokens)
            dst_context_mask.append(dst_mask)

        if not (len(src) == len(dst) == len(t) == len(src_context_tokens) == len(dst_context_tokens)):
            raise RuntimeError(
                "Temporal feature alignment failed: mismatched lengths between structural and context sequences"
            )

        t_tensor = torch.tensor(t, dtype=torch.long)
        if t_tensor.numel() > 1 and not torch.all(t_tensor[1:] >= t_tensor[:-1]):
            logger.warning(
                "Events for day %s were not strictly non-decreasing in timestamp after ordering; "
                "downstream temporal alignment may be affected.",
                day,
            )

        dataset.src = torch.tensor(src)
        dataset.dst = torch.tensor(dst)
        dataset.t = t_tensor
        dataset.msg = torch.vstack(msg)
        dataset.src_context_tokens = torch.stack(src_context_tokens)
        dataset.src_context_mask = torch.stack(src_context_mask)
        dataset.dst_context_tokens = torch.stack(dst_context_tokens)
        dataset.dst_context_mask = torch.stack(dst_context_mask)
        dataset.context_event_index = torch.arange(dataset.t.numel(), dtype=torch.long)
        dataset.src = dataset.src.to(torch.long)
        dataset.dst = dataset.dst.to(torch.long)
        dataset.msg = dataset.msg.to(torch.float)
        dataset.t = dataset.t.to(torch.long)
        dataset.src_context_tokens = dataset.src_context_tokens.to(torch.long)
        dataset.dst_context_tokens = dataset.dst_context_tokens.to(torch.long)
        dataset.src_context_mask = dataset.src_context_mask.to(torch.bool)
        dataset.dst_context_mask = dataset.dst_context_mask.to(torch.bool)
        dataset.context_event_index = dataset.context_event_index.to(torch.long)
        torch.save(dataset, graphs_dir + "/graph_4_" + str(day) + ".TemporalData.simple")

if __name__ == "__main__":
    logger.info("Start logging.")

    os.system(f"mkdir -p {graphs_dir}")

    cur, _ = init_database_connection()
    nodeid2msg = gen_nodeid2msg(cur=cur)
    node2higvec = gen_feature(cur=cur, nodeid2msg=nodeid2msg)
    rel2vec = gen_relation_onehot()
    gen_vectorized_graphs(cur=cur, node2higvec=node2higvec, rel2vec=rel2vec, nodeid2msg=nodeid2msg, logger=logger)

