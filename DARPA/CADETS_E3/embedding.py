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


CMD_CONTEXT_KEYS = (
    "cmd",
    "command",
    "argv",
    "arg",
    "subject",
    "process",
    "exe",
)

PATH_CONTEXT_KEYS = (
    "file",
    "path",
    "netflow",
    "directory",
    "dir",
    "ip",
)


def _separate_context_parts(node_context):
    cmd_parts = []
    path_parts = []

    if isinstance(node_context, dict):
        for key, value in node_context.items():
            if value is None:
                continue
            key_lower = str(key).lower()
            value_str = value if isinstance(value, str) else str(value)
            if any(tag in key_lower for tag in CMD_CONTEXT_KEYS):
                cmd_parts.append(value_str)
            if any(tag in key_lower for tag in PATH_CONTEXT_KEYS):
                path_parts.append(value_str)
        if not cmd_parts and not path_parts and node_context:
            fallback = " ".join(str(v) for v in node_context.values() if v)
            if fallback:
                cmd_parts.append(fallback)
    elif node_context is not None:
        value_str = node_context if isinstance(node_context, str) else str(node_context)
        cmd_parts.append(value_str)

    return cmd_parts, path_parts


def _build_context_tensors(parts):
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

def _context_record_to_mapping(node_type, node_msg):
    """Build a lightweight mapping the tokenizer understands from SQL rows."""
    if node_type is None and node_msg is None:
        return {}

    key = str(node_type).lower() if node_type is not None else "context"
    value = node_msg if node_msg is not None else ""
    return {key: value}


def gen_vectorized_graphs(cur, node2higvec, rel2vec, logger):
    for day in tqdm(range(2, 14)):
        start_timestamp = datetime_to_ns_time_US('2018-04-' + str(day) + ' 00:00:00')
        end_timestamp = datetime_to_ns_time_US('2018-04-' + str(day + 1) + ' 00:00:00')
        sql = """
        SELECT
            e.src_index_id::bigint AS src_index_id,
            e.dst_index_id::bigint AS dst_index_id,
            e.operation,
            e.timestamp_rec,
            src_meta.node_type   AS src_type,
            src_meta.msg         AS src_msg,
            dst_meta.node_type   AS dst_type,
            dst_meta.msg         AS dst_msg
        FROM event_table e
        JOIN node2id src_meta ON src_meta.index_id = e.src_index_id::bigint
        JOIN node2id dst_meta ON dst_meta.index_id = e.dst_index_id::bigint
        WHERE e.timestamp_rec > '%s' AND e.timestamp_rec < '%s'
        ORDER BY e.timestamp_rec;
        """ % (start_timestamp, end_timestamp)
        cur.execute(sql)
        events = cur.fetchall()
        logger.info(f'2018-04-{day}, events count: {len(events)}')
        edge_list = []
        for e in events:
            edge_temp = [int(e[0]), int(e[1]), e[2], e[3], e[4], e[5], e[6], e[7]]
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
        src_cmd_tokens = []
        src_cmd_mask = []
        src_path_tokens = []
        src_path_mask = []
        dst_cmd_tokens = []
        dst_cmd_mask = []
        dst_path_tokens = []
        dst_path_mask = []

        for i in edge_list:
            src_idx = int(i[0])
            dst_idx = int(i[1])
            src.append(src_idx)
            dst.append(dst_idx)
            msg.append(
                torch.cat([torch.from_numpy(node2higvec[src_idx]), rel2vec[i[2]], torch.from_numpy(node2higvec[dst_idx])]))
            t.append(int(i[3]))

            src_context = _context_record_to_mapping(i[4], i[5])
            dst_context = _context_record_to_mapping(i[6], i[7])

            src_cmd_parts, src_path_parts = _separate_context_parts(src_context)
            dst_cmd_parts, dst_path_parts = _separate_context_parts(dst_context)

            src_cmd_tensor, src_cmd_mask_tensor = _build_context_tensors(src_cmd_parts)
            src_path_tensor, src_path_mask_tensor = _build_context_tensors(src_path_parts)
            dst_cmd_tensor, dst_cmd_mask_tensor = _build_context_tensors(dst_cmd_parts)
            dst_path_tensor, dst_path_mask_tensor = _build_context_tensors(dst_path_parts)

            src_cmd_tokens.append(src_cmd_tensor)
            src_cmd_mask.append(src_cmd_mask_tensor)
            src_path_tokens.append(src_path_tensor)
            src_path_mask.append(src_path_mask_tensor)
            dst_cmd_tokens.append(dst_cmd_tensor)
            dst_cmd_mask.append(dst_cmd_mask_tensor)
            dst_path_tokens.append(dst_path_tensor)
            dst_path_mask.append(dst_path_mask_tensor)

        if not (
            len(src)
            == len(dst)
            == len(t)
            == len(src_cmd_tokens)
            == len(dst_cmd_tokens)
            == len(src_path_tokens)
            == len(dst_path_tokens)
        ):
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
        dataset.src_cmd_tokens = torch.stack(src_cmd_tokens)
        dataset.src_cmd_mask = torch.stack(src_cmd_mask)
        dataset.src_path_tokens = torch.stack(src_path_tokens)
        dataset.src_path_mask = torch.stack(src_path_mask)
        dataset.dst_cmd_tokens = torch.stack(dst_cmd_tokens)
        dataset.dst_cmd_mask = torch.stack(dst_cmd_mask)
        dataset.dst_path_tokens = torch.stack(dst_path_tokens)
        dataset.dst_path_mask = torch.stack(dst_path_mask)
        dataset.context_event_index = torch.arange(dataset.t.numel(), dtype=torch.long)
        dataset.src = dataset.src.to(torch.long)
        dataset.dst = dataset.dst.to(torch.long)
        dataset.msg = dataset.msg.to(torch.float)
        dataset.t = dataset.t.to(torch.long)
        dataset.src_cmd_tokens = dataset.src_cmd_tokens.to(torch.long)
        dataset.src_path_tokens = dataset.src_path_tokens.to(torch.long)
        dataset.dst_cmd_tokens = dataset.dst_cmd_tokens.to(torch.long)
        dataset.dst_path_tokens = dataset.dst_path_tokens.to(torch.long)
        dataset.src_cmd_mask = dataset.src_cmd_mask.to(torch.bool)
        dataset.src_path_mask = dataset.src_path_mask.to(torch.bool)
        dataset.dst_cmd_mask = dataset.dst_cmd_mask.to(torch.bool)
        dataset.dst_path_mask = dataset.dst_path_mask.to(torch.bool)
        dataset.context_event_index = dataset.context_event_index.to(torch.long)
        torch.save(dataset, graphs_dir + "/graph_4_" + str(day) + ".TemporalData.simple")

if __name__ == "__main__":
    logger.info("Start logging.")

    os.system(f"mkdir -p {graphs_dir}")

    cur, _ = init_database_connection()
    nodeid2msg = gen_nodeid2msg(cur=cur)
    node2higvec = gen_feature(cur=cur, nodeid2msg=nodeid2msg)
    rel2vec = gen_relation_onehot()
    gen_vectorized_graphs(cur=cur, node2higvec=node2higvec, rel2vec=rel2vec, logger=logger)

