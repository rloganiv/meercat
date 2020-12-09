import argparse
import collections
import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import List
from xml.etree import ElementTree


# Train/dev/test splits from Barhom/Cybulska/Kenyan-Dean on topics.
DEV_TOPICS = set((2, 5, 12, 18, 21, 23, 34, 35))
TRAIN_TOPICS = set(range(1, 36)) - DEV_TOPICS


@dataclass
class Token:
    t_id: str
    sentence: str
    number: str
    text: str

    @classmethod
    def from_xml(cls, node):
        return cls(
            t_id=node.attrib['t_id'],
            sentence=node.attrib['sentence'],
            number=node.attrib['number'],
            text=node.text,
        )


@dataclass
class Mention:
    tag: str
    m_id: str
    t_ids: List[str] = field(default_factory=list)

    @classmethod
    def from_xml(cls, node):
        t_ids = [x.attrib['t_id'] for x in node.findall('token_anchor')]
        return cls(
            tag=node.tag,
            m_id=node.attrib['m_id'],
            t_ids=t_ids,
        )


@dataclass
class Relation:
    r_id: str
    m_ids: List[str] = field(default_factory=list)

    @classmethod
    def from_xml(cls, node):
        m_ids = [x.attrib['m_id'] for x in node]
        return cls(
            r_id=node.attrib['r_id'],
            m_ids=m_ids,
        )


@dataclass
class GoldSentence:
    split: str = None
    sentences: List[str] = field(default_factory=list)


def _to_lookup(iterable, key):
    return {key(x): x for x in iterable}


def _is_valid(mention):
    # Non-entity tags
    if 'ACT' in mention.tag or 'NEG' in mention.tag:
        return False
    # Empty-mention
    if not mention.t_ids:
        return False
    return True


def process_xml(
    xml_filename,
    gold_sentences,
):
    tree = ElementTree.parse(xml_filename)
    root = tree.getroot()
    tokens = _to_lookup(
        (Token.from_xml(t) for t in root.iter('token')),
        key=lambda x: x.t_id,
    )
    mentions = _to_lookup(
        (Mention.from_xml(m) for m in root.find('Markables')),
        key=lambda x: x.m_id,
    )
    relations = _to_lookup(
        (Relation.from_xml(r) for r in root.find('Relations')),
        key=lambda x: x.r_id,
    )

    full_text = [t.text for t in tokens.values()]
    for r_id, relation in relations.items():
        for m_id in relation.m_ids:
            mention = mentions[m_id]
            if not _is_valid(mention):
                continue
            token_start = tokens[mention.t_ids[0]]
            token_end = tokens[mention.t_ids[-1]]
            if token_start.sentence in gold_sentences and \
                    token_end.sentence in gold_sentences:
                # REMINDER: Tokens are 1-indexed.
                start_idx = int(token_start.t_id) - 1
                end_idx = int(token_end.t_id)
                left_context = ' '.join(full_text[:start_idx])
                mention = ' '.join(full_text[start_idx:end_idx])
                right_context = ' '.join(full_text[end_idx:])
                yield {
                    'left_context': left_context,
                    'mention': mention,
                    'right_context': right_context,
                    'entity_id': r_id,
                }


def get_split(topic):
    if topic in TRAIN_TOPICS:
        return 'train'
    elif topic in DEV_TOPICS:
        return 'dev'
    else:
        return 'test'


def read_gold_sentences(fname):
    gold_sentences = collections.defaultdict(GoldSentence)
    with open(fname, 'r') as f:
        reader = csv.DictReader(f)
        for line in reader:
            fname = '{Topic}/{Topic}_{File}.xml'.format(**line)
            gold_sentences[fname].split = get_split(int(line['Topic']))
            gold_sentences[fname].sentences.append(line['Sentence Number'])
    return gold_sentences


def main(args):
    entity_ids = collections.Counter()
    assert args.input.is_dir()

    out_files = {
        'train': open(args.output / 'train.jsonl', 'w'),
        'dev': open(args.output / 'dev.jsonl', 'w'),
        'test': open(args.output / 'test.jsonl', 'w'),
    }
    gold_sentences = read_gold_sentences(args.gold_csv)

    for fname, gold_sentence in gold_sentences.items():
        for out_dict in process_xml(args.input / fname, gold_sentence.sentences):
            if gold_sentence.split == 'train':
                entity_ids[out_dict['entity_id']] += 1
            out_files[gold_sentence.split].write(
                json.dumps(out_dict) + '\n'
            )

    with open(args.output / 'entity_vocab.txt', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(('[PAD]', 0))
        for line in entity_ids.items():
            writer.writerow(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gold_csv', type=Path, required=True)
    args = parser.parse_args()

    main(args)

