"""
Not sure whether to name training data BLINK or KILT.
"""
import argparse
import collections
import csv
import json
import logging


logger = logging.getLogger(__name__)


def main(args):
    entity_ids_train = collections.Counter()
    entity_ids_dev = collections.Counter()

    logger.info('Processing train data')
    with open(args.train, 'r') as f, \
         open(args.prefix + 'train.jsonl', 'w') as g:
        for line in f:
            data = json.loads(line)
            entity_id = data['output'][0]['answer']
            json_obj = {
                'left_context': data['meta']['left_context'],
                'right_context': data['meta']['right_context'],
                'mention': data['meta']['mention'],
                'entity_id': entity_id,
            }
            g.write(json.dumps(json_obj) + '\n')
            entity_ids_train[entity_id] += 1

    logger.info('Processing dev data')
    with open(args.dev, 'r') as f, \
         open(args.prefix + 'dev.jsonl', 'w') as g:
        for line in f:
            data = json.loads(line)
            entity_id = data['output'][0]['answer']
            json_obj = {
                'left_context': data['meta']['left_context'],
                'right_context': data['meta']['right_context'],
                'mention': data['meta']['mention'],
                'entity_id': entity_id,
            }
            g.write(json.dumps(json_obj) + '\n')
            entity_ids_dev[entity_id] += 1

    logger.info('Serializing entity vocab')
    counts = sorted(entity_ids_train.items(), key=lambda x: x[1], reverse=True)
    with open(args.prefix + 'entity_vocab.txt', 'w') as g:
        writer = csv.writer(g)
        writer.writerow(('[PAD]', 0))
        for entity_id, count in counts:
            writer.writerow((entity_id, count))

    unseen = set(entity_ids_dev.keys()) - set(entity_ids_train.keys())
    logger.info('Number of unseen dev entities: %s', len(unseen))
    logger.info('Number of total dev entities: %s', len(entity_ids_dev))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='Training data')
    parser.add_argument('--dev', type=str, help='Dev data')
    parser.add_argument('-o', '--prefix', type=str,
                        help='Output prefix.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)

