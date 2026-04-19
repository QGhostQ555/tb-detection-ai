import os, hashlib

def file_hash(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def hashes_in_dir(root):
    hashes = {}
    for cls in ['TB','NORMAL']:
        d = os.path.join(root, cls)
        for f in os.listdir(d):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                hashes[p] = file_hash(p)
    return hashes

train_h = hashes_in_dir('../data/train')
test_h  = hashes_in_dir('../data/test')

train_hashes = set(train_h.values())
test_hashes  = set(test_h.values())

dups = train_hashes.intersection(test_hashes)
print("Duplicados entre train y test:", len(dups))