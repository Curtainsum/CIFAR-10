from _sha1 import sha1
from . import cache
#from cifar import newcifar


CACHED = True


def get_result(img_data):
    hash = get_hash(img_data)
    if CACHED:
        return do_predict(img_data)
    else:
        result, anm, vhc = cache.check_exists(hash)
        if not result:
            result, anm, vhc = do_predict(img_data)
            cache.add_result(hash, anm, vhc)
        return result, anm, vhc


def get_hash(data):
    sha1_obj = sha1()
    sha1_obj.update(data)
    hash = sha1_obj.hexdigest()
    print(hash)
    return hash


def do_predict(img_data):
    return True,0.7 ,0.6
