# -*- coding: utf-8 -*-

def connect_mongo(mongo_addr='doraemon.iis.sinica.edu.tw', db_name='LJ40K'):
    import pymongo
    db = pymongo.Connection(mongo_addr)[db_name]
    return db

def toNumber(string, **kwargs):
    """
    Convert a string to number
    string:
        <str>, input string
    kwargs:
        NaN:
            <int>/<float>, convert "NaN" to certain value
        toFloat:
            "auto"/True/False, set auto to detect "." in the input string automatically
    """

    NaN = -1 if 'NaN' not in kwargs else kwargs['NaN']
    toFloat = "auto" if 'toFloat' not in kwargs else kwargs['toFloat']

    string = string.strip()
    if "nan" in string.lower():
        return NaN
    elif toFloat == True or toFloat == "auto" and "." in string:
        return float(string)
    elif toFloat == False:
        return int(string)

def find_missing(root):
    """
    find documents without features in emotion-image output
    Parameter:
        root: the root path of the nested folders
    Returns:
        a dict containing missing documents
    """
    import os
    from collections import defaultdict
    Missing = {}
    for i, folder in enumerate(filter(lambda x:x[0]!='.', os.listdir(root))):
        files = set(map(lambda x: int(x.split('/')[-1].split('.')[0]), os.listdir( os.path.join(root, folder) )))

        ## global     local
        ## -----------------
        ## 0~999      0~999
        ## 1000~1999  0~999
        ## ...        ...
        label = folder
        Missing[label] = defaultdict(list)
        for (_local_id, _global_id) in enumerate(range(i*1000,(i+1)*1000)):
             if _global_id not in files:
                Missing[label]['global'].append(_global_id)
                Missing[label]['local'].append(_local_id)
    return Missing

def load_csv(path, **kwargs):

    LINE = "\n" if 'LINE' not in kwargs else kwargs['LINE']
    ITEM = "," if 'ITEM' not in kwargs else kwargs['ITEM']
    number = True if 'number' not in kwargs else kwargs['number']
 
    doc = open(path).read()
    lines_raw = doc.strip().split(LINE)

    if number:
        lines = [ map(lambda x: toNumber(x, **kwargs), line.split(ITEM) ) for line in lines_raw]
    else:
        lines = [line.split(ITEM) for line in lines_raw]
    return lines

def amend(missing, input_path, output_path, **kwargs):
    import os
    # e.g.,
    # input_path = '/Users/Maxis/projects/emotion-detection-modules/dev/image/emotion_imgs_threshold_1x1_rbga_out/out_f1'
    # output_path = '/Users/Maxis/projects/emotion-detection-modules/dev/image/emotion_imgs_threshold_1x1_rbga_out_amend/out_f1'
    """
    input_path: <str: folder> 
    output_path: <str: folder>
    """
    for fn in os.listdir(input_path):

        fn_path = os.path.join(input_path, fn)
        out_path = os.path.join(output_path, fn)
        ## get label
        label, feature_name = fn_path.split('/')[-1].split('.')[0].split('_')

        lines = load_csv(fn_path, **kwargs)

        new_lines = []
        for line in lines:
            for idx in missing[label]['local']:
                line.insert(idx, 'NaN')
            new_lines.append( line )

        content = LINE.join([ ITEM.join(nl) for nl in new_lines])

        with open(out_path, 'w') as fw:
            fw.write(content)



        




