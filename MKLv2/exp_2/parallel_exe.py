from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from subprocess import call

def run(eid):
    
    train_data_root = '/home/doug919/projects/data/MKLv2/2000samples_4/train'
    test_data_root = '/home/doug919/projects/data/MKLv2/2000samples_4/test_8000'
    train_data_tag = '800p800n_Xy'
    test_data_tag = 'full.Xy'
    output_prefix = 'Thread%d_E22_8000' % (eid)
    nclass_neg = 39;

    cmd = 'matlab -r "addpath(\'../common\');mklv2_load_seed;mklv2_exp_2(%d, \'%s\', {\'TFIDF\', \'keyword\'}, \'%s\', \'%s\', \'%s\', \'%s\', %d, 10);exit;" > log/log_thread_%d' % \
        (eid, output_prefix, train_data_root, test_data_root, train_data_tag, test_data_tag, nclass_neg, eid)
    #cmd = 'matlab -r "addpath(\'../common\');mklv2_load_seed;mklv2_exp_2(%d, \'%s\', {\'image_rgba_gist\', \'image_rgba_phog\'}, \'%s\', \'%s\', \'%s\', \'%s\', %d, 10);exit;" > log/log_thread_%d' % \
    #    (eid, output_prefix, train_data_root, test_data_root, train_data_tag, test_data_tag, nclass_neg, eid)

    print '> run:',cmd
    call(cmd, shell=True)

if __name__ == "__main__":

    eids = range(1, 21) 
    #eids = range(21, 41) 
    pool = ThreadPool(len(eids))
    res = pool.map(run, eids)
    pool.close()
    pool.join()

