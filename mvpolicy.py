import os
import pickle
import tensorflow as tf

def mvfiles(itr, num, pde, eig):
    filepathpre = './experiments/lto'
    if eig < 0 and pde > 0:
        datafile = '/data_files_pde/'
    elif eig > 0 and pde < 0:
        datafile = '/data_files_eig/'
    filepath1 = filepathpre + datafile + 'policy_itr_' + '%02d' % itr + '_tf_data.ckpt.data-00000-of-00001'
    filepath2 = filepathpre + datafile + 'policy_itr_' + '%02d' % itr + '_tf_data.ckpt.index'
    filepath3 = filepathpre + datafile + 'policy_itr_' + '%02d' % itr + '_tf_data.ckpt.meta'
    filepath1to = filepathpre + datafile + 'policy_itr_' + '%02d' % itr + '_tf_data_' + '%02d' % num + '.ckpt.data-00000-of-00001'
    filepath2to = filepathpre + datafile + 'policy_itr_' + '%02d' % itr + '_tf_data_' + '%02d' % num + '.ckpt.index'
    filepath3to = filepathpre + datafile + 'policy_itr_' + '%02d' % itr + '_tf_data_' + '%02d' % num + '.ckpt.meta'
    os.system('mv ' + ' ' + filepath1 + ' ' + filepath1to)
    os.system('mv ' + ' ' + filepath2 + ' ' + filepath2to)
    os.system('mv ' + ' ' + filepath3 + ' ' + filepath3to)
    

    print 'files renaming completed.'

def main():
    itr = 7
    num = 1
    pde = -1.0
    eig = 1.0
    mvfiles(itr, num, pde ,eig)

if __name__ == "__main__":
    main()
