import numpy as np
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import evaluate.linear_probe.liblinearsvm.liblinearutil as liblinearsvm
import json


def main():
    parser = argparse.ArgumentParser('svm_perf')
    parser.add_argument('--output-dir', type=str, default='output/eval_output_linear')
    parser.add_argument('--num_replica', type=int, default=4)
    parser.add_argument('--cost', type=float, default=1.0)
    args = parser.parse_args()

    # check feature files
    for i in range(args.num_replica):
        os.path.exists(os.path.join(args.output_dir, 'feature_train_{}.npy'.format(i)))
        os.path.exists(os.path.join(args.output_dir, 'feature_train_cls_{}.npy'.format(i)))
        os.path.exists(os.path.join(args.output_dir, 'feature_test_{}.npy'.format(i)))
        os.path.exists(os.path.join(args.output_dir, 'feature_test_cls_{}.npy'.format(i)))
        os.path.exists(os.path.join(args.output_dir, 'vid_num.npy'))

    # load feature index
    vid_num_train = np.load(os.path.join(args.output_dir, 'vid_num_train.npy'))
    train_padding_num = vid_num_train[0] % args.num_replica
    vid_num_test = np.load(os.path.join(args.output_dir, 'vid_num_test.npy'))
    test_padding_num = vid_num_test[0] % args.num_replica

    # load feature and GT: training set
    feat_train = []
    feat_train_cls = []
    for i in range(args.num_replica):
        feat_train.append(np.load(os.path.join(args.output_dir, 'feature_train_{}.npy'.format(i))))
        feat_train_cls.append(np.load(os.path.join(args.output_dir, 'feature_train_cls_{}.npy'.format(i))))
    if train_padding_num > 0:
        for i in range(train_padding_num, args.num_replica):
            feat_train[i] = feat_train[i][:-1, :]
            feat_train_cls[i] = feat_train_cls[i][:-1]
    feat_train = np.concatenate(feat_train, axis=0).squeeze()
    feat_train_cls = np.concatenate(feat_train_cls, axis=0).squeeze()
    print('feat_train: {}'.format(feat_train.shape))
    print('feat_train_cls: {}'.format(feat_train_cls.shape))

    # load feature and GT: test set
    feat_test = []
    feat_test_cls = []
    for i in range(args.num_replica):
        feat_test.append(np.load(os.path.join(args.output_dir, 'feature_test_{}.npy'.format(i))))
        feat_test_cls.append(np.load(os.path.join(args.output_dir, 'feature_test_cls_{}.npy'.format(i))))

    if test_padding_num > 0:
        for i in range(test_padding_num, args.num_replica):
            feat_test[i] = feat_test[i][:-1, :]
            feat_test_cls[i] = feat_test_cls[i][:-1]
    feat_test = np.concatenate(feat_test, axis=0)
    feat_test_cls = np.concatenate(feat_test_cls, axis=0)
    print('feat_test: {}'.format(feat_test.shape))
    print('feat_test_cls: {}'.format(feat_test_cls.shape))

    # solving SVM
    print('form svm problem')
    svm_problem = liblinearsvm.problem(feat_train_cls, feat_train)
    print('L2-regularized L2-loss support vector classification (primal), cost={}'.format(args.cost))
    svm_parameter = liblinearsvm.parameter('-s 2 -n 32 -c {}'.format(args.cost))
    svm_filename = 'multicore_linearsvm_primal_c{}.svmmodel'.format(args.cost)

    print('train svm')
    svm_model = liblinearsvm.train(svm_problem, svm_parameter)
    print('save svm')
    liblinearsvm.save_model(os.path.join(args.output_dir, svm_filename), svm_model)
    print('eval svm')
    pd_label, pd_acc, pd_test = liblinearsvm.predict(feat_test_cls, feat_test, svm_model)
    eval_acc, eval_mse, eval_scc = liblinearsvm.evaluations(feat_test_cls, pd_label)
    print('{}/{}'.format(pd_acc, eval_acc))


    with open(os.path.join(args.output_dir, svm_filename + '.txt'), 'w') as f:
        f.write('{}/{}'.format(pd_acc, eval_acc))
    print('Done')


if __name__ == '__main__':
    main()