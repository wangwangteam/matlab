clear;
% ����ѵ����
traindir='E:\cifar-100\RCNNnet\flowers\train\';
[filenames,labels,data]=makedata(traindir);
batch_label=char('training batch 1 of 1');
save train.mat batch_label labels data filenames;
% ���ɲ��Լ�
testdir='E:\cifar-100\RCNNnet\flowers\test\';
[filenames,labels,data]=makedata(testdir);
batch_label=char('training batch 1 of 1');
save test.mat batch_label labels data filenames;
% �Զ�����𣬲�����˵���ļ�
label_names={'daisy','dandelion','rose','sunflower','tulip'}';
save meta.mat label_names