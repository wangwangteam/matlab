clear;
% 生成训练集
traindir='E:\cifar-100\RCNNnet\flowers\train\';
[filenames,labels,data]=makedata(traindir);
batch_label=char('training batch 1 of 1');
save train.mat batch_label labels data filenames;
% 生成测试集
testdir='E:\cifar-100\RCNNnet\flowers\test\';
[filenames,labels,data]=makedata(testdir);
batch_label=char('training batch 1 of 1');
save test.mat batch_label labels data filenames;
% 自定义类别，并声称说明文件
label_names={'daisy','dandelion','rose','sunflower','tulip'}';
save meta.mat label_names