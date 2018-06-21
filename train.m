% 已经下载好的数据集拿过来
cifar100Data ='E:\cifar-100\';
% 加载CIFAR 100的数据集，helperCIFAR10Data.m需要复制过来才可以使用该函数
[trainingImages, trainingLabels, testImages, testLabels] = helperAlexnetData.load(cifar100Data);
size(trainingImages)
numImageCategories = 5;
categories(trainingLabels)
% Display a few of the training images, resizing them for display.
figure
thumbnails = trainingImages(:,:,:,1:100);
thumbnails = imresize(thumbnails, [64 64]);
montage(thumbnails)
% Create the image input layer for 32x32x3 CIFAR-10 images
[height, width, numChannels, ~] = size(trainingImages);

imageSize = [height width numChannels];
inputLayer = imageInputLayer(imageSize,'name','data')
% Convolutional layer parameters
filterSize = [11 11];
numFilters = 96;

middleLayers = [
% conv1
convolution2dLayer(filterSize,numFilters,'Stride',4, 'Padding', 0,'name','conv_1')
reluLayer('name','relu_1')
crossChannelNormalizationLayer(5,'name','norm_1')
maxPooling2dLayer(3, 'Stride',2,'Padding',0,'name','pool_1')
% conv2
convolution2dLayer(5,256,'Stride',1,'Padding', 2,'name','conv_2')
reluLayer('name','relu_2')
crossChannelNormalizationLayer(5,'name','norm_2')
maxPooling2dLayer(3, 'Stride',2,'Padding',0,'name','pool_2')
% conv3
convolution2dLayer(3,384,'Stride',1, 'Padding', 1,'name','conv_3')
reluLayer('name','relu_2')
% conv4
convolution2dLayer(3,384,'Stride',1, 'Padding', 1,'name','conv_4')
reluLayer('name','relu_4')
% conv5
convolution2dLayer(3,256,'Stride',1, 'Padding', 1,'name','conv_5')
reluLayer('name','relu_5')
maxPooling2dLayer(3, 'Stride',2,'Padding',0,'name','pool_5')
]

finalLayers = [
% fc6
fullyConnectedLayer(1024,'name','fc_6')
reluLayer('name','relu_6')
dropoutLayer(0.5,'name','drop_6')
% fc7
fullyConnectedLayer(1024,'name','fc_7')
reluLayer('name','relu_7')
dropoutLayer(0.5,'name','drop_7')
% fc8
fullyConnectedLayer(numImageCategories,'name','fc_8')
softmaxLayer('name','prob')
classificationLayer('name','output')
]

layers = [
    inputLayer
    middleLayers
    finalLayers
    ]
layers(2).Weights = 0.0001 * randn([[11,11] numChannels 96]);
% Set the network training options
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod',8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs',80, ...
    'MiniBatchSize', 16, ...
    'Plots','training-progress', ...
    'Verbose', true);
% A trained network is loaded from disk to save time when running the
% example. Set this flag to true to train the network.
doTraining = true;

if doTraining
    % Train a network.
    cifar100Net = trainNetwork(trainingImages, trainingLabels, layers, opts);
% else
%     % Load pre-trained detector for the example.
%     load('rcnnStopSigns.mat','cifar100Net')
end
save cifar100.mat cifar100Net;
% Extract the first convolutional layer weights
% w = cifar100Net.Layers(2).Weights;
% 
% % rescale and resize the weights for better visualization
% w = mat2gray(w);
% w = imresize(w, [100 100]);
% 
% figure
% montage(w)
% Run the network on the test set.
YTest = classify(cifar100Net, testImages);

% Calculate the accuracy.
accuracy = sum(YTest == testLabels)/numel(testLabels)
% Load the ground truth data
% data = load('stopSignsAndCars.mat', 'stopSignsAndCars');
% stopSignsAndCars = data.stopSignsAndCars;
% 
% % Update the path to the image files to match the local file system
% visiondata = fullfile(toolboxdir('vision'),'visiondata');
% stopSignsAndCars.imageFilename = fullfile(visiondata, stopSignsAndCars.imageFilename);
% 
% % Display a summary of the ground truth data
% summary(stopSignsAndCars)
% 
% % Only keep the image file names and the stop sign ROI labels
% stopSigns = stopSignsAndCars(:, {'imageFilename','stopSign'});
% 
% % Display one training image and the ground truth bounding boxes
% I = imread(stopSigns.imageFilename{1});
% I = insertObjectAnnotation(I, 'Rectangle', stopSigns.stopSign{1}, 'stop sign', 'LineWidth', 4);
% 
% figure
% imshow(I)
% % A trained detector is loaded from disk to save time when running the
% % example. Set this flag to true to train the detector.
% doTraining = false;
% 
% if doTraining
% 
%     % Set training options
%     options = trainingOptions('sgdm', ...
%         'MiniBatchSize', 128, ...
%         'InitialLearnRate', 1e-3, ...
%         'LearnRateSchedule', 'piecewise', ...
%         'LearnRateDropFactor', 0.1, ...
%         'LearnRateDropPeriod', 100, ...
%         'MaxEpochs', 100, ...
%         'Verbose', true);
% 
%     % Train an R-CNN object detector. This will take several minutes.
%     rcnn = trainRCNNObjectDetector(stopSigns, cifar10Net, options, ...
%     'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange',[0.5 1])
% else
%     % Load pre-trained network for the example.
%     load('rcnnStopSigns.mat','rcnn')
% end
% % Read test image
% testImage = imread('image013.jpg');
% 
% % Detect stop signs
% [bboxes, score, label] = detect(rcnn, testImage, 'MiniBatchSize', 128)
% % Display the detection results
% [score, idx] = max(score);
% 
% bbox = bboxes(idx, :);
% annotation = sprintf('%s: (Confidence = %f)', label(idx), score);
% 
% outputImage = insertObjectAnnotation(testImage, 'rectangle', bbox, annotation);
% 
% figure
% imshow(outputImage)
% % The trained network is stored within the R-CNN detector
% rcnn.Network
% featureMap = activations(rcnn.Network, testImage, 'softmax', 'OutputAs', 'channels');
% 
% % The softmax activations are stored in a 3-D array.
% size(featureMap)
% rcnn.ClassNames
% stopSignMap = featureMap(:, :, 1);
% % Resize stopSignMap for visualization
% [height, width, ~] = size(testImage);
% stopSignMap = imresize(stopSignMap, [height, width]);
% 
% % Visualize the feature map superimposed on the test image.
% featureMapOnImage = imfuse(testImage, stopSignMap);
% 
% figure
% imshow(featureMapOnImage)