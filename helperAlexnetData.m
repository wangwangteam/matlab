% This is helper class to download and import the CIFAR-10 dataset. The
% dataset is downloaded from:
%
%  https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz
%
% References
% ----------
% Krizhevsky, Alex, and Geoffrey Hinton. "Learning multiple layers of
% features from tiny images." (2009).

classdef helperAlexnetData
    
    methods(Static)
        
        %------------------------------------------------------------------
        function download(url, destination)
            if nargin == 1
                url = 'https://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz';
            end        
            
            unpackedData = fullfile(destination, 'cifar-100-matlab');
            if ~exist(unpackedData, 'dir')
                fprintf('Downloading CIFAR-100 dataset...');     
                untar(url, destination); 
                fprintf('done.\n\n');
            end
        end
        
        %------------------------------------------------------------------
        % Return CIFAR-100 Training and Test data.
        function [XTrain, TTrain, XTest, TTest] = load(dataLocation)         
            location = fullfile(dataLocation, 'AlexNet');
            [XTrain,TTrain] = loadBatchAsFourDimensionalArray(location,'train.mat');
            [XTest, TTest] = loadBatchAsFourDimensionalArray(location, 'test.mat');
                      
        end
    end
end

function [XBatch, TBatch] = loadBatchAsFourDimensionalArray(location, batchFileName)
load(fullfile(location,batchFileName));
XBatch = data';
XBatch = reshape(XBatch, 64,64,3,[]);
XBatch = permute(XBatch, [2 1 3 4]);
TBatch = convertLabelsToCategorical(location,labels);
end

function categoricalLabels = convertLabelsToCategorical(location, integerLabels)
load(fullfile(location,'meta.mat'));
categoricalLabels = categorical(integerLabels, 0:4, label_names);
end

