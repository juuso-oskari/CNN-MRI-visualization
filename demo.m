%% Create an image datastore object out of the brain-tumor images
% for linux '/home/juuso/Dropbox/demo'
imds = imageDatastore('/home/juuso/Dropbox/demo/volumes',...
'IncludeSubfolders',true,'FileExtensions',{'.jpg','.png', '.jpeg', '.JPG'},'LabelSource','foldernames');
[imdsTrain,imdsValidation, imdsTest] = splitEachLabel(imds,0.8, 0.1);

%% Set up the CNN, using transfer learning 
%net = vgg16('Weights', 'None'); % The nontrained version
net = vgg16;
%analyzeNetwork(net) % to analyze the chosen network
%% modify input and first convolutional layer to accept grayscale
inputName = 'input';
conv1Name = 'conv1_1';

if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 
% get weights and biases of old conv layer
oldWeights = lgraph.Layers(2).Weights;
oldBias = lgraph.Layers(2).Bias;
% get the old mean for input normalization
oldMean = lgraph.Layers(1).Mean;
% get the old stride and padding
oldStride = lgraph.Layers(2).Stride;
oldPadding = lgraph.Layers(2).Padding;
% get new weigths for new conv layer and new mean for new input layer, by
% averaging
newMean = mean(oldMean, 3);
newWeights = mean(oldWeights, 3);
newConvLayer = convolution2dLayer([3 3], 64, 'NumChannels', 1, 'Name', 'conv1', 'Weights', newWeights, 'Stride', oldStride, 'Padding', oldPadding, 'Bias', oldBias);
lgraph = replaceLayer(lgraph, inputName, imageInputLayer([224 224 1], 'Name', 'data', 'Normalization', 'zerocenter', 'Mean', newMean));
lgraph = replaceLayer(lgraph, conv1Name, newConvLayer);
%% Reduce fc layers into one
numClasses = numel(categories(imdsTrain.Labels));
lrFactor = 1; % learning factor for the fc layers

layers = lgraph.Layers(1:32);
connections = lgraph.Connections(1:31, :);

lgraph = createLgraphUsingConnections(layers,connections);
drop = dropoutLayer('Name', 'drop1');
newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',lrFactor, ...
        'BiasLearnRateFactor',lrFactor);
softmax = softmaxLayer('Name', 'prob');
classLayer = classificationLayer('Name', 'classoutput');
lgraph = addLayers(lgraph, [drop; newLearnableLayer;softmax;classLayer]);
lgraph = connectLayers(lgraph, 'pool5', 'drop1');

%%
analyzeNetwork(lgraph)

%% Resize our images to the input size of the network + augmentation
inputSize = [224 224]; % to get the size of the input image
pixelRange = [-15 15];
scaleRange = [0.85 1.15];
% augmentation (rotating, shifting and scaling) 
% to prevent overfitting and memorization
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandRotation', [-15, 15], ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain, ...
    'DataAugmentation',imageAugmenter, 'ColorPreprocessing', 'rgb2gray');

% no augmentation needed for the validation set
augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation, 'ColorPreprocessing', 'rgb2gray');
augimdsTest = augmentedImageDatastore(inputSize,imdsTest, 'ColorPreprocessing', 'rgb2gray');

%% training options
miniBatchSize = 20;
trainOpts.initLearnRate   = 0.0001;
trainOpts.valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
trainOpts.miniBatchSize   = miniBatchSize; % rounding needed in case offline data augmentation is disabled 
trainOpts.maxEpochs       = 100;

options = trainingOptions('sgdm', ...
    'MiniBatchSize',trainOpts.miniBatchSize, ...
    'MaxEpochs',trainOpts.maxEpochs, ...
    'InitialLearnRate',trainOpts.initLearnRate, ...
    'Shuffle','every-epoch', ... % this handles the case where the mini-batch size doesn't evenly divide the number of training images
    'ValidationData',augimdsValidation, ... % source of validation data to evaluate learning during training
    'ValidationFrequency',trainOpts.valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress',... % display a plot of progress during training
    'ValidationPatience', 10); % 'OutputFcn',@(info)stopIfAccuracyNotImproving(info,10)stop training if val acc not improved in 15 epochs or reached 100%

%% training and testing

[trainednet, info] = trainNetwork(augimdsTrain,lgraph,options);

%% test accuracy on test set
[YPred,probs] = classify(trainednet,augimdsTest);
accuracy = mean(YPred == imdsTest.Labels)
%%
plotconfusion(imdsTest.Labels, YPred)
%% OR (NOT VALID ANYMORE)
% load pretrained network
%load('vgg16_98.mat');
%[YPred,probs] = classify(net,augimdsTest);
%accuracy = mean(YPred == imdsTest.Labels);
%% Visualization of the classification (Grad-CAM)


destFolder = '/home/juuso/Dropbox/texmf/tex/latex/misc/viz';

type gradcam.m
nfig = 5;
idx = find(imdsTest.Labels=='yes');
idx = idx(randperm(length(idx),nfig));
% or the misclassifications
%idx = find(YPred ~= imdsTest.Labels);
%nfig = length(idx);
%%
figure;
for i = 1:nfig
    img = readByIndex(augimdsTest, idx(i)).input{1,1};
    [classfn,score] = classify(trainednet,img);
    %figure;
    %subplot(1,2,1)
    ax = subplot(2,nfig,i);
    imshow(img);
    %imwrite(fullfile(destFolder, strcat('o',i,'.jpg')), img);
    colormap gray
    

    lgraph2 = layerGraph(trainednet);
    %To access the data that GoogLeNet uses for classification, remove its final classification layer.
    lgraph2 = removeLayers(lgraph2, lgraph2.Layers(end).Name);
    %Create a dlnetwork from the layer graph.
    dlnet = dlnetwork(lgraph2);
    %Specify the names of the softmax and feature map layers to use with the Grad-CAM helper function. For the feature map layer, specify either the last ReLU layer with non-singleton spatial dimensions, or the last layer that gathers the outputs of ReLU layers (such as a depth concatenation or an addition layer). If your network does not contain any ReLU layers, specify the name of the final convolutional layer that has non-singleton spatial dimensions in the output. Use the function analyzeNetwork to examine your network and select the correct layers. For GoogLeNet, the name of the softmax layer is 'prob' and the depth concatenation layer is 'inception_5b-output'.
    softmaxName = 'prob';
    featureLayerName = 'relu5_3';
    %To use automatic differentiation, convert the sherlock image to a dlarray.
    dlImg = dlarray(single(img),'SSC');
    %Compute the Grad-CAM gradient for the image by calling dlfeval on the gradcam function.
    [featureMap, dScoresdMap] = dlfeval(@gradcam, dlnet, dlImg, softmaxName, featureLayerName, classfn);
    %Resize the gradient map to the GoogLeNet image size, and scale the scores to the appropriate levels for display.
    gradcamMap = sum(featureMap .* sum(dScoresdMap, [1 2]), 3);
    gradcamMap = extractdata(gradcamMap);
    gradcamMap = rescale(gradcamMap);
    gradcamMap = imresize(gradcamMap, inputSize, 'Method', 'bilinear');
    %Show the Grad-CAM levels on top of the image by using an 'AlphaData' value of 0.5. The 'jet' colormap has deep blue as the lowest value and deep red as the highest.
    subplot(2,nfig, i + nfig)
    %figure
    CAMshow(img, gradcamMap);
    %title(sprintf("predicted class: %s (score: %.2f)\ntrue class: %s ", classfn, score(classfn), imdsTest.Labels(idx(i))), 'FontSize',15);
end
%%



%% Occlusion visualization

nfig = 1;
%idx = randperm(numel(imdsTest.Files),nfig);
idx = find(imdsTest.Labels=='yes');
idx = idx(randperm(length(idx),nfig));
%idx = find(YPred ~= imdsTest.Labels);
%nfig = length(idx);
figure;
for i = 1:nfig
    img = readByIndex(augimdsTest, idx(i)).input{1,1};
    [classfn,score] = classify(net,img);
    subplot(nfig,2,1+(i-1)*2)
    imshow(img);
    title(sprintf("y: %s score: (%.2f)\ntarget y: %s ", classfn, score(classfn), imdsTest.Labels(idx(i))));

    map = occlusionSensitivity(net,img,classfn,'ExecutionEnvironment', 'cpu');
    c = subplot(nfig,2,1+(i-1)*2 + 1);
    imshow(img,'InitialMagnification', 150)
    hold on
    imagesc(map,'AlphaData',0.5)
    colormap(c, 'jet')
    colorbar

    title(sprintf("Occlusion sensitivity (%s)", ...
        classfn))
end

%%

function ret_im = CAMshow(im,CAM)
imSize = size(im);
CAM = imresize(CAM,imSize(1:2));
CAM(CAM<0) = 0;
CAM = normalizeImage(CAM);
%ReLU
%CAM(CAM<0) = 0;
cmap = jet(255).*linspace(0,1,255)';
CAM = ind2rgb(uint8(CAM*255),cmap)*255;

combinedImage = double(im)/2 + CAM;
combinedImage = normalizeImage(combinedImage)*255;
imshow(uint8(combinedImage));
end

function N = normalizeImage(I)
minimum = min(I(:));
maximum = max(I(:));
N = (I-minimum)/(maximum-minimum);
end

