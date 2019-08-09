% this matlab script creates grayscale images the Matlab way.
% This is done to have perfect comparability to older works which use
% Matlab, and especially "Beyond a Gaussian Denoiser", Zhang et al. '17
% 
% This is not strictly necessary, only if you want to use the training set
% trainingZhang or the test set testing68Zhang which can also be found at
% https://github.com/cszn/DnCNN/tree/master/testsets
%
% The effect on PSNR in comparison to the PIL RGB->Gray conversion is
% measurable, but less than 0.1 PSNR and not necessarily better.

path = 'data/BSDS300/images/train/';
path_new = 'data/BSDS300/images/train_gray_matlab/';

img_list = dir(path) %#ok<NOPTS>

for i=3:length(img_list)
    img = imread([path, img_list(i).name]);
    img_gray = rgb2gray(img);
    imwrite(img_gray, [path_new, img_list(i).name(1:end-4),'.png']);
end