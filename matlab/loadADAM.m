function [x_train, y_train, x_test, y_test] = loadADAM(root, cutoff)

fovea_df = readtable(root + "fovea_location.csv");
len = numel(fovea_df.imgName);
cutoff = floor(cutoff * len);
adam_imgs = double.empty(256,256,3,0);

for img_idx = 1:len
    img_name = fovea_df.imgName{img_idx};
    if img_name(1) == 'A'
        img = imread(root + "AMD/" + img_name);
    else
        img = imread(root + "Non-AMD/" + img_name);
    end

    img = im2double(img);
    adam_imgs(:,:,:,img_idx) = img;
end

x_train = adam_imgs(:,:,:,1:cutoff);
x_test = adam_imgs(:,:,:,cutoff:len);

y_all = [fovea_df.Fovea_X(:), fovea_df.Fovea_Y(:)];
y_train = y_all(1:cutoff,:);
y_test = y_all(cutoff:len,:);
% transposition here might be necessary but well see

end

