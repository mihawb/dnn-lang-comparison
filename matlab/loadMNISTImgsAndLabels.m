function [imgs, labels] = loadMNISTImgsAndLabels(imgs_path,labels_path)

%% loading in files
i_hand = fopen(imgs_path);
l_hand = fopen(labels_path);
fseek(i_hand, 4, 'bof'); % skipping "magic" numbers
fseek(l_hand, 4, 'bof');

%% retrieving necessary dimension...
n_imgs = fread(i_hand, 1, 'int32', 0, 'b');
% ...and skipping the rest
fseek(i_hand, 8, 'cof');
fseek(l_hand, 4, 'cof');

%% parsing 
imgs = fread(i_hand, inf, 'uint8');
imgs = double(255 - imgs) / 255;
imgs = reshape(imgs, 28, 28, 1, n_imgs);
imgs = permute(imgs, [2 1 3 4]);

labels = fread(l_hand, inf, 'uint8');
labels = categorical(labels);

%% free 
fclose(i_hand);
fclose(l_hand);

end

 