function [celeba_array] = loadCELEBA(root,img_size,limit)

celeba_files = dir(root);
celeba_files = string({celeba_files(3:end).name});
index = 1;
celeba_array = double.empty(img_size,img_size,3,0);

for img_name = celeba_files

    img = imread(root + img_name);
    img = im2double(img);
    resized = imresize(img, [img_size img_size]);
    celeba_array(:,:,:,index) = resized;
    index = index + 1;

    if (mod(index, 1000) == 0)
        fprintf("%d/%d (%f%%)\n", index, limit, index/limit);
    end

    % hopefully only for debug
    if (index > limit)
        break
    end
end

end

