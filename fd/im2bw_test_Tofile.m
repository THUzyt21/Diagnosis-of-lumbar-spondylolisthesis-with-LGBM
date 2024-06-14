file_search = [pwd '/*.jpg'];
dirData = dir(file_search)
fid = fopen('./diction.txt','w+');
for i = 1: length(dirData)
    img = imread(dirData(i).name);
    bwimg = im2bw(rgb2gray(img), graythresh(img));
    %图像开运算去除小的噪点
    se = strel('disk',100);
    imclosed = imclose(bwimg, se);

    label = judgeDirection(imclosed);

    fprintf(fid,[dirData(i).name, '|',num2str(label), '\n']);
end