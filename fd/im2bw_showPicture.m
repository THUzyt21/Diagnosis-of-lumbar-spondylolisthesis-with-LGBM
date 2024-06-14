file_search = [pwd '/*.jpg'];
dirData = dir(file_search)
for i = 900: 920
    img = imread(dirData(i).name);
    grayed = rgb2gray(img);
    histeqed = histeq(grayed);
    figure
    subplot(131)
    imshow(grayed)
    title(dirData(i).name);
    subplot(132)
    imshow(histeqed)
    bwimg = im2bw(histeqed, graythresh(histeqed));
    %图像闭运算去除小的噪点
    se = strel('disk',100);
    imclosed = imclose(bwimg, se);
    subplot(133)
    imshow(imclosed);
    label = judgeDirection(imclosed);
    title(label);
end