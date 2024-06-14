%Input: 二值化的图像， Output:方向{-1,1}表示左右
function direction = judgeDirection(pic)
    s = size(pic);
    len = s(1,2);
    half_len = len/2;
    left_pic = pic(1:s(1,1)/2,1:half_len);
    right_pic = pic(1:s(1,1)/2,half_len+1:len);
    left_area = sum(sum(left_pic));
    right_area = sum(sum(right_pic));
    if left_area > right_area
        direction = -1;
    else 
        direction = 1;
    end
end

