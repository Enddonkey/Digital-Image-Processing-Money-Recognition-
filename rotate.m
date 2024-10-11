function angle=rotate(img,num)
%读取图像
img1=rgb2gray(img);
%figure(1);imshow(img1);title('原图');
grayimg=medfilt2(img1,[5,5]);
%figure(2);imshow(grayimg);title('滤波图');
%%
%图像的分割
BW = edge(grayimg, 'canny');                     %3.利用边缘检测，减少干扰
%figure(3);imshow(BW);title('调用Matlab的graythresh函数分割结果');

%形态学处理
BW1=imclose(BW,strel('disk',5));
BW2=bwareaopen(BW1,100);
BW3=imfill(BW2,'holes');
%figure(4);imshow(BW3);title('删除结果');

[L, num] = bwlabel(BW3);
stats = regionprops(L,'Area');
[b,index]=sort([stats.Area],'descend');
if length(stats)<3
    bw=L;
else
    bw=ismember(L,index(1:3));
end

%%
% 应用连通域分析进行图像分割
connectedComponents = bwconncomp(bw);
% 获取每个连通域的像素索引
pixelIdxList = connectedComponents.PixelIdxList;
% 遍历连通域并显示每个区域
for i = 1:length(pixelIdxList)
    regionImage = zeros(size(bw));
    regionImage(pixelIdxList{i}) = 1;
    Img = edge(regionImage, 'canny'); 
    figure;
    imshow(Img);
    theta = 1:180;                              %4.theta就是要投影方向的角度
    [R,xp] = radon(Img,theta);          %5.沿某个方向theta做radon变换，结果是向量
    %所得R(p,alph)矩阵的每一个点为对I3基于（p,alph）的线积分,其每一个投影的方向对应一个列向量
    [r,c] = find(R>=max(max(R)));  %检索矩阵R中最大值所在位置，提取行列标 
    % max(R)找出每个角度对应的最大投影角度 然在对其取最大值，即为最大的倾斜角即90度
    J=c;  %由于R的列标就是对应的投影角度
    angle(i)=90-c; %计算倾斜角
    rotatedimg = imrotate(img,angle(i),'bilinear','crop');        %4.图像进行位置矫正
    %取值为负值向右旋转 并选区双线性插值 并输出同样尺寸的图像
%     figure;
%     imshow(rotatedimg);
    title(['区域 ', num2str(i)]);
end

