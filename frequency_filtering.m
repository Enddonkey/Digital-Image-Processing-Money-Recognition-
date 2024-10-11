function g = frequency_filtering(img,filter_type,filter_param)
if length(size(img))>2
    img = rgb2gray(img);
end

subplot(1,2,1);
imshow(img);
title('Original image','fontsize',18);

%% choose filter
if strcmpi(filter_type,'ILPF')
    frequency_filter = ILPF(img,filter_param);
end

%% set A, B, P, and Q
A = size(img,1);
B = size(img,2);
P = 2*A;
Q = 2*B;

%% zero padding
img_p = zeros(P,Q);
img_p(1:A,1:B) = double(img);

%% filtering
img_p = img_p.*shift(img_p); % centering
fft_img_p = fft2(img_p);
G = fft_img_p.*frequency_filter;
g_p = real(ifft2(G));
g_p = g_p.*shift(g_p); % de-centering

g = g_p(1:A,1:B);
subplot(1,2,2);
imshow(uint8(g));
title('Filtered image','fontsize',18);
g = uint8(g);

%% shift to the center
function shift_xy = shift(array)
[X,Y] = meshgrid([1:size(array,2)],[1:size(array,1)]);
shift_xy = ((-1).^X).*((-1).^Y);

%% ideal lowpass filter 
function fil = ILPF(img,D0)
A = size(img,1);
B = size(img,2);
P = 2*A;
Q = 2*B;
[x, y] = meshgrid(1:Q, 1:P);
mg = sqrt((x-B).^2 + (y-A).^2);
fil = double(mg<=D0);



