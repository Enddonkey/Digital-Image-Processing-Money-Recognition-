function g = frequency_filtering_2(img,filter_type,filter_param)
close all;
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

if strcmpi(filter_type,'IHPF')
    frequency_filter = IHPF(img,filter_param);
end

if strcmpi(filter_type,'BLPF')
    frequency_filter = BLPF(img,filter_param);
end

if strcmpi(filter_type,'BHPF')
    frequency_filter = BHPF(img,filter_param);
end

if strcmpi(filter_type,'GLPF')
    frequency_filter = GLPF(img,filter_param);
end

if strcmpi(filter_type,'GHPF')
    frequency_filter = GHPF(img,filter_param);
end

frequency_filter = ifftshift(frequency_filter);
%% filtering
fft_img_p = fft2(img);
G = fft_img_p.*frequency_filter;
g = real(ifft2(G));
subplot(1,2,2);
imshow(uint8(g));
title('Filtered image','fontsize',18);
g = uint8(g);

%% ideal lowpass filter 
function fil = ILPF(img,param)
D0 = param.D0;
A = size(img,1);
B = size(img,2);
[x, y] = meshgrid(1:B, 1:A);
cx = B/2;
cy = A/2;
mg = sqrt((x-cx).^2 + (y-cy).^2);
fil = double(mg<=D0);

%% ideal highpass filter 
function fil = IHPF(img,param)
D0 = param.D0;
A = size(img,1);
B = size(img,2);
[x, y] = meshgrid(1:B, 1:A);
cx = B/2;
cy = A/2;
mg = sqrt((x-cx).^2 + (y-cy).^2);
fil = double(mg>D0);

%% Butterworth lowpass filter 
function fil = BLPF(img,param)
D0 = param.D0;
order = param.order;
A = size(img,1);
B = size(img,2);
[x, y] = meshgrid(1:B, 1:A);
cx = B/2;
cy = A/2;
mg = sqrt((x-cx).^2 + (y-cy).^2);
mg = (mg/D0).^(2*order);
mg = 1./(1+mg);
fil = mg;

%% Butterworth highpass filter 
function fil = BHPF(img,param)
D0 = param.D0;
order = param.order;
A = size(img,1);
B = size(img,2);
[x, y] = meshgrid(1:B, 1:A);
cx = B/2;
cy = A/2;
mg = sqrt((x-cx).^2 + (y-cy).^2);
mg = (D0./(mg+1e-5)).^(2*order);
mg = 1./(1+mg);
fil = mg;

%% Gaussian lowpass filter 
function fil = GLPF(img,param)
D0 = param.D0;
A = size(img,1);
B = size(img,2);
[x, y] = meshgrid(1:B, 1:A);
cx = B/2;
cy = A/2;
mg = (x-cx).^2 + (y-cy).^2;
mg = -mg/(2*D0^2);
fil = exp(mg);

%% Gaussian high filter 
function fil = GHPF(img,param)
D0 = param.D0;
A = size(img,1);
B = size(img,2);
[x, y] = meshgrid(1:B, 1:A);
cx = B/2;
cy = A/2;
mg = (x-cx).^2 + (y-cy).^2;
mg = -mg/(2*D0^2);
fil = 1-exp(mg);

