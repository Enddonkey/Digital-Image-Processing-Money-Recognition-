%% Grayscale and Denoising
%This section is mainly used to read and grayscale the image.
img = imread('search_noise.png'); 
img1 = rgb2gray(img); % change the image from rgb to gray
figure(1); imshow(img1); title('gray image');% show figure

%Image Denoising
%This part is mainly used for image denoising
grayimg = medfilt2(img1, [5,5]); % Processing of images using median filtering
figure(2); imshow(grayimg); title('The denoised image');% show figure

%% Binarization Masking
% Detecting image edges
BW = edge(grayimg, 'canny'); %Find edges using the Canny method.
figure(3); imshow(BW); title('Canny edge detect result');

% Morphological processing
BW1 = imclose(BW, strel('disk', 25)); %closed the region
BW2 = bwareaopen(BW1, 2e5);% delete the noise from the enviroment 
BW3 = imfill(BW2, 'holes');% fill the hole in the closed region
figure(4); imshow(BW3); title('morphological processing result');

% Connected domain analysis
[L, num] = bwlabel(BW3);
stats = regionprops(L, 'Area', 'PixelIdxList', 'BoundingBox', 'Orientation', 'Centroid');
[~, index] = sort([stats.Area], 'descend');
% Select the top three regions in terms of number of pixels (if they exist)
numRegions = min(3, length(stats));
%Process the template image in the same way
%Read template image 
template = imread('aud-notes.jpg');
template_gray = rgb2gray(template);

% Create an array of tuples to store information about each region
regions = cell(1, numRegions);
%% Rotation Correction  
figure;
for i = 1:numRegions
    % Get current area
    currentRegion = false(size(BW3));
    currentRegion(stats(index(i)).PixelIdxList) = true;
    
    % Get the direction of the area
    angle = stats(index(i)).Orientation;
    
    % rotation correction
    rotatedRegion = imrotate(currentRegion, -angle, 'bilinear', 'crop');
    rotatedImg = imrotate(img, -angle, 'bilinear', 'crop');
    
    % Find the exact boundary of the rotated area
    [rows, cols] = find(rotatedRegion);
    minRow = min(rows);
    maxRow = max(rows);
    minCol = min(cols);
    maxCol = max(cols);
    
    % Keying out the current area
    croppedImg = rotatedImg(minRow:maxRow, minCol:maxCol, :);
    
    % Resize to match template
    resizedImg = imresize(croppedImg, size(template_gray));
    
    % Storage area information
    regions{i}.image = croppedImg;
    regions{i}.resized = resizedImg;
    regions{i}.centroid = stats(index(i)).Centroid;
    regions{i}.bbox = stats(index(i)).BoundingBox;
    
    % Getting Boundaries
    regions{i}.boundary = bwboundaries(currentRegion);
    
    % Framing the area of interest on the rotationally corrected image
    subplot(2, numRegions, i);
    imshow(rotatedImg);
    hold on;
    rectangle('Position', [minCol, minRow, maxCol-minCol+1, maxRow-minRow+1], ...
              'EdgeColor', 'r', 'LineWidth', 2);
    text(minCol, minRow-10, sprintf('area %d', i), 'Color', 'red', ...
         'FontSize', 12, 'BackgroundColor', 'white');
    title(sprintf('ROI %d (Rotated)', i));
    hold off;
    
    % Show keyed area
    subplot(2, numRegions, i + numRegions);
    imshow(croppedImg);
    title(sprintf('Keying out the area %d', i));
end
%% Correlation Comparison 
% Adjusting the image layout
set(gcf, 'Position', get(0, 'Screensize')); % 全屏显示

% value recognition
denominations = [10, 20, 100, 50, 5]; % 正确的澳元面值顺序
template_regions = cell(1, length(denominations));

% Preprocessing of template images
template_edge = edge(template_gray, 'canny');
template_closed = imclose(template_edge, strel('disk', 5));
template_filled = imfill(template_closed, 'holes');

% Find the bill area in the template image
[L_template, num_template] = bwlabel(template_filled);
stats_template = regionprops(L_template, 'Area', 'BoundingBox');
[~, index_template] = sort([stats_template.Area], 'descend');

figure;
for i = 1:length(denominations)
    % Get the bounding box of the current bill
    bbox = stats_template(index_template(i)).BoundingBox;
    
    % Crop out the current bill area
    template_regions{i} = imcrop(template_gray, bbox);
    
    % Resize to ensure all templates have the same dimensions
    template_regions{i} = imresize(template_regions{i}, [200, 400]);  % Can be resized as needed
    
    subplot(1, length(denominations), i);
    imshow(template_regions{i});
    title(sprintf('$%d Template', denominations(i)));
end

% value judgment for each identified region
figure;
best_match = zeros(1, numRegions);
for i = 1:numRegions
    subplot(1, numRegions, i);
    region_gray = rgb2gray(regions{i}.resized);
    imshow(region_gray);
    hold on;
    
    max_correlation = -inf;
    
    for j = 1:length(denominations)
        % Ensure that the area image and the template have the same dimensions
        resized_region = imresize(region_gray, size(template_regions{j}));
        
        % Calculating Correlation
        correlation = corr2(resized_region, template_regions{j});
        
        if correlation > max_correlation
            max_correlation = correlation;
            best_match(i) = j;
        end
    end
    
    title(sprintf('Area %d: $%d', i, denominations(best_match(i))));
    hold off;
end

% Adjusting the image layout
set(gcf, 'Position', get(0, 'Screensize')); 
% full screen display

% Labeling the recognition results on the original image
figure;
imshow(img);
hold on;
%
for i = 1:numRegions
    % Get all outlines of the current area
    boundaries = regions{i}.boundary;
    
    % Find the longest outline (usually the outer border)
    [~, maxIndex] = max(cellfun(@length, boundaries));
    outerBoundary = boundaries{maxIndex};
    
    % Drawing the outer border
    plot(outerBoundary(:,2), outerBoundary(:,1), 'r', 'LineWidth', 2);
    
    % Get the center of the area
    centroid = regions{i}.centroid;
    
    % Displaying the identified denomination in the center of the area
    text(centroid(1), centroid(2), sprintf('$%d', denominations(best_match(i))), ...
         'Color', 'r', 'FontSize', 14, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
end

title('Result');
hold off;

% Adjusting the image layout
set(gcf, 'Position', get(0, 'Screensize')); % full screen display

% Output recognition results
fprintf('\nResult：\n');
for i = 1:numRegions
    bbox = regions{i}.bbox;
    fprintf('Area %d: Position [x=%.1f, y=%.1f, Length=%.1f, Height=%.1f], Value $%d\n', ...
            i, bbox(1), bbox(2), bbox(3), bbox(4), denominations(best_match(i)));
end

