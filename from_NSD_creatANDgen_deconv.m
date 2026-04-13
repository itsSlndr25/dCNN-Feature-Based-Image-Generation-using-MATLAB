%% from NSD create image
% 用途: 透過VGG-16的卷積層生成正交於兩張圖片特徵向量差異的新特徵，並反向重建影像
% 作者: I-HANG CHEN
% 日期: 2023.SEP

%% ========== 初始化設定 ==========
% 設定隨機種子以確保可重現性
rng(0);

% 添加必要路徑
addpath('./conv1_1');
addpath('./conv5_3');
vl_compilenn;
run ./matconvnet/matlab/vl_setupnn;

% 載入VGG-16網路
net = vgg16;
inputSize = net.Layers(1).InputSize;

% 共用參數設定
vector_amount = 5; % 每個filter生成的圖片數量

% 圖片路徑 (example, change to real target)
imgPath_A = 'forest.png';
imgPath_B = 'house.png';

%% ========== Section 1: conv1_1 處理 ==========
fprintf('========== 開始處理 conv1_1 層 ==========\n');

% 選擇卷積層
nlayers = 2;

% 載入並預處理圖片
A = loadAndPreprocessImage(imgPath_A, inputSize);
B = loadAndPreprocessImage(imgPath_B, inputSize);

% 提取該層的filters
filters = net.Layers(nlayers).Weights;

% feedforward through conv1_1
featureA = vl_nnconv(A, filters, [], 'Pad', 1);
featureB = vl_nnconv(B, filters, [], 'Pad', 1);
[dim1, dim2, nfilter] = size(featureA);

% 輸出路徑
outputPath_conv1_1 = './reverse_img_generation/conv1_1_forestVSsign/';

% 對每個filter進行處理
for f = 1:nfilter
    fprintf('處理第 %d/%d 個filter\n', f, nfilter);
    
    % flatten特徵圖成向量
    vectorA = reshape(featureA(:,:,f), [1, dim1*dim2]);
    vectorB = reshape(featureB(:,:,f), [1, dim1*dim2]);
    
    % 計算兩向量間的差異向量及其長度
    deltaL = vectorB - vectorA;
    deltaL_length = norm(deltaL);
    
    % 計算差異向量(deltaL)的null space(正交補空間)
    orthSpace = null(deltaL);
    
    % 生成多個正交向量並重建圖片
    for k = 1:vector_amount
        fprintf('  生成第 %d/%d 張圖片\n', k, vector_amount);
        
        % 標準化正交向量使其長度等於deltaL_length
        orth_temp = orthSpace(:,k) / norm(orthSpace(:,k)) * deltaL_length;
        
        % 將正交向量加到vectorA上，得到新的特徵向量
        orth_vector = vectorA + transpose(orth_temp);
        orth_vector = reshape(orth_vector, [dim1, dim2]);
        
        % 使用轉置卷積重建圖片
        reconstructedImage = vl_nnconvt(single(orth_vector), filters(:,:,:,f), [], 'Crop', 1);
        
        % 儲存圖片
        filename = fullfile(outputPath_conv1_1, sprintf('filter%d_no%d.png', f, k));
        imwrite(reconstructedImage, filename);
    end
    
    % 清除暫存變數釋放記憶體
    clear orthSpace orth_temp orth_vector;
end

fprintf('conv1_1 完成!\n');
finished1 = datetime("now");
fprintf('完成時間: %s\n\n', finished1);

% 清除不需要的變數，但保留net和共用參數
clearvars -except net inputSize vector_amount imgPath_A imgPath_B;

%% ========== Section 2: conv1_2 處理 ==========
fprintf('========== 開始處理 conv1_2 層 ==========\n');

% 選擇兩個連續的卷積層
nlayers = [2, 4];

% 載入並處理圖片
A = loadAndPreprocessImage(imgPath_A, inputSize);
B = loadAndPreprocessImage(imgPath_B, inputSize);

% feedforward through conv1_1
filters_1 = net.Layers(nlayers(1)).Weights;
featureA = vl_nnconv(A, filters_1, [], 'Pad', 1);
featureA = relu(dlarray(featureA, "SSCB"));

featureB = vl_nnconv(B, filters_1, [], 'Pad', 1);
featureB = relu(dlarray(featureB, "SSCB"));

% feedforward through conv1_2
filters_2 = net.Layers(nlayers(2)).Weights;
featureA = vl_nnconv(extractdata(featureA), filters_2, [], 'Pad', 1);
featureB = vl_nnconv(extractdata(featureB), filters_2, [], 'Pad', 1);

[dim1, dim2, nfilter] = size(featureA);

% 輸出路徑
outputPath_conv1_2 = './reversed_image/conv1_2/';

% 對每個filter進行處理
for f = 1:nfilter
    fprintf('處理第 %d/%d 個filter\n', f, nfilter);
    
    % flatten特徵圖為成向量
    vectorA = reshape(featureA(:,:,f), [1, dim1*dim2]);
    vectorB = reshape(featureB(:,:,f), [1, dim1*dim2]);
    
    % 計算差異向量
    deltaL = vectorB - vectorA;
    deltaL_length = norm(deltaL);
    
    % 計算正交空間
    orthSpace = null(deltaL);
    
    % 生成多個正交向量並重建圖片
    for k = 1:vector_amount
        fprintf('  生成第 %d/%d 張圖片\n', k, vector_amount);
        
        % 標準化正交向量
        orth_temp = orthSpace(:,k) / norm(orthSpace(:,k)) * deltaL_length;
        orth_vector = vectorA + transpose(orth_temp);
        orth_vector = reshape(orth_vector, [dim1, dim2]);
        
        % 反向重建: conv1_2 -> relu -> conv1_1
        reconstructed1 = vl_nnconvt(single(orth_vector), filters_2(:,:,:,f), [], 'Crop', 1);
        reconstructed2 = relu(dlarray(reconstructed1, "SSCB"));
        reconstructedImage = vl_nnconvt(extractdata(reconstructed2), filters_1, [], 'Crop', 1);
        
        % 儲存圖片
        filename = fullfile(outputPath_conv1_2, sprintf('filter%d_no%d.png', f, k));
        imwrite(reconstructedImage, filename);
    end
    
    % 清除暫存變數
    clear orthSpace orth_temp orth_vector reconstructed1 reconstructed2;
end

fprintf('conv1_2 完成!\n');
finished2 = datetime("now");
fprintf('完成時間: %s\n\n', finished2);

% 清除不需要的變數
clearvars -except net inputSize vector_amount imgPath_A imgPath_B;

%% ========== Section 3: 深層卷積層處理 (conv4_3 或更高層) ==========
fprintf('========== 開始處理較深層卷積層 ==========\n');

% 選擇目標層 (9: conv2_2, 16: conv3_3, 23: conv4_3)
nlayers = 23;

% 載入並預處理圖片
A = loadAndPreprocessImage(imgPath_A, inputSize);
B = loadAndPreprocessImage(imgPath_B, inputSize);

% 取得目標層名稱
baseLayer = net.Layers(nlayers).Name;
fprintf('目標層: %s\n', baseLayer);

% 前向傳播到目標層
featureA = activations(net, A, baseLayer);
featureB = activations(net, B, baseLayer);

[dim1, dim2, nfilter] = size(featureA);

% 建立層名稱array (排除input層)
filterarray = strings(1, 29);
for r = 1:29
    filterarray(r) = net.Layers(r+1).Name;
end

% 預先計算所有maxpool層的indices (用於unpooling)
fprintf('計算maxpool indices...\n');
indx_cell = cell(1, 4);

poolLayers = [4, 9, 16, 23]; % pool1, pool2, pool3, pool4的位置
for i = 1:length(poolLayers)
    for_pool = activations(net, A, filterarray(poolLayers(i)));
    for_pool = dlarray(for_pool, 'SSCB');
    [~, indx, ~] = maxpool(for_pool, 2, 'Stride', 2);
    indx_cell{i} = indx;
end

% 輸出路徑
outputPath_conv4_3 = './reversed_image/conv4_3/';

% 對每個filter進行處理
for f = 1:nfilter
    fprintf('處理第 %d/%d 個filter\n', f, nfilter);
    
    % flatten特徵圖
    vectorA = reshape(featureA(:,:,f), [1, dim1*dim2]);
    vectorB = reshape(featureB(:,:,f), [1, dim1*dim2]);
    
    % 計算差異向量
    deltaL = vectorB - vectorA;
    deltaL_length = norm(deltaL);
    
    % 計算正交空間
    orthSpace = null(deltaL);
    
    % 生成多個正交向量並重建圖片
    for k = 1:vector_amount
        fprintf('  生成第 %d/%d 張圖片\n', k, vector_amount);
        
        % 標準化正交向量
        orth_temp = orthSpace(:,k) / norm(orthSpace(:,k)) * deltaL_length;
        % Note: 這裡用 deltaL 而非 vectorA (原版)
        orth_vector = deltaL + transpose(orth_temp);
        reverseFeature = reshape(orth_vector, [dim1, dim2]);
        
        % 初始化反向傳播
        filters = net.Layers(nlayers).Weights;
        
        % 初始化unpooling的輸出大小和pool索引
        outputSize = [28, 28, 512, 1];
        poolIdx = 4;
        
        % 第一步反卷積
        reverseFeature = vl_nnconvt(reverseFeature, filters(:,:,:,f), [], 'Crop', 1);
        
        % 反向傳播通過各層 (根據目標層決定迭代次數)
        numLayersToReverse = 28; % 對於conv4_3
        
        for t = 8:numLayersToReverse
            layerIdx = 29 - t;
            currentLayer = filterarray(layerIdx);
            
            if contains(currentLayer, "conv")
                % 反卷積層
                filters = net.Layers(layerIdx + 1).Weights; % +1因為有input層
                reverseFeature = vl_nnconvt(extractdata(reverseFeature), filters, [], 'Crop', 1);
                
            elseif contains(currentLayer, "pool")
                % Unpooling層
                x = dlarray(reverseFeature, 'SSCB');
                indx = indx_cell{poolIdx};
                
                % 計算當前unpooling的輸出大小
                outputSize = outputSize .* [2, 2, 1/2, 1];
                reverseFeature = maxunpool(x, indx, outputSize);
                
                poolIdx = poolIdx - 1;
                
            elseif contains(currentLayer, "relu")
                % ReLU層
                x = dlarray(reverseFeature, 'SSCB');
                reverseFeature = relu(x);
            end
        end
        
        % 儲存重建的圖片
        filename = fullfile(outputPath_conv4_3, sprintf('filter%d_no%d.png', f, k));
        imwrite(reverseFeature, filename);
    end
    
    % 清除暫存變數
    clear orthSpace orth_temp orth_vector reverseFeature;
end

fprintf('conv4_3 完成!\n');
finished3 = datetime("now");
fprintf('完成時間: %s\n\n', finished3);

%% ========== Section 4: 視覺化各層特徵 ==========
fprintf('========== 開始視覺化各層特徵 ==========\n');

outputPath_viz = './reversed_image/for_layer_visualisation/';

for layer = 2:29
    layerName = net.Layers(layer).Name;
    fprintf('視覺化層: %s\n', layerName);
    
    channels = 1:64;
    featureImg = deepDreamImage(net, layerName, channels, 'PyramidLevels', 1);
    
    % 產生縮圖網格
    featureImg = imtile(featureImg, 'ThumbnailSize', [64, 64]);
    
    % 繪製並儲存
    figure('Visible', 'off'); % 不顯示視窗以加速
    imshow(featureImg);
    title(['Layer ', layerName, ' Features'], 'Interpreter', 'none');
    
    saveas(gcf, fullfile(outputPath_viz, layerName), 'jpg');
    close(gcf);
end

fprintf('特徵視覺化完成!\n\n');

%% ========== Section 5: 視覺化第一層Filters ==========
fprintf('========== 視覺化第一層Filters ==========\n');

filters = net.Layers(2).Weights;

% 正規化filter值到0-1範圍以便視覺化
f_min = min(filters(:));
f_max = max(filters(:));
filters_norm = (filters - f_min) ./ (f_max - f_min);

% 產生filter網格圖
filter_show = imtile(filters_norm(:,:,1,:), 'BorderSize', 1, 'BackgroundColor', 'w');
filter_show = imresize(filter_show, [800, 800], 'nearest');

% 繪製前6個filters的RGB channel
figure;
n_filters = 6;
for i = 1:n_filters
    f = filters_norm(:, :, :, i);
    
    % 分別繪製每個color channel
    for j = 1:3
        subplot(n_filters, 3, (i-1)*3 + j);
        
        % 移除座標軸刻度
        set(gca, 'XTick', []);
        set(gca, 'YTick', []);
        
        % 以greyscale顯示channel
        imshow(f(:, :, j));
    end
end

sgtitle('第二層Filters視覺化');

fprintf('Filters視覺化完成!\n\n');

%% ========== Section 6: 使用deepDreamImage視覺化Filter學習的特徵 ==========
fprintf('========== 使用deepDreamImage視覺化特徵 ==========\n');

layer = net.Layers(2).Name;
channels = 1:25;

% 使用圖片B作為初始圖片生成deep dream
B = loadAndPreprocessImage(imgPath_B, inputSize);
I = deepDreamImage(net, layer, channels, ...
    'PyramidLevels', 1, ...
    'Verbose', 0, ...
    'InitialImage', B);

% 繪製結果
figure;
for i = 1:25
    subplot(5, 5, i);
    imshow(I(:,:,:,i));
    title(sprintf('Ch %d', i));
end
sgtitle('DeepDream視覺化 (基於圖片B)');

fprintf('DeepDream視覺化完成!\n');
fprintf('========== 所有處理完成 ==========\n');

%% ========== 輔助函數 ==========

function img = loadAndPreprocessImage(imgPath, targetSize)
    % 載入並處理圖片以符合input要求
    % 輸入:
    %   imgPath: 圖片路徑
    %   targetSize: 目標尺寸 [height, width, channels]
    % 輸出:
    %   img: 預處理後的圖片 (single類型)
    
    img = im2single(imread(imgPath));
    img = imresize(img, targetSize(1:2), 'nearest');
    
    % convert to greyscale # if needed
    % if size(img, 3) == 1
    %     img = cat(3, img, img, img);
    % end
end