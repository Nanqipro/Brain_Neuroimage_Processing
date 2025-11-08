%% input
clc;clear;
addpath(genpath(pwd));
srcDir = 'C:\Users\ym376\SCN\CalWaveCode\';
darkDir = 'data\dark\';
% calDir = 'data\calWave\';
calDir = 'data\calWave\';
resDir = 'results\';
mkdir([srcDir, resDir]);

% %% read data (all frames)
% % dark data
% sz = [512,512,43,17];
% darkData = zeros(sz);
% calData = zeros(sz);
% imgList = dir(fullfile([srcDir, darkDir, '*.tif']));
% for i = 1:length(imgList)
%     imgName = imgList(i).name;
%     darkData(:,:,:,i) = openMovie([srcDir, darkDir, imgName]);
% end
% darkData(find(~darkData(:,:,:,1))) = 1;
% darkData(:,:,:,:) = repmat(darkData(:,:,:,1),1,1,1, sz(4));
% 
% % calcium wave data
% imgList = dir(fullfile([srcDir, calDir, '*.tif']));
% for i = 1:length(imgList)
%     imgName = imgList(i).name;
%     calData(:,:,:,i) = openMovie([srcDir, calDir, imgName]);
% end

%% read data (frame 1-22)
% dark data
sz = [512,512,22,17];
image = zeros(512,512,43);
darkData = zeros(sz);
calData = zeros(sz);
imgList = dir(fullfile([srcDir, darkDir, '*.tif']));
for i = 1:length(imgList)
    imgName = imgList(i).name;
    image = openMovie([srcDir, darkDir, imgName]);
    darkData(:,:,:,i) = image(:,:,1:22);
end
darkData(~darkData(:,:,:,1)) = 1;
darkData(:,:,:,:) = repmat(darkData(:,:,:,1),1,1,1, sz(4));

% calcium wave data
imgList = dir(fullfile([srcDir, calDir, '*.tif']));
for i = 1:length(imgList)
    imgName = imgList(i).name;
    image = openMovie([srcDir, calDir, imgName]);
    calData(:,:,:,i) = image(:,:,1:22);
end

%% Read ROI
roiRMask = zeros(sz(1),sz(2),sz(3));
roiLMask = zeros(sz(1),sz(2),sz(3));
roiAllMask = zeros(sz(1),sz(2),sz(3));
roiDir = [srcDir, 'data/outline_wave_L_1-22/'];
roiList = dir(fullfile([roiDir, '*.roi']));
for ri = 1:length(roiList) 
    roiName = roiList(ri).name;
    roi = ReadImageJROI([roiDir, roiName]);
    mask = poly2mask(roi.mnCoordinates(:, 1), roi.mnCoordinates(:, 2), sz(1), sz(2));
    % reg exp
    expression = 'L(?<layer>\d+)_(?<direction>L|R)_wave.roi';
    expNames = regexp(roiName, expression, 'names');
    if (expNames.direction == 'L') 
        roiLMask(:,:,str2num(expNames.layer)) = mask;
    elseif (expNames.direction == 'R')
        roiRMask(:,:,str2num(expNames.layer)) = mask;
    end 
end 

for zi = 1:sz(3) 
    mask = zeros(sz(1), sz(2));
    mask(find(roiLMask(:,:,zi) == 1 | roiRMask(:,:,zi) == 1)) = 1;
    roiAllMask(:,:,zi) = mask;
end

roiSum = sum(roiAllMask, 'all');
roiNum = 0;
roiCell = cell(roiSum, 2);
for zi = 1:sz(3)
    for yi = 1:sz(2)
        for xi = 1:sz(1)
            if (roiAllMask(xi,yi,zi) == 1)
                roiNum = roiNum + 1;
                roiCell(roiNum,1) = {[xi,yi,zi]};
                if (roiLMask(xi,yi,zi) == 1 && roiRMask(xi,yi,zi) == 1)
                    roiCell(roiNum,2) = {'A'};
                elseif (roiLMask(xi,yi,zi) == 1)
                    roiCell(roiNum,2) = {'L'};
                elseif (roiRMask(xi,yi,zi) == 1)
                    roiCell(roiNum,2) = {'R'};
                    
                end 
            end 
        end
    end
end
%% Mask the ori img
% TODO: 只对 ROI 进行计算
% (x, y, z, t)
darkSigma = 30;
calSigma = 15;
waveImg = calRatio(darkData, calData, darkSigma, calSigma);
fprintf('Calculate ROI image...\n');
% clear darkData calData;
waveROImg = zeros(sz);
for ti = 1:sz(4) 
    waveROImg(:,:,:,ti) = waveImg(:,:,:,ti) .* roiAllMask;
end 
waveROImg(waveROImg == 0) = nan;
% Image processing
% 去除过曝区域; delete over exposure (DOE)
% Todo: 手工逐帧去除过曝结果
waveDOE = waveROImg;
% medMul = 5; % DOE取高的中间值
% for zi = 1:sz(3)
%     zMed = median(waveImg(:,:,zi,:), 'all');
%     waveDOE(waveImg > medMul*zMed) = nan;
% end
%% Get wave front
waveAvg = squeeze(mean(mean(waveDOE, 1, 'omitnan'), 2, 'omitnan'));
waveRatio = zeros(sz);
waveDDF = zeros(sz(1), sz(2), sz(3), sz(4)-2);
waveColor = zeros(sz(1), sz(2), sz(3));
fprintf('Begin to calculate ratio\n');
% Z-score will not be better than ratio
for xi = 1:sz(1)
    for yi = 1:sz(2)
        for zi = 1:sz(3)
            for ti = 1:sz(4)
%                 if (waveAvg(zi,ti) < 1) 
%                     fAvg = 1;
%                 else 
%                     fAvg = ;
%                 end 
                 l1Img = waveDOE(xi,yi,zi,ti);
                 l1Avg = waveAvg(zi, ti);
                 l1Bug = waveDOE(xi,yi,zi,ti) / waveAvg(zi, ti);
                 waveRatio(xi, yi, zi, ti) = l1Bug;
            end
        end
    end
end
% for ri = 1:length(roiCell) 
%     xi = roiCell{ri,1}(1);
%     yi = roiCell{ri,1}(2);
%     zi = roiCell{ri,1}(3);
%     for ti = 1:sz(4)
%         l1Img = waveDOE(xi,yi,zi,ti);
%         l1Avg = waveAvg(zi, ti);
%         l1Bug = waveDOE(xi,yi,zi,ti) / waveAvg(zi, ti);
%         waveRatio(xi, yi, zi, ti) = l1Bug;
%     end 
% end 

%% 
fprintf('Begin to calculate wave\n');
% layer 11-21, works well 
% waveMinBar = -0.7;
% waveRatioMaxBar = 1.1;
waveMinBar = -0.52;
waveRatioMaxBar = 1.1;
for ri = 1:length(roiCell) 
    xi = roiCell{ri,1}(1);
    yi = roiCell{ri,1}(2);
    zi = roiCell{ri,1}(3);
% for xi = 1:sz(1)
%     for yi = 1:sz(2)
%         for zi = 1:sz(3)
            % calculate ddf
            waveDDF(xi, yi, zi, :) = diff(diff(squeeze(waveRatio(xi,yi,zi,:))));
            % 取极小值，定义为时间，画出时间图；
            % Todo: 极大值也要看, ratio最大值也考量
            [waveDDfMin, waveDDfTime] = min(waveDDF(xi, yi, zi, :));
            [waveRatioMax, waveRMaxTime] = max(waveRatio(xi, yi, zi, :));
            if (waveDDfMin < waveMinBar)
                waveColor(xi, yi, zi) = waveDDfTime + 1;
            elseif (waveRatioMax > waveRatioMaxBar && waveRMaxTime > 2)
                waveColor(xi, yi, zi) = waveRMaxTime;
            else
                waveColor(xi, yi, zi) = 0;
            end
%         end
%     end
end

fprintf('Finish Calculation\n');
% clear waveDDF waveRatio waveAvg;
save([srcDir, resDir, 'waveColor/waveColor-calSig', num2str(calSigma),'.mat'], "waveColor", '-mat');
%% imshow wave color
wCDir = [srcDir, resDir, 'waveColor/waveColor-maxChange-calGauss-', num2str(calSigma),'/'];
mkdir(wCDir);
maxTime = 17;
for zNum = 1:22
    figure(zNum);
    waveExg = squeeze(waveColor(:,:,zNum, 1));
    imshow(waveExg, []);
    colormap("jet");
    imwrite(squeeze(waveColor(:,:,zNum, 1))./maxTime, ...
        [wCDir, 'waveColor-z-', num2str(zNum), '.tif']);
end

%% 
% xi =225; yi = 86; zi = 12;
% figure(102);
% plot(squeeze(waveRatio(xi, yi, zi,:)), 'Color','b');
% hold on 
% plot(squeeze(waveDDF(xi, yi,zi,:)), 'Color','r');
% ylim([-1.5 2]);
% xlim([0 18]);
% % hold on 
% % plot(squeeze(waveROImg(xi, yi,zi,:)));
% % hold on 
% % plot(squeeze(waveAvg(zi,:)));
% hold off 
% % plot(waveRatio); % use dff?
% % hold on
% % waveZScore = zscore(squeeze(waveDOE(xi,yi,zNum,:)));
% % plot(waveZScore);
% % hold off
% xi =312; yi = 340; zi = 12;
% figure(103);
% plot(squeeze(waveRatio(xi, yi, zi,:)), 'Color','b');
% hold on 
% plot(squeeze(waveDDF(xi, yi,zi,:)), 'Color','r');
% ylim([-1.5 2]);
% xlim([0 18]);
% % hold on 
% % plot(squeeze(waveROImg(xi, yi,zi,:)));
% % hold on 
% % plot(squeeze(waveAvg(zi,:)));
% hold off 


%% write img
% medShow = 2.5;
% zFileDir = [srcDir, resDir, 'ROI-medShow-', num2str(medShow), '-waveImg/'];
% mkdir(zFileDir);
% for zi = 1:sz(3)
%     ziDir = [zFileDir, 'z-', num2str(zi), '/'];
%     mkdir(ziDir);
%     zMed = median(waveImg(:,:,zi,:), 'all');
%     for ti = 1:sz(4)
%         imwrite(waveROImg(:,:,zi,ti)./(zMed*medShow), ...
%             [ziDir, 'z-', num2str(zi), '_t-', num2str(ti), '.tif']);
%     end
% end

% %% Visualization using scatter (Wrong!)
% xUpSample = 1.27;
% yUpsample = 1.27;
% zUpsample = 6.96;
% vis_3d(waveColor, sz, xUpSample, yUpsample, zUpsample);
% 

%% Paint neurons
% for fi = 1:2 
%     if (fi == 1)
%         load('C:\Users\ym376\SCN\data\20210916-neuron\SCN-only\20210916_POI.mat');
%         szPoiSCN = size(POIMatrix);
%         SCNPoiMat = POIMatrix;
%     elseif (fi == 2)
%         load('C:\Users\ym376\SCN\data\20210916-neuron\SCN_with_extra\20210916_non_POI.mat');
%         szPoiExtra = size(POIMatrix);
%         extSCNPoiMat = POIMatrix;
%     end 
% end 
% 
% neuronPos = zeros(szPoiSCN(1) + szPoiExtra(1),3,'uint8'); % (x,y,z)
% lenAll = szPoiSCN(1) + szPoiExtra(1);
% % read pos from SCN POI
% for i = 1:szPoiSCN(1)
%     pos = SCNPoiMat{i,15};
%     if (isempty(pos)) 
%         pos = [0,0,0];
%         num = 0;
%         for ti = 1:szPoiSCN(2)
%             if (~isempty(SCNPoiMat{i,ti}))
%                 pos = pos + SCNPoiMat{i,ti};
%                 num = num +1;
%             end 
%         end 
%         pos = pos./num;
%     end 
%     neuronPos(i,1) = uint8(pos(1));
%     neuronPos(i,2) = uint8(pos(2));
%     neuronPos(i,3) = uint8(pos(3));
% end 
% % read pos from extra SCN
% i = i+1; j =1 ;
% while (i <= lenAll) 
%     pos = extSCNPoiMat{j,15};
%     if (isempty(pos)) 
%         pos = [0,0,0];
%         num = 0;
%         for ti = 1:szPoiExtra(2)
%             if (~isempty(extSCNPoiMat{j,ti}))
%                 pos = pos + extSCNPoiMat{j,ti};
%                 num = num +1;
%             end 
%         end 
%         pos = pos./num;
%     end 
%     neuronPos(i,1) = uint8(pos(1));
%     neuronPos(i,2) = uint8(pos(2));
%     neuronPos(i,3) = uint8(pos(3));
%     i = i+1; j = j+1;
% end 
% % paint neurons color
% neuronColor= zeros(lenAll, 1, 'uint8');
% for ni = 1:lenAll
%     if (neuronPos(ni,3) < 23)
%         neuronColor(ni) = waveColor(neuronPos(ni,1),...
%             neuronPos(ni,2), neuronPos(ni,3)); % matlab invert x and y
%     else 
%         neuronColor(ni) = 0;
%     end 
% end 
% save([srcDir, resDir, 'neuronColorAll.mat'],"neuronColor", '-mat');
%% write img
% medShow = 3; % median multi show in the img
% zFileDir = [srcDir, resDir, 'deOE-', num2str(medMul), '-med-', num2str(medShow) '-waveImg/'];
% mkdir(zFileDir);
% for zi = 1:sz(3)
%     ziDir = [zFileDir, 'z-', num2str(zi), '/'];
%     mkdir(ziDir);
%     % zMed = median(waveImg(:,:,zi,:), 'all');
%     for ti = 1:sz(4)
%         imwrite(waveDOE(:,:,zi,ti)./(medShow*zMed), ...
%             [ziDir, 'z-', num2str(zi), '_t-', num2str(ti), '.tif']);
%     end
% end


% % 测试钙波出现的具体时间。
% medMul = 3;
% zFileDir = [srcDir, resDir, 'Frame', num2str(sz(4)),'-7-23-z-', num2str(medMul), 'med-waveImg/'];
% mkdir(zFileDir);
% for zi = 1:sz(3)
%     ziDir = [zFileDir, 'z-', num2str(zi), '/'];
%     mkdir(ziDir);
%     zMed = median(waveImg(:,:,zi,:), 'all');
%     for ti = 1:sz(4)
%         imwrite(waveImg(:,:,zi,ti)./(medMul*zMed), ...
%             [ziDir, 'z-', num2str(zi), '_t-', num2str(ti), '.tif']);
%     end
% end








