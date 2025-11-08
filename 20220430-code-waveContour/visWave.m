%%
clear, clc;
dir = 'C:\Users\ym376\SCN\CalWaveCode\results\waveColor\';
load([dir, 'waveColor-maxChange-calSig15.mat']);
sz = size(waveColor);
% %%
% % 112
% colorMergeMap112 = [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9];
% % 122
% colorMergeMap122 = [1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9];
% waveColor(waveColor == 0) = nan;
% waveColorMerge112 = waveColor;
% waveColorMerge122 = waveColor;
% szWC = size(waveColor);
% for i = 1:length(colorMergeMap112)
%     waveColorMerge112(waveColor == i) = colorMergeMap112(i);
%     waveColorMerge122(waveColor == i) = colorMergeMap122(i);
% end 
% %%
% wCDir = [dir, 'waveColor-gauss20-merge-122/'];
% mkdir(wCDir);
% maxTimeMerge = 9;
% for zNum = 1:22
%     figure(100+zNum);
%     waveExg = squeeze(waveColorMerge(:,:,zNum, 1));
%     imshow(waveExg, []);
%     colormap("jet");
%     imwrite(squeeze(waveColorMerge(:,:,zNum, 1))./maxTimeMerge, ...
%         [wCDir, 'waveColor-z-', num2str(zNum), '.tif']);
% end
%% set z of wave color
%               1,2,3,4,5,6,7,8,9,0,11,12,13,14,5,6,7,
colorMarkMap = [0,2,3,4,5,6,6,8,9,9,11,11,13,14,0,0,0];
waveColorMark = waveColor;
for i = 1:length(colorMarkMap)
    waveColorMark(waveColor == i) = colorMarkMap(i);
end 
waveColorMark(waveColorMark==0) = nan;
maxTime = 17;
wcDir = [dir, 'waveColorTrans\'];
mkdir(wcDir);
for zi = 1:sz(3)
    figure(zi);
    waveExg = squeeze(waveColorMark(:,:,zi));
    % 透明
    alpha = ones(sz(1),sz(2));
    alpha(isnan(waveExg)) = 0;
    imshow(waveExg, []);
    colormap("jet");
%     imwrite(waveExg, jet(14), [wcDir, 'transit-z-', num2str(zi), '.png']);
%     [rImg,myMap] = imread([wcDir, 'transit-z-',num2str(zi), '.png']);
%     imwrite(rImg,myMap,[wcDir, 'waveColorMarkTran-z-', num2str(zi), '.png'], 'Alpha', alpha);
end 

%% contour: (colori, zi, set{xi,yi})
% for zi = 1:22
%     contourf(flipud(waveColorMerge(:,:,zi)),'ShowText','on');
% end 
zi = 14;
mul = 1000;
interval = 200;
% for zi = 1:22
    figure(100+zi);
    % contour(flipud(waveColorMark(:,:,12)), flipud(z(:,:,12)),'ShowText','on');
    contourf(flipud(waveColorMark(:,:,zi)),[2,3,4,5,6,8,9,11,13,14],'ShowText','on');
    colormap("parula");
    figure(200+zi);
    [dx,dy] = gradient(flipud(waveColorMark(:,:,zi)),interval, interval);
    quiver(dx.*mul, dy.*mul,0);
% end

% zLayer = 12;

% saveas(gcf, [dir, 'waveContour.jpg']);
% figure(101+zLayer);
% contourf(flipud(waveColor(:,:,zLayer)),'ShowText','on');
% colormap("jet");

% maxTime = 17;
% sz = size(waveColor);
% waveContour = cell(maxTime, sz(3));
% waveContPointNum = zeros(maxTime, sz(3), 'uint16');
% conMax = 1100;
% for ti = 1:maxTime
%     for zi = 1:sz(3)
%         numContour = 0;
%         zContour = zeros(conMax, 2);
%         [colorRaw, colorCol] = find(waveColor(:,:,zi) == ti);
%         for ri = 1:length(colorRaw)
%             if(colorRaw(ri) == 1 || colorRaw(ri) == 255 || ...
%                     colorCol(ri) == 1 || colorCol(ri) == 255)
%                 continue;
%             elseif (ti < maxTime &&  ...
%                     (waveColor(colorRaw(ri) + 1, colorCol(ri),zi) > ti || ...
%                     waveColor(colorRaw(ri) - 1, colorCol(ri),zi) > ti || ...
%                     waveColor(colorRaw(ri), colorCol(ri) + 1,zi) > ti || ...
%                     waveColor(colorRaw(ri), colorCol(ri) - 1,zi) > ti))
%                 numContour = numContour + 1;
%                 zContour(numContour, :) = [colorRaw(ri), colorCol(ri)]; 
%             elseif (ti == maxTime && ...
%                     (waveColor(colorRaw(ri) + 1, colorCol(ri),zi) == 0 || ...
%                     waveColor(colorRaw(ri) - 1, colorCol(ri),zi) == 0 || ...
%                     waveColor(colorRaw(ri), colorCol(ri) + 1,zi) == 0 || ...
%                     waveColor(colorRaw(ri), colorCol(ri) - 1,zi) == 0))
%                 numContour = numContour + 1;
%                 zContour(numContour, :) = [colorRaw(ri), colorCol(ri)];
%             end 
%         end
%         waveContPointNum(ti,zi) = numContour;
%         % zConLen = zContour(1:numContour,:);
%         waveContour(ti,zi) = {zContour(1:numContour,:)};
%     end 
% end 
% 
% %% Contour visualization
% 
% for zi = 1:sz(3)
%     figure(200+zi);
%     for ti = 1:maxTime
%         plot(waveContour{ti,zi}(:,1), waveContour{ti,zi}(:,2));
%         % contour(waveContour{ti,zi},ti);
%         hold on;
%     end
%     hold off;
% end


