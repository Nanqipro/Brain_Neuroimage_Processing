function vis_3d(waveColor,sz, xUpsample, yUpsample, zUpsample)
%VIS_3D 此处显示有关此函数的摘要
%   此处显示详细说明
figure(25);
waveColorMap = parula(sz(4));
for zi = 1:sz(3)
    for xi = 1:sz(1)
        for yi = 1:sz(2)
            if (waveColor(xi,yi,zi) ~= 0)
                colorTime = waveColor(xi,yi,zi);
                colorNum = [waveColorMap(colorTime,1), waveColorMap(colorTime,2),...
                    waveColorMap(colorTime,3)];
                scatter3(xi*xUpsample, yi*yUpsample, zi*zUpsample, 'filled',...
                    'MarkerFaceColor',colorNum);
            end 
        end
    end
end 


end

