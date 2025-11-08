function waveImg = calRatio(darkImg, calImg, darkSigma, calSigma)

dim = size(darkImg);
darkGau = zeros(dim);
calGau = zeros(dim);
% sigma = 20;
fprintf('Begin Gaussian blur...\n');
% dark img gaussian blur
for ti = 1:dim(4) 
    for zi = 1:dim(3) 
        darkGau(:,:, zi, ti) = imgaussfilt(darkImg(:,:,zi, ti), darkSigma);
    end 
end 

% calcium wave gaussian blur 
for ti = 1:dim(4) 
    for zi = 1:dim(3) 
        calGau(:,:, zi, ti) = imgaussfilt(calImg(:,:,zi, ti), calSigma);
    end 
end 

waveImg = calGau./darkGau;
fprintf('End Gaussian blur...\n');

% figure(1);
% imshow(waveImg(:,:,22,1),[0 7]);
% figure(2);
% imshow(waveImg(:,:,22,2),[0 7]);
% figure(3);
% imshow(waveImg(:,:,22,3),[0 7]);
% figure(4);
% imshow(waveImg(:,:,22,4),[0 7]);
% figure(5);
% imshow(waveImg(:,:,22,5),[0 7]);
% figure(6);
% imshow(waveImg(:,:,22,6),[0 7]);
% figure(7);
% imshow(waveImg(:,:,22,7),[0 7]);
% figure(8);
% imshow(waveImg(:,:,22,8),[0 7]);
% figure(9);
% imshow(waveImg(:,:,22,9),[0 7]);
% figure(10);
% imshow(waveImg(:,:,22,10),[0 7]);
% figure(11);
% imshow(waveImg(:,:,22,11),[0 7]);
% figure(12);
% imshow(waveImg(:,:,22,12),[0 7]);


end

