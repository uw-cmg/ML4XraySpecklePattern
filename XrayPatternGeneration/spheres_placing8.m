%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Author Dina, Mingren
%Created:  2020-12-16
%Modified: 2020-12-16

% v1: Dina's initial code
% v2: modify to read in layout and output pattern iamges
% v3: modify system parameters to get better results
% v4: 
%       modify output to the raw data of FFT
%       drop log scale
%       using better parameters
% v5: modify the system layout plot using direct pattern not log scale
% v6: Fixing the scatter area to constant
%     r = sqrt(2.0/N) * r2
% v7: processing polydispersity disks
% v8: 2 times bigger system to check size effects

clear all;
close all;

SZ = 4000;
xs=0:1:SZ;
ys=0:1:SZ;
pad = 8000;
%R2 = 90;

% Number of Configures
% testing 100 configurations
NumConfigures = 100;

% Number of Beads
% need to be changed for each bead number
NumBeads = 2 * 4;

%RR = sqrt( 2.0 / NumBeads) * R2;

% Let fixing rshow ==> img size is (2 * rshow + 1)
rshow = 256;
%rshow = round((2*pi/(2*RR))/(2*pi/(length(xs)+2*pad)));
%display(rshow);

% loop all Configures
for frame = 0:NumConfigures-1
    % Reset System
    [x, y] = meshgrid(xs, ys);
    D2_matrix = zeros(length(x));
    
    % Read In Beads Positions
    fullFileName = strcat('Beads_',int2str(NumBeads),'/BeadsNum_',int2str(NumBeads),'Config_',int2str(frame),'.txt');
    fprintf(1, 'Now reading %s\n', fullFileName);
    AA = readtable(fullFileName,'ReadVariableNames',false);
    AAvect = table2array(AA);
    xp = AAvect(:,1);
    yp = AAvect(:,2);
    RR = AAvect(:,3);
    a(:,1) = xp;
    a(:,2) = yp;
    
    % Generate Speckle Pattern
    nn = 1:length(xp);
    R_all = zeros(size(xp));
    R_all(:) = RR;
    all_centres = a;

    for count = 1:1:length(all_centres(:,1))
        R = R_all(count);
        xc = all_centres(count,1);
        yc = all_centres(count,2);
        for i = 1:1:length(x)
            for j = 1:1:length(y)
                r = sqrt((i-xc)^2+(j-yc)^2);
                if r<=R
                    D2_matrix(i,j) = 1;
                end
            end
        end 
    end

    f = figure('visible','off');
    subplot(1,2,1)
    cla
    imagesc(D2_matrix)
    axis image

    subplot(1,2,2)
    cla
    B = padarray(D2_matrix,[pad,pad]);

    I = (fftshift(abs(fft2(B))));
    I = I.^2;
    % I = I/mean(I(:));
    sI = size(I)/2;
    pattern_data = I(sI(1)- 3 * rshow:sI(1)+ 3 * rshow,sI(1)-3 * rshow:sI(1)+ 3 * rshow);
    save(strcat('SpeckleBeads_',int2str(NumBeads),'/Beads_',int2str(NumBeads),'.',int2str(frame),'.mat'),'pattern_data');
    % loading data into Python
    % import scipy.io as sio
    % mat_Acontents = sio.loadmat("Beads_2.0.mat")
    % AA = mat_Acontents['pattern_data']
    % AA.shape == > (768 x 768)
    imagesc(pattern_data);
    axis image
    % save image
    saveas(f, strcat('SystemLayoutBeads_',int2str(NumBeads),'/Beads_',int2str(NumBeads),'.',int2str(frame),'.jpg'));
    
    % close all to save memory space
    close all;
end