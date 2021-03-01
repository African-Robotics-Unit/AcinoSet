% Written by Ricky Jericevich
% Requirements: MATLAB 2020b or later, Computer Vision Toolbox

% Ensure that:
% - MATLAB's current folder is in AcinoSet/data
% - the calibration frames are extracted and stored in folders as per the AcinoSet convention
% - main_dir points to the day/scene you want to calibrate
% - board_square_len holds the checkerboard's square size

% Method:
% - In section '1 - Initialise Vars', change main_dir and board_square_len as necessary
% - Run '1 - Initialise Vars'
% - Run '2 - Get Checkerboard Points for Camera Pairs'
% - Ensure that all cam pairs finished processing (alarm will sound when all are complete)
% - For each camera pairs' Stereo Camera Calibration:
%    - Carefully check each image pair to make sure the detected points are very accurate.
%      If one or both of the detected points in the image pair are inaccurate, delete the frame pair. To assist
%      with this process, do the following:
%       - Select 'Use Fixed Intrinsics'
%       - Select 'Load Intrinsics' and select intrinsic_a and intrinsic_b
%       - Select 'Calibrate'
%       - Use the 'Reprojection Errors' graph to help you find inaccurate frames
%         NB: The calibration that is obtained here is not very accurate, so the Reprojection
%             Errors graph should only be used as a guide. If an image pair's detected checkerboard
%             points are visually very accurate but the graph says it has a high error, DO NOT
%             DELETE THE FRAME PAIR. Always favour your visual inspection over the graph's numbers
%    - Click save and name it calibrationSessionA&B where A is the name of the 1st camera in the
%      camera pair and B is the 2nd. Eg:
%      calibrationSession1&2 or calibrationSession2&3 or calibrationSession5&6 or calibrationSession6&1
%      Ensure that the files are saved in AcinoSet/data/(scene)/extrinsic_calib/points
% - Once all the calibrationSessions have been saved, run '3 - Extract Data'

% Done! Now you can calibrate the extrinsics accurately using AcinoSet's calib_with_gui.ipynb
% Note: Always double check the points*.json files for any errors

%% 1 - Initialise Vars

clear; clc;

main_dir = '2017_12_14/bottom';
board_square_len = 55; % millimeters

% Do not modify any code below

extrinsic_dir = [main_dir '/extrinsic_calib'];

year = '2017';
if contains(main_dir, '2019')
    year = '2019';
end

% obtain intrinsic params
fid = fopen(['intrinsic_calib/' year '/camera.json'], 'r');
intrinsics = jsondecode(fscanf(fid, '%s'));
fclose(fid);

intrinsic_a = cameraIntrinsics(intrinsics.k([1,5]),... % focal length [fx, fy]
             intrinsics.k([1,2],3).',... % principle point [cx, cy]
             flip(intrinsics.camera_resolution.'),... 
             'RadialDistortion', intrinsics.d(1:3).'); % accepts max of 3 distortion coefficients

intrinsic_b = intrinsic_a;

%% 2 - Get Checkerboard Points for Camera Pairs

cd([extrinsic_dir '/frames']);

img_folders = {dir().name}; % folders that contain the calibration frames
img_folders = img_folders(~contains(img_folders, '.')); % ignore parent directories
img_folders = img_folders(~contains(img_folders, '&')); % ignore any 'a&b' folders

n_cams = length(img_folders);

for i = 0:n_cams-1
    idxs(1) = mod(i,n_cams) + 1;
    idxs(2) = mod(i+1,n_cams) + 1;
    cam_nums{1} = img_folders{idxs(1)};
    cam_nums{2} = img_folders{idxs(2)};
    
    img_names_a = {dir(cam_nums{1}).name}; % includes parent directories '.' and '..'
    img_names_b = {dir(cam_nums{2}).name};
    
    common_imgs = intersect(img_names_a(3:end), img_names_b(3:end));  % ignore parent directories!
    
    cd('../frames');
    if ~isempty(common_imgs)
        cam_pair_dir = [cam_nums{1} '&' cam_nums{2}]
        
        img_dirs{1} = [cam_pair_dir '/' cam_nums{1}];
        img_dirs{2} = [cam_pair_dir '/' cam_nums{2}];

        mkdir(img_dirs{1});
        mkdir(img_dirs{2});

        for k = common_imgs
            % copy common images to new folders
            copyfile([cam_nums{1} '/' k{:}], img_dirs{1});
            copyfile([cam_nums{2} '/' k{:}], img_dirs{2});
        end
        
        % get checkerboard points
        stereoCameraCalibrator(img_dirs{1}, img_dirs{2}, board_square_len,'millimeters')
    end
    
    if n_cams == 2
        break;
    end
end

cd('../points')

% alarm sounds when processing is complete
load handel
sound(y,Fs)

%% Delete Duplicated Frames

% original_dir = cd('../frames');
% img_folders = {dir('*&*').name};
% for folder = img_folders
%     rmdir(folder{:}, 's');
% end
% 
% cd(original_dir);

%% 3 - Extract Data 
% extracts data from calibrationSession files and saves as JSON for AcinoSet
% MATLAB's current folder must be in AcinoSet/data/(scene)/extrinsic_calib/points to run this

% cd([extrinsic_dir '/points']);

files = {dir('calibrationSession*&*.mat').name};
for file = files
    calib_sesh = load(file{:}).calibrationSession.BoardSet; % extract the relevant struct
    cam_nums = regexp(file{:}, '\d*', 'Match'); % extract cam names from file name
    
    for i = 1:length(cam_nums)
        
        cam = ['cam' cam_nums{i}];
        
        s.(cam).timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF');
        s.(cam).board_shape = calib_sesh.BoardSize - 1;
        s.(cam).board_square_len = calib_sesh.SquareSize / 1000;
        s.(cam).camera_resolution = flip(calib_sesh.ImageSize);
        
        for img_i = 1:calib_sesh.NumBoards
            img_name = calib_sesh.BoardLabels{img_i};
            pts = calib_sesh.BoardPoints(:, :, img_i, i); % BoardPoints shape: (# checkerboard points, 2, # images, # cams)
            [img_name, pts] = reformatData(img_name, pts, s.(cam).board_shape);
            s.(cam).points.(img_name) = pts;
        end
        
        s.(cam).board_shape = flip(s.(cam).board_shape);
    end    
end

cams = fieldnames(s);
for i = 1:length(cams)
    fpath = ['points' cams{i}(end) '.json'];
    disp(fpath);
    
    json_txt = replace(jsonencode(s.(cams{i})), '__', '.'); % revert image names back to original names
    json_txt = replace(json_txt, {',', '":'}, {', ', '": '}); % add spaces to ensure json is *exactly* like AcinoSet's json output
    
    % save json file
    fid = fopen(fpath, 'w');
    fprintf(fid, '%s', json_txt);
    fclose(fid);
end

function [img_name, pts] = reformatData(img_name, pts, board_shape)
    img_name = img_name(1:strfind(img_name,' ')-1); % remove all chars after the 1st space
    img_name = replace(img_name, '.', '__'); % convert to valid matlab field name
    
    % reshape checkerboard points to match AcinoSet convention
    pts = pagetranspose(reshape(pts, [board_shape, 2]));
    pts = pagetranspose(reshape(pts, [board_shape, 2])); % this repetition is not a mistake
end