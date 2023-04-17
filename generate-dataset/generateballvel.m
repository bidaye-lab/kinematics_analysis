%% get all files (separated by folder/file into separate structure cells)
% fileName = struct2cell(dir('Z:\BallSystem_RawData\19_UAS-CSChrim-BPN-S1\BallTracking\*.mat'));
fileName = struct2cell(dir('Z:\BallSystem_RawData\21_P9-RightTurning\BallTracking\*.mat'));

% construct all file paths and save each to cell array
allFiles = [];
arr_size = size(fileName);
for i=1:arr_size(2)
    temp_file = fileName{1,i};
    temp_folder = fileName{2,i};
    allFiles = [allFiles, strcat(temp_folder,"\", temp_file)];
end

all_flies = [];
for i=1:length(allFiles)
    curr_fly = allFiles(i);
    videoDuration = 3500;

    % get current fly
    curr_flyspl = split(curr_fly, "_");
    curr_flyspl = curr_flyspl(end-1);
    curr_flyspl = split(curr_flyspl, ".");
    curr_fly_num = curr_flyspl(1);
    disp(curr_fly_num)
    all_flies = [all_flies curr_fly_num];

    [timeArr, x, y, z] = extract_all_data(curr_fly);

    ballvel_table = table(x, y, z);
    writetable(ballvel_table, join(["Z:\BallSystem_RawData\21_P9-RightTurning\BallTracking\", string(curr_fly_num), "ballvel.csv"], ''));
end

%% 

function [timeArr, forward_x, sideways_y, angVel_z] = extract_all_data(fileName)

    BallTrack = matfile(fileName);
    disp(fileName)
    
    BallStruct = BallTrack.sensorData;
    ballRotations = BallStruct.bufferRotations;

    ballRotations = ballRotations(all(~isnan(ballRotations),2),:);

    ballRotationX = ballRotations(:,1)';
    ballRotationXUp = interp(ballRotationX,4);
    ballRotationY = ballRotations(:,2)';
    ballRotationYUp = interp(ballRotationY,4);
    ballRotationZ = ballRotations(:,3)';
    ballRotationZUp = interp(ballRotationZ,4);

    timeArr = 1:length(ballRotationZUp);
    forward_x=smooth(ballRotationXUp,50);
    sideways_y=smooth(ballRotationYUp,50);
    angVel_z=smooth(ballRotationZUp,50);
    
end