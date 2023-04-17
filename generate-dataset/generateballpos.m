%% get all files (separated by folder/file into separate structure cells)
% fileName = struct2cell(dir('Z:\BallSystem_RawData\19_UAS-CSChrim-BPN-S1\BallTracking\*.mat'));
fileName = struct2cell(dir('Z:\BallSystem_RawData\19_UAS-CSChrim-BPN-S1\Set1_HighInt_0.35\BallTracking\*.mat'));

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

    plot_velocities(timeArr, x, y, z)

    ballpos_table = table(x, y, z);
    writetable(ballpos_table, join(["Z:\BallSystem_RawData\19_UAS-CSChrim-BPN-S1\Set1_HighInt_0.35\FlyTrajectories\", string(curr_fly_num), "ballpos.csv"], ''));
end

%% 

function [timeArr, x, y, z] = extract_all_data(fileName)

    BallTrack = matfile(fileName);
    disp(fileName)
    
    BallStruct = BallTrack.sensorData;
    ballPositions = BallStruct.bufferFly;

    ballPositions = ballPositions(all(~isnan(ballPositions),2),:);

    ballPositionsX = ballPositions(:,1)';
    ballPositionsXUp = interp(ballPositionsX,4);
    ballPositionsY = ballPositions(:,2)';
    ballPositionsYUp = interp(ballPositionsY,4);
    ballPositionsZ = ballPositions(:,3)';
    ballPositionsZUp = interp(ballPositionsZ,4);

    timeArr = 1:length(ballPositionsZUp);
    x = smooth(ballPositionsXUp,50);
    y = smooth(ballPositionsYUp,50);
    z = smooth(ballPositionsZUp,50);
    
end

function plot_velocities(timeArr, x, y, z)

    figure;
    plot3(x, y, z);

end