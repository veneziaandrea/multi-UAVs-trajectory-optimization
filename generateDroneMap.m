% GENERATORE MAPPA 3D CON CILINDRI PER FLEET DI DRONI
% Ambiente geometrico continuo (collision objects) - NO planner richiesto

function [environment, droneStarts] = generateDroneMap(areaSize, density, maxHeight, numObstacles, numDrones, manualDronePositions)
%
% INPUT
% areaSize             : side length of the map (meters)
% density              : density of obstacles (0.1 - 0.6 recommended)
% maxHeight            : maximum building height (meters)
% numDrones            : number of drones
% numObstacles         : optional exact number of obstacles
% manualDronePositions : optional [Nx3] initial positions


if nargin < 5
    manualDronePositions = [];
end

if nargin < 6
    cityArea = areaSize^2;
    numObstacles = round(cityArea * density);
end

% Parameters for obstacles
minRadius = 0.5;
maxRadius = 2.5;
minHeight = 4;
safetyMargin = 0.5; % minimum distance between obstacles/drones
environment = {};

%% 1. Place drones 

if ~isempty(manualDronePositions)
    droneStarts = manualDronePositions;
else
    droneStarts = zeros(numDrones,3);
    centerX = areaSize/2;
    centerY = areaSize/2;
    pz = 0; % starting altitude
    inflation = 0.6; % minimum separation
    
    % spiral/grid offsets
    offsets = [0 0]; % start with first drone at center
    step = inflation;
    n = 1;
    
    while size(offsets,1) < numDrones
        layer = ceil((sqrt(size(offsets,1))+1)/2); % layer of spiral
        % generate candidate positions for this layer
        candidates = [
            layer*step 0;
            -layer*step 0;
            0 layer*step;
            0 -layer*step;
            layer*step layer*step;
            layer*step -layer*step;
            -layer*step layer*step;
            -layer*step -layer*step;
        ];
        % append only if not exceeding required number of drones
        for c = 1:size(candidates,1)
            if size(offsets,1) < numDrones
                offsets = [offsets; candidates(c,:)]; %#ok<AGROW>
            else
                break;
            end
        end
    end
    
    % assign positions
    for d = 1:numDrones
        droneStarts(d,:) = [centerX + offsets(d,1), centerY + offsets(d,2), pz];
    end
end


% 2. Generate obstacles (avoid drones)
occupiedZones = [droneStarts(:,1:2) 0.6*ones(numDrones,1)]; % include drones as occupied

for i = 1:numObstacles
    valid = false;
    attempts = 0;
    
    while ~valid && attempts < 50
        attempts = attempts + 1;
        
        r = minRadius + rand*(maxRadius-minRadius);
        h = minHeight + rand*(maxHeight-minHeight);
        x = rand * areaSize;
        y = rand * areaSize;
        z = h/2;
        
        % check collision with existing zones (drones + other obstacles)
        distances = vecnorm(occupiedZones(:,1:2) - [x y], 2, 2);
        if all(distances > (occupiedZones(:,3) + r + safetyMargin))
            valid = true;
        end
    end
    
    if valid
        cyl = collisionCylinder(r, h);
        cyl.Pose = trvec2tform([x y z]);
        environment{end+1} = cyl;
        occupiedZones(end+1,:) = [x y r];
    end
end

% 3. Visualization
figure; hold on; grid on; axis equal
xlim([0 areaSize]); ylim([0 areaSize]); zlim([0 maxHeight+2])
title('3D Map for Drone Fleet')

% show obstacles
for i = 1:numel(environment)
    show(environment{i});
end

% show drones
for d = 1:numDrones
    drone = collisionSphere(0.3);
    drone.Pose = trvec2tform(droneStarts(d,:));
    show(drone);
end

xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
view(3)
end

%[env, starts] = generateDroneMap(40, 0.1, 15, 20, 10, []); example