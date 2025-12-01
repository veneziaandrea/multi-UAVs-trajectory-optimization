function [Z, seen_points] = DWA(ind, Z, bounds, r_FOV, numDrones, seen_points) 
% Function in which is implemented the optimization problem for waypoint
% generation;
% INPUT: - current state of the drone, upper/lower bounds
% OUTPUT: - future state of the drone

% create circumference of seen points and add them to the relative set (da sistemare)
alpha= linspace(0:2*pi-pi/180);
    for j = 1:numDrones
        seen_points(j, :)= alpha*r_FOV + Z(j, :);
    end
end
end