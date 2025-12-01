function [Z, seen_points, t_present] = DWA(map, Z, bounds, r_FOV, numDrones, Ts, N, prev_t) 
% Function in which is implemented the optimization problem for waypoint
% generation;
% INPUT: - map, current state of the drone, upper/lower bounds, FOV radius,
% number of drones, sampling time, time horizon
% OUTPUT: - future state of the drone, 

% create sphere of seen points and add them to the relative set
[phi, theta] = meshgrid(linspace(0, pi, 20), linspace(0, 2*pi, 20));
x = r_FOV * sin(phi) .* cos(theta);
y = r_FOV * sin(phi) .* sin(theta);
circle_points = [x(:), y(:)];
seen_points= zeros(size(circle_points));
% Generate the seen points for each drone based on their current position
for j = 1:numDrones
    seen_points(j, :) = circle_points + Z(j, 1:2);
end

% Find intersections of seen points
J_int= FindIntersections(seen_points); % cost function value for intersections

% create possible accelerations vector
a_min= bounds(4);
a_max= bounds(3);
a_step= 0.1;
a_vec= [a_min:a_step:a_max]'*ones(1, 3); % tentative accelerations in x,y,z directions
time= prev_t:Ts:prev_t+N*Ts;
% Define cost for obstacle avoidance and mean distance between drones
J_obs= ;
J_dist= ;

% evaluate the total cost function
J= J_obs + J_dist + J_int;
if J < minJ
    minJ= J;
    ... % select these inputs as best ones ecc
end

t_present= prev_t+N*Ts;
end