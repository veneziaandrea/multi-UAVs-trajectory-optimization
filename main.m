% MAIN SCRIPT FOR THE PROJECT

clc
clear all

% Import path of other folders
currentDir= fullfile('main.m');
utilsPath = fullfile('utils');
resultsPath = fullfile('results');
mpcPath = fullfile('mpc');
docsPath = fullfile('docs');
dataPath = fullfile('data');
addpath(utilsPath);
addpath(resultsPath);
addpath(mpcPath);
addpath(docsPath);
addpath(dataPath);
%% MAP CREATION
% define the map parameters
l= 50; % length of one side of the map [m]
density= 0.5; % how much of the area you want occupied; default= 0.5
maxH= 10; % max height of the obstacles [m]
numObs= 30; % if the number of obstacles wants to be exactly defined; overwrites density
numDrones= 5; % number of drones
manualDPos= []; % if is desired to manually place drones inside the map; starting position [x y z] for each drone

% create the map 
map = generateDroneMap(l, density, maxH, numObs, numDrones, manualDPos); % will also plot it

%% Save the generated map to a .mat file for future use (optional)
save('generated_drone_map.mat', 'map');
% Optionally, load the generated map for verification
% loadedMap = load('generated_drone_map.mat');
% disp('Map loaded successfully. Displaying the map again for verification.');

%% OPTIMIZATION PARAMETERS (WAYPOINT GENERATION + MAX DISTANCE)
% in this stage a Dynamic Window Approach is needed, since the waypoints
% are still unknown and the trajectory optimizer can't be (still) used.
% To run the drone model, which needs as input vector the accelerations,
% somehow we need to generate these, and the DWA approach to simulate a
% possible state evolution in this FHOCP seemed the most sensible choice.
% Then, once all the drones generated their chosen waypoint these are
% evaluated in another cost function, which aims to maximize the mean distance
% between drones 

maxIterations = 100; % maximum number of iterations for the optimization process
tolerance = 1e-3; % convergence tolerance for the optimization

r_FOV= 4; % range of vision of a single drone, 360 degrees [m]
safeDist= 0.6; % inflation of the drone size to ensure that no collisions occure between drones and obstacles [m]
N= 20; % horizon of the FHOCP
Ts= 1e-2; % sampling time
n= 6; % system's order (drone model)

Z_i= zeros(n, N); % state vector of one drone
a_max= 3; % upper bound for the acceleration [m/s^2]
a_min= -3; % lower bound for the acceleration [m/s^2]
v_max= 15; % [m/s]
v_min= 0;  % [m/s]
bounds= [v_max, v_min, a_max, a_min];

seen_points= [];