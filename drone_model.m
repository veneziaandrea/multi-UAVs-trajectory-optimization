function [z] = drone_model(dt, u)
% Discrete 3D-model of a single drone. 
% UAV is modeled as a point-mass flying in a pre-determined 3D space. 
% The mass of the single drone is assumed to be constant. 
% The model consists in a integrator whith time step dt.

% STATE: 

% x_i,k      x-coordinate of position of vehicle i at time step k           [m]
% y_i,k      y-coordinate of position of vehicle i at time step k           [m]
% z_i,k      z-coordinate of position of vehicle i at time step k           [m]

% vx_i,k    x-component of velocity vector of vehicle i at time step k      [m/s]
% vy_i,k    y-component of velocity vector of vehicle i at time step k      [m/s]
% vz_i,k    z-component of velocity vector of vehicle i at time step k      [m/s]

% INPUT: 

% dt        time step lenght                                                [s]
% u_i: 
% ax_i,k    x-component of acceleration vector of vehicle i at time step k  [m/s^2]
% ax_i,k    y-component of acceleration vector of vehicle i at time step k  [m/s^2]
% ax_i,k    z-component of acceleration vector of vehicle i at time step k  [m/s^2]



%% INITIALIZATION: 

% Initialize state variables
z           = zeros(6, 1);                                                 % [x_i; y_i; z_i; vx_i; vy_i; vz_i]

%Initaialize state matrix A and input matix B
A           = eye(6);
B           = zeros(3, 6); 

% Define the state transition matrix A and input matrix B
A(1:3, 4:6) = eye(3);                                                      % Assigning identity matrix for velocity to position
B(:, 1:3)   = dt * eye(3);                                                 % Assigning identity matrix for acceleration inputs


%% MODEL UPDATE: 
% pos_kplus1 = pos_k + dt * vel_k + 0 * acc_k
% vel_kplus1 = 0 * pos_k + vel_k + dt * acc_k

z = A * z + B * u;
end