function [H, c, Aineq, bineq, Aeq, beq, lb, ub] = QP_mats(A, B, Q, R, N, x0, Xref, a_min, a_max, v_min, v_max, pos_min, pos_max)
% Constructs QP matrices for linear MPC/FHOCP:
% min 0.5*U'*H*U + f'*U
% s.t. Aineq * U <= bineq  (we use form <=)
%      Aeq * U = beq
% U stacked vector of size m*N
%
% Inputs:
%  - A,B : discrete model
%  - Q,R : state and input weights (Q is n x n, R is m x m)
%  - N   : prediction horizon
%  - x0  : current state (n x 1)
%  - Xref: desired states along horizon as stack [x_ref(0); ... ; x_ref(N)] (n*(N+1) x 1)
%  - a_min,a_max: mx1
%  - v_min,v_max: velocities bounds in form [vx_min; vy_min; vz_min] and max (3x1)
%  - pos_min,pos_max: position bounds (3x1) min and max
%
% Outputs: H,f,Aineq,bineq,Aeq,beq,lb,ub

n           = size(A, 1);
m           = size(B, 2);

% Build Prediction Matrices: 
[Lambda, Gamma] = pred_mats(A, B, N);

% Build block diagonal Qbar and Rbar
Qbar = kron(eye(N+1), Q);                                                  % (n*(N+1)) x (n*(N+1))
Rbar = kron(eye(N), R);                                                    % (m*N) x (m*N)

% Cost matrices H and f (from derivation)
H = (Gamma' * Qbar * Gamma) + Rbar;
c = (Gamma' * Qbar * (Lambda * x0 - Xref));                              % linear term (QP expects 1/2 x'Hx + c'x)


% Define equality constraints
Aeq = []; 
beq = [];

% Inequality constraints: box on inputs U (ax, ay, az)
lb = repmat(a_min, N, 1);                                                  % input lower bound constraint
ub = repmat(a_max, N, 1);                                                  % input upper bound constraint
    
% linear inequalities for velocity & position

% number of constraints per each time step:
% 3 (pos<=max) + 3 (-pos<=-min) + 3 (vel<=max) + 3 (-vel<=-min) = 12
nc_per_step = 12;
num_constraints = nc_per_step * (N+1);

% Define Inequality Constraints: 
Aineq = zeros(num_constraints, m*N);
bineq = zeros(num_constraints, 1);

row = 1; % row counter for inserting constraints

C_pos = [eye(3), zeros(3, n-3)];   % posizione
C_vel = [zeros(3,3), eye(3)];      % velocità

for k = 0:N
    
    % get correct block indices for Z_k
    Lambda_k = Lambda(k*n+1:(k+1)*n, :);
    Gamma_k  = Gamma(k*n+1:(k+1)*n, :);

    % ---- posizione: pos <= pos_max ----
    Aineq(row:row+2, :) = C_pos * Gamma_k;
    bineq(row:row+2)    = pos_max - C_pos * (Lambda_k*x0);
    row = row + 3;

    % ---- posizione: pos >= pos_min  ( -pos <= -pos_min ) ----
    Aineq(row:row+2, :) = -C_pos * Gamma_k;
    bineq(row:row+2)    = -pos_min + C_pos * (Lambda_k*x0);
    row = row + 3;

    % ---- velocità: vel <= v_max ----
    Aineq(row:row+2, :) = C_vel * Gamma_k;
    bineq(row:row+2)    = v_max - C_vel * (Lambda_k*x0);
    row = row + 3;

    % ---- velocità: vel >= v_min  ( -vel <= -v_min ) ----
    Aineq(row:row+2, :) = -C_vel * Gamma_k;
    bineq(row:row+2)    = -v_min + C_vel * (Lambda_k*x0);
    row = row + 3;

end

end