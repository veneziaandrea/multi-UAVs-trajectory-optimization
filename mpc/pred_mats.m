function [Lambda, Gamma] = pred_mats(N, A, B)
% Builds prediction matrices:
% Z = Lambda * x0 + Gamma * U
% Z = [x(0); x(1); ...; x(N)] (stack of states)
% U = [u(0); ...; u(N-1)]

n = size(A,1);
m = size(B,2);

% Initialization: 
Lambda = zeros(n*(N+1), n);
Gamma  = zeros(n*(N+1), m*N);

for k = 0:N
    Lambda(n*k+1:n*(k+1), :) = A^k; % Fill Lambda with powers of A
    if k > 0
        Gamma(n*k+1:n*(k+1), (k-1)*m+1:k*m) = B; % Fill Gamma with B
    end
end

end
