function x_n = mapNonLinear(x,d)
% Inputs:
% x - a single column vector (N x 1)
% d - integer (>= 0)
% Outputs:
% x_n - (N x (d+1))

%Adding 1st column as 1 for d = 0.
x_n = ones(size(x,1),1);

for i = 1:d
    mat = x.^i;
    x_n = [x_n mat]; 
end

end
