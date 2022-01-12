function [x, it, resvec] = kac(A, b, tol, maxit)
    % KAC: Kaczmarz solver
    %
    % [x, it, resvec] = kac(A, b, tol, maxit)
    
    [m, n] = size(A);

    if length(b) ~= m
        error('Invalid b');
    end

    % initialize x arbitrarily
    x = rand(n, 1);

    % initialize residual vector
    resvec = zeros(maxit + 1, 1);
    resvec(1) = norm(b - A * x);

    % iterate
    for it = 1:maxit
        r = mod(it, m) + 1; % select row
        x = x + A(r, :)' * (b(r) - A(r, :) * x) / (norm(A(r, :)) ^ 2);

        % calculate residual
        res = norm(b - A * x);
        resvec(it + 1) = res;

        if res < tol
            resvec = resvec(1:it + 1);
            return
        end
    end
end
