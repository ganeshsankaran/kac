function [x, it, resvec] = rkac(A, b, p, tol, maxit)
    % RKAC: Randomized Kaczmarz solver
    %
    % [x, it, resvec] = rkac(A, b, p, tol, maxit)

    [m, n] = size(A);

    if length(b) ~= m
        error('Invalid b');
    elseif length(p(:)) ~= m || abs(1 - sum(p)) > 1e-6
        error('Invalid p');
    end

    % initialize x arbitrarily
    x = rand(n, 1);

    % initialize residual vector
    resvec = zeros(maxit + 1, 1);
    resvec(1) = norm(b - A * x);

    % iterate
    for it = 1:maxit
        r = find(cumsum([0; p(:)]) < rand, 1, 'last'); % select row
        x = x + A(r, :) * (b(r) - A(r, :) * x) / (norm(A (r, :)) ^ 2);

        % calculate residual
        res = norm(b - A * x);
        resvec(it + 1) = res;
        
        if res < tol
            resvec = resvec(1:it + 1);
            return
        end
    end
end
