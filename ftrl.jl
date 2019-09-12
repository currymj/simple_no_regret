rps_payoff = [0.0 -1 1;1 0.0 -1;-1 1 0.0]

function hedge_fictplay(matrix, initial_x, initial_y; n_iters=1000, η=0.01)
    current_x = copy(initial_x)
    current_y = copy(initial_y)
    cumulative_x = copy(initial_x)
    cumulative_y = copy(initial_y)

    for i=1:n_iters
        x_update = exp.(η*matrix*current_y)
        current_x .= current_x .* (x_update / sum(current_x .* x_update))
        cumulative_x += current_x
        y_update = exp.(-η*matrix'*current_x)
        current_y .= current_y .* (y_update / sum(current_y .* y_update))
        cumulative_y += current_y
    end

    # must use average iterates
    (cumulative_x / n_iters, cumulative_y / n_iters)
end

function optimistic_hedge_fictplay(matrix, initial_x, initial_y; n_iters=1000, η=0.01)
    current_x = copy(initial_x)
    current_y = copy(initial_y)
    prev_x = copy(initial_x)
    prev_y = copy(initial_y)
    cumulative_x = copy(initial_x)
    cumulative_y = copy(initial_y)

    for i=1:n_iters
        x_update = exp.(2*η*matrix*current_y - η*matrix*prev_y)
        prev_x .= current_x
        current_x .= current_x .* (x_update / sum(current_x .* x_update))
        cumulative_x += current_x
        y_update = exp.(-2*η*matrix'*current_x + η*matrix'*prev_x)
        prev_y .= current_y
        current_y .= current_y .* (y_update / sum(current_y .* y_update))
        cumulative_y += current_y
    end

    # with optimism, last iterates converge!
    current_x, current_y
end
