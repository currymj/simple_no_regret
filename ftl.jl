rps_payoff = [0.0 -1 1;1 0.0 -1;-1 1 0.0]

function empiricalstrategy(movehist)
    movecounts = [0.0,0.0,0.0]
    for i in movehist
        movecounts[i] += 1
    end
    movecounts / sum(movecounts)
end

bestresponse(matrix, oppstrategy) = argmax(matrix*oppstrategy)


function fictplay(matrix, xmoves, ymoves; n_iters=1000)
    # brown's fictitious play (1951)
    for i=1:n_iters
        xmove = bestresponse(matrix, empiricalstrategy(ymoves))
        push!(xmoves, xmove)
        ymove = bestresponse(-matrix', empiricalstrategy(xmoves))
        push!(ymoves, ymove)
    end
    (empiricalstrategy(xmoves), empiricalstrategy(ymoves))
end


ftl(total_regrets) = argmax(-total_regrets)

function ftl_fictplay(matrix; n_iters=1000)
    x_observed_regrets = [0.0,0.0,0.0]
    y_observed_regrets = [0.0,0.0,0.0]
    x_emp_strategy = [0.0,0.0,0.0]
    y_emp_strategy = [0.0,0.0,0.0]
    for i=1:n_iters
        xmove = ftl(x_observed_regrets)
        x_emp_strategy[xmove] += 1
        y_observed_regrets += rps_payoff[xmove,:]
        ymove = ftl(y_observed_regrets)
        y_emp_strategy[ymove] += 1
        x_observed_regrets += -rps_payoff[:,ymove]
    end
    (x_emp_strategy / sum(x_emp_strategy), y_emp_strategy / sum(y_emp_strategy))
end
