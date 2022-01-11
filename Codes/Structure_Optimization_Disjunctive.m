% BRB Structure Optimization (Disjunctive Approach)

% Finding Optimal Number of Reference Values 
% Number of Rules Depends on Number of Reference Values.
% So, Finding optimal Number of Reference Values will lead to Optimal Number of Rules

% Training Dataset
M = 10; % No of Inputs-Output Pair
T = 3; % No of Attributes
train_input = [0.8 0.6 0.2;
    0.8 0.2 0.5;
    0.4 0.4 0.3;
    0.2 0.8 0.7;
    0.4 0.0 0.2;
    1.0 0.1 0.7;
    0.1 0.4 0.3;
    0.5 0.6 0.2;
    0.3 0.9 0.2;
    0.4 0.5 0.2];

train_output = [0.9;
    0.7;
    0.5;
    0.4;
    0.3;
    0.8;
    0.2;
    0.7;
    0.7;
    0.6];

no_of_solution_candidate_considered = 6; % Number of solution candidate
solution_candidates = 3:(3 + no_of_solution_candidate_considered - 1); % Solution Candidates

seed = rng; % Preserve seed of random generator, so that we can generate the same belief degree later

% Initialize Variables
calculated_output = zeros(M,1);
differences = zeros(M,1);
result = zeros(no_of_solution_candidate_considered, 2);

for x = 1:no_of_solution_candidate_considered 
    
    N = solution_candidates(1,x); % No of referential values
    L = N; % Number of rules (Because, We Are Considering Disjunctive Inference)
    
    % Referential Values
    ref_val = generate_ref_val(N,1,0);
    
    % Initial Belief Degree
    belief_degree = generate_belief_degree(N,L,seed);  % Initial Belief Degrees of size (N,L)
    
    % Inference Methodology
    for i = 1:M
        % Get Rule Weights
        weights = get_rule_weights(train_input,i,T,N,ref_val); 

        % Calculate Aggregated Belief Degree and Compute Y
        aggregated_belief_degree = calc_aggregated_belief_degree(weights, belief_degree, N, L);

        % Calculate Difference Between Outputs
        calculated_output(i,1) = calculateY(aggregated_belief_degree,ref_val,N);
        differences(i,1) = abs(calculated_output(i,1) - train_output(i,1));
    end
    rms_error = sum((differences).^2) / M;
    result(x,:) = [N, rms_error];   % Number of ref val - total_difference result pair
end

disp("Number of Reference Value - Error Pair:")
disp(result)

optimal_ref_val_number = find_optimal_ref_val_number(result, no_of_solution_candidate_considered)
disp("Optimized Belief Degrees (Left to Right One Row, Represents For One Rule)")
belief_degree_used = generate_belief_degree(optimal_ref_val_number, optimal_ref_val_number,seed)'


function arr = generate_ref_val(no_of_ref_val, upper, lower)
    arr = zeros(1, no_of_ref_val);
    value_difference = (upper - lower)/(no_of_ref_val - 1);
    for i = 1:no_of_ref_val
        if i == 1
            arr(1,i) = upper;
        elseif i == no_of_ref_val
            arr(1,i) = lower;
        else
            arr(1,i) = (upper - (value_difference * (i-1)));
        end
         
    end
end

function arr = generate_belief_degree(N, L,seed)
    rng(seed);
    belief_generator = rand(N,L);
    temp_gen_col_total = zeros(L,1);
    arr = zeros(N,L);
    for col = 1:L
        for row = 1:N
            temp_gen_col_total(col,1) = temp_gen_col_total(col,1) + belief_generator(row, col);
        end
    end 

    for row = 1:N
        for col = 1:L
            arr(row,col)  = belief_generator(row,col) ./ temp_gen_col_total(col,1);
        end 
    end
end

function arr = get_rule_weights(train_input,input_no,no_of_attributes,no_of_ref_val, ref_vals)
    % Input Transformation
    transformed_input = transform_input(train_input(input_no,:),no_of_attributes, no_of_ref_val, ref_vals);
    
    % Rule Activation Weight Calculation    
    matching_degrees = calc_matching_degrees(transformed_input, no_of_attributes, no_of_ref_val); % Calculate Matching Degree    
    combined_matching_degree = calc_combined_matching_degrees(matching_degrees,no_of_ref_val); % Calculate Combined Matching Degree    
    arr = (matching_degrees) ./ (combined_matching_degree); % Calculate Activation Weight
end

function arr = transform_input (input,no_of_attr,no_of_ref_val,ref_vals)
    arr = zeros(no_of_attr,no_of_ref_val); % Initialize with row_number x column_number dummy values
    % Calculate and Populate with original values
    for i = 1:no_of_attr
        for j = 1:(no_of_ref_val - 1)
            if (input(1,i)>= ref_vals(1,(j+1)) && input(1,i) <= ref_vals(1,j))
              arr(i,(j+1)) = (ref_vals(1,j) - input(1,i))/(ref_vals(1,j) - ref_vals(1,j+1));  
              arr(i,j) = 1 - arr(i,(j+1));
            end
        end
    end
end

function arr = calc_matching_degrees(individual_matching_degree, no_of_attributes, no_of_ref_val)
    arr = zeros(no_of_ref_val,1);
    for i = 1:no_of_ref_val
        for j = 1:no_of_attributes
            arr(i,1) = arr(i,1) + individual_matching_degree(j,i);
        end
        
    end
end

function val = calc_combined_matching_degrees(matching_degrees, no_of_rules)
    val = 0;
    for i = 1:no_of_rules
       val = val + matching_degrees(i,1);        
    end
end

function arr = calc_aggregated_belief_degree(activation_weight, belief_degree, no_of_ref_val, no_of_rules)
    arr = zeros(no_of_ref_val,1);

    partA = calc_Part_A(activation_weight, belief_degree, no_of_ref_val, no_of_rules);
    partB = calc_Part_B(activation_weight, belief_degree, no_of_ref_val, no_of_rules);
    partC = calc_Part_C(activation_weight, no_of_rules);
    
    combined_partA = 0;
    for i = 1:no_of_ref_val
        combined_partA = combined_partA + partA(i,1);
    end
    
    for j = 1:no_of_ref_val
        arr(j,1) = (partA(j,1) - partB)/((combined_partA - ((no_of_ref_val - 1) * partB)) - partC);
    end    
end

function arr = calc_Part_A(activation_weight, belief_degree, no_of_ref_val, no_of_rules)
    arr = ones(no_of_ref_val,1);
    for i = 1:no_of_ref_val
        for j = 1:no_of_rules
            part1 = activation_weight(j,1) * belief_degree(i,j);
            
            temp = 0;
            for k = 1:no_of_ref_val
                temp = temp + belief_degree(k,j);
            end
            part2 = (1 - (activation_weight(j,1)*temp));
            
            temp_val = part1 + part2;
            arr(i,1) = arr(i,1) * temp_val;
        end
    end
end

function val = calc_Part_B(activation_weight, belief_degree, no_of_ref_val, no_of_rules)
    val = 1;
    for i = 1:no_of_rules
        
        temp_total_belief = 0;
        
        for j = 1:no_of_ref_val
            temp_total_belief = temp_total_belief + belief_degree(j,i);
        end
        
        temp = activation_weight(i,1) * temp_total_belief;
        val = val * (1 - temp);
    end
end

function val = calc_Part_C(activation_weight, no_of_rules)
    val = 1;
    for i = 1:no_of_rules
        val = val * (1 - activation_weight(i,1));
    end
end

function val = calculateY(agg_bel_val, ref_vals,no_ref_val)
    val = 0;
    for i = 1: no_ref_val
        val = val + (agg_bel_val(i,1)*ref_vals(1,i));
    end
end

function no_of_ref_val = find_optimal_ref_val_number(result_pair_array, no_of_solution_candidate_considered)
    value = result_pair_array(1,2);
    ref_no = result_pair_array(1,1);
    for i = 2:no_of_solution_candidate_considered
        if result_pair_array(i,2) <= value
            ref_no = result_pair_array(i,1);
            value = result_pair_array(i,2);
        end
    end
    no_of_ref_val = ref_no;
end
