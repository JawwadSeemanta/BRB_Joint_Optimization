% Initial Inputs
x0 = [0.6235    0.1584    0.2181;
0.2039    0.3235    0.4726;
0.1606    0.3051    0.5343;
0.1574    0.7454    0.0972;
0.3713    0.2194    0.4093;
0.5104    0.4878    0.0018;
0.3766    0.4769    0.1465;
0.3867    0.4327    0.1806;
0.0834    0.4079    0.5087;
0.5652    0.1170    0.3178;
0.3123    0.2375    0.4502;
0.1775    0.5747    0.2478;
0.1426    0.0863    0.7711;
0.3528    0.3608    0.2865;
0.2894    0.2553    0.4554;
0.4265    0.1663    0.4072;
0.0904    0.5010    0.4086;
0.5469    0.3846    0.0685;
0.3901    0.5606    0.0493;
0.0171    0.2332    0.7497;
0.2908    0.4952    0.2139;
0.3826    0.2313    0.3861;
0.1011    0.2086    0.6903;
0.4837    0.0019    0.5144;
0.3578    0.4396    0.2026;
0.1498    0.4248    0.4254;
0.1485    0.8031    0.0484]';

% Bound Constraints
lb = zeros(3,3^3); % Lower Bound of size (N,L)
ub = ones(3,3^3); % Upper Bound of size (N,L)

% Equality Constraints
% Equality Constraints are taken as row vector "Aeq" sized m*n. 
% m = number of equality constraints
% n = number of elements in x0/solution
% Solver converts x0/solution into x0(:)/solution(:) to impose constraints
% beq is a column vector of m elements
aeq = zeros(3^3,3*(3^3)); % of size (L,L*N)
for i = 1:3^3
    temp1 = (i-1)*3 + 1; % (i-1)*N + 1
    temp2 = (i-1)*3 + 3; % (i-1)*N + N
    aeq(i,temp1:temp2) = ones(1,3); % ones(1,N)
end
beq = ones(3^3,1); % of size (L,1)


% Set nondefault solver options
options = optimoptions('fmincon','PlotFcn','optimplotfvalconstr');

% Solve
[solution,objectiveValue] = fmincon(@objectiveFcn,x0,[],[],aeq,beq,lb,ub,[],...
    options);

% Clear variables
clearvars options

% Display Optimized Values
disp("Least Mean Square Error = ") 
disp(objectiveValue)
disp("Optimized Belief Degrees (Left to Right One Row, Represents For One Rule) = ") 
disp(solution')

function f = objectiveFcn(optimInput)
% Training Dataset

M = 10; % No of Inputs-Output Pair
T = 3; % No of Attributes
N = 3; % No of referencial values
L = N^T; % Number of rules (Disjunctive Assumption)
train_input = [0.8 0.6 0.4;
    0.8 0.2 0.5;
    0.4 0.4 0.3;
    0.2 0.8 0.7;
    0.4 0.0 0.2;
    1.0 0.1 0.7;
    0.8 0.4 0.3;
    0.5 0.6 0.2;
    0.3 0.9 0.2;
    0.4 0.5 0.2];

train_output = [1.0;
    0.7;
    0.5;
    0.4;
    0.3;
    0.8;
    0.9;
    0.7;
    0.7;
    0.6];


% Define the variables
belief_degrees = optimInput;

ref_val = generate_ref_val(N,1,0);

calculated_output = zeros(M,1);
differences = zeros(M,1);

for i = 1:M
    weights = get_rule_weights(train_input,i,T,N,ref_val); % Rule Weights

    % Calculate Aggregated Belief Degree and Compute Y
    aggregated_belief_degree = calc_aggregated_belief_degree(weights, belief_degrees, N, L);

    calculated_output(i,1) = calculateY(aggregated_belief_degree,ref_val,N);
    differences(i,1) = abs(calculated_output(i,1) - train_output(i,1));
end


% Define Objective Function
f = sum((differences).^2) / M;
end

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

function arr = generate_belief_degree(N, L)
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
    combined_matching_degree = calc_combined_matching_degrees(matching_degrees,no_of_ref_val^no_of_attributes); % Calculate Combined Matching Degree    
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
    no_of_matching_degree = no_of_ref_val^no_of_attributes;
    arr = ones(no_of_matching_degree, 1);
    counter = 0;
    for i = 1:no_of_ref_val
        for j = 1:no_of_ref_val
            for k = 1:no_of_ref_val
                counter = counter + 1;
                arr(counter,1) = individual_matching_degree(1,i) * individual_matching_degree(2,j) * individual_matching_degree(3,k); 
            end
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