% clear
% clc
% close all hidden
%
% addpath('../frap_matlab/');
%
% random_seed = 1;%sum( 1e6 * clock() );
% random_stream = RandStream('mt19937ar', 'Seed', random_seed);
% RandStream.setGlobalStream(random_stream);
%
% %% Parallel pool.
%
% delete(gcp('nocreate'));
% local_cluster = parcluster();
% number_of_workers = 44;
% local_cluster.NumWorkers = number_of_workers;
% local_pool = parpool(local_cluster, number_of_workers);

%% Experimental and simulation parameters.
function success = generate_batch_for_python(noise_upper_bound, batch_size, directory, number_of_workers)

exp_sim_param = struct();

exp_sim_param.pixel_size = 7.5e-07; % m
exp_sim_param.number_of_pixels = 256;

exp_sim_param.number_of_prebleach_frames = 10;
exp_sim_param.number_of_bleach_frames = 1;
exp_sim_param.number_of_postbleach_frames = 100;
exp_sim_param.delta_t = 0.2; % s

exp_sim_param.number_of_pad_pixels = 128;

exp_sim_param.bleach_region.shape = "circle";
exp_sim_param.bleach_region.x = 128; % pixels
exp_sim_param.bleach_region.y = 128; % pixels
exp_sim_param.bleach_region.r = 15e-6 / exp_sim_param.pixel_size; % pixels

%% System parameters.

mobile_fraction = 1.0; % dimensionless
beta = 1.0; % a.u. imaging bleach factor
gamma = 0; % bleach profile spread.
b = 0;

%% Data set sizes.
% Set batch size
number_of_simulations = batch_size; %2^5;

lb_D = 5e-12;
ub_D = 5e-10;

lb_a = 0;
ub_a = noise_upper_bound;

%% Generate data.
newDir = directory+"/data";
[status, msg, msgID] = rmdir(newDir, 's');
[status, msg, msgID] = mkdir(newDir);

number_of_simulations_per_worker = double(floor(number_of_simulations/number_of_workers)) * ones(number_of_workers, 1);
number_of_simulations_remaining = number_of_simulations - sum(number_of_simulations_per_worker);
number_of_simulations_per_worker(1:number_of_simulations_remaining) = number_of_simulations_per_worker(1:number_of_simulations_remaining) + 1;

t_start = tic();
x = cell(1, 1, 1, number_of_workers);
y = cell(1, number_of_workers);
first = cell(1, number_of_workers);
second = cell(1, number_of_workers);
varval    = cell(1, number_of_workers);
parfor current_worker = 1:number_of_workers
    
    random_seed = 1;%sum( 1e6 * clock() );
    random_stream = RandStream('mt19937ar', 'Seed', random_seed);
    RandStream.setGlobalStream(random_stream);
    
    x{current_worker} = nan(exp_sim_param.number_of_pixels, exp_sim_param.number_of_pixels, exp_sim_param.number_of_prebleach_frames + exp_sim_param.number_of_postbleach_frames, number_of_simulations_per_worker(current_worker));
    y{current_worker} = nan(3, number_of_simulations_per_worker(current_worker));
    
    first{current_worker}   = nan(exp_sim_param.number_of_prebleach_frames + exp_sim_param.number_of_postbleach_frames, number_of_simulations_per_worker(current_worker)); 
    second{current_worker}  = nan(exp_sim_param.number_of_prebleach_frames + exp_sim_param.number_of_postbleach_frames, number_of_simulations_per_worker(current_worker)); 
    varval{current_worker}     = nan(exp_sim_param.number_of_prebleach_frames + exp_sim_param.number_of_postbleach_frames, number_of_simulations_per_worker(current_worker)); 

    for current_simulation = 1:number_of_simulations_per_worker(current_worker)
        %([i current_worker current_simulation])
        
        % Simulate data.
        log10D_SI = log10(lb_D) + (log10(ub_D) - log10(lb_D)) * rand();
        D_SI = 10^log10D_SI;
        C0 = 0.5 + 0.5 * rand(); % a.u. original concentration
        alpha = 0.5 + 0.35 * rand(); % a.u. bleach factor
        
        D = D_SI / exp_sim_param.pixel_size^2; % pixels^2 / s
        a = lb_a + (ub_a - lb_a) * rand();
        
        
        sys_param = [D, mobile_fraction, C0, alpha, beta, gamma];
        [C_prebleach, C_postbleach] = signal_d(sys_param, exp_sim_param);
        C_prebleach = C_prebleach + sqrt(a) * randn(size(C_prebleach));
        C_postbleach = C_postbleach + sqrt(a) * randn(size(C_postbleach));
        
        [first_moment, second_moment, variance] = moments(C_prebleach, C_postbleach, exp_sim_param);
        
        % Store data.
        %         [rc_prebleach, rc_postbleach] = recovery_curve(C_prebleach, C_postbleach, exp_sim_param);
        
        x{current_worker}(:, :, :, current_simulation) = cat(3, C_prebleach, C_postbleach);
        y{current_worker}(:, current_simulation) = [log10(D) ; C0 ; alpha];
        
        first{current_worker}(:, current_simulation) = first_moment;
        second{current_worker}(:, current_simulation) = second_moment;
        varval{current_worker}(:, current_simulation) = variance;
    end
end




x       = cell2mat(x);
y       = cell2mat(y);
first   = cell2mat(first);
second  = cell2mat(second);
varval  = cell2mat(varval);

x = reshape(x, numel(x), 1);
y = reshape(y, numel(y), 1);
first = reshape(first, numel(first), 1);
second = reshape(second, numel(second), 1);
varval = reshape(varval, numel(varval), 1);

file_id = fopen([newDir+"/x.bin"], 'w');
fwrite(file_id, x, 'float32');
fclose(file_id);

file_id = fopen([newDir+"/y.bin"], 'w');
fwrite(file_id, y, 'float32');
fclose(file_id);

file_id = fopen([newDir+"/rcs.bin"], 'w');
fwrite(file_id, first, 'float32');
fclose(file_id);

file_id = fopen([newDir+"/second.bin"], 'w');
fwrite(file_id, second, 'float32');
fclose(file_id);

file_id = fopen([newDir+"/var.bin"], 'w');
fwrite(file_id, varval, 'float32');
fclose(file_id);

t_exec = toc(t_start);



success = 1;
end