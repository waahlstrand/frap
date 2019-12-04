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
function success = generate_batch(noise_upper_bound, batch_size, directory, number_of_workers, augment, spatiotemporal)

% Assert no already started cluster
% delete(gcp('nocreate'));

% Start local cluster
%local_cluster = parcluster();

% Choose number of workers
%number_of_workers = 32;
%local_cluster.NumWorkers = number_of_workers;

% Create pool of workers on cluster
%local_pool = parpool(local_cluster, number_of_workers);


%%%% NUMBER OF PIXELS AND SIZE %%%%%%%%%%%
pixel_size = 7.5e-07; % m
number_of_pixels = 256; %256


%%%% SAMPLING SIZE AND RESOLUTION %%%%%%%%
number_of_prebleach_frames = 10;
number_of_bleach_frames = 4;
number_of_postbleach_frames = 100;
delta_t = 0.2; % s
number_of_pad_pixels = 128; %128

%%%% BLEACH REGION, SIZE AND POSITION %%%%%
shape = "circle";

displacement = 0;%10; % pixels
%exp_sim_param.bleach_region.x = 128 + randi([-displacement displacement],1,1); % pixels 128
%exp_sim_param.bleach_region.y = 128 + randi([-displacement displacement],1,1); % pixels 128

size_var = 0;%0.20; % fraction (20 %)
%exp_sim_param.bleach_region.r = unirnd(1-size_var, 1+size_var, 1)*(15e-6 / exp_sim_param.pixel_size); % pixels 15e-6 / exp_sim_param.pixel_size


%% System parameters.

mobile_fraction = 1.0; % dimensionless
beta            = 1.0; % a.u. imaging bleach factor
gamma           = 0; % bleach profile spread.

lb_b            = 0.01;
ub_b            = noise_upper_bound;

lb_a = 0.01;
ub_a = noise_upper_bound;

%% Data set sizes.
% Set batch size
number_of_simulations = batch_size; %2^5;

%% Inference parameters
lb_c = 0.5;
ub_c = 1;

lb_D = 1e-12;
ub_D = 1e-9;

lb_alpha = 0.5;
ub_alpha = 0.95;

%% Generate data.
newDir = directory+"/data";
if ~augment
    [status, msg, msgID] = rmdir(newDir, 's');
    [status, msg, msgID] = mkdir(newDir);
    suffix = "";
    
else
    
    suffix = "temp";
    
end

% Split the simulations of the parallel workers
number_of_simulations_per_worker = double(floor(number_of_simulations/number_of_workers)) * ones(number_of_workers, 1);
number_of_simulations_remaining = number_of_simulations - sum(number_of_simulations_per_worker);
number_of_simulations_per_worker(1:number_of_simulations_remaining) = number_of_simulations_per_worker(1:number_of_simulations_remaining) + 1;

t_start = tic();

parfor current_worker = 1:number_of_workers
    
    random_seed = sum( 1e6 * clock() );
    random_stream = RandStream('mt19937ar', 'Seed', random_seed);
    RandStream.setGlobalStream(random_stream);
    
    
    for current_simulation = 1:number_of_simulations_per_worker(current_worker)
        
        
        exp_sim_param = struct();
        
        %%%% NUMBER OF PIXELS AND SIZE %%%%%%%%%%%
        exp_sim_param.pixel_size =pixel_size; % m
        exp_sim_param.number_of_pixels = number_of_pixels; %256
        
        
        %%%% SAMPLING SIZE AND RESOLUTION %%%%%%%%
        exp_sim_param.number_of_prebleach_frames = number_of_prebleach_frames;
        exp_sim_param.number_of_bleach_frames = number_of_bleach_frames;
        exp_sim_param.number_of_postbleach_frames = number_of_postbleach_frames;
        exp_sim_param.delta_t = delta_t; % s
        exp_sim_param.number_of_pad_pixels = number_of_pad_pixels; %128
        
        %%%% BLEACH REGION, SIZE AND POSITION %%%%%
        exp_sim_param.bleach_region.shape = shape;
        
        %%%%%% VARY THE SIZE AND DISPLACEMENT OF ROI %%%%%%%%%%
        exp_sim_param.bleach_region.x = 128 + randi([-displacement displacement],1,1); % pixels 128
        exp_sim_param.bleach_region.y = 128 + randi([-displacement displacement],1,1); % pixels 128
        exp_sim_param.bleach_region.r = unirnd(1-size_var, 1+size_var, 1)*(15e-6 / exp_sim_param.pixel_size); % pixels 15e-6 / exp_sim_param.pixel_size
        
        %%%%%% DIFFUSION PARAMETER %%%%%%%%%%
        D_SI = logunirnd(lb_D, ub_D, 1);
        %D_SI      = 10^log10D_SI;
        D         = D_SI / exp_sim_param.pixel_size^2; % pixels^2 / s
        
        %%%%%% INITIAL CONCENTRATION %%%%%%%%
        C0      = unirnd(lb_c, ub_c, 1)    % a.u. original concentration
        
        %%%%%% BLEACH STRENGTH %%%%%%%%%%%%%%
        alpha   = unirnd(lb_alpha, ub_alpha, 1); % a.u. bleach factor
        
        % Amount of Gaussian noise
        a       = logunirnd(lb_a, ub_a, 1);
        b       = logunirnd(lb_b, ub_b, 1);
        
        % Simulate data signal.
        sys_param = [D, mobile_fraction, C0, alpha, beta, gamma];
        [C_prebleach, C_postbleach] = signal_d(sys_param, exp_sim_param);
        
        % Augment with Gaussian noise
        noise_prebleach = a + b*C_prebleach
        C_prebleach = C_prebleach + sqrt(noise_prebleach).* randn(size(C_prebleach));
        noise_postbleach = a + b*C_postbleach
        C_postbleach = C_postbleach + sqrt(noise_postbleach) .* randn(size(C_postbleach));
        
        
        
        % Save the data to multiple individual files
        if spatiotemporal
            
            x_to_save = cat(3, C_prebleach, C_postbleach);
            y_to_save = [log10(D) ; C0 ; alpha];
            
            
            
        else
            
            [rc_prebleach, rc_postbleach] = recovery_curve(C_prebleach, C_postbleach, exp_sim_param);
            x_to_save = cat(2, rc_prebleach', rc_postbleach');
            y_to_save = [log10(D) ; C0 ; alpha];
            
            
            
        end
        
        x_to_save = reshape(x_to_save, numel(x_to_save), 1);
        y_to_save = reshape(y_to_save, numel(y_to_save), 1);
        
        
        file_id1 = fopen(directory+"/data/x" + suffix +"_"+string(current_worker)+"."+string(current_simulation)+".bin", 'w');
        fwrite(file_id1, x_to_save, 'float32');
        fclose(file_id1);
        
        file_id2 = fopen(directory+"/data/y" + suffix +"_"+string(current_worker)+"."+string(current_simulation)+".bin", 'w');
        fwrite(file_id2, y_to_save, 'float32');
        fclose(file_id2);
        
        
    end
end

t_exec = toc(t_start);

success = 1;
end